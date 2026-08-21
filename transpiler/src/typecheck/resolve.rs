//! Step 1: Symbol Resolution
//!
//! Walks a `TranslationUnit`, building the `SymbolTable` scope-by-scope
//! (global scope, then a function's params+body scope, then a nested scope
//! per `{ ... }` block/`for` loop) and resolving every ordinary-namespace
//! identifier reference (`Expr::Ident`) against it. `Expr::Member` field
//! names are deliberately *not* looked up here -- resolving a field name
//! needs the base expression's type, which is Step 3's job (type checking),
//! not this step's purely lexical one. Label names (`goto`/`Labeled`) are
//! their own function-local namespace in C89 and are likewise out of scope.
//!
//! Reused across steps: `register_decl_specifiers` and the declarator
//! walkers here are what makes every array-size expression, bit-field
//! width, and enum-constant value get resolved too, not just "loose"
//! statement-level expressions.

use crate::parser::ast::*;
use crate::typecheck::scope::{Symbol, SymbolKind, SymbolTable, Tag, TagKind};

/// An identifier reference that didn't resolve to any declaration visible
/// at its use site. Name-only, matching the rest of this pipeline's current
/// error granularity -- the AST carries no source spans yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedIdent {
    pub name: String,
}

pub struct ResolveResult {
    pub table: SymbolTable,
    pub unresolved: Vec<UnresolvedIdent>,
}

pub fn resolve_translation_unit(unit: &TranslationUnit) -> ResolveResult {
    let mut r = Resolver {
        table: SymbolTable::new(),
        unresolved: Vec::new(),
    };
    for item in &unit.items {
        r.external_decl(item);
    }
    ResolveResult {
        table: r.table,
        unresolved: r.unresolved,
    }
}

struct Resolver {
    table: SymbolTable,
    unresolved: Vec<UnresolvedIdent>,
}

impl Resolver {
    fn external_decl(&mut self, item: &ExternalDecl) {
        match item {
            ExternalDecl::Declaration(decl) => self.declaration(decl),
            ExternalDecl::FunctionDef(f) => self.function_def(f),
        }
    }

    fn function_def(&mut self, f: &FunctionDef) {
        self.decl_specifiers(&f.specifiers);
        if let Some(name) = grammar_names::declarator_name(&f.declarator) {
            self.table.declare(Symbol {
                name,
                kind: SymbolKind::Function,
                storage: f.specifiers.storage,
            });
        }
        // C89: function parameters live in the same scope as the body's
        // own top-level declarations, not a separate nested one.
        self.table.enter_scope();
        self.declare_params_from_declarator(&f.declarator.direct);
        for item in &f.body.items {
            self.block_item(item);
        }
        self.table.exit_scope();
    }

    fn declare_params_from_declarator(&mut self, d: &DirectDeclarator) {
        // The declarator tree for `f(int a, char *b)` is
        // `Function(Ident("f"), params)` -- walk down to the outermost
        // `Function` node (there's only ever one for an actual function
        // definition) and declare its parameters into the current scope.
        if let DirectDeclarator::Function(_, params) = d {
            self.declare_params(params);
        }
    }

    fn declare_params(&mut self, params: &ParamList) {
        for p in &params.params {
            self.decl_specifiers(&p.specifiers);
            if let ParamDeclarator::Named(declarator) = &p.declarator {
                self.declarator_arrays(declarator);
                if let Some(name) = grammar_names::declarator_name(declarator) {
                    self.table.declare(Symbol {
                        name,
                        kind: SymbolKind::Parameter,
                        storage: None,
                    });
                }
            } else if let ParamDeclarator::Abstract(ad) = &p.declarator {
                self.abstract_declarator_arrays(ad);
            }
        }
    }

    fn block_item(&mut self, item: &BlockItem) {
        match item {
            BlockItem::Decl(decl) => self.declaration(decl),
            BlockItem::Stmt(stmt) => self.stmt(stmt),
        }
    }

    fn declaration(&mut self, decl: &Declaration) {
        self.decl_specifiers(&decl.specifiers);
        let is_typedef = decl.specifiers.storage == Some(StorageClass::Typedef);
        for init_decl in &decl.declarators {
            self.declarator_arrays(&init_decl.declarator);
            if let Some(name) = grammar_names::declarator_name(&init_decl.declarator) {
                self.table.declare(Symbol {
                    name,
                    kind: if is_typedef {
                        SymbolKind::Typedef
                    } else {
                        SymbolKind::Variable
                    },
                    storage: decl.specifiers.storage,
                });
            }
            if let Some(initializer) = &init_decl.initializer {
                self.initializer(initializer);
            }
        }
    }

    fn initializer(&mut self, init: &Initializer) {
        match init {
            Initializer::Expr(e) => self.expr(e),
            Initializer::List(items) => {
                for i in items {
                    self.initializer(i);
                }
            }
        }
    }

    /// Registers any struct/union/enum tag this specifier list declares or
    /// references, recurses into a defining struct/union's fields, and
    /// declares+resolves a defining enum's constants -- all namespace-only
    /// bookkeeping, independent of whether the specifiers sit on a
    /// declaration, a parameter, a cast, or a `sizeof` type name.
    fn decl_specifiers(&mut self, specs: &DeclSpecifiers) {
        for ts in &specs.type_specifiers {
            match ts {
                TypeSpecifier::Struct(spec) => self.struct_or_union(spec, TagKind::Struct),
                TypeSpecifier::Union(spec) => self.struct_or_union(spec, TagKind::Union),
                TypeSpecifier::Enum(spec) => self.enum_spec(spec),
                _ => {}
            }
        }
    }

    fn struct_or_union(&mut self, spec: &StructOrUnionSpec, kind: TagKind) {
        if let Some(name) = &spec.name {
            self.table.declare_tag(Tag {
                name: name.clone(),
                kind,
                defined: spec.fields.is_some(),
            });
        }
        if let Some(fields) = &spec.fields {
            for field in fields {
                self.decl_specifiers(&field.specifiers);
                for (declarator, bitwidth) in &field.declarators {
                    if let Some(d) = declarator {
                        self.declarator_arrays(d);
                    }
                    if let Some(w) = bitwidth {
                        self.expr(w);
                    }
                }
            }
        }
    }

    fn enum_spec(&mut self, spec: &EnumSpec) {
        if let Some(name) = &spec.name {
            self.table.declare_tag(Tag {
                name: name.clone(),
                kind: TagKind::Enum,
                defined: spec.variants.is_some(),
            });
        }
        if let Some(variants) = &spec.variants {
            for (name, value) in variants {
                if let Some(e) = value {
                    self.expr(e);
                }
                self.table.declare(Symbol {
                    name: name.clone(),
                    kind: SymbolKind::EnumConstant,
                    storage: None,
                });
            }
        }
    }

    /// Walks a (named) declarator's array-size expressions and any nested
    /// parameter-list types, without declaring anything from those nested
    /// parameter lists -- a prototype's parameter names don't leak into the
    /// enclosing scope (C89 "prototype scope").
    fn declarator_arrays(&mut self, d: &Declarator) {
        self.direct_declarator_arrays(&d.direct);
    }

    fn direct_declarator_arrays(&mut self, d: &DirectDeclarator) {
        match d {
            DirectDeclarator::Ident(_) => {}
            DirectDeclarator::Paren(inner) => self.declarator_arrays(inner),
            DirectDeclarator::Array(base, size) => {
                self.direct_declarator_arrays(base);
                if let Some(e) = size {
                    self.expr(e);
                }
            }
            DirectDeclarator::Function(base, params) => {
                self.direct_declarator_arrays(base);
                self.walk_param_types_only(params);
            }
        }
    }

    /// Like `declare_params`, but for a parameter list that's part of a
    /// type (a prototype, a cast/sizeof function-pointer type, ...) rather
    /// than an actual function definition being parsed -- resolves what
    /// needs resolving (nested tags, array sizes) without declaring
    /// anything into the enclosing scope.
    fn walk_param_types_only(&mut self, params: &ParamList) {
        for p in &params.params {
            self.decl_specifiers(&p.specifiers);
            match &p.declarator {
                ParamDeclarator::Named(d) => self.declarator_arrays(d),
                ParamDeclarator::Abstract(ad) => self.abstract_declarator_arrays(ad),
                ParamDeclarator::Bare => {}
            }
        }
    }

    fn abstract_declarator_arrays(&mut self, ad: &AbstractDeclarator) {
        if let Some(d) = &ad.direct {
            self.direct_abstract_declarator_arrays(d);
        }
    }

    fn direct_abstract_declarator_arrays(&mut self, d: &DirectAbstractDeclarator) {
        match d {
            DirectAbstractDeclarator::Paren(inner) => self.abstract_declarator_arrays(inner),
            DirectAbstractDeclarator::Array(base, size) => {
                if let Some(b) = base {
                    self.direct_abstract_declarator_arrays(b);
                }
                if let Some(e) = size {
                    self.expr(e);
                }
            }
            DirectAbstractDeclarator::Function(base, params) => {
                if let Some(b) = base {
                    self.direct_abstract_declarator_arrays(b);
                }
                self.walk_param_types_only(params);
            }
        }
    }

    fn type_name(&mut self, t: &TypeName) {
        self.decl_specifiers(&t.specifiers);
        if let Some(ad) = &t.abstract_declarator {
            self.abstract_declarator_arrays(ad);
        }
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr(Some(e)) => self.expr(e),
            Stmt::Expr(None) => {}
            Stmt::Compound(cs) => {
                self.table.enter_scope();
                for item in &cs.items {
                    self.block_item(item);
                }
                self.table.exit_scope();
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.expr(cond);
                self.stmt(then_branch);
                if let Some(e) = else_branch {
                    self.stmt(e);
                }
            }
            Stmt::Switch { cond, body } => {
                self.expr(cond);
                self.stmt(body);
            }
            Stmt::Case { expr, stmt } => {
                self.expr(expr);
                self.stmt(stmt);
            }
            Stmt::Default(stmt) => self.stmt(stmt),
            Stmt::While { cond, body } => {
                self.expr(cond);
                self.stmt(body);
            }
            Stmt::DoWhile { body, cond } => {
                self.stmt(body);
                self.expr(cond);
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
            } => {
                self.table.enter_scope();
                match init {
                    Some(ForInit::Decl(d)) => self.declaration(d),
                    Some(ForInit::Expr(e)) => self.expr(e),
                    None => {}
                }
                if let Some(e) = cond {
                    self.expr(e);
                }
                if let Some(e) = step {
                    self.expr(e);
                }
                self.stmt(body);
                self.table.exit_scope();
            }
            Stmt::Goto(_) | Stmt::Continue | Stmt::Break => {}
            Stmt::Return(Some(e)) => self.expr(e),
            Stmt::Return(None) => {}
            Stmt::Labeled { stmt, .. } => self.stmt(stmt),
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Ident(name) => {
                if self.table.lookup(name).is_none() {
                    self.unresolved.push(UnresolvedIdent { name: name.clone() });
                }
            }
            Expr::IntLiteral(_)
            | Expr::FloatLiteral(_)
            | Expr::StringLiteral(_)
            | Expr::CharLiteral(_) => {}
            Expr::Unary { expr, .. } => self.expr(expr),
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            Expr::Assign { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            Expr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => {
                self.expr(cond);
                self.expr(then_expr);
                self.expr(else_expr);
            }
            Expr::Comma(a, b) => {
                self.expr(a);
                self.expr(b);
            }
            Expr::Call { callee, args } => {
                self.expr(callee);
                for a in args {
                    self.expr(a);
                }
            }
            Expr::Index { base, index } => {
                self.expr(base);
                self.expr(index);
            }
            // Field names live in the struct/union's own namespace, not the
            // ordinary one -- resolving them needs `base`'s type (Step 3).
            Expr::Member { base, .. } => self.expr(base),
            Expr::PostIncDec { expr, .. } | Expr::PreIncDec { expr, .. } => self.expr(expr),
            Expr::Cast { type_name, expr } => {
                self.type_name(type_name);
                self.expr(expr);
            }
            Expr::Sizeof(SizeofArg::Expr(e)) => self.expr(e),
            Expr::Sizeof(SizeofArg::Type(t)) => self.type_name(t),
        }
    }
}

/// Small re-export shim so this module doesn't reach three levels deep into
/// `crate::parser::grammar` at every call site.
mod grammar_names {
    pub(super) use crate::parser::grammar::declarator_name;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse_full, parse_translation_unit};

    fn parse(src: &str) -> TranslationUnit {
        let (spliced, chunks) = crate::parser::parse_chunks(src);
        let mut env = crate::parser::PreprocessorEnv::linux_doom_defaults();
        let resolved = crate::parser::resolve_conditionals(&chunks, &mut env).unwrap();
        let entries = crate::parser::lex_chunks(&resolved).unwrap();
        let stream = crate::parser::attach_comments(entries);
        let _ = spliced;
        parse_translation_unit(&stream).unwrap()
    }

    #[test]
    fn test_resolves_global_and_local_variables() {
        let unit = parse("int g; void f(int p) { int l = p + g; }");
        let result = resolve_translation_unit(&unit);
        assert!(
            result.unresolved.is_empty(),
            "unexpected unresolved: {:?}",
            result.unresolved
        );
    }

    #[test]
    fn test_flags_truly_unresolved_identifier() {
        let unit = parse("void f() { int l = totally_unknown; }");
        let result = resolve_translation_unit(&unit);
        assert_eq!(
            result.unresolved,
            vec![UnresolvedIdent {
                name: "totally_unknown".to_string()
            }]
        );
    }

    #[test]
    fn test_block_scope_does_not_leak() {
        let unit = parse("void f() { { int inner = 1; } int outer = inner; }");
        let result = resolve_translation_unit(&unit);
        assert_eq!(
            result.unresolved,
            vec![UnresolvedIdent {
                name: "inner".to_string()
            }]
        );
    }

    #[test]
    fn test_enum_constants_and_struct_field_names() {
        // `field` must NOT be flagged unresolved -- member access isn't a
        // scope lookup (Step 3's job), only `s` and `RED` are.
        let unit = parse(
            "enum Color { RED, GREEN }; \
             struct S { int field; }; \
             void f() { struct S s; int c = RED; int x = s.field; }",
        );
        let result = resolve_translation_unit(&unit);
        assert!(
            result.unresolved.is_empty(),
            "unexpected unresolved: {:?}",
            result.unresolved
        );
    }

    #[test]
    fn test_array_size_macro_like_constant_is_resolved() {
        let unit = parse("enum { SIZE = 4 }; void f() { int arr[SIZE]; }");
        let result = resolve_translation_unit(&unit);
        assert!(
            result.unresolved.is_empty(),
            "unexpected unresolved: {:?}",
            result.unresolved
        );
    }

    #[test]
    fn test_prototype_param_names_do_not_leak() {
        let unit = parse("void f(int n); void g() { int x = n; }");
        let result = resolve_translation_unit(&unit);
        assert_eq!(
            result.unresolved,
            vec![UnresolvedIdent {
                name: "n".to_string()
            }]
        );
    }

    #[test]
    fn test_corpus_resolution_coverage() {
        // Not a pass/fail assertion (matching Step 4b/7's "measure actual
        // scope before deciding it needs more" methodology): this
        // translation-unit-local symbol table only knows about a file's own
        // declarations plus its own typedef names (Step 6b) -- it doesn't
        // yet pull in function/variable declarations transitively from
        // `#include`d headers (unlike typedefs, Step 6b never collected
        // those), so real corpus code that calls e.g. libc functions or
        // references another header's globals is expected to show up here
        // as unresolved. Prints the actual rate for follow-up.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        files.sort();
        assert!(files.len() > 50, "expected the full Doom .c corpus");

        let mut clean_files = 0;
        let mut total_unresolved = 0;
        for path in &files {
            if let Ok((_, unit)) = parse_full(path.to_str().unwrap()) {
                let result = resolve_translation_unit(&unit);
                if result.unresolved.is_empty() {
                    clean_files += 1;
                }
                total_unresolved += result.unresolved.len();
            }
        }
        eprintln!(
            "symbol resolution over {} files: {clean_files} fully resolved, \
             {total_unresolved} unresolved identifier references total \
             (expected: cross-header function/variable declarations aren't \
             imported yet, only typedefs)",
            files.len()
        );
    }
}
