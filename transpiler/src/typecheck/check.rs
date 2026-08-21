//! Step 3: Type Checking & Promotion Resolution (docs/02_TYPECHECKER.md Step 3)
//!
//! The step that finally puts every earlier one to work: walks a
//! translation unit exactly like Step 1's `Resolver` does (same scopes,
//! same declaration/statement/expression shapes), but where Step 1 only
//! resolved *names*, this computes a `Type` for every expression --
//! resolving Doom's typedefs (`fixed_t`, `byte`, `boolean`, ...) down to
//! their underlying representation via Step 3's own `DeclaredTypes`
//! (`declared_types.rs`), typing struct/union member access via its field
//! layouts (which Step 2 always left `Unknown`, not having them), and
//! reporting every assignment/call-argument site whose value type isn't
//! compatible with its target (`types::is_assignment_compatible`).
//!
//! **Local scope, not `scope::SymbolTable`**: Step 1's `SymbolTable` tracks
//! *that* a name is declared, not *what type* it has -- retrofitting a type
//! field onto its `Symbol` would touch every one of Step 0/1's existing
//! construction sites for a step that (like Step 0 before it) needs richer
//! information than the original design carried. A small local `TypeScope`
//! (name -> `Type`, block-scoped) fills that gap instead, mirroring
//! `SymbolTable`'s shape without disturbing it. It's still handed a
//! resolved `SymbolTable` (Step 1's real output, seeded the same way Step
//! 1 itself is) purely to recognize enum-constant identifiers, the same
//! way `MacroTyper` does.
//!
//! **Macro references get typed here too, not by delegating to
//! `MacroTyper`**: a macro's own body typically can't see the calling
//! function's locals or struct fields, but its *arguments* at a real call
//! site can (`SHORT(mobj->x)` -- `mobj->x` is only typeable once this step
//! knows `mobj_t`'s field layout). Reusing `MacroTyper` as-is would lose
//! that, since its own `type_of_expr` has no access to this step's richer
//! context -- so this module reimplements the same small
//! substitute-then-type dance (`type_of_object_macro`/`type_of_macro_call`)
//! directly against its own `type_of_expr`, closing exactly the gap
//! `docs/KNOWN_LIMITATIONS.md`'s Step 2 entry measured and flagged for
//! follow-up.

use crate::parser::MacroBody;
use crate::parser::ast::*;
use crate::typecheck::declared_types::DeclaredTypes;
use crate::typecheck::macro_types::substitute;
use crate::typecheck::scope::{SymbolKind, SymbolTable};
use crate::typecheck::types::{
    Type, is_assignment_compatible, param_type, type_from_declarator, type_from_specifiers,
    type_from_type_name, type_of_additive, type_of_float_literal, type_of_int_literal,
    unary_arith_result, usual_arithmetic_conversions,
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticKind {
    /// A declaration's initializer, or the right-hand side of an `=`,
    /// isn't assignment-compatible with its target's type.
    Assignment,
    /// A call argument isn't assignment-compatible with the callee's
    /// declared parameter type at that position.
    CallArgument,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDiagnostic {
    pub kind: DiagnosticKind,
    pub target: Type,
    pub value: Type,
}

/// A block-scoped name -> `Type` stack, mirroring `scope::SymbolTable`'s
/// shape (see module docs for why this is separate from it).
#[derive(Default)]
struct TypeScope {
    scopes: Vec<HashMap<String, Type>>,
}

impl TypeScope {
    fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    fn enter(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit(&mut self) {
        self.scopes.pop();
    }

    fn declare(&mut self, name: String, ty: Type) {
        self.scopes.last_mut().unwrap().insert(name, ty);
    }

    fn lookup(&self, name: &str) -> Option<&Type> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }
}

#[derive(Debug)]
pub struct TypeCheckResult {
    pub diagnostics: Vec<TypeDiagnostic>,
    /// Every expression this pass typed, whether it resolved to a concrete
    /// `Type` or came back `Unknown` -- Step 3's "every expression gets a
    /// resolved type" validation criterion is measured from these two
    /// counts, the same way Step 0-2 measured their own corpus coverage.
    pub expr_count: usize,
    pub unknown_count: usize,
}

pub fn check_translation_unit(
    unit: &TranslationUnit,
    declared: &DeclaredTypes,
    macros: &HashMap<String, MacroBody>,
    table: &SymbolTable,
) -> TypeCheckResult {
    let mut c = Checker {
        declared,
        macros,
        table,
        scope: TypeScope::new(),
        visiting_macros: HashSet::new(),
        object_macro_cache: HashMap::new(),
        diagnostics: Vec::new(),
        expr_count: 0,
        unknown_count: 0,
    };
    for item in &unit.items {
        c.external_decl(item);
    }
    TypeCheckResult {
        diagnostics: c.diagnostics,
        expr_count: c.expr_count,
        unknown_count: c.unknown_count,
    }
}

struct Checker<'a> {
    declared: &'a DeclaredTypes,
    macros: &'a HashMap<String, MacroBody>,
    table: &'a SymbolTable,
    scope: TypeScope,
    visiting_macros: HashSet<String>,
    object_macro_cache: HashMap<String, Type>,
    diagnostics: Vec<TypeDiagnostic>,
    expr_count: usize,
    unknown_count: usize,
}

impl Checker<'_> {
    fn external_decl(&mut self, item: &ExternalDecl) {
        match item {
            ExternalDecl::Declaration(decl) => self.declaration(decl),
            ExternalDecl::FunctionDef(f) => {
                let base = type_from_specifiers(&f.specifiers);
                self.scope.enter();
                self.declare_params(&f.declarator.direct);
                for item in &f.body.items {
                    self.block_item(item);
                }
                self.scope.exit();
                let _ = base; // return-type checking isn't in this pass's scope (see module docs).
            }
        }
    }

    fn declare_params(&mut self, d: &DirectDeclarator) {
        if let DirectDeclarator::Function(_, params) = d {
            for p in &params.params {
                if let ParamDeclarator::Named(pd) = &p.declarator
                    && let Some(name) = crate::parser::grammar::declarator_name(pd)
                {
                    self.scope.declare(name, param_type(p));
                }
            }
        }
    }

    fn declaration(&mut self, decl: &Declaration) {
        let base = type_from_specifiers(&decl.specifiers);
        let is_typedef = decl.specifiers.storage == Some(StorageClass::Typedef);
        for init_decl in &decl.declarators {
            let ty = type_from_declarator(base.clone(), &init_decl.declarator);
            if !is_typedef
                && let Some(name) = crate::parser::grammar::declarator_name(&init_decl.declarator)
            {
                self.scope.declare(name, ty.clone());
            }
            if let Some(init) = &init_decl.initializer {
                self.initializer(init, &ty);
            }
        }
    }

    /// Checks a declaration's initializer against its declared type --
    /// only for a single `Expr` initializer. An aggregate (`{ ... }`)
    /// initializer's own per-member compatibility isn't checked in this
    /// pass (would need to walk it alongside the target's own field/
    /// element types in lockstep) -- its sub-expressions are still typed,
    /// just not checked, so they still count toward the "every expression
    /// gets a type" measurement.
    fn initializer(&mut self, init: &Initializer, target: &Type) {
        match init {
            Initializer::Expr(e) => {
                let value = self.expr(e);
                if !is_assignment_compatible(
                    &self.declared.normalize(target),
                    &self.declared.normalize(&value),
                ) {
                    self.diagnostics.push(TypeDiagnostic {
                        kind: DiagnosticKind::Assignment,
                        target: target.clone(),
                        value,
                    });
                }
            }
            Initializer::List(items) => {
                for i in items {
                    self.initializer(i, &Type::Unknown);
                }
            }
        }
    }

    fn decl_specifiers(&mut self, specs: &DeclSpecifiers) {
        for ts in &specs.type_specifiers {
            match ts {
                TypeSpecifier::Struct(spec) | TypeSpecifier::Union(spec) => {
                    for field in spec.fields.iter().flatten() {
                        self.decl_specifiers(&field.specifiers);
                        for (_, bitwidth) in &field.declarators {
                            if let Some(w) = bitwidth {
                                self.expr(w);
                            }
                        }
                    }
                }
                TypeSpecifier::Enum(spec) => {
                    for (_, value) in spec.variants.iter().flatten() {
                        if let Some(e) = value {
                            self.expr(e);
                        }
                    }
                }
                _ => {}
            }
        }
    }

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

    fn block_item(&mut self, item: &BlockItem) {
        match item {
            BlockItem::Decl(decl) => self.declaration(decl),
            BlockItem::Stmt(stmt) => self.stmt(stmt),
        }
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr(Some(e)) => {
                self.expr(e);
            }
            Stmt::Expr(None) => {}
            Stmt::Compound(cs) => {
                self.scope.enter();
                for item in &cs.items {
                    self.block_item(item);
                }
                self.scope.exit();
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
                self.scope.enter();
                match init {
                    Some(ForInit::Decl(d)) => self.declaration(d),
                    Some(ForInit::Expr(e)) => {
                        self.expr(e);
                    }
                    None => {}
                }
                if let Some(e) = cond {
                    self.expr(e);
                }
                if let Some(e) = step {
                    self.expr(e);
                }
                self.stmt(body);
                self.scope.exit();
            }
            Stmt::Goto(_) | Stmt::Continue | Stmt::Break => {}
            Stmt::Return(Some(e)) => {
                self.expr(e);
            }
            Stmt::Return(None) => {}
            Stmt::Labeled { stmt, .. } => self.stmt(stmt),
        }
    }

    fn record(&mut self, ty: Type) -> Type {
        self.expr_count += 1;
        if ty == Type::Unknown {
            self.unknown_count += 1;
        }
        ty
    }

    fn type_of_ident(&mut self, name: &str) -> Type {
        if let Some(t) = self.scope.lookup(name) {
            return t.clone();
        }
        if let Some((t, _)) = self.declared.variables.get(name) {
            return t.clone();
        }
        if let Some((sig, _)) = self.declared.functions.get(name) {
            return Type::Function(Box::new(sig.ret.clone()));
        }
        match self.macros.get(name) {
            Some(MacroBody::Object(_)) => return self.type_of_object_macro(name),
            Some(_) => return Type::Unknown,
            None => {}
        }
        if let Some(SymbolKind::EnumConstant) = self.table.lookup(name).map(|s| s.kind) {
            return Type::Int;
        }
        Type::Unknown
    }

    fn type_of_object_macro(&mut self, name: &str) -> Type {
        if let Some(t) = self.object_macro_cache.get(name) {
            return t.clone();
        }
        if !self.visiting_macros.insert(name.to_string()) {
            return Type::Unknown;
        }
        let ty = match self.macros.get(name) {
            Some(MacroBody::Object(expr)) => {
                let expr = expr.clone();
                self.expr(&expr)
            }
            _ => Type::Unknown,
        };
        self.visiting_macros.remove(name);
        self.object_macro_cache.insert(name.to_string(), ty.clone());
        ty
    }

    fn type_of_macro_call(&mut self, name: &str, args: &[Expr]) -> Type {
        let Some(MacroBody::Function { params, body }) = self.macros.get(name) else {
            return Type::Unknown;
        };
        if !self.visiting_macros.insert(name.to_string()) {
            return Type::Unknown;
        }
        let substituted = substitute(body, params, args);
        let ty = self.expr(&substituted);
        self.visiting_macros.remove(name);
        ty
    }

    /// Types field access on `base_ty` (already dereferenced through a
    /// pointer for `->`), looking the tag up in `declared.fields` after
    /// resolving through any typedef chain.
    fn type_of_member(&self, base_ty: &Type, field: &str) -> Type {
        let resolved = self.declared.resolve_typedef(base_ty);
        let tag = match &resolved {
            Type::Struct(name) | Type::Union(name) => name,
            _ => return Type::Unknown,
        };
        self.declared
            .fields
            .get(tag)
            .and_then(|fields| fields.iter().find(|(n, _)| n == field))
            .map(|(_, t)| t.clone())
            .unwrap_or(Type::Unknown)
    }

    fn expr(&mut self, e: &Expr) -> Type {
        let ty = self.expr_inner(e);
        self.record(ty)
    }

    fn expr_inner(&mut self, e: &Expr) -> Type {
        match e {
            Expr::IntLiteral(s) => type_of_int_literal(s),
            Expr::FloatLiteral(s) => type_of_float_literal(s),
            Expr::CharLiteral(_) => Type::Int,
            Expr::StringLiteral(_) => Type::Pointer(Box::new(Type::Char)),
            Expr::Ident(name) => self.type_of_ident(name),
            Expr::Unary { op, expr } => {
                let t = self.expr(expr);
                match op {
                    UnaryOp::Deref => match self.declared.resolve_typedef(&t) {
                        Type::Pointer(inner) | Type::Array(inner) => *inner,
                        _ => Type::Unknown,
                    },
                    UnaryOp::AddrOf => Type::Pointer(Box::new(t)),
                    UnaryOp::Not => Type::Int,
                    UnaryOp::Plus | UnaryOp::Minus | UnaryOp::BitNot => unary_arith_result(&t),
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let lt = self.expr(lhs);
                let rt = self.expr(rhs);
                self.type_of_binary(*op, &lt, &rt)
            }
            Expr::Assign { lhs, rhs, .. } => {
                let target = self.expr(lhs);
                let value = self.expr(rhs);
                if !is_assignment_compatible(
                    &self.declared.normalize(&target),
                    &self.declared.normalize(&value),
                ) {
                    self.diagnostics.push(TypeDiagnostic {
                        kind: DiagnosticKind::Assignment,
                        target: target.clone(),
                        value,
                    });
                }
                target
            }
            Expr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => {
                self.expr(cond);
                let t1 = self.expr(then_expr);
                let t2 = self.expr(else_expr);
                if t1 == t2 {
                    t1
                } else if t1.is_arithmetic() && t2.is_arithmetic() {
                    usual_arithmetic_conversions(&t1, &t2)
                } else {
                    Type::Unknown
                }
            }
            Expr::Comma(a, b) => {
                self.expr(a);
                self.expr(b)
            }
            Expr::Call { callee, args } => self.type_of_call(callee, args),
            Expr::Index { base, index } => {
                self.expr(index);
                let base_ty = self.expr(base);
                match self.declared.resolve_typedef(&base_ty) {
                    Type::Pointer(inner) | Type::Array(inner) => *inner,
                    _ => Type::Unknown,
                }
            }
            Expr::Member { base, field, arrow } => {
                let base_ty = self.expr(base);
                let target_ty = if *arrow {
                    match self.declared.resolve_typedef(&base_ty) {
                        Type::Pointer(inner) => *inner,
                        _ => return Type::Unknown,
                    }
                } else {
                    base_ty
                };
                self.type_of_member(&target_ty, field)
            }
            Expr::PostIncDec { expr, .. } | Expr::PreIncDec { expr, .. } => self.expr(expr),
            Expr::Cast { type_name, expr } => {
                self.expr(expr);
                type_from_type_name(type_name)
            }
            Expr::Sizeof(SizeofArg::Expr(e)) => {
                self.expr(e);
                Type::UInt
            }
            Expr::Sizeof(SizeofArg::Type(t)) => {
                self.type_name(t);
                Type::UInt
            }
        }
    }

    fn type_of_binary(&self, op: BinaryOp, lt: &Type, rt: &Type) -> Type {
        use BinaryOp::*;
        match op {
            Lt | Le | Gt | Ge | Eq | Ne | LogAnd | LogOr => Type::Int,
            Shl | Shr => {
                if lt.is_integer() {
                    crate::typecheck::types::integer_promote(lt)
                } else {
                    Type::Unknown
                }
            }
            Add | Sub => type_of_additive(op, lt, rt),
            Mul | Div | Mod | BitAnd | BitXor | BitOr => {
                if lt.is_arithmetic() && rt.is_arithmetic() {
                    usual_arithmetic_conversions(lt, rt)
                } else {
                    Type::Unknown
                }
            }
        }
    }

    fn type_of_call(&mut self, callee: &Expr, args: &[Expr]) -> Type {
        if let Expr::Ident(name) = callee {
            if self.macros.contains_key(name) {
                for a in args {
                    self.expr(a);
                }
                return self.type_of_macro_call(name, args);
            }
            if let Some((sig, _)) = self.declared.functions.get(name).cloned() {
                for (i, arg) in args.iter().enumerate() {
                    let value = self.expr(arg);
                    if let Some(target) = sig.params.get(i)
                        && !is_assignment_compatible(
                            &self.declared.normalize(target),
                            &self.declared.normalize(&value),
                        )
                    {
                        self.diagnostics.push(TypeDiagnostic {
                            kind: DiagnosticKind::CallArgument,
                            target: target.clone(),
                            value,
                        });
                    }
                }
                return sig.ret;
            }
        }
        self.expr(callee);
        for a in args {
            self.expr(a);
        }
        Type::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{
        MacroBodyResolver, PreprocessorEnv, parse_full, parse_translation_unit,
        resolve_conditionals,
    };
    use crate::typecheck::declared_types::DeclaredTypesResolver;
    use crate::typecheck::exports::ExportResolver;
    use crate::typecheck::resolve::resolve_translation_unit_seeded;

    fn parse(src: &str) -> TranslationUnit {
        let (_, chunks) = crate::parser::parse_chunks(src);
        let mut env = PreprocessorEnv::linux_doom_defaults();
        let resolved = resolve_conditionals(&chunks, &mut env).unwrap();
        let entries = crate::parser::lex_chunks(&resolved).unwrap();
        let stream = crate::parser::attach_comments(entries);
        parse_translation_unit(&stream).unwrap()
    }

    fn check(src: &str) -> TypeCheckResult {
        let unit = parse(src);
        let declared = DeclaredTypes::default();
        let macros = HashMap::new();
        let table = resolve_translation_unit_seeded(&unit, Default::default()).table;
        check_translation_unit(&unit, &declared, &macros, &table)
    }

    #[test]
    fn test_compatible_assignment_is_not_flagged() {
        let result = check("void f(void) { int x; x = 1; }");
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    }

    #[test]
    fn test_arithmetic_assignment_across_types_is_not_flagged() {
        // C89 allows implicit conversion here (may narrow) -- not an error.
        let result = check("void f(void) { double d; int x; d = x; x = d; }");
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    }

    #[test]
    fn test_pointer_to_unrelated_pointer_assignment_is_flagged() {
        let result = check("void f(void) { int *p; char *c; p = c; }");
        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(result.diagnostics[0].kind, DiagnosticKind::Assignment);
    }

    #[test]
    fn test_void_pointer_assignment_is_not_flagged() {
        let result = check("void f(void) { int *p; void *v; p = v; v = p; }");
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    }

    #[test]
    fn test_initializer_incompatibility_is_flagged() {
        let result = check("void f(void) { int *p = 3.5; }");
        // 3.5 is a double (arithmetic), p is a pointer -- not integer, so
        // not covered by the int/pointer allowance.
        assert_eq!(result.diagnostics.len(), 1);
    }

    #[test]
    fn test_struct_field_access_is_typed() {
        let src = "struct point_t { int x; char *label; }; \
                    void f(void) { struct point_t s; int a = s.x; char *b = s.label; }";
        let unit = parse(src);
        let mut declared = DeclaredTypes::default();
        declared.fields.insert(
            "point_t".to_string(),
            vec![
                ("x".to_string(), Type::Int),
                ("label".to_string(), Type::Pointer(Box::new(Type::Char))),
            ],
        );
        let macros = HashMap::new();
        let table = resolve_translation_unit_seeded(&unit, Default::default()).table;
        let result = check_translation_unit(&unit, &declared, &macros, &table);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.unknown_count, 0, "{result:?}");
    }

    #[test]
    fn test_struct_field_access_through_pointer() {
        let src = "struct point_t { int x; }; \
                    void f(struct point_t *s) { int a = s->x; }";
        let unit = parse(src);
        let mut declared = DeclaredTypes::default();
        declared
            .fields
            .insert("point_t".to_string(), vec![("x".to_string(), Type::Int)]);
        let macros = HashMap::new();
        let table = resolve_translation_unit_seeded(&unit, Default::default()).table;
        let result = check_translation_unit(&unit, &declared, &macros, &table);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.unknown_count, 0, "{result:?}");
    }

    #[test]
    fn test_call_argument_mismatch_is_flagged() {
        let mut declared = DeclaredTypes::default();
        declared.functions.insert(
            "g".to_string(),
            (
                crate::typecheck::types::FunctionSignature {
                    ret: Type::Void,
                    params: vec![Type::Pointer(Box::new(Type::Int))],
                    variadic: false,
                },
                None,
            ),
        );
        // A `double` argument for a pointer parameter isn't covered by the
        // pointer/integer allowance (see `is_assignment_compatible`'s
        // docs) -- unlike e.g. a `char`, which C89 treats as an integer
        // type and so is deliberately *not* flagged (too easily a `0`/
        // `NULL`-shaped false positive without constant-value tracking).
        let unit = parse("void f(void) { double d; g(d); }");
        let macros = HashMap::new();
        let table = resolve_translation_unit_seeded(&unit, Default::default()).table;
        let result = check_translation_unit(&unit, &declared, &macros, &table);
        assert_eq!(result.diagnostics.len(), 1, "{:?}", result.diagnostics);
        assert_eq!(result.diagnostics[0].kind, DiagnosticKind::CallArgument);
    }

    #[test]
    fn test_macro_reference_with_struct_member_argument_is_typed() {
        // The exact gap Step 2's KNOWN_LIMITATIONS.md entry measured and
        // flagged for follow-up: SHORT(mobj->x) needs mobj_t's field
        // layout to type at all, which only this step's DeclaredTypes has.
        let mut declared = DeclaredTypes::default();
        declared
            .fields
            .insert("mobj_t".to_string(), vec![("x".to_string(), Type::Int)]);
        let mut macros = HashMap::new();
        macros.insert(
            "SHORT".to_string(),
            MacroBody::Function {
                params: vec!["a".to_string()],
                body: Expr::Ident("a".to_string()),
            },
        );
        let unit = parse(
            "struct mobj_t { int x; }; \
             void f(struct mobj_t *mobj) { int v = SHORT(mobj->x); }",
        );
        let table = resolve_translation_unit_seeded(&unit, Default::default()).table;
        let result = check_translation_unit(&unit, &declared, &macros, &table);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.unknown_count, 0);
    }

    #[test]
    fn test_typedef_pointer_field_access_resolves_through_the_chain() {
        let mut declared = DeclaredTypes::default();
        declared
            .typedefs
            .insert("mobj_t".to_string(), Type::Struct("mobj_s".to_string()));
        declared
            .fields
            .insert("mobj_s".to_string(), vec![("x".to_string(), Type::Int)]);
        // The typedef itself must be declared before use so the parser's
        // own single-pass typedef tracking (not this test's `declared`,
        // which the *checker* consults) recognizes `mobj_t` as a type name
        // rather than an undeclared identifier.
        let unit = parse("typedef struct mobj_s mobj_t; void f(mobj_t *mo) { int v = mo->x; }");
        let macros = HashMap::new();
        let table = resolve_translation_unit_seeded(&unit, Default::default()).table;
        let result = check_translation_unit(&unit, &declared, &macros, &table);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.unknown_count, 0);
    }

    #[test]
    fn test_corpus_type_checking_coverage() {
        // Not a pass/fail assertion (matching this project's "measure,
        // don't assume" methodology, same as every prior step) -- runs the
        // full checker over the corpus and reports how much of it typed
        // cleanly, and how many assignment/call-argument sites it flagged.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        files.sort();
        assert!(files.len() > 50, "expected the full Doom .c corpus");

        let mut declared_resolver = DeclaredTypesResolver::new();
        let mut export_resolver = ExportResolver::new();
        let mut macro_resolver = MacroBodyResolver::new();
        let mut total_expr = 0usize;
        let mut total_unknown = 0usize;
        let mut total_diagnostics = 0usize;
        let mut assignment_diags = 0usize;
        let mut call_diags = 0usize;
        for path in &files {
            let Ok((_, unit)) = parse_full(path.to_str().unwrap()) else {
                continue;
            };
            let declared = declared_resolver.resolve(path);
            let macros = macro_resolver.resolve(path);
            let seed = export_resolver.resolve(path);
            let table = resolve_translation_unit_seeded(&unit, seed).table;
            let result = check_translation_unit(&unit, &declared, &macros, &table);
            total_expr += result.expr_count;
            total_unknown += result.unknown_count;
            total_diagnostics += result.diagnostics.len();
            for d in &result.diagnostics {
                match d.kind {
                    DiagnosticKind::Assignment => assignment_diags += 1,
                    DiagnosticKind::CallArgument => call_diags += 1,
                }
            }
        }
        eprintln!(
            "type checking over {} files: {total_expr} expressions typed, \
             {total_unknown} left Unknown ({:.1}% resolved); {total_diagnostics} \
             compatibility diagnostics ({assignment_diags} assignment, {call_diags} \
             call-argument)",
            files.len(),
            100.0 * (total_expr - total_unknown) as f64 / total_expr.max(1) as f64,
        );
        assert!(total_expr > 0);
    }
}
