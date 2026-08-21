//! Step 2: Macro Typing (docs/02_TYPECHECKER.md Step 2)
//!
//! Parser Step 7 (`crate::parser::macro_body`) already turns each `#define`'s
//! replacement text into a structured `MacroBody`, purely syntactically --
//! this step is the first consumer that actually needs a *type*, which is
//! why `types.rs`'s `Type` representation is introduced right here rather
//! than earlier.
//!
//! Two kinds of macro, two typing strategies:
//! - `MacroBody::Object(expr)`: typecheck `expr` directly, resolving any
//!   macro-to-macro references recursively (memoized, cycle-guarded --
//!   `MacroTyper::type_of_object_macro`).
//! - `MacroBody::Function { params, body }`: no single fixed signature, like
//!   a template -- at each real call site, substitute the actual argument
//!   expressions for `params` inside `body` (`substitute`) and typecheck
//!   the substituted expression, rather than typechecking the macro once in
//!   isolation.
//!
//! `MacroBody::Empty`/`Statements`/`Unparseable` bodies have no expression
//! value to type -- referencing one of those (or a function-like macro used
//! bare, without a call) yields `Type::Unknown`, flagged for follow-up
//! rather than a hard error (matching Step 4b/Step 0's "measure, don't
//! assume" policy).
//!
//! `collect_macro_uses` finds every place *real code* (not just another
//! macro's body) references a known macro -- a bare `Expr::Ident` matching
//! an object-like macro, or an `Expr::Call` whose callee matches a
//! function-like one -- by walking the same expression-bearing AST shapes
//! Step 1's `Resolver` does (function bodies, declaration initializers,
//! array-size/bit-field-width/enum-value expressions), just without any
//! scope bookkeeping, since macro-ness doesn't depend on scope: a real C
//! preprocessor macro always wins over any identically-named declaration,
//! so a name simply being present in the macro map is authoritative here.

use crate::parser::MacroBody;
use crate::parser::ast::*;
use crate::typecheck::scope::{SymbolKind, SymbolTable};
use crate::typecheck::types::{
    Type, type_from_type_name, type_of_additive, type_of_float_literal, type_of_int_literal,
    unary_arith_result, usual_arithmetic_conversions,
};
use std::collections::{HashMap, HashSet};

/// A real-code reference to a known macro: a bare mention of an object-like
/// one, or a call to a function-like one with its (unsubstituted) argument
/// expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum MacroUse {
    Object(String),
    Call { name: String, args: Vec<Expr> },
}

/// Substitutes `args` for `params` throughout `body`'s `Expr` tree --
/// structural, not textual: every `Expr::Ident` matching a parameter name
/// is replaced with a deep clone of that parameter's actual argument
/// expression; everything else is walked but otherwise left alone. Doing
/// this before typing (rather than threading a substitution environment
/// through `type_of_expr`) means a nested macro call inside `body` gets
/// typed the same way a top-level one does -- no separate case needed.
pub fn substitute(body: &Expr, params: &[String], args: &[Expr]) -> Expr {
    let sub = |e: &Expr| substitute(e, params, args);
    match body {
        Expr::Ident(name) => match params.iter().position(|p| p == name) {
            Some(i) => args.get(i).cloned().unwrap_or_else(|| body.clone()),
            None => body.clone(),
        },
        Expr::IntLiteral(_)
        | Expr::FloatLiteral(_)
        | Expr::StringLiteral(_)
        | Expr::CharLiteral(_) => body.clone(),
        Expr::Unary { op, expr } => Expr::Unary {
            op: *op,
            expr: Box::new(sub(expr)),
        },
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op: *op,
            lhs: Box::new(sub(lhs)),
            rhs: Box::new(sub(rhs)),
        },
        Expr::Assign { op, lhs, rhs } => Expr::Assign {
            op: *op,
            lhs: Box::new(sub(lhs)),
            rhs: Box::new(sub(rhs)),
        },
        Expr::Conditional {
            cond,
            then_expr,
            else_expr,
        } => Expr::Conditional {
            cond: Box::new(sub(cond)),
            then_expr: Box::new(sub(then_expr)),
            else_expr: Box::new(sub(else_expr)),
        },
        Expr::Comma(a, b) => Expr::Comma(Box::new(sub(a)), Box::new(sub(b))),
        Expr::Call {
            callee,
            args: cargs,
        } => Expr::Call {
            callee: Box::new(sub(callee)),
            args: cargs.iter().map(sub).collect(),
        },
        Expr::Index { base, index } => Expr::Index {
            base: Box::new(sub(base)),
            index: Box::new(sub(index)),
        },
        Expr::Member { base, field, arrow } => Expr::Member {
            base: Box::new(sub(base)),
            field: field.clone(),
            arrow: *arrow,
        },
        Expr::PostIncDec { expr, op } => Expr::PostIncDec {
            expr: Box::new(sub(expr)),
            op: *op,
        },
        Expr::PreIncDec { expr, op } => Expr::PreIncDec {
            expr: Box::new(sub(expr)),
            op: *op,
        },
        Expr::Cast { type_name, expr } => Expr::Cast {
            type_name: type_name.clone(),
            expr: Box::new(sub(expr)),
        },
        Expr::Sizeof(SizeofArg::Expr(e)) => Expr::Sizeof(SizeofArg::Expr(Box::new(sub(e)))),
        Expr::Sizeof(SizeofArg::Type(t)) => Expr::Sizeof(SizeofArg::Type(t.clone())),
    }
}

/// Types expressions against a fixed set of macros visible to one
/// translation unit (plus, optionally, its resolved `SymbolTable`, used
/// only to type enum-constant identifiers a macro body happens to
/// reference). Memoizes each object-like macro's type and cycle-guards
/// against a macro that (directly or through others) references itself.
pub struct MacroTyper<'a> {
    macros: &'a HashMap<String, MacroBody>,
    table: Option<&'a SymbolTable>,
    visiting: HashSet<String>,
    object_cache: HashMap<String, Type>,
}

impl<'a> MacroTyper<'a> {
    pub fn new(macros: &'a HashMap<String, MacroBody>, table: Option<&'a SymbolTable>) -> Self {
        Self {
            macros,
            table,
            visiting: HashSet::new(),
            object_cache: HashMap::new(),
        }
    }

    /// The type of object-like macro `name`, or `Type::Unknown` if it isn't
    /// one (a function-like/empty/statement/unparseable body, an unknown
    /// name, or a cyclic reference).
    pub fn type_of_object_macro(&mut self, name: &str) -> Type {
        if let Some(t) = self.object_cache.get(name) {
            return t.clone();
        }
        if !self.visiting.insert(name.to_string()) {
            return Type::Unknown; // cyclic macro reference
        }
        let ty = match self.macros.get(name) {
            Some(MacroBody::Object(expr)) => {
                let expr = expr.clone();
                self.type_of_expr(&expr)
            }
            _ => Type::Unknown,
        };
        self.visiting.remove(name);
        self.object_cache.insert(name.to_string(), ty.clone());
        ty
    }

    /// The type of calling function-like macro `name` with `args`, via
    /// substitution (see `substitute`'s docs).
    pub fn type_of_macro_call(&mut self, name: &str, args: &[Expr]) -> Type {
        let Some(MacroBody::Function { params, body }) = self.macros.get(name) else {
            return Type::Unknown;
        };
        if !self.visiting.insert(name.to_string()) {
            return Type::Unknown; // cyclic macro reference
        }
        let substituted = substitute(body, params, args);
        let ty = self.type_of_expr(&substituted);
        self.visiting.remove(name);
        ty
    }

    pub fn type_of_expr(&mut self, e: &Expr) -> Type {
        match e {
            Expr::IntLiteral(s) => type_of_int_literal(s),
            Expr::FloatLiteral(s) => type_of_float_literal(s),
            // C89 character constants have type `int`, not `char`.
            Expr::CharLiteral(_) => Type::Int,
            Expr::StringLiteral(_) => Type::Pointer(Box::new(Type::Char)),
            Expr::Ident(name) => self.type_of_ident(name),
            Expr::Unary { op, expr } => {
                let t = self.type_of_expr(expr);
                match op {
                    UnaryOp::Deref => match t {
                        Type::Pointer(inner) | Type::Array(inner) => *inner,
                        _ => Type::Unknown,
                    },
                    UnaryOp::AddrOf => Type::Pointer(Box::new(t)),
                    UnaryOp::Not => Type::Int,
                    UnaryOp::Plus | UnaryOp::Minus | UnaryOp::BitNot => unary_arith_result(&t),
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let lt = self.type_of_expr(lhs);
                let rt = self.type_of_expr(rhs);
                self.type_of_binary(*op, &lt, &rt)
            }
            // The value of an assignment expression is its left operand's
            // (converted) type.
            Expr::Assign { lhs, .. } => self.type_of_expr(lhs),
            Expr::Conditional {
                then_expr,
                else_expr,
                ..
            } => {
                let t1 = self.type_of_expr(then_expr);
                let t2 = self.type_of_expr(else_expr);
                if t1 == t2 {
                    t1
                } else if t1.is_arithmetic() && t2.is_arithmetic() {
                    usual_arithmetic_conversions(&t1, &t2)
                } else {
                    Type::Unknown
                }
            }
            Expr::Comma(_, rhs) => self.type_of_expr(rhs),
            Expr::Call { callee, args } => {
                // Still visit args even when the callee isn't a macro (or
                // is one but of the wrong shape) -- they may reference
                // other macros in their own right, and `type_of_expr` is
                // also how object-macro-cache warming happens.
                for a in args {
                    self.type_of_expr(a);
                }
                match callee.as_ref() {
                    Expr::Ident(name) if self.macros.contains_key(name) => {
                        self.type_of_macro_call(name, args)
                    }
                    _ => Type::Unknown,
                }
            }
            Expr::Index { base, index } => {
                self.type_of_expr(index);
                match self.type_of_expr(base) {
                    Type::Pointer(inner) | Type::Array(inner) => *inner,
                    _ => Type::Unknown,
                }
            }
            // Field layouts aren't modeled (Step 0 only collected coarse
            // tag kinds, not member lists) -- Step 3's job.
            Expr::Member { base, .. } => {
                self.type_of_expr(base);
                Type::Unknown
            }
            Expr::PostIncDec { expr, .. } | Expr::PreIncDec { expr, .. } => self.type_of_expr(expr),
            Expr::Cast { type_name, expr } => {
                self.type_of_expr(expr);
                type_from_type_name(type_name)
            }
            // size_t ~= `unsigned int` on the ILP32 target (see
            // `types.rs`'s module docs).
            Expr::Sizeof(SizeofArg::Expr(e)) => {
                self.type_of_expr(e);
                Type::UInt
            }
            Expr::Sizeof(SizeofArg::Type(_)) => Type::UInt,
        }
    }

    fn type_of_ident(&mut self, name: &str) -> Type {
        match self.macros.get(name) {
            Some(MacroBody::Object(_)) => self.type_of_object_macro(name),
            // A function-like macro referenced bare (not called), or one
            // whose body is `Empty`/`Statements`/`Unparseable` -- no
            // expression value to report.
            Some(_) => Type::Unknown,
            None => {
                if let Some(SymbolKind::EnumConstant) =
                    self.table.and_then(|t| t.lookup(name)).map(|s| s.kind)
                {
                    Type::Int
                } else {
                    Type::Unknown
                }
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
}

/// Walks every expression-bearing site in `unit` (function bodies,
/// declaration initializers, array-size/bit-field-width/enum-value
/// expressions -- the same shapes Step 1's `Resolver` reaches) collecting
/// each reference to a name in `macros`. No scope tracking: a macro name is
/// authoritative wherever it textually appears (see module docs).
pub fn collect_macro_uses(
    unit: &TranslationUnit,
    macros: &HashMap<String, MacroBody>,
) -> Vec<MacroUse> {
    let mut uses = Vec::new();
    let mut w = UseWalker {
        macros,
        uses: &mut uses,
    };
    for item in &unit.items {
        w.external_decl(item);
    }
    uses
}

struct UseWalker<'a> {
    macros: &'a HashMap<String, MacroBody>,
    uses: &'a mut Vec<MacroUse>,
}

impl UseWalker<'_> {
    fn external_decl(&mut self, item: &ExternalDecl) {
        match item {
            ExternalDecl::Declaration(decl) => self.declaration(decl),
            ExternalDecl::FunctionDef(f) => {
                self.decl_specifiers(&f.specifiers);
                self.declarator(&f.declarator);
                for item in &f.body.items {
                    self.block_item(item);
                }
            }
        }
    }

    fn declaration(&mut self, decl: &Declaration) {
        self.decl_specifiers(&decl.specifiers);
        for init_decl in &decl.declarators {
            self.declarator(&init_decl.declarator);
            if let Some(init) = &init_decl.initializer {
                self.initializer(init);
            }
        }
    }

    fn initializer(&mut self, init: &Initializer) {
        match init {
            Initializer::Expr(e) => self.expr(e),
            Initializer::List(items) => items.iter().for_each(|i| self.initializer(i)),
        }
    }

    fn decl_specifiers(&mut self, specs: &DeclSpecifiers) {
        for ts in &specs.type_specifiers {
            match ts {
                TypeSpecifier::Struct(s) | TypeSpecifier::Union(s) => {
                    for field in s.fields.iter().flatten() {
                        self.decl_specifiers(&field.specifiers);
                        for (declarator, bitwidth) in &field.declarators {
                            if let Some(d) = declarator {
                                self.declarator(d);
                            }
                            if let Some(w) = bitwidth {
                                self.expr(w);
                            }
                        }
                    }
                }
                TypeSpecifier::Enum(s) => {
                    for (_, value) in s.variants.iter().flatten() {
                        if let Some(e) = value {
                            self.expr(e);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn declarator(&mut self, d: &Declarator) {
        self.direct_declarator(&d.direct);
    }

    fn direct_declarator(&mut self, d: &DirectDeclarator) {
        match d {
            DirectDeclarator::Ident(_) => {}
            DirectDeclarator::Paren(inner) => self.declarator(inner),
            DirectDeclarator::Array(base, size) => {
                self.direct_declarator(base);
                if let Some(e) = size {
                    self.expr(e);
                }
            }
            DirectDeclarator::Function(base, params) => {
                self.direct_declarator(base);
                self.param_list(params);
            }
        }
    }

    fn param_list(&mut self, params: &ParamList) {
        for p in &params.params {
            self.decl_specifiers(&p.specifiers);
            match &p.declarator {
                ParamDeclarator::Named(d) => self.declarator(d),
                ParamDeclarator::Abstract(ad) => self.abstract_declarator(ad),
                ParamDeclarator::Bare => {}
            }
        }
    }

    fn abstract_declarator(&mut self, ad: &AbstractDeclarator) {
        if let Some(d) = &ad.direct {
            self.direct_abstract_declarator(d);
        }
    }

    fn direct_abstract_declarator(&mut self, d: &DirectAbstractDeclarator) {
        match d {
            DirectAbstractDeclarator::Paren(inner) => self.abstract_declarator(inner),
            DirectAbstractDeclarator::Array(base, size) => {
                if let Some(b) = base {
                    self.direct_abstract_declarator(b);
                }
                if let Some(e) = size {
                    self.expr(e);
                }
            }
            DirectAbstractDeclarator::Function(base, params) => {
                if let Some(b) = base {
                    self.direct_abstract_declarator(b);
                }
                self.param_list(params);
            }
        }
    }

    fn type_name(&mut self, t: &TypeName) {
        self.decl_specifiers(&t.specifiers);
        if let Some(ad) = &t.abstract_declarator {
            self.abstract_declarator(ad);
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
            Stmt::Expr(Some(e)) => self.expr(e),
            Stmt::Expr(None) => {}
            Stmt::Compound(cs) => cs.items.iter().for_each(|i| self.block_item(i)),
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
                if self.macros.contains_key(name) {
                    self.uses.push(MacroUse::Object(name.clone()));
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
                let is_macro_call = match callee.as_ref() {
                    Expr::Ident(name) if self.macros.contains_key(name) => {
                        self.uses.push(MacroUse::Call {
                            name: name.clone(),
                            args: args.clone(),
                        });
                        true
                    }
                    _ => false,
                };
                if !is_macro_call {
                    self.expr(callee);
                }
                for a in args {
                    self.expr(a);
                }
            }
            Expr::Index { base, index } => {
                self.expr(base);
                self.expr(index);
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{
        MacroBodyResolver, parse_expr_from_tokens, parse_full, parse_translation_unit,
    };
    use std::collections::HashSet as HSet;

    fn expr(src: &str) -> Expr {
        let (_, chunks) = crate::parser::parse_chunks(src);
        let entries = crate::parser::lex_chunks(&chunks).unwrap();
        let tokens: Vec<_> = entries
            .into_iter()
            .filter_map(|e| match e.item {
                crate::parser::LexItem::Token(t) => Some(t),
                _ => None,
            })
            .collect();
        parse_expr_from_tokens(tokens, HSet::new()).unwrap()
    }

    fn parse(src: &str) -> TranslationUnit {
        let (_, chunks) = crate::parser::parse_chunks(src);
        let mut env = crate::parser::PreprocessorEnv::linux_doom_defaults();
        let resolved = crate::parser::resolve_conditionals(&chunks, &mut env).unwrap();
        let entries = crate::parser::lex_chunks(&resolved).unwrap();
        let stream = crate::parser::attach_comments(entries);
        parse_translation_unit(&stream).unwrap()
    }

    fn macros(pairs: &[(&str, MacroBody)]) -> HashMap<String, MacroBody> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn test_object_macro_types_from_int_literal() {
        let m = macros(&[("FRACBITS", MacroBody::Object(expr("16")))]);
        let mut typer = MacroTyper::new(&m, None);
        assert_eq!(typer.type_of_object_macro("FRACBITS"), Type::Int);
    }

    #[test]
    fn test_object_macro_resolves_through_another_macro() {
        // FRACUNIT -> (1<<FRACBITS) -> int, after FRACBITS itself resolves.
        let m = macros(&[
            ("FRACBITS", MacroBody::Object(expr("16"))),
            ("FRACUNIT", MacroBody::Object(expr("(1<<FRACBITS)"))),
        ]);
        let mut typer = MacroTyper::new(&m, None);
        assert_eq!(typer.type_of_object_macro("FRACUNIT"), Type::Int);
    }

    #[test]
    fn test_cyclic_macro_reference_yields_unknown_not_infinite_loop() {
        let m = macros(&[
            ("A", MacroBody::Object(expr("B"))),
            ("B", MacroBody::Object(expr("A"))),
        ]);
        let mut typer = MacroTyper::new(&m, None);
        assert_eq!(typer.type_of_object_macro("A"), Type::Unknown);
    }

    #[test]
    fn test_function_like_macro_typed_per_call_site() {
        let m = macros(&[(
            "ADD",
            MacroBody::Function {
                params: vec!["a".to_string(), "b".to_string()],
                body: expr("(a) + (b)"),
            },
        )]);
        let mut typer = MacroTyper::new(&m, None);
        let args = vec![expr("1"), expr("2.0")];
        assert_eq!(typer.type_of_macro_call("ADD", &args), Type::Double);
    }

    #[test]
    fn test_function_like_macro_referenced_bare_is_unknown() {
        let m = macros(&[(
            "ADD",
            MacroBody::Function {
                params: vec!["a".to_string()],
                body: expr("a"),
            },
        )]);
        let mut typer = MacroTyper::new(&m, None);
        assert_eq!(
            typer.type_of_expr(&Expr::Ident("ADD".to_string())),
            Type::Unknown
        );
    }

    #[test]
    fn test_string_literal_macro_types_as_char_pointer() {
        let m = macros(&[("GREETING", MacroBody::Object(expr("\"hi\"")))]);
        let mut typer = MacroTyper::new(&m, None);
        assert_eq!(
            typer.type_of_object_macro("GREETING"),
            Type::Pointer(Box::new(Type::Char))
        );
    }

    #[test]
    fn test_cast_macro_uses_the_cast_type() {
        let m = macros(&[("AS_INT", MacroBody::Object(expr("(int)(3.0)")))]);
        let mut typer = MacroTyper::new(&m, None);
        assert_eq!(typer.type_of_object_macro("AS_INT"), Type::Int);
    }

    #[test]
    fn test_collect_macro_uses_finds_object_and_call_sites() {
        let unit = parse("int f(void) { int x = FRACUNIT; return ADD(x, 1); }");
        let m = macros(&[
            ("FRACUNIT", MacroBody::Object(expr("65536"))),
            (
                "ADD",
                MacroBody::Function {
                    params: vec!["a".to_string(), "b".to_string()],
                    body: expr("(a)+(b)"),
                },
            ),
        ]);
        let uses = collect_macro_uses(&unit, &m);
        assert!(uses.contains(&MacroUse::Object("FRACUNIT".to_string())));
        assert!(
            uses.iter()
                .any(|u| matches!(u, MacroUse::Call { name, .. } if name == "ADD"))
        );
    }

    #[test]
    fn test_resolve_fracunit_type_from_real_corpus_file() {
        // FRACUNIT is `#define FRACUNIT (1<<FRACBITS)` in m_fixed.h, with
        // `#define FRACBITS 16` alongside it.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let path = dir.join("m_fixed.c");
        let mut resolver = MacroBodyResolver::new();
        let macros = resolver.resolve(&path);
        let mut typer = MacroTyper::new(&macros, None);
        assert_eq!(typer.type_of_object_macro("FRACUNIT"), Type::Int);
    }

    #[test]
    fn test_corpus_macro_typing_coverage() {
        // Not a pass/fail assertion (matching this project's "measure
        // actual scope before deciding it needs more" methodology, same as
        // Step 0/1's own corpus tests) -- types every macro use found in
        // real code across the corpus and reports the typed/unknown split.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        files.sort();
        assert!(files.len() > 50, "expected the full Doom .c corpus");

        let mut body_resolver = MacroBodyResolver::new();
        let mut total_uses = 0usize;
        let mut typed = 0usize;
        let mut unknown = 0usize;
        for path in &files {
            let Ok((_, unit)) = parse_full(path.to_str().unwrap()) else {
                continue;
            };
            let macros = body_resolver.resolve(path);
            let uses = collect_macro_uses(&unit, &macros);
            let mut typer = MacroTyper::new(&macros, None);
            for u in &uses {
                total_uses += 1;
                let ty = match u {
                    MacroUse::Object(name) => typer.type_of_object_macro(name),
                    MacroUse::Call { name, args } => typer.type_of_macro_call(name, args),
                };
                if ty == Type::Unknown {
                    unknown += 1;
                } else {
                    typed += 1;
                }
            }
        }
        eprintln!(
            "macro typing over {} files: {total_uses} real-code macro references, \
             {typed} typed, {unknown} left Unknown (flagged for follow-up)",
            files.len()
        );
        assert!(total_uses > 0, "expected at least some macro references");
    }
}
