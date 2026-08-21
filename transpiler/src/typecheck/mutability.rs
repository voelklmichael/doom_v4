//! Step 5: Pointer Mutability Analysis (docs/02_TYPECHECKER.md Step 5)
//!
//! Infers, for every pointer-typed parameter of a corpus-defined function,
//! whether the function (directly, or transitively through calls) ever
//! mutates *through* it -- the C/Rust equivalent of choosing `&T` vs
//! `&mut T`. Two things count as evidence, both keyed by `ParamKey`
//! (`array_shape.rs`'s `(function name, parameter index)`):
//!
//! - **A direct write** (`collect_body_evidence`): an assignment or
//!   increment/decrement whose lvalue is reached from the parameter by
//!   dereferencing, indexing, or a member access (`*p = x`, `p[i]++`,
//!   `p->field += 1`) -- reassigning the pointer variable itself (`p = x;`)
//!   does *not* count, since that changes what `p` points to, not what it
//!   currently points at.
//! - **A call that could write through it** (`collect_call_evidence`): any
//!   argument reached from the parameter the same way (with or without a
//!   further `&`, e.g. `f(p)` or `f(&p->field)` or `f(pp->next)` for a
//!   pointer-to-pointer chain) is forwarding *some* pointer derived from
//!   the parameter into another function. If that function is a known
//!   corpus function, this becomes a forwarding edge resolved by the same
//!   kind of bounded fixpoint `array_shape.rs`'s `analyze` uses -- once the
//!   callee's corresponding parameter is known to be mutated through,
//!   that's real evidence the caller's parameter is too, however deep the
//!   chain. If the callee *isn't* a known corpus function (an indirect
//!   call through a function pointer, or an external/library function this
//!   project doesn't have a body for), its behavior can't be verified --
//!   per the spec's own "fall back to the conservative answer (mutable)
//!   rather than under-report" policy, that's immediate `Mutable` evidence,
//!   not a shrug.
//!
//! **No `Ambiguous` bucket, unlike Step 4**: mutability is binary. Absence
//! of any evidence classifies a parameter `Immutable` (a pointer only ever
//! read through is exactly that); any evidence at all -- a write, a
//! resolved forward, or an unresolved/conservative call -- classifies it
//! `Mutable`. Every pointer parameter of every corpus-defined function
//! gets one of the two, never left unclassified (unlike Step 4's "no
//! evidence found" outcome, which this step's spec doesn't allow for).
//!
//! **Scope, matching Step 4's own documented boundary**: only *syntactic*
//! derivation chains within a single expression are tracked (dereference,
//! index, member access, address-of, cast) -- an intermediate local
//! variable that copies the pointer first (`int *q = *pp; *q = 5;`) is
//! general local-variable data-flow, not a derivation chain or a
//! parameter-forwarding edge, and is out of scope here for the same reason
//! it was for Step 4 (see `docs/KNOWN_LIMITATIONS.md`).

use crate::parser::ast::*;
use crate::typecheck::declared_types::DeclaredTypes;
use crate::typecheck::types::{Type, param_type};
use std::collections::{HashMap, HashSet};

pub use crate::typecheck::array_shape::ParamKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Mutable,
    Immutable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Evidence {
    BodyWrite,
    ForwardedMutable { from: ParamKey },
    ConservativeIndirectCall { caller: String },
}

pub type EvidenceMap = HashMap<ParamKey, Vec<Evidence>>;
pub type ForwardEdges = Vec<(ParamKey, ParamKey)>;

/// A named function definition's pointer-typed parameters, in declared
/// order: `(name, pointee-including Type)` for each pointer parameter,
/// `None` for a non-pointer one (kept as a placeholder so positions still
/// line up with the declared signature).
fn pointer_params(f: &FunctionDef) -> Vec<Option<(String, Type)>> {
    let DirectDeclarator::Function(_, params) = &f.declarator.direct else {
        return Vec::new();
    };
    params
        .params
        .iter()
        .map(|p| {
            let ty = param_type(p);
            if !matches!(ty, Type::Pointer(_)) {
                return None;
            }
            match &p.declarator {
                ParamDeclarator::Named(d) => {
                    crate::parser::grammar::declarator_name(d).map(|n| (n, ty))
                }
                _ => None,
            }
        })
        .collect()
}

/// Peels off `Deref`/`AddrOf`/`Index`/`Member`/`Cast` layers to find the
/// identifier a chain is ultimately rooted at -- `None` if `e` isn't such a
/// chain at all (a literal, a call result, an unrelated computation, ...).
fn root_ident(e: &Expr) -> Option<&str> {
    match e {
        Expr::Ident(name) => Some(name),
        Expr::Unary {
            op: UnaryOp::Deref | UnaryOp::AddrOf,
            expr,
        } => root_ident(expr),
        Expr::Index { base, .. } => root_ident(base),
        Expr::Member { base, .. } => root_ident(base),
        Expr::Cast { expr, .. } => root_ident(expr),
        _ => None,
    }
}

/// Computes `e`'s `Type`, given `e`'s root is known to be `param_name` with
/// declared type `param_ty` -- mirrors `check.rs`'s expression typing, but
/// only for the handful of node kinds a derivation chain can be built from
/// (see `root_ident`), so it needs no general scope/locals tracking.
fn type_along_chain(
    e: &Expr,
    param_name: &str,
    param_ty: &Type,
    declared: &DeclaredTypes,
) -> Option<Type> {
    match e {
        Expr::Ident(name) if name == param_name => Some(param_ty.clone()),
        Expr::Unary {
            op: UnaryOp::Deref,
            expr,
        } => {
            let t = type_along_chain(expr, param_name, param_ty, declared)?;
            match declared.normalize(&t) {
                Type::Pointer(inner) | Type::Array(inner) => Some(*inner),
                _ => None,
            }
        }
        Expr::Unary {
            op: UnaryOp::AddrOf,
            expr,
        } => {
            let t = type_along_chain(expr, param_name, param_ty, declared)?;
            Some(Type::Pointer(Box::new(t)))
        }
        Expr::Index { base, .. } => {
            let t = type_along_chain(base, param_name, param_ty, declared)?;
            match declared.normalize(&t) {
                Type::Pointer(inner) | Type::Array(inner) => Some(*inner),
                _ => None,
            }
        }
        Expr::Member { base, field, arrow } => {
            let t = type_along_chain(base, param_name, param_ty, declared)?;
            let target = if *arrow {
                match declared.normalize(&t) {
                    Type::Pointer(inner) => *inner,
                    _ => return None,
                }
            } else {
                t
            };
            let tag = match declared.resolve_typedef(&target) {
                Type::Struct(n) | Type::Union(n) => n,
                _ => return None,
            };
            declared
                .fields
                .get(&tag)?
                .iter()
                .find(|(n, _)| n == field)
                .map(|(_, t)| t.clone())
        }
        Expr::Cast { expr, .. } => type_along_chain(expr, param_name, param_ty, declared),
        _ => None,
    }
}

struct BodyWalker<'w> {
    watched: &'w HashSet<&'w str>,
    written: HashSet<&'w str>,
}

impl<'w> BodyWalker<'w> {
    fn mark_lvalue(&mut self, e: &Expr) {
        // Reassigning the pointer variable itself (a bare `Ident`) doesn't
        // count -- only an actual dereference/index/member layer does.
        if matches!(e, Expr::Ident(_)) {
            return;
        }
        if let Some(name) = root_ident(e)
            && let Some(&key) = self.watched.get(name)
        {
            self.written.insert(key);
        }
    }

    fn block_item(&mut self, item: &BlockItem) {
        match item {
            BlockItem::Decl(d) => self.declaration(d),
            BlockItem::Stmt(s) => self.stmt(s),
        }
    }

    fn declaration(&mut self, decl: &Declaration) {
        for init_decl in &decl.declarators {
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
            Expr::Ident(_)
            | Expr::IntLiteral(_)
            | Expr::FloatLiteral(_)
            | Expr::StringLiteral(_)
            | Expr::CharLiteral(_) => {}
            Expr::Unary { expr, .. } => self.expr(expr),
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            Expr::Assign { lhs, rhs, .. } => {
                self.mark_lvalue(lhs);
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
                args.iter().for_each(|a| self.expr(a));
            }
            Expr::Index { base, index } => {
                self.expr(base);
                self.expr(index);
            }
            Expr::Member { base, .. } => self.expr(base),
            Expr::PostIncDec { expr, .. } | Expr::PreIncDec { expr, .. } => {
                self.mark_lvalue(expr);
                self.expr(expr);
            }
            Expr::Cast { expr, .. } => self.expr(expr),
            Expr::Sizeof(SizeofArg::Expr(e)) => self.expr(e),
            Expr::Sizeof(SizeofArg::Type(_)) => {}
        }
    }
}

/// Scans every function body in `unit` for a direct write through one of
/// its own pointer parameters (see module docs' first evidence kind).
pub fn collect_body_evidence(unit: &TranslationUnit) -> EvidenceMap {
    let mut out = HashMap::new();
    for item in &unit.items {
        let ExternalDecl::FunctionDef(f) = item else {
            continue;
        };
        let Some(name) = crate::parser::grammar::declarator_name(&f.declarator) else {
            continue;
        };
        let params = pointer_params(f);
        let watched: HashSet<&str> = params.iter().flatten().map(|(n, _)| n.as_str()).collect();
        if watched.is_empty() {
            continue;
        }
        let mut w = BodyWalker {
            watched: &watched,
            written: HashSet::new(),
        };
        for item in &f.body.items {
            w.block_item(item);
        }
        for (i, param) in params.iter().enumerate() {
            let Some((pname, _)) = param else { continue };
            if w.written.contains(pname.as_str()) {
                out.insert((name.clone(), i), vec![Evidence::BodyWrite]);
            }
        }
    }
    out
}

struct CallWalker<'a> {
    declared: &'a DeclaredTypes,
    corpus_functions: &'a HashSet<String>,
    current_function: Option<String>,
    current_params: Vec<Option<(String, Type)>>,
    evidence: EvidenceMap,
    forwards: ForwardEdges,
}

impl CallWalker<'_> {
    fn external_decl(&mut self, item: &ExternalDecl) {
        let ExternalDecl::FunctionDef(f) = item else {
            return;
        };
        let Some(name) = crate::parser::grammar::declarator_name(&f.declarator) else {
            return;
        };
        self.current_function = Some(name);
        self.current_params = pointer_params(f);
        for item in &f.body.items {
            self.block_item(item);
        }
        self.current_function = None;
        self.current_params.clear();
    }

    fn block_item(&mut self, item: &BlockItem) {
        match item {
            BlockItem::Decl(d) => {
                for init_decl in &d.declarators {
                    if let Some(init) = &init_decl.initializer {
                        self.initializer(init);
                    }
                }
            }
            BlockItem::Stmt(s) => self.stmt(s),
        }
    }

    fn initializer(&mut self, init: &Initializer) {
        match init {
            Initializer::Expr(e) => self.expr(e),
            Initializer::List(items) => items.iter().for_each(|i| self.initializer(i)),
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
                if let Some(ForInit::Expr(e)) = init {
                    self.expr(e);
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

    /// If `arg` is a pointer-shaped chain rooted at one of the current
    /// function's own watched parameters, records either a forwarding edge
    /// (callee is a known corpus function with a pointer parameter at this
    /// position) or immediate conservative evidence (anything else --
    /// see module docs).
    fn handle_arg(&mut self, arg: &Expr, callee: &Expr, position: usize) {
        let Some(root) = root_ident(arg) else { return };
        let Some(caller) = self.current_function.clone() else {
            return;
        };
        let Some((_, param_ty)) = self
            .current_params
            .iter()
            .flatten()
            .find(|(n, _)| n == root)
        else {
            return;
        };
        let Some(arg_ty) = type_along_chain(arg, root, param_ty, self.declared) else {
            return;
        };
        if !matches!(self.declared.normalize(&arg_ty), Type::Pointer(_)) {
            return; // not actually pointer-shaped (e.g. `*p` where p isn't pointer-to-pointer)
        }
        let param_index = self
            .current_params
            .iter()
            .position(|p| p.as_ref().is_some_and(|(n, _)| n == root))
            .unwrap();
        let caller_key = (caller.clone(), param_index);

        let resolved = match callee {
            Expr::Ident(callee_name) if self.corpus_functions.contains(callee_name) => self
                .declared
                .functions
                .get(callee_name)
                .and_then(|(sig, _)| sig.params.get(position))
                .filter(|t| matches!(t, Type::Pointer(_)))
                .map(|_| (callee_name.clone(), position)),
            _ => None,
        };
        match resolved {
            Some(callee_key) => self.forwards.push((caller_key, callee_key)),
            None => self
                .evidence
                .entry(caller_key)
                .or_default()
                .push(Evidence::ConservativeIndirectCall { caller }),
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Ident(_)
            | Expr::IntLiteral(_)
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
                for (i, a) in args.iter().enumerate() {
                    self.handle_arg(a, callee, i);
                    self.expr(a);
                }
                self.expr(callee);
            }
            Expr::Index { base, index } => {
                self.expr(base);
                self.expr(index);
            }
            Expr::Member { base, .. } => self.expr(base),
            Expr::PostIncDec { expr, .. } | Expr::PreIncDec { expr, .. } => self.expr(expr),
            Expr::Cast { expr, .. } => self.expr(expr),
            Expr::Sizeof(SizeofArg::Expr(e)) => self.expr(e),
            Expr::Sizeof(SizeofArg::Type(_)) => {}
        }
    }
}

/// Scans every call site in `unit` for an argument derived from one of the
/// enclosing function's own pointer parameters (see module docs' second
/// evidence kind).
pub fn collect_call_evidence(
    unit: &TranslationUnit,
    declared: &DeclaredTypes,
    corpus_functions: &HashSet<String>,
) -> (EvidenceMap, ForwardEdges) {
    let mut w = CallWalker {
        declared,
        corpus_functions,
        current_function: None,
        current_params: Vec::new(),
        evidence: HashMap::new(),
        forwards: Vec::new(),
    };
    for item in &unit.items {
        w.external_decl(item);
    }
    (w.evidence, w.forwards)
}

pub struct MutabilityAnalysis {
    pub evidence: EvidenceMap,
    pub mutability: HashMap<ParamKey, Mutability>,
}

/// Combines direct evidence with forwarding edges, iterating to a bounded
/// fixpoint the same way `array_shape.rs`'s `analyze` does: a resolved
/// `Mutable` callee parameter becomes `ForwardedMutable` evidence for
/// whatever caller parameter forwards to it. Every key in `all_params` gets
/// a classification -- `Immutable` by default when no evidence was ever
/// found, matching Step 5's spec (unlike Step 4, there's no "no evidence"
/// outcome here).
pub fn analyze(
    mut evidence: EvidenceMap,
    forwards: &[(ParamKey, ParamKey)],
    all_params: &HashSet<ParamKey>,
) -> MutabilityAnalysis {
    let mut mutable: HashSet<ParamKey> = evidence
        .iter()
        .filter(|(_, v)| !v.is_empty())
        .map(|(k, _)| k.clone())
        .collect();

    for _round in 0..25 {
        let mut changed = false;
        for (from, to) in forwards {
            if mutable.contains(from) || !mutable.contains(to) {
                continue;
            }
            let new_evidence = Evidence::ForwardedMutable { from: to.clone() };
            let entry = evidence.entry(from.clone()).or_default();
            if entry.contains(&new_evidence) {
                continue;
            }
            entry.push(new_evidence);
            mutable.insert(from.clone());
            changed = true;
        }
        if !changed {
            break;
        }
    }

    let mutability = all_params
        .iter()
        .map(|k| {
            let m = if mutable.contains(k) {
                Mutability::Mutable
            } else {
                Mutability::Immutable
            };
            (k.clone(), m)
        })
        .collect();

    MutabilityAnalysis {
        evidence,
        mutability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{
        PreprocessorEnv, parse_full, parse_translation_unit, resolve_conditionals,
    };
    use crate::typecheck::declared_types::DeclaredTypesResolver;
    use crate::typecheck::types::FunctionSignature;

    fn parse(src: &str) -> TranslationUnit {
        let (_, chunks) = crate::parser::parse_chunks(src);
        let mut env = PreprocessorEnv::linux_doom_defaults();
        let resolved = resolve_conditionals(&chunks, &mut env).unwrap();
        let entries = crate::parser::lex_chunks(&resolved).unwrap();
        let stream = crate::parser::attach_comments(entries);
        parse_translation_unit(&stream).unwrap()
    }

    fn classify(unit: &TranslationUnit, declared: &DeclaredTypes) -> HashMap<ParamKey, Mutability> {
        let mut corpus_functions = HashSet::new();
        for item in &unit.items {
            if let ExternalDecl::FunctionDef(f) = item
                && let Some(name) = crate::parser::grammar::declarator_name(&f.declarator)
            {
                corpus_functions.insert(name);
            }
        }
        let mut all_params = HashSet::new();
        for item in &unit.items {
            if let ExternalDecl::FunctionDef(f) = item
                && let Some(name) = crate::parser::grammar::declarator_name(&f.declarator)
            {
                for (i, p) in pointer_params(f).iter().enumerate() {
                    if p.is_some() {
                        all_params.insert((name.clone(), i));
                    }
                }
            }
        }
        let mut evidence = collect_body_evidence(unit);
        let (call_evidence, forwards) = collect_call_evidence(unit, declared, &corpus_functions);
        for (k, v) in call_evidence {
            evidence.entry(k).or_default().extend(v);
        }
        analyze(evidence, &forwards, &all_params).mutability
    }

    #[test]
    fn test_direct_dereference_write_is_mutable() {
        let unit = parse("void f(int *p) { *p = 1; }");
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Mutable);
    }

    #[test]
    fn test_index_write_is_mutable() {
        let unit = parse("void f(int *p) { p[0] = 1; }");
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Mutable);
    }

    #[test]
    fn test_reassigning_the_pointer_itself_is_not_mutation() {
        let unit = parse("void f(int *p) { p = 0; }");
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Immutable);
    }

    #[test]
    fn test_read_only_dereference_is_immutable() {
        let unit = parse("void f(int *p) { int x = *p + 1; }");
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Immutable);
    }

    #[test]
    fn test_member_write_through_arrow_is_mutable() {
        let src = "struct s_t { int x; }; void f(struct s_t *p) { p->x = 1; }";
        let unit = parse(src);
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Mutable);
    }

    #[test]
    fn test_compound_assignment_and_incdec_count_as_writes() {
        let unit = parse("void f(int *p) { *p += 1; }");
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Mutable);

        let unit2 = parse("void f(int *p) { (*p)++; }");
        let m2 = classify(&unit2, &DeclaredTypes::default());
        assert_eq!(m2[&("f".to_string(), 0)], Mutability::Mutable);
    }

    #[test]
    fn test_forwarding_unchanged_pointer_inherits_callee_mutation() {
        let unit = parse(
            "void B(int *q) { *q = 1; } \
             void A(int *p) { B(p); }",
        );
        let mut declared = DeclaredTypes::default();
        declared.functions.insert(
            "B".to_string(),
            (
                FunctionSignature {
                    ret: Type::Void,
                    params: vec![Type::Pointer(Box::new(Type::Int))],
                    variadic: false,
                },
                None,
            ),
        );
        let m = classify(&unit, &declared);
        assert_eq!(m[&("B".to_string(), 0)], Mutability::Mutable);
        assert_eq!(m[&("A".to_string(), 0)], Mutability::Mutable);
    }

    #[test]
    fn test_forwarding_address_of_field_inherits_callee_mutation() {
        // The spec's own example: `&(*p).field` passed to a function that
        // mutates through it counts as mutating through `p`.
        let unit = parse(
            "struct s_t { int x; }; \
             void B(int *q) { *q = 1; } \
             void A(struct s_t *p) { B(&p->x); }",
        );
        let mut declared = DeclaredTypes::default();
        declared
            .fields
            .insert("s_t".to_string(), vec![("x".to_string(), Type::Int)]);
        declared.functions.insert(
            "B".to_string(),
            (
                FunctionSignature {
                    ret: Type::Void,
                    params: vec![Type::Pointer(Box::new(Type::Int))],
                    variadic: false,
                },
                None,
            ),
        );
        let m = classify(&unit, &declared);
        assert_eq!(m[&("A".to_string(), 0)], Mutability::Mutable);
    }

    #[test]
    fn test_call_to_unknown_function_is_conservatively_mutable() {
        // `g` isn't a corpus function -- its behavior can't be verified,
        // so the spec's own conservative-fallback policy applies.
        let unit = parse("void f(int *p) { g(p); }");
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Mutable);
    }

    #[test]
    fn test_indirect_call_through_function_pointer_is_conservatively_mutable() {
        let unit = parse("void f(int *p, void (*fp)(int *)) { fp(p); }");
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Mutable);
    }

    #[test]
    fn test_pointer_to_pointer_double_dereference_write_is_mutable() {
        let unit = parse("void f(int **pp) { **pp = 1; }");
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Mutable);
    }

    #[test]
    fn test_dereferencing_a_value_argument_is_not_forwarding() {
        // `g(*p)` passes an `int` by value, not a pointer -- shouldn't be
        // treated as forwarding or trigger the conservative fallback.
        let unit = parse("void f(int *p) { g(*p); }");
        let m = classify(&unit, &DeclaredTypes::default());
        assert_eq!(m[&("f".to_string(), 0)], Mutability::Immutable);
    }

    #[test]
    fn test_corpus_mutability_inference_coverage() {
        // Not a pass/fail assertion (matching this project's "measure,
        // don't assume" methodology, same as every prior step).
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        files.sort();
        assert!(files.len() > 50, "expected the full Doom .c corpus");

        let units: Vec<_> = files
            .iter()
            .filter_map(|p| parse_full(p.to_str().unwrap()).ok().map(|(_, u)| u))
            .collect();

        let mut corpus_functions = HashSet::new();
        for unit in &units {
            corpus_functions.extend(crate::typecheck::array_shape::functions_with_bodies(unit));
        }

        let mut declared_resolver = DeclaredTypesResolver::new();
        let mut all_evidence: EvidenceMap = HashMap::new();
        let mut all_forwards = ForwardEdges::new();
        let mut all_params: HashSet<ParamKey> = HashSet::new();
        for (path, unit) in files.iter().zip(&units) {
            for (k, v) in collect_body_evidence(unit) {
                all_evidence.entry(k).or_default().extend(v);
            }
            let declared = declared_resolver.resolve(path);
            let (call_ev, fwd) = collect_call_evidence(unit, &declared, &corpus_functions);
            for (k, v) in call_ev {
                all_evidence.entry(k).or_default().extend(v);
            }
            all_forwards.extend(fwd);
            for item in &unit.items {
                if let ExternalDecl::FunctionDef(f) = item
                    && let Some(name) = crate::parser::grammar::declarator_name(&f.declarator)
                {
                    for (i, p) in pointer_params(f).iter().enumerate() {
                        if p.is_some() {
                            all_params.insert((name.clone(), i));
                        }
                    }
                }
            }
        }

        let analysis = analyze(all_evidence, &all_forwards, &all_params);
        let mutable_count = analysis
            .mutability
            .values()
            .filter(|m| **m == Mutability::Mutable)
            .count();
        let immutable_count = all_params.len() - mutable_count;
        eprintln!(
            "mutability inference over {} files: {} pointer parameters classified -- \
             {mutable_count} mutable, {immutable_count} immutable",
            files.len(),
            all_params.len(),
        );
        assert!(!all_params.is_empty());
    }
}
