//! Step 6: Pointer Nullability Analysis (docs/02_TYPECHECKER.md Step 6)
//!
//! Infers, for every pointer-typed parameter of a corpus-defined function,
//! whether it can actually be null in practice -- `&T` vs. `Option<&T>` in
//! Rust terms. Unlike Steps 4/5, this step keeps its two evidence sources
//! *separate* all the way to the final result, per the spec's own
//! "classification... from each evidence source independently, plus a
//! combined verdict": a parameter that the body unconditionally
//! dereferences but some call site passes a literal `NULL` to isn't
//! silently resolved either way -- that's a `Verdict::Conflict`, a real
//! finding.
//!
//! - **Call-site evidence** (`collect_call_evidence`): a call site passing
//!   a literal `0`/`NULL`, a local variable declared without an
//!   initializer, or `&x`/an array name/a string literal (inherently
//!   non-null) is direct evidence. A call site passing the *calling*
//!   function's own parameter unchanged is a forwarding edge instead --
//!   resolved by the same bounded fixpoint `array_shape.rs`/`mutability.rs`
//!   use, propagating the caller parameter's own eventual verdict forward.
//! - **Body evidence** (`collect_body_evidence`): a null-check on the
//!   parameter (`if (p)`, `if (!p)`, `if (p == NULL)`, ...) anywhere in the
//!   body is `Nullable` evidence -- the author thought it could be null.
//!   A dereference/index/member access reached with no preceding
//!   null-check *guarding* it is `NonNullable` evidence. "Guarding" here
//!   means one specific, dominant real idiom: a guard clause (`if
//!   (!p) return;` and friends) that diverges (`return`/`goto`/`break`/
//!   `continue`) marks the parameter safe for the rest of that block, and
//!   a positive check's `then`-branch (`if (p) { ... }`) marks it safe
//!   just within that branch. Anything not covered by one of those two
//!   shapes is conservatively still "unconditional" for this analysis --
//!   see `docs/KNOWN_LIMITATIONS.md` for what that misses.
//!
//! **Out of scope, per the spec's own words**: global/static pointers
//! reached from within a function body -- this is about function
//! *arguments* only, both at the call site and within the body.

use crate::parser::ast::*;
use crate::typecheck::declared_types::DeclaredTypes;
use crate::typecheck::types::{Type, param_type};
use std::collections::{HashMap, HashSet};

pub use crate::typecheck::array_shape::ParamKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Nullable,
    NonNullable,
    Conflict,
    NoEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    CallSite,
    Body,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Polarity {
    Nullable,
    NonNullable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Evidence {
    CallSiteNullLiteral { caller: String },
    CallSiteUninitializedVariable { caller: String },
    CallSiteNonNullAddress { caller: String },
    CallSiteForwardedNullable { from: ParamKey },
    BodyNullChecked,
    BodyUnconditionalDeref,
}

impl Evidence {
    fn source(&self) -> Source {
        match self {
            Evidence::CallSiteNullLiteral { .. }
            | Evidence::CallSiteUninitializedVariable { .. }
            | Evidence::CallSiteNonNullAddress { .. }
            | Evidence::CallSiteForwardedNullable { .. } => Source::CallSite,
            Evidence::BodyNullChecked | Evidence::BodyUnconditionalDeref => Source::Body,
        }
    }

    fn polarity(&self) -> Polarity {
        match self {
            Evidence::CallSiteNullLiteral { .. }
            | Evidence::CallSiteUninitializedVariable { .. }
            | Evidence::CallSiteForwardedNullable { .. }
            | Evidence::BodyNullChecked => Polarity::Nullable,
            Evidence::CallSiteNonNullAddress { .. } | Evidence::BodyUnconditionalDeref => {
                Polarity::NonNullable
            }
        }
    }
}

pub type EvidenceMap = HashMap<ParamKey, Vec<Evidence>>;
pub type ForwardEdges = Vec<(ParamKey, ParamKey)>;

fn verdict_of(evidence: &[Evidence], source: Source) -> Verdict {
    let mut nullable = false;
    let mut non_nullable = false;
    for e in evidence.iter().filter(|e| e.source() == source) {
        match e.polarity() {
            Polarity::Nullable => nullable = true,
            Polarity::NonNullable => non_nullable = true,
        }
    }
    match (nullable, non_nullable) {
        (true, true) => Verdict::Conflict,
        (true, false) => Verdict::Nullable,
        (false, true) => Verdict::NonNullable,
        (false, false) => Verdict::NoEvidence,
    }
}

/// The per-parameter result: each source's own verdict, plus the combined
/// one (matching the spec's own three-part validation criterion).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamVerdict {
    pub call_site: Verdict,
    pub body: Verdict,
    pub combined: Verdict,
}

fn combine(call_site: Verdict, body: Verdict) -> Verdict {
    use Verdict::*;
    match (call_site, body) {
        (Conflict, _) | (_, Conflict) => Conflict,
        (NoEvidence, other) | (other, NoEvidence) => other,
        (a, b) if a == b => a,
        _ => Conflict, // one says Nullable, the other NonNullable
    }
}

/// A named function definition's pointer-typed parameters, in declared
/// order -- `None` for a non-pointer one (kept as a placeholder so
/// positions still line up with the declared signature).
fn pointer_params(f: &FunctionDef) -> Vec<Option<String>> {
    let DirectDeclarator::Function(_, params) = &f.declarator.direct else {
        return Vec::new();
    };
    params
        .params
        .iter()
        .map(|p| {
            if !matches!(param_type(p), Type::Pointer(_)) {
                return None;
            }
            match &p.declarator {
                ParamDeclarator::Named(d) => crate::parser::grammar::declarator_name(d),
                _ => None,
            }
        })
        .collect()
}

fn is_null_literal(e: &Expr) -> bool {
    matches!(e, Expr::IntLiteral(s) if s == "0") || matches!(e, Expr::Ident(n) if n == "NULL")
}

/// If `cond` is a null-check on some identifier, returns `(name, true)`
/// when the check's *true* branch means "non-null" (`if (p)`, `if (p !=
/// NULL)`) or `(name, false)` when it means "null" (`if (!p)`, `if (p ==
/// NULL)`).
fn null_check_target(cond: &Expr) -> Option<(&str, bool)> {
    match cond {
        Expr::Ident(name) => Some((name, true)),
        Expr::Unary {
            op: UnaryOp::Not,
            expr,
        } => match expr.as_ref() {
            Expr::Ident(name) => Some((name, false)),
            _ => None,
        },
        Expr::Binary {
            op: BinaryOp::Eq,
            lhs,
            rhs,
        } => null_cmp(lhs, rhs, false),
        Expr::Binary {
            op: BinaryOp::Ne,
            lhs,
            rhs,
        } => null_cmp(lhs, rhs, true),
        _ => None,
    }
}

fn null_cmp<'e>(lhs: &'e Expr, rhs: &'e Expr, true_means_nonnull: bool) -> Option<(&'e str, bool)> {
    match (lhs, rhs) {
        (Expr::Ident(name), other) | (other, Expr::Ident(name)) if is_null_literal(other) => {
            Some((name, true_means_nonnull))
        }
        _ => None,
    }
}

/// Does `stmt` unconditionally leave the enclosing block (a guard clause's
/// `then`-branch shape)?
fn diverges(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Return(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue => true,
        // `I_Error` is linuxdoom-1.10's own fatal-error idiom (prints and
        // exits) -- doesn't syntactically `return`, but a guard clause
        // shaped `if (!p) I_Error(...);` is exactly as terminal for this
        // analysis's purposes as one that does. Corpus-measured to matter:
        // 84 `if (...) ... I_Error(...)` sites, dwarfing the handful of
        // bare-identifier NULL checks this codebase otherwise uses.
        Stmt::Expr(Some(Expr::Call { callee, .. })) => {
            matches!(callee.as_ref(), Expr::Ident(name) if name == "I_Error")
        }
        Stmt::Compound(cs) => cs.items.last().is_some_and(|item| match item {
            BlockItem::Stmt(s) => diverges(s),
            BlockItem::Decl(_) => false,
        }),
        _ => false,
    }
}

struct BodyWalker<'w> {
    watched: &'w HashSet<&'w str>,
    null_checked: HashSet<&'w str>,
    unconditional_deref: HashSet<&'w str>,
}

impl<'w> BodyWalker<'w> {
    fn note_null_check(&mut self, name: &str) {
        if let Some(&key) = self.watched.get(name) {
            self.null_checked.insert(key);
        }
    }

    fn note_deref(&mut self, name: &str, guarded: &HashSet<&str>) {
        if guarded.contains(name) {
            return;
        }
        if let Some(&key) = self.watched.get(name) {
            self.unconditional_deref.insert(key);
        }
    }

    /// Walks a sequence of block items left to right, threading `guarded`
    /// forward across sibling statements so a guard clause's effect
    /// persists for the rest of the block (see module docs).
    fn block_items(&mut self, items: &[BlockItem], guarded: &HashSet<&'w str>) {
        let mut guarded = guarded.clone();
        for item in items {
            match item {
                BlockItem::Decl(d) => self.declaration(d, &guarded),
                BlockItem::Stmt(s) => {
                    if let Some(name) = self.guard_clause_target(s) {
                        guarded.insert(name);
                    }
                    self.stmt(s, &guarded);
                }
            }
        }
    }

    /// If `s` is `if (<param is null>) { <diverges> }` (with no `else`),
    /// returns the guarded parameter name.
    fn guard_clause_target(&self, s: &Stmt) -> Option<&'w str> {
        let Stmt::If {
            cond,
            then_branch,
            else_branch: None,
        } = s
        else {
            return None;
        };
        let (name, means_nonnull) = null_check_target(cond)?;
        if means_nonnull || !diverges(then_branch) {
            return None;
        }
        self.watched.get(name).copied()
    }

    fn declaration(&mut self, decl: &Declaration, guarded: &HashSet<&str>) {
        for init_decl in &decl.declarators {
            if let Some(init) = &init_decl.initializer {
                self.initializer(init, guarded);
            }
        }
    }

    fn initializer(&mut self, init: &Initializer, guarded: &HashSet<&str>) {
        match init {
            Initializer::Expr(e) => self.expr(e, guarded),
            Initializer::List(items) => items.iter().for_each(|i| self.initializer(i, guarded)),
        }
    }

    fn stmt(&mut self, stmt: &Stmt, guarded: &HashSet<&'w str>) {
        match stmt {
            Stmt::Expr(Some(e)) => self.expr(e, guarded),
            Stmt::Expr(None) => {}
            Stmt::Compound(cs) => self.block_items(&cs.items, guarded),
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.cond_expr(cond, guarded);
                if let Some((name, means_nonnull)) = null_check_target(cond) {
                    let mut then_guarded = guarded.clone();
                    if means_nonnull && let Some(&k) = self.watched.get(name) {
                        then_guarded.insert(k);
                    }
                    self.stmt(then_branch, &then_guarded);
                    if let Some(else_branch) = else_branch {
                        let mut else_guarded = guarded.clone();
                        if !means_nonnull && let Some(&k) = self.watched.get(name) {
                            else_guarded.insert(k);
                        }
                        self.stmt(else_branch, &else_guarded);
                    }
                } else {
                    self.stmt(then_branch, guarded);
                    if let Some(e) = else_branch {
                        self.stmt(e, guarded);
                    }
                }
            }
            Stmt::Switch { cond, body } => {
                self.expr(cond, guarded);
                self.stmt(body, guarded);
            }
            Stmt::Case { expr, stmt } => {
                self.expr(expr, guarded);
                self.stmt(stmt, guarded);
            }
            Stmt::Default(stmt) => self.stmt(stmt, guarded),
            Stmt::While { cond, body } => {
                self.cond_expr(cond, guarded);
                self.stmt(body, guarded);
            }
            Stmt::DoWhile { body, cond } => {
                self.stmt(body, guarded);
                self.cond_expr(cond, guarded);
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
            } => {
                match init {
                    Some(ForInit::Decl(d)) => self.declaration(d, guarded),
                    Some(ForInit::Expr(e)) => self.expr(e, guarded),
                    None => {}
                }
                if let Some(e) = cond {
                    self.cond_expr(e, guarded);
                }
                if let Some(e) = step {
                    self.expr(e, guarded);
                }
                self.stmt(body, guarded);
            }
            Stmt::Goto(_) | Stmt::Continue | Stmt::Break => {}
            Stmt::Return(Some(e)) => self.expr(e, guarded),
            Stmt::Return(None) => {}
            Stmt::Labeled { stmt, .. } => self.stmt(stmt, guarded),
        }
    }

    /// A condition expression: recorded as a null-check (if it's one) in
    /// addition to being walked for its own derefs.
    fn cond_expr(&mut self, e: &Expr, guarded: &HashSet<&str>) {
        if let Some((name, _)) = null_check_target(e) {
            self.note_null_check(name);
        }
        self.expr(e, guarded);
    }

    fn expr(&mut self, e: &Expr, guarded: &HashSet<&str>) {
        match e {
            Expr::Ident(_)
            | Expr::IntLiteral(_)
            | Expr::FloatLiteral(_)
            | Expr::StringLiteral(_)
            | Expr::CharLiteral(_) => {}
            Expr::Unary {
                op: UnaryOp::Deref,
                expr,
            } => {
                if let Expr::Ident(name) = expr.as_ref() {
                    self.note_deref(name, guarded);
                }
                self.expr(expr, guarded);
            }
            Expr::Unary { expr, .. } => self.expr(expr, guarded),
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(lhs, guarded);
                self.expr(rhs, guarded);
            }
            Expr::Assign { lhs, rhs, .. } => {
                self.expr(lhs, guarded);
                self.expr(rhs, guarded);
            }
            Expr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => {
                self.cond_expr(cond, guarded);
                self.expr(then_expr, guarded);
                self.expr(else_expr, guarded);
            }
            Expr::Comma(a, b) => {
                self.expr(a, guarded);
                self.expr(b, guarded);
            }
            Expr::Call { callee, args } => {
                self.expr(callee, guarded);
                args.iter().for_each(|a| self.expr(a, guarded));
            }
            Expr::Index { base, index } => {
                if let Expr::Ident(name) = base.as_ref() {
                    self.note_deref(name, guarded);
                }
                self.expr(base, guarded);
                self.expr(index, guarded);
            }
            Expr::Member { base, arrow, .. } => {
                if *arrow && let Expr::Ident(name) = base.as_ref() {
                    self.note_deref(name, guarded);
                }
                self.expr(base, guarded);
            }
            Expr::PostIncDec { expr, .. } | Expr::PreIncDec { expr, .. } => {
                self.expr(expr, guarded)
            }
            Expr::Cast { expr, .. } => self.expr(expr, guarded),
            Expr::Sizeof(SizeofArg::Expr(e)) => self.expr(e, guarded),
            Expr::Sizeof(SizeofArg::Type(_)) => {}
        }
    }
}

/// Scans every function body in `unit` for null-checks and unconditional
/// dereferences of its own pointer parameters (see module docs' second
/// evidence kind).
pub fn collect_body_evidence(unit: &TranslationUnit) -> EvidenceMap {
    let mut out = HashMap::new();
    for item in &unit.items {
        let ExternalDecl::FunctionDef(f) = item else {
            continue;
        };
        let Some(name) = crate::parser::grammar::declarator_name(&f.declarator) else {
            continue;
        };
        let names = pointer_params(f);
        let watched: HashSet<&str> = names.iter().flatten().map(|s| s.as_str()).collect();
        if watched.is_empty() {
            continue;
        }
        let mut w = BodyWalker {
            watched: &watched,
            null_checked: HashSet::new(),
            unconditional_deref: HashSet::new(),
        };
        w.block_items(&f.body.items, &HashSet::new());
        for (i, param_name) in names.iter().enumerate() {
            let Some(param_name) = param_name else {
                continue;
            };
            let mut ev = Vec::new();
            if w.null_checked.contains(param_name.as_str()) {
                ev.push(Evidence::BodyNullChecked);
            }
            if w.unconditional_deref.contains(param_name.as_str()) {
                ev.push(Evidence::BodyUnconditionalDeref);
            }
            if !ev.is_empty() {
                out.insert((name.clone(), i), ev);
            }
        }
    }
    out
}

enum ArgShape {
    NullLiteral,
    UninitializedVariable,
    NonNullAddress,
    Forward(usize),
    Unknown,
}

struct CallWalker<'a> {
    declared: &'a DeclaredTypes,
    corpus_functions: &'a HashSet<String>,
    current_function: Option<String>,
    current_params: Vec<Option<String>>,
    uninitialized_locals: HashSet<String>,
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
        self.uninitialized_locals.clear();
        for item in &f.body.items {
            self.block_item(item);
        }
        self.current_function = None;
        self.current_params.clear();
    }

    fn block_item(&mut self, item: &BlockItem) {
        match item {
            BlockItem::Decl(d) => self.declaration(d),
            BlockItem::Stmt(s) => self.stmt(s),
        }
    }

    fn declaration(&mut self, decl: &Declaration) {
        for init_decl in &decl.declarators {
            let Some(name) = crate::parser::grammar::declarator_name(&init_decl.declarator) else {
                continue;
            };
            let is_ptr = matches!(
                crate::typecheck::types::type_from_declarator(
                    crate::typecheck::types::type_from_specifiers(&decl.specifiers),
                    &init_decl.declarator,
                ),
                Type::Pointer(_)
            );
            match &init_decl.initializer {
                None if is_ptr => {
                    self.uninitialized_locals.insert(name);
                }
                Some(init) => {
                    self.uninitialized_locals.remove(&name);
                    self.initializer(init);
                }
                None => {}
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

    fn classify_arg(&self, arg: &Expr) -> ArgShape {
        if is_null_literal(arg) {
            return ArgShape::NullLiteral;
        }
        if let Expr::Ident(name) = arg {
            if let Some(pos) = self
                .current_params
                .iter()
                .position(|p| p.as_deref() == Some(name.as_str()))
            {
                return ArgShape::Forward(pos);
            }
            if self.uninitialized_locals.contains(name) {
                return ArgShape::UninitializedVariable;
            }
            if matches!(
                self.declared.variables.get(name).map(|(t, _)| t),
                Some(Type::Array(_))
            ) {
                return ArgShape::NonNullAddress;
            }
            return ArgShape::Unknown;
        }
        match arg {
            Expr::StringLiteral(_) => ArgShape::NonNullAddress,
            Expr::Unary {
                op: UnaryOp::AddrOf,
                ..
            } => ArgShape::NonNullAddress,
            _ => ArgShape::Unknown,
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
                if let Expr::Ident(name) = lhs.as_ref() {
                    self.uninitialized_locals.remove(name);
                }
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
                if let Expr::Ident(callee_name) = callee.as_ref()
                    && self.corpus_functions.contains(callee_name)
                    && let Some((sig, _)) = self.declared.functions.get(callee_name)
                {
                    for (i, param_ty) in sig.params.iter().enumerate() {
                        if !matches!(param_ty, Type::Pointer(_)) {
                            continue;
                        }
                        let Some(arg) = args.get(i) else { continue };
                        let Some(caller) = self.current_function.clone() else {
                            continue;
                        };
                        let callee_key = (callee_name.clone(), i);
                        match self.classify_arg(arg) {
                            ArgShape::NullLiteral => {
                                self.evidence
                                    .entry(callee_key)
                                    .or_default()
                                    .push(Evidence::CallSiteNullLiteral { caller });
                            }
                            ArgShape::UninitializedVariable => {
                                self.evidence
                                    .entry(callee_key)
                                    .or_default()
                                    .push(Evidence::CallSiteUninitializedVariable { caller });
                            }
                            ArgShape::NonNullAddress => {
                                self.evidence
                                    .entry(callee_key)
                                    .or_default()
                                    .push(Evidence::CallSiteNonNullAddress { caller });
                            }
                            ArgShape::Forward(caller_idx) => {
                                self.forwards.push(((caller, caller_idx), callee_key));
                            }
                            ArgShape::Unknown => {}
                        }
                    }
                }
                self.expr(callee);
                args.iter().for_each(|a| self.expr(a));
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

/// Scans every call site in `unit` for an argument at a pointer-parameter
/// position of a corpus function, classifying its nullability shape (see
/// module docs' first evidence kind).
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
        uninitialized_locals: HashSet::new(),
        evidence: HashMap::new(),
        forwards: Vec::new(),
    };
    for item in &unit.items {
        w.external_decl(item);
    }
    (w.evidence, w.forwards)
}

pub struct NullabilityAnalysis {
    pub evidence: EvidenceMap,
    pub verdicts: HashMap<ParamKey, ParamVerdict>,
}

/// Combines direct evidence with forwarding edges to a bounded fixpoint:
/// once a forwarding edge's target has a resolved `combined` verdict of
/// `Nullable`, that becomes `CallSiteForwardedNullable` evidence for the
/// edge's source. Every key in `all_params` gets a full `ParamVerdict`.
pub fn analyze(
    mut evidence: EvidenceMap,
    forwards: &[(ParamKey, ParamKey)],
    all_params: &HashSet<ParamKey>,
) -> NullabilityAnalysis {
    let verdict_for = |evidence: &EvidenceMap, key: &ParamKey| -> Verdict {
        let ev = evidence.get(key).map(Vec::as_slice).unwrap_or(&[]);
        combine(
            verdict_of(ev, Source::CallSite),
            verdict_of(ev, Source::Body),
        )
    };

    for _round in 0..25 {
        let mut changed = false;
        // Note the direction here is the *opposite* of array_shape.rs/
        // mutability.rs's fixpoint: there, a callee's resolved behavior
        // informs the caller (whatever the callee does with the pointer,
        // the caller's copy of it must support). Here, it's the caller's
        // *already-known* nullability that informs the callee: if the
        // caller's parameter can be null and gets passed straight through
        // unchanged, the callee's corresponding parameter can be null too
        // (see module docs' call-site evidence: "another parameter/
        // variable already classified as nullable").
        for (from, to) in forwards {
            if verdict_for(&evidence, from) != Verdict::Nullable {
                continue;
            }
            let new_evidence = Evidence::CallSiteForwardedNullable { from: from.clone() };
            let entry = evidence.entry(to.clone()).or_default();
            if entry.contains(&new_evidence) {
                continue;
            }
            entry.push(new_evidence);
            changed = true;
        }
        if !changed {
            break;
        }
    }

    let verdicts = all_params
        .iter()
        .map(|k| {
            let ev = evidence.get(k).map(Vec::as_slice).unwrap_or(&[]);
            let call_site = verdict_of(ev, Source::CallSite);
            let body = verdict_of(ev, Source::Body);
            let combined = combine(call_site, body);
            (
                k.clone(),
                ParamVerdict {
                    call_site,
                    body,
                    combined,
                },
            )
        })
        .collect();

    NullabilityAnalysis { evidence, verdicts }
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

    fn classify(
        unit: &TranslationUnit,
        declared: &DeclaredTypes,
    ) -> HashMap<ParamKey, ParamVerdict> {
        let mut corpus_functions = HashSet::new();
        let mut all_params = HashSet::new();
        for item in &unit.items {
            if let ExternalDecl::FunctionDef(f) = item
                && let Some(name) = crate::parser::grammar::declarator_name(&f.declarator)
            {
                corpus_functions.insert(name.clone());
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
        analyze(evidence, &forwards, &all_params).verdicts
    }

    #[test]
    fn test_body_null_check_gives_nullable_body_verdict() {
        let unit = parse("void f(int *p) { if (p) { *p = 1; } }");
        let v = classify(&unit, &DeclaredTypes::default())[&("f".to_string(), 0)];
        assert_eq!(v.body, Verdict::Nullable);
    }

    #[test]
    fn test_unconditional_dereference_gives_nonnullable_body_verdict() {
        let unit = parse("void f(int *p) { *p = 1; }");
        let v = classify(&unit, &DeclaredTypes::default())[&("f".to_string(), 0)];
        assert_eq!(v.body, Verdict::NonNullable);
    }

    #[test]
    fn test_guard_clause_protects_rest_of_block_from_unconditional_flag() {
        let unit = parse("void f(int *p) { if (!p) return; *p = 1; }");
        let v = classify(&unit, &DeclaredTypes::default())[&("f".to_string(), 0)];
        // The guard clause itself is null-check evidence; the *subsequent*
        // dereference is protected by it and shouldn't also flag
        // NonNullable (that would wrongly manufacture a Conflict).
        assert_eq!(v.body, Verdict::Nullable);
    }

    #[test]
    fn test_positive_check_then_branch_is_guarded() {
        let unit = parse("void f(int *p) { if (p) { *p = 1; } }");
        let v = classify(&unit, &DeclaredTypes::default())[&("f".to_string(), 0)];
        assert_eq!(v.body, Verdict::Nullable);
    }

    #[test]
    fn test_unguarded_deref_after_non_diverging_check_is_a_conflict() {
        let unit = parse("void f(int *p) { if (p) { *p = 1; } *p = 2; }");
        let v = classify(&unit, &DeclaredTypes::default())[&("f".to_string(), 0)];
        assert_eq!(v.body, Verdict::Conflict);
    }

    #[test]
    fn test_call_site_null_literal_is_nullable() {
        let unit = parse("void g(int *p) {} void f(void) { g(0); }");
        let mut declared = DeclaredTypes::default();
        declared.functions.insert(
            "g".to_string(),
            (
                FunctionSignature {
                    ret: Type::Void,
                    params: vec![Type::Pointer(Box::new(Type::Int))],
                    variadic: false,
                },
                None,
            ),
        );
        let v = classify(&unit, &declared)[&("g".to_string(), 0)];
        assert_eq!(v.call_site, Verdict::Nullable);
    }

    #[test]
    fn test_call_site_uninitialized_local_is_nullable() {
        let unit = parse("void g(int *p) {} void f(void) { int *q; g(q); }");
        let mut declared = DeclaredTypes::default();
        declared.functions.insert(
            "g".to_string(),
            (
                FunctionSignature {
                    ret: Type::Void,
                    params: vec![Type::Pointer(Box::new(Type::Int))],
                    variadic: false,
                },
                None,
            ),
        );
        let v = classify(&unit, &declared)[&("g".to_string(), 0)];
        assert_eq!(v.call_site, Verdict::Nullable);
    }

    #[test]
    fn test_call_site_address_of_local_is_nonnullable() {
        let unit = parse("void g(int *p) {} void f(void) { int x; g(&x); }");
        let mut declared = DeclaredTypes::default();
        declared.functions.insert(
            "g".to_string(),
            (
                FunctionSignature {
                    ret: Type::Void,
                    params: vec![Type::Pointer(Box::new(Type::Int))],
                    variadic: false,
                },
                None,
            ),
        );
        let v = classify(&unit, &declared)[&("g".to_string(), 0)];
        assert_eq!(v.call_site, Verdict::NonNullable);
    }

    #[test]
    fn test_forwarding_nullable_parameter_propagates() {
        let unit = parse(
            "void B(int *q) {} \
             void A(int *p) { if (p) {} B(p); }",
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
        let verdicts = classify(&unit, &declared);
        assert_eq!(verdicts[&("A".to_string(), 0)].combined, Verdict::Nullable);
        assert_eq!(verdicts[&("B".to_string(), 0)].call_site, Verdict::Nullable);
    }

    #[test]
    fn test_disagreeing_sources_yield_conflict_combined_verdict() {
        // The spec's own example: body unconditionally dereferences, but a
        // call site passes a literal NULL.
        let unit = parse("void g(int *p) { *p = 1; } void f(void) { g(0); }");
        let mut declared = DeclaredTypes::default();
        declared.functions.insert(
            "g".to_string(),
            (
                FunctionSignature {
                    ret: Type::Void,
                    params: vec![Type::Pointer(Box::new(Type::Int))],
                    variadic: false,
                },
                None,
            ),
        );
        let v = classify(&unit, &declared)[&("g".to_string(), 0)];
        assert_eq!(v.body, Verdict::NonNullable);
        assert_eq!(v.call_site, Verdict::Nullable);
        assert_eq!(v.combined, Verdict::Conflict);
    }

    #[test]
    fn test_corpus_nullability_inference_coverage() {
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
        let mut combined_counts: HashMap<&str, usize> = HashMap::new();
        let mut call_site_counts: HashMap<&str, usize> = HashMap::new();
        let mut body_counts: HashMap<&str, usize> = HashMap::new();
        let verdict_name = |v: Verdict| match v {
            Verdict::Nullable => "nullable",
            Verdict::NonNullable => "non-nullable",
            Verdict::Conflict => "conflict",
            Verdict::NoEvidence => "no-evidence",
        };
        for pv in analysis.verdicts.values() {
            *combined_counts
                .entry(verdict_name(pv.combined))
                .or_insert(0) += 1;
            *call_site_counts
                .entry(verdict_name(pv.call_site))
                .or_insert(0) += 1;
            *body_counts.entry(verdict_name(pv.body)).or_insert(0) += 1;
        }
        eprintln!(
            "nullability inference over {} files: {} pointer parameters classified\n  \
             combined: {combined_counts:?}\n  call-site: {call_site_counts:?}\n  body: {body_counts:?}",
            files.len(),
            all_params.len(),
        );
        assert!(!all_params.is_empty());
    }
}
