//! Step 4: Pointer-to-Array Parameter Inference (docs/02_TYPECHECKER.md Step 4)
//!
//! A C `T *p` function parameter is, at the language level, indistinguishable
//! between "pointer to one `T`" and "pointer to the first element of a `T`
//! array" -- this step decides, per parameter, which one the source actually
//! means, from two independent kinds of evidence:
//! - **Body evidence** (`collect_body_evidence`): does the function's own
//!   body index the parameter (`p[i]`) or do pointer arithmetic on it
//!   (`p + i`, `*(p + i)`) -- beyond a single, plain `*p` dereference, which
//!   implies nothing either way?
//! - **Call-site evidence** (`collect_call_evidence`): across *every* file in
//!   the corpus (not just the parameter's own defining file -- a Doom
//!   function is typically declared in a header and called from many `.c`
//!   files), what shape is the actual argument at each call site: an array
//!   name (decays), `&array[i]`, a string literal (also array-shaped), or
//!   `&x` (a single object's address)?
//!
//! **Forwarding chains** (`analyze`'s fixpoint loop): when a call site's
//! argument is itself the *calling* function's own parameter (unchanged
//! forwarding, e.g. `void A(byte *p) { B(p); }`), that call site has no
//! shape evidence of its own -- but once `B`'s corresponding parameter's
//! shape is known, it becomes real evidence for `A`'s parameter too. This
//! can chain arbitrarily deep, so `analyze` doesn't stop at the first call
//! boundary: it iterates every forwarding edge to a fixpoint (bounded, as a
//! safety net against a pathological cycle rather than because deep chains
//! are expected in a 62-file corpus).
//!
//! **Conflicting evidence is a real finding, not noise**: matching the
//! spec's own framing, a parameter with both array and single-object
//! evidence (from different call sites, or body vs. call site) comes back
//! `Ambiguous` rather than picking a side silently.
//!
//! **Scope**: only parameters of functions that actually have a body
//! somewhere in the corpus are analyzed -- a libc function's parameters
//! aren't this project's to classify. A parameter declared with array
//! syntax (`T arr[]`) is unambiguous by construction and isn't run through
//! this inference at all (see `is_interesting_param`).

use crate::parser::ast::*;
use crate::typecheck::declared_types::DeclaredTypes;
use crate::typecheck::types::{Type, param_type, type_from_declarator, type_from_specifiers};
use std::collections::{HashMap, HashSet};

/// `(function name, parameter index)` -- the unit every piece of evidence
/// and every final shape is keyed by.
pub type ParamKey = (String, usize);

/// Every piece of collected evidence, keyed by the parameter it's about.
pub type EvidenceMap = HashMap<ParamKey, Vec<Evidence>>;

/// A forwarding edge: `(caller's parameter, callee's parameter)` -- the
/// caller's parameter should inherit whatever shape the callee's ends up
/// with (see `analyze`'s fixpoint).
pub type ForwardEdges = Vec<(ParamKey, ParamKey)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayShape {
    ArrayShaped,
    SingleObject,
    Ambiguous,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Evidence {
    BodyIndexed,
    BodyPointerArithmetic,
    CallSiteArray { caller: String },
    CallSiteSingleObject { caller: String },
    ForwardedArray { from: ParamKey },
    ForwardedSingleObject { from: ParamKey },
}

impl Evidence {
    fn is_array(&self) -> bool {
        matches!(
            self,
            Evidence::BodyIndexed
                | Evidence::BodyPointerArithmetic
                | Evidence::CallSiteArray { .. }
                | Evidence::ForwardedArray { .. }
        )
    }

    fn is_single_object(&self) -> bool {
        matches!(
            self,
            Evidence::CallSiteSingleObject { .. } | Evidence::ForwardedSingleObject { .. }
        )
    }
}

/// Every function name with a body (`FunctionDef`) somewhere in `unit` --
/// the scope boundary for this whole analysis (see module docs).
pub fn functions_with_bodies(unit: &TranslationUnit) -> HashSet<String> {
    unit.items
        .iter()
        .filter_map(|item| match item {
            ExternalDecl::FunctionDef(f) => crate::parser::grammar::declarator_name(&f.declarator),
            ExternalDecl::Declaration(_) => None,
        })
        .collect()
}

/// A plain named function definition's pointer-typed parameter names, in
/// order -- `None` for a non-pointer parameter (kept as a placeholder so
/// positions still line up with the declared signature). Array-declared
/// parameters (`T arr[]`) are excluded on purpose (see module docs).
fn pointer_param_names(f: &FunctionDef) -> Vec<Option<String>> {
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
                ParamDeclarator::Named(d) => crate::parser::grammar::declarator_name(d),
                _ => None,
            }
        })
        .collect()
}

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

/// Scans every function body in `unit` for direct evidence of how each of
/// its own pointer parameters is used (see module docs' "Body evidence").
pub fn collect_body_evidence(unit: &TranslationUnit) -> EvidenceMap {
    let mut out = HashMap::new();
    for item in &unit.items {
        let ExternalDecl::FunctionDef(f) = item else {
            continue;
        };
        let Some(name) = crate::parser::grammar::declarator_name(&f.declarator) else {
            continue;
        };
        let names = pointer_param_names(f);
        let watched: HashSet<&str> = names.iter().flatten().map(|s| s.as_str()).collect();
        if watched.is_empty() {
            continue;
        }
        let mut flags: HashMap<&str, (bool, bool)> = HashMap::new();
        {
            let mut w = BodyEvidenceWalker {
                watched: &watched,
                flags: &mut flags,
            };
            for item in &f.body.items {
                w.block_item(item);
            }
        }
        for (i, param_name) in names.iter().enumerate() {
            let Some(param_name) = param_name else {
                continue;
            };
            let (indexed, arithmetic) = flags.get(param_name.as_str()).copied().unwrap_or_default();
            let mut ev = Vec::new();
            if indexed {
                ev.push(Evidence::BodyIndexed);
            }
            if arithmetic {
                ev.push(Evidence::BodyPointerArithmetic);
            }
            if !ev.is_empty() {
                out.insert((name.clone(), i), ev);
            }
        }
    }
    out
}

struct BodyEvidenceWalker<'w, 'f> {
    watched: &'w HashSet<&'w str>,
    flags: &'f mut HashMap<&'w str, (bool, bool)>,
}

impl BodyEvidenceWalker<'_, '_> {
    fn mark(&mut self, name: &str, indexed: bool, arithmetic: bool) {
        let Some(&key) = self.watched.get(name) else {
            return;
        };
        let entry = self.flags.entry(key).or_default();
        entry.0 |= indexed;
        entry.1 |= arithmetic;
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
            Expr::Binary { op, lhs, rhs } => {
                if matches!(op, BinaryOp::Add | BinaryOp::Sub) {
                    if let Expr::Ident(name) = lhs.as_ref() {
                        self.mark(name, false, true);
                    }
                    if let Expr::Ident(name) = rhs.as_ref() {
                        self.mark(name, false, true);
                    }
                }
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
                args.iter().for_each(|a| self.expr(a));
            }
            Expr::Index { base, index } => {
                if let Expr::Ident(name) = base.as_ref() {
                    self.mark(name, true, false);
                }
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

/// The shape a call argument's own expression syntactically suggests.
enum ArgShape {
    Array,
    SingleObject,
    Forward(usize),
    Unknown,
}

/// Scans every call site in `unit` to a function in `corpus_functions`,
/// classifying each pointer-parameter argument's shape (see module docs'
/// "Call-site evidence" and "Forwarding chains"). Returns direct evidence
/// plus every forwarding edge found, to be resolved by `analyze`'s
/// fixpoint once every file's evidence is combined.
pub fn collect_call_evidence(
    unit: &TranslationUnit,
    declared: &DeclaredTypes,
    corpus_functions: &HashSet<String>,
) -> (EvidenceMap, ForwardEdges) {
    let mut w = CallEvidenceWalker {
        declared,
        corpus_functions,
        scope: TypeScope::new(),
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

struct CallEvidenceWalker<'a> {
    declared: &'a DeclaredTypes,
    corpus_functions: &'a HashSet<String>,
    scope: TypeScope,
    current_function: Option<String>,
    current_params: Vec<Option<String>>,
    evidence: EvidenceMap,
    forwards: ForwardEdges,
}

impl CallEvidenceWalker<'_> {
    fn external_decl(&mut self, item: &ExternalDecl) {
        let ExternalDecl::FunctionDef(f) = item else {
            return;
        };
        let base = type_from_specifiers(&f.specifiers);
        let Some(name) = crate::parser::grammar::declarator_name(&f.declarator) else {
            return;
        };
        self.current_function = Some(name);
        self.current_params = pointer_param_names(f);
        self.scope.enter();
        if let DirectDeclarator::Function(_, params) = &f.declarator.direct {
            for p in &params.params {
                if let ParamDeclarator::Named(pd) = &p.declarator
                    && let Some(pname) = crate::parser::grammar::declarator_name(pd)
                {
                    self.scope.declare(pname, param_type(p));
                }
            }
        }
        for item in &f.body.items {
            self.block_item(item);
        }
        self.scope.exit();
        let _ = base;
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
        let base = type_from_specifiers(&decl.specifiers);
        let is_typedef = decl.specifiers.storage == Some(StorageClass::Typedef);
        for init_decl in &decl.declarators {
            let ty = type_from_declarator(base.clone(), &init_decl.declarator);
            if !is_typedef
                && let Some(name) = crate::parser::grammar::declarator_name(&init_decl.declarator)
            {
                self.scope.declare(name, ty);
            }
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
            Stmt::Compound(cs) => {
                self.scope.enter();
                cs.items.iter().for_each(|i| self.block_item(i));
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
                self.scope.exit();
            }
            Stmt::Goto(_) | Stmt::Continue | Stmt::Break => {}
            Stmt::Return(Some(e)) => self.expr(e),
            Stmt::Return(None) => {}
            Stmt::Labeled { stmt, .. } => self.stmt(stmt),
        }
    }

    fn classify_arg(&self, arg: &Expr) -> ArgShape {
        if let Expr::Ident(name) = arg
            && let Some(pos) = self
                .current_params
                .iter()
                .position(|p| p.as_deref() == Some(name.as_str()))
        {
            return ArgShape::Forward(pos);
        }
        match arg {
            Expr::Ident(name) => match self
                .scope
                .lookup(name)
                .or_else(|| self.declared.variables.get(name).map(|(t, _)| t))
            {
                Some(Type::Array(_)) => ArgShape::Array,
                _ => ArgShape::Unknown,
            },
            Expr::StringLiteral(_) => ArgShape::Array,
            Expr::Unary {
                op: UnaryOp::AddrOf,
                expr,
            } => match expr.as_ref() {
                Expr::Index { .. } => ArgShape::Array,
                _ => ArgShape::SingleObject,
            },
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
                        let callee_key = (callee_name.clone(), i);
                        match self.classify_arg(arg) {
                            ArgShape::Array => {
                                let Some(caller) = self.current_function.clone() else {
                                    continue;
                                };
                                self.evidence
                                    .entry(callee_key)
                                    .or_default()
                                    .push(Evidence::CallSiteArray { caller });
                            }
                            ArgShape::SingleObject => {
                                let Some(caller) = self.current_function.clone() else {
                                    continue;
                                };
                                self.evidence
                                    .entry(callee_key)
                                    .or_default()
                                    .push(Evidence::CallSiteSingleObject { caller });
                            }
                            ArgShape::Forward(caller_param_idx) => {
                                let Some(caller) = self.current_function.clone() else {
                                    continue;
                                };
                                self.forwards.push(((caller, caller_param_idx), callee_key));
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

fn combine(evidence: &[Evidence]) -> Option<ArrayShape> {
    let has_array = evidence.iter().any(Evidence::is_array);
    let has_single = evidence.iter().any(Evidence::is_single_object);
    match (has_array, has_single) {
        (true, true) => Some(ArrayShape::Ambiguous),
        (true, false) => Some(ArrayShape::ArrayShaped),
        (false, true) => Some(ArrayShape::SingleObject),
        (false, false) => None,
    }
}

pub struct ArrayShapeAnalysis {
    pub evidence: EvidenceMap,
    pub shapes: HashMap<ParamKey, ArrayShape>,
}

/// Combines every file's direct evidence with the forwarding edges found
/// across the whole corpus, then iterates to a fixpoint: whenever a
/// forwarding edge's target now has a known shape, that becomes new
/// evidence for the edge's source, which may in turn change *its* shape
/// and feed further edges -- capped at a generous round limit as a safety
/// net against a pathological cycle, not because real chains run that deep.
pub fn analyze(mut evidence: EvidenceMap, forwards: &[(ParamKey, ParamKey)]) -> ArrayShapeAnalysis {
    let mut shapes: HashMap<ParamKey, ArrayShape> = evidence
        .iter()
        .filter_map(|(k, v)| combine(v).map(|s| (k.clone(), s)))
        .collect();

    for _round in 0..25 {
        let mut changed = false;
        for (from, to) in forwards {
            let Some(&to_shape) = shapes.get(to) else {
                continue;
            };
            let new_evidence = match to_shape {
                ArrayShape::ArrayShaped => Evidence::ForwardedArray { from: to.clone() },
                ArrayShape::SingleObject => Evidence::ForwardedSingleObject { from: to.clone() },
                ArrayShape::Ambiguous => continue,
            };
            let entry = evidence.entry(from.clone()).or_default();
            if entry.contains(&new_evidence) {
                continue;
            }
            entry.push(new_evidence);
            let new_shape = combine(entry);
            if shapes.get(from) != new_shape.as_ref() {
                if let Some(s) = new_shape {
                    shapes.insert(from.clone(), s);
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    ArrayShapeAnalysis { evidence, shapes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{
        PreprocessorEnv, parse_full, parse_translation_unit, resolve_conditionals,
    };
    use crate::typecheck::declared_types::DeclaredTypesResolver;

    fn parse(src: &str) -> TranslationUnit {
        let (_, chunks) = crate::parser::parse_chunks(src);
        let mut env = PreprocessorEnv::linux_doom_defaults();
        let resolved = resolve_conditionals(&chunks, &mut env).unwrap();
        let entries = crate::parser::lex_chunks(&resolved).unwrap();
        let stream = crate::parser::attach_comments(entries);
        parse_translation_unit(&stream).unwrap()
    }

    #[test]
    fn test_indexed_parameter_is_array_shaped() {
        let unit = parse("void f(int *p) { p[0] = 1; }");
        let evidence = collect_body_evidence(&unit);
        let analysis = analyze(evidence, &[]);
        assert_eq!(
            analysis.shapes.get(&("f".to_string(), 0)),
            Some(&ArrayShape::ArrayShaped)
        );
    }

    #[test]
    fn test_pointer_arithmetic_is_array_shaped() {
        let unit = parse("void f(int *p) { int x = *(p + 1); }");
        let evidence = collect_body_evidence(&unit);
        let analysis = analyze(evidence, &[]);
        assert_eq!(
            analysis.shapes.get(&("f".to_string(), 0)),
            Some(&ArrayShape::ArrayShaped)
        );
    }

    #[test]
    fn test_plain_dereference_gives_no_body_evidence() {
        let unit = parse("void f(int *p) { int x = *p; }");
        let evidence = collect_body_evidence(&unit);
        assert!(!evidence.contains_key(&("f".to_string(), 0)));
    }

    #[test]
    fn test_call_site_array_argument() {
        let unit = parse("void g(int *p); void f(void) { int arr[4]; g(arr); }");
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
        let corpus_functions = ["g".to_string()].into_iter().collect();
        let (evidence, forwards) = collect_call_evidence(&unit, &declared, &corpus_functions);
        assert!(forwards.is_empty());
        let analysis = analyze(evidence, &forwards);
        assert_eq!(
            analysis.shapes.get(&("g".to_string(), 0)),
            Some(&ArrayShape::ArrayShaped)
        );
    }

    #[test]
    fn test_call_site_address_of_index_is_array_evidence() {
        let unit = parse("void g(int *p); void f(void) { int arr[4]; g(&arr[0]); }");
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
        let corpus_functions = ["g".to_string()].into_iter().collect();
        let (evidence, forwards) = collect_call_evidence(&unit, &declared, &corpus_functions);
        let analysis = analyze(evidence, &forwards);
        assert_eq!(
            analysis.shapes.get(&("g".to_string(), 0)),
            Some(&ArrayShape::ArrayShaped)
        );
    }

    #[test]
    fn test_call_site_address_of_variable_is_single_object_evidence() {
        let unit = parse("void g(int *p); void f(void) { int x; g(&x); }");
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
        let corpus_functions = ["g".to_string()].into_iter().collect();
        let (evidence, forwards) = collect_call_evidence(&unit, &declared, &corpus_functions);
        let analysis = analyze(evidence, &forwards);
        assert_eq!(
            analysis.shapes.get(&("g".to_string(), 0)),
            Some(&ArrayShape::SingleObject)
        );
    }

    #[test]
    fn test_conflicting_evidence_is_ambiguous() {
        let mut evidence = HashMap::new();
        evidence.insert(
            ("g".to_string(), 0),
            vec![
                Evidence::CallSiteArray {
                    caller: "a".to_string(),
                },
                Evidence::CallSiteSingleObject {
                    caller: "b".to_string(),
                },
            ],
        );
        let analysis = analyze(evidence, &[]);
        assert_eq!(
            analysis.shapes.get(&("g".to_string(), 0)),
            Some(&ArrayShape::Ambiguous)
        );
    }

    #[test]
    fn test_forwarding_chain_propagates_through_fixpoint() {
        // A(byte *p) just forwards p to B(byte *q), and only B's body
        // actually indexes it -- A's own parameter has no direct evidence
        // of its own, only via the forwarding edge.
        let unit = parse(
            "void B(int *q) { q[0] = 1; } \
             void A(int *p) { B(p); }",
        );
        let mut declared = DeclaredTypes::default();
        declared.functions.insert(
            "B".to_string(),
            (
                crate::typecheck::types::FunctionSignature {
                    ret: Type::Void,
                    params: vec![Type::Pointer(Box::new(Type::Int))],
                    variadic: false,
                },
                None,
            ),
        );
        let corpus_functions = ["A".to_string(), "B".to_string()].into_iter().collect();
        let body_evidence = collect_body_evidence(&unit);
        let (call_evidence, forwards) = collect_call_evidence(&unit, &declared, &corpus_functions);
        assert_eq!(forwards, vec![(("A".to_string(), 0), ("B".to_string(), 0))]);
        let mut all_evidence = body_evidence;
        for (k, v) in call_evidence {
            all_evidence.entry(k).or_default().extend(v);
        }
        let analysis = analyze(all_evidence, &forwards);
        assert_eq!(
            analysis.shapes.get(&("B".to_string(), 0)),
            Some(&ArrayShape::ArrayShaped)
        );
        assert_eq!(
            analysis.shapes.get(&("A".to_string(), 0)),
            Some(&ArrayShape::ArrayShaped),
            "A's parameter should inherit B's shape through the forwarding edge"
        );
    }

    #[test]
    fn test_array_declared_parameter_is_excluded_from_inference() {
        // `int arr[]` is unambiguous by construction -- not a `T *`
        // parameter this step needs to infer anything about.
        let unit = parse("void f(int arr[]) { arr[0] = 1; }");
        let evidence = collect_body_evidence(&unit);
        assert!(evidence.is_empty());
    }

    #[test]
    fn test_corpus_array_shape_inference_coverage() {
        // Not a pass/fail assertion (matching this project's "measure,
        // don't assume" methodology, same as every prior step) -- runs the
        // full interprocedural analysis over the corpus and reports the
        // array/single-object/ambiguous/no-evidence split.
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
        assert!(units.len() > 50);

        let mut corpus_functions = HashSet::new();
        for unit in &units {
            corpus_functions.extend(functions_with_bodies(unit));
        }

        let mut declared_resolver = DeclaredTypesResolver::new();
        let mut all_evidence: EvidenceMap = HashMap::new();
        let mut all_forwards = Vec::new();
        let mut interesting_params: HashSet<ParamKey> = HashSet::new();
        for (path, unit) in files.iter().zip(&units) {
            for (k, v) in collect_body_evidence(unit) {
                interesting_params.insert(k.clone());
                all_evidence.entry(k).or_default().extend(v);
            }
            let declared = declared_resolver.resolve(path);
            let (call_ev, fwd) = collect_call_evidence(unit, &declared, &corpus_functions);
            for (k, v) in call_ev {
                all_evidence.entry(k).or_default().extend(v);
            }
            all_forwards.extend(fwd);
        }
        // Every pointer parameter of every corpus-defined function is "in
        // scope" for the measurement, whether or not any evidence was
        // found for it -- a missing entry is itself a measured outcome
        // ("no evidence"), not something to skip silently.
        for unit in &units {
            for item in &unit.items {
                if let ExternalDecl::FunctionDef(f) = item
                    && let Some(name) = crate::parser::grammar::declarator_name(&f.declarator)
                {
                    for (i, p) in pointer_param_names(f).iter().enumerate() {
                        if p.is_some() {
                            interesting_params.insert((name.clone(), i));
                        }
                    }
                }
            }
        }

        let analysis = analyze(all_evidence, &all_forwards);
        let mut array_count = 0;
        let mut single_count = 0;
        let mut ambiguous_count = 0;
        let mut no_evidence_count = 0;
        for key in &interesting_params {
            match analysis.shapes.get(key) {
                Some(ArrayShape::ArrayShaped) => array_count += 1,
                Some(ArrayShape::SingleObject) => single_count += 1,
                Some(ArrayShape::Ambiguous) => ambiguous_count += 1,
                None => no_evidence_count += 1,
            }
        }
        eprintln!(
            "array-shape inference over {} files: {} pointer parameters analyzed -- \
             {array_count} array-shaped, {single_count} single-object, {ambiguous_count} \
             ambiguous, {no_evidence_count} no evidence found",
            files.len(),
            interesting_params.len(),
        );
        assert!(!interesting_params.is_empty());
    }
}
