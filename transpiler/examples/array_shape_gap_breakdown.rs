use std::collections::HashSet;
use transpiler::parser::parse_full;
use transpiler::typecheck::declared_types::DeclaredTypesResolver;
use transpiler::typecheck::{
    analyze, collect_body_evidence, collect_call_evidence, functions_with_bodies,
};

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
        .collect();
    files.sort();

    let units: Vec<_> = files
        .iter()
        .filter_map(|p| parse_full(p.to_str().unwrap()).ok().map(|(_, u)| u))
        .collect();

    let mut corpus_functions = HashSet::new();
    for unit in &units {
        corpus_functions.extend(functions_with_bodies(unit));
    }

    // Count how many corpus-defined functions are ever the callee of a
    // direct `Ident(name)(args)` call site anywhere in the corpus.
    let mut ever_called_directly: HashSet<String> = HashSet::new();
    for unit in &units {
        collect_direct_callees(unit, &mut ever_called_directly);
    }

    let mut declared_resolver = DeclaredTypesResolver::new();
    let mut all_evidence = std::collections::HashMap::new();
    let mut all_forwards = Vec::new();
    for (path, unit) in files.iter().zip(&units) {
        for (k, v) in collect_body_evidence(unit) {
            all_evidence.entry(k).or_insert_with(Vec::new).extend(v);
        }
        let declared = declared_resolver.resolve(path);
        let (call_ev, fwd) = collect_call_evidence(unit, &declared, &corpus_functions);
        for (k, v) in call_ev {
            all_evidence.entry(k).or_insert_with(Vec::new).extend(v);
        }
        all_forwards.extend(fwd);
    }

    let mut interesting: HashSet<(String, usize)> = HashSet::new();
    let mut fn_of_param: std::collections::HashMap<(String, usize), String> =
        std::collections::HashMap::new();
    for unit in &units {
        for item in &unit.items {
            if let transpiler::parser::ast::ExternalDecl::FunctionDef(f) = item
                && let Some(name) = decl_name(&f.declarator)
            {
                for (i, p) in transpiler_pointer_param_names(f).iter().enumerate() {
                    if p.is_some() {
                        interesting.insert((name.clone(), i));
                        fn_of_param.insert((name.clone(), i), name.clone());
                    }
                }
            }
        }
    }

    let analysis = analyze(all_evidence, &all_forwards);

    let mut no_evidence_never_called_directly = 0;
    let mut no_evidence_but_called_directly = 0;
    for key in &interesting {
        if analysis.shapes.contains_key(key) {
            continue;
        }
        let fname = &fn_of_param[key];
        if ever_called_directly.contains(fname) {
            no_evidence_but_called_directly += 1;
        } else {
            no_evidence_never_called_directly += 1;
        }
    }
    println!(
        "no-evidence params: {} whose function is never called directly anywhere in the corpus \
         (likely invoked only through a function-pointer table), {} whose function IS called \
         directly but this specific parameter still got no shape evidence (e.g. only \
         single-dereferenced, or only ever passed already-typed pointer variables)",
        no_evidence_never_called_directly, no_evidence_but_called_directly
    );
}

fn collect_direct_callees(
    unit: &transpiler::parser::ast::TranslationUnit,
    out: &mut HashSet<String>,
) {
    use transpiler::parser::ast::*;
    fn expr(e: &Expr, out: &mut HashSet<String>) {
        if let Expr::Call { callee, args } = e {
            if let Expr::Ident(name) = callee.as_ref() {
                out.insert(name.clone());
            }
            expr(callee, out);
            for a in args {
                expr(a, out);
            }
            return;
        }
        match e {
            Expr::Unary { expr: inner, .. } => expr(inner, out),
            Expr::Binary { lhs, rhs, .. } | Expr::Assign { lhs, rhs, .. } => {
                expr(lhs, out);
                expr(rhs, out);
            }
            Expr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => {
                expr(cond, out);
                expr(then_expr, out);
                expr(else_expr, out);
            }
            Expr::Comma(a, b) => {
                expr(a, out);
                expr(b, out);
            }
            Expr::Index { base, index } => {
                expr(base, out);
                expr(index, out);
            }
            Expr::Member { base, .. } => expr(base, out),
            Expr::PostIncDec { expr: inner, .. } | Expr::PreIncDec { expr: inner, .. } => {
                expr(inner, out)
            }
            Expr::Cast { expr: inner, .. } => expr(inner, out),
            Expr::Sizeof(SizeofArg::Expr(inner)) => expr(inner, out),
            _ => {}
        }
    }
    fn stmt(s: &Stmt, out: &mut HashSet<String>) {
        match s {
            Stmt::Expr(Some(e)) => expr(e, out),
            Stmt::Compound(cs) => cs.items.iter().for_each(|i| block_item(i, out)),
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                expr(cond, out);
                stmt(then_branch, out);
                if let Some(e) = else_branch {
                    stmt(e, out);
                }
            }
            Stmt::Switch { cond, body } => {
                expr(cond, out);
                stmt(body, out);
            }
            Stmt::Case { expr: e, stmt: s } => {
                expr(e, out);
                stmt(s, out);
            }
            Stmt::Default(s) => stmt(s, out),
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                expr(cond, out);
                stmt(body, out);
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
            } => {
                if let Some(ForInit::Expr(e)) = init {
                    expr(e, out);
                }
                if let Some(e) = cond {
                    expr(e, out);
                }
                if let Some(e) = step {
                    expr(e, out);
                }
                stmt(body, out);
            }
            Stmt::Return(Some(e)) => expr(e, out),
            Stmt::Labeled { stmt: s, .. } => stmt(s, out),
            _ => {}
        }
    }
    fn block_item(b: &BlockItem, out: &mut HashSet<String>) {
        if let BlockItem::Stmt(s) = b {
            stmt(s, out);
        }
    }
    for item in &unit.items {
        if let ExternalDecl::FunctionDef(f) = item {
            for i in &f.body.items {
                block_item(i, out);
            }
        }
    }
}

fn transpiler_pointer_param_names(f: &transpiler::parser::ast::FunctionDef) -> Vec<Option<String>> {
    use transpiler::parser::ast::*;
    use transpiler::typecheck::types::{Type, param_type};
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
                ParamDeclarator::Named(d) => decl_name(d),
                _ => None,
            }
        })
        .collect()
}

fn decl_name(d: &transpiler::parser::ast::Declarator) -> Option<String> {
    use transpiler::parser::ast::DirectDeclarator;
    match &d.direct {
        DirectDeclarator::Ident(name) => Some(name.clone()),
        DirectDeclarator::Paren(inner) => decl_name(inner),
        DirectDeclarator::Array(base, _) | DirectDeclarator::Function(base, _) => {
            decl_name_direct(base)
        }
    }
}

fn decl_name_direct(d: &transpiler::parser::ast::DirectDeclarator) -> Option<String> {
    use transpiler::parser::ast::DirectDeclarator;
    match d {
        DirectDeclarator::Ident(name) => Some(name.clone()),
        DirectDeclarator::Paren(inner) => decl_name(inner),
        DirectDeclarator::Array(base, _) | DirectDeclarator::Function(base, _) => {
            decl_name_direct(base)
        }
    }
}
