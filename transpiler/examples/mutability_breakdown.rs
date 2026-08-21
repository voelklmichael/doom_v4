use std::collections::{HashMap, HashSet};
use transpiler::parser::ast::ExternalDecl;
use transpiler::parser::parse_full;
use transpiler::typecheck::array_shape::functions_with_bodies;
use transpiler::typecheck::declared_types::DeclaredTypesResolver;
use transpiler::typecheck::mutability::{
    Evidence, EvidenceMap, ForwardEdges, Mutability, ParamKey, analyze, collect_body_evidence,
    collect_call_evidence,
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

    let mut declared_resolver = DeclaredTypesResolver::new();
    let mut all_evidence: EvidenceMap = HashMap::new();
    let mut all_forwards: ForwardEdges = Vec::new();
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
                && let Some(name) = decl_name(&f.declarator)
            {
                for (i, p) in transpiler_pointer_params(f).iter().enumerate() {
                    if p.is_some() {
                        all_params.insert((name.clone(), i));
                    }
                }
            }
        }
    }

    let analysis = analyze(all_evidence, &all_forwards, &all_params);

    let mut direct_write = 0;
    let mut forwarded_only = 0;
    let mut conservative_only = 0;
    let mut mixed = 0;
    for key in all_params
        .iter()
        .filter(|k| analysis.mutability[*k] == Mutability::Mutable)
    {
        let ev = &analysis.evidence[key];
        let has_write = ev.iter().any(|e| matches!(e, Evidence::BodyWrite));
        let has_forward = ev
            .iter()
            .any(|e| matches!(e, Evidence::ForwardedMutable { .. }));
        let has_conservative = ev
            .iter()
            .any(|e| matches!(e, Evidence::ConservativeIndirectCall { .. }));
        match (has_write, has_forward, has_conservative) {
            (true, false, false) => direct_write += 1,
            (false, true, false) => forwarded_only += 1,
            (false, false, true) => conservative_only += 1,
            _ => mixed += 1,
        }
    }
    println!(
        "of {} mutable params: {direct_write} direct-write-only, {forwarded_only} \
         forwarded-only, {conservative_only} conservative-fallback-only, {mixed} mixed evidence",
        direct_write + forwarded_only + conservative_only + mixed
    );

    // Sample a few conservative-only call sites' callee names to see what's
    // actually driving them.
    let mut callee_names: HashMap<String, usize> = HashMap::new();
    for key in &all_params {
        for e in analysis.evidence.get(key).into_iter().flatten() {
            if let Evidence::ConservativeIndirectCall { caller } = e {
                *callee_names.entry(caller.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut by_count: Vec<_> = callee_names.into_iter().collect();
    by_count.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    println!("--- top callers triggering the conservative fallback ---");
    for (name, count) in by_count.into_iter().take(15) {
        println!("{count:5}  {name}");
    }
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

fn transpiler_pointer_params(
    f: &transpiler::parser::ast::FunctionDef,
) -> Vec<Option<(String, transpiler::typecheck::Type)>> {
    use transpiler::parser::ast::*;
    use transpiler::typecheck::Type;
    use transpiler::typecheck::types::param_type;
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
                ParamDeclarator::Named(d) => decl_name(d).map(|n| (n, ty)),
                _ => None,
            }
        })
        .collect()
}
