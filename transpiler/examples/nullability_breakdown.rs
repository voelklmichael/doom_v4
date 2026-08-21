use std::collections::{HashMap, HashSet};
use transpiler::parser::ast::{
    Declarator, DirectDeclarator, ExternalDecl, FunctionDef, ParamDeclarator,
};
use transpiler::parser::parse_full;
use transpiler::typecheck::array_shape::functions_with_bodies;
use transpiler::typecheck::declared_types::DeclaredTypesResolver;
use transpiler::typecheck::nullability::{
    EvidenceMap, ForwardEdges, ParamKey, Verdict, analyze, collect_body_evidence,
    collect_call_evidence,
};
use transpiler::typecheck::types::{Type, param_type};

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
    let mut param_name_of: HashMap<ParamKey, String> = HashMap::new();
    let mut file_of: HashMap<String, String> = HashMap::new();

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
                file_of.insert(
                    name.clone(),
                    path.file_name().unwrap().to_string_lossy().to_string(),
                );
                for (i, p) in pointer_params(f).iter().enumerate() {
                    if let Some(pname) = p {
                        all_params.insert((name.clone(), i));
                        param_name_of.insert((name.clone(), i), pname.clone());
                    }
                }
            }
        }
    }

    let analysis = analyze(all_evidence, &all_forwards, &all_params);

    let mut nullable: Vec<_> = all_params
        .iter()
        .filter(|k| analysis.verdicts[*k].combined == Verdict::Nullable)
        .collect();
    let mut conflict: Vec<_> = all_params
        .iter()
        .filter(|k| analysis.verdicts[*k].combined == Verdict::Conflict)
        .collect();
    nullable.sort();
    conflict.sort();

    println!("--- combined = Nullable ({}) ---", nullable.len());
    for key in &nullable {
        print_row(key, &param_name_of, &file_of, &analysis);
    }
    println!("\n--- combined = Conflict ({}) ---", conflict.len());
    for key in &conflict {
        print_row(key, &param_name_of, &file_of, &analysis);
    }
}

fn print_row(
    key: &ParamKey,
    param_name_of: &HashMap<ParamKey, String>,
    file_of: &HashMap<String, String>,
    analysis: &transpiler::typecheck::nullability::NullabilityAnalysis,
) {
    let (func, idx) = key;
    let pname = param_name_of.get(key).map(String::as_str).unwrap_or("?");
    let file = file_of.get(func).map(String::as_str).unwrap_or("?");
    let v = &analysis.verdicts[key];
    println!(
        "{file}: {func}(param #{idx} `{pname}`) -- call_site={:?} body={:?}",
        v.call_site, v.body
    );
    if let Some(ev) = analysis.evidence.get(key) {
        for e in ev {
            println!("    {e:?}");
        }
    }
}

fn decl_name(d: &Declarator) -> Option<String> {
    match &d.direct {
        DirectDeclarator::Ident(name) => Some(name.clone()),
        DirectDeclarator::Paren(inner) => decl_name(inner),
        DirectDeclarator::Array(base, _) | DirectDeclarator::Function(base, _) => {
            decl_name_direct(base)
        }
    }
}

fn decl_name_direct(d: &DirectDeclarator) -> Option<String> {
    match d {
        DirectDeclarator::Ident(name) => Some(name.clone()),
        DirectDeclarator::Paren(inner) => decl_name(inner),
        DirectDeclarator::Array(base, _) | DirectDeclarator::Function(base, _) => {
            decl_name_direct(base)
        }
    }
}

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
                ParamDeclarator::Named(d) => decl_name(d),
                _ => None,
            }
        })
        .collect()
}
