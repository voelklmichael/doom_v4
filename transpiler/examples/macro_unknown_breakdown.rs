use std::collections::BTreeMap;
use transpiler::parser::{MacroBody, MacroBodyResolver, parse_full};
use transpiler::typecheck::{MacroTyper, MacroUse, Type, collect_macro_uses};

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
        .collect();
    files.sort();

    let mut body_resolver = MacroBodyResolver::new();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut names: BTreeMap<String, usize> = BTreeMap::new();
    for path in &files {
        let Ok((_, unit)) = parse_full(path.to_str().unwrap()) else {
            continue;
        };
        let macros = body_resolver.resolve(path);
        let uses = collect_macro_uses(&unit, &macros);
        let mut typer = MacroTyper::new(&macros, None);
        for u in &uses {
            let (name, ty) = match u {
                MacroUse::Object(name) => (name.clone(), typer.type_of_object_macro(name)),
                MacroUse::Call { name, args } => {
                    (name.clone(), typer.type_of_macro_call(name, args))
                }
            };
            if ty != Type::Unknown {
                continue;
            }
            *names.entry(name.clone()).or_insert(0) += 1;
            let reason = match macros.get(&name) {
                Some(MacroBody::Object(_)) => "object-but-inner-unknown".to_string(),
                Some(MacroBody::Function { .. }) => {
                    "function-macro-call-substitution-unknown".to_string()
                }
                Some(MacroBody::Empty { .. }) => "empty-body".to_string(),
                Some(MacroBody::Statements { .. }) => "statements-body".to_string(),
                Some(MacroBody::Unparseable(_)) => "unparseable-body".to_string(),
                None => "not-in-macro-map(bug?)".to_string(),
            };
            *reasons.entry(reason).or_insert(0) += 1;
        }
    }
    println!("--- reasons ---");
    for (k, v) in &reasons {
        println!("{v:5}  {k}");
    }
    println!("--- top unknown macro names ---");
    let mut by_count: Vec<_> = names.into_iter().collect();
    by_count.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (name, count) in by_count.into_iter().take(25) {
        println!("{count:5}  {name}");
    }
}
