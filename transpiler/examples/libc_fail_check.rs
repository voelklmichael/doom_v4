use std::collections::BTreeMap;
use transpiler::parser::parse_full;
use transpiler::typecheck::{ExportResolver, resolve_translation_unit_seeded};

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
        .collect();
    files.sort();

    let mut exports = ExportResolver::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for path in &files {
        if let Ok((_, unit)) = parse_full(path.to_str().unwrap()) {
            let seed = exports.resolve(path);
            let result = resolve_translation_unit_seeded(&unit, seed);
            for u in result.unresolved {
                *counts.entry(u.name).or_insert(0) += 1;
            }
        }
    }
    println!("{} unique unresolved names total", counts.len());
    let names: Vec<_> = counts.keys().cloned().collect();
    println!("NAMES_START");
    for n in &names {
        println!("{n}");
    }
    println!("NAMES_END");
    let _ = counts;
}
