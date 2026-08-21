use std::collections::BTreeMap;
use transpiler::parser::{MacroBodyResolver, parse_full};
use transpiler::typecheck::resolve::resolve_translation_unit_seeded;
use transpiler::typecheck::{
    DeclaredTypesResolver, DiagnosticKind, ExportResolver, check_translation_unit,
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

    let mut declared_resolver = DeclaredTypesResolver::new();
    let mut export_resolver = ExportResolver::new();
    let mut macro_resolver = MacroBodyResolver::new();
    let mut shapes: BTreeMap<String, usize> = BTreeMap::new();
    let mut samples: Vec<String> = Vec::new();
    for path in &files {
        let Ok((_, unit)) = parse_full(path.to_str().unwrap()) else {
            continue;
        };
        let declared = declared_resolver.resolve(path);
        let macros = macro_resolver.resolve(path);
        let seed = export_resolver.resolve(path);
        let table = resolve_translation_unit_seeded(&unit, seed).table;
        let result = check_translation_unit(&unit, &declared, &macros, &table);
        for d in &result.diagnostics {
            let kind = match d.kind {
                DiagnosticKind::Assignment => "assign",
                DiagnosticKind::CallArgument => "call-arg",
            };
            let shape = format!("{kind}: {:?} <- {:?}", d.target, d.value);
            *shapes.entry(shape.clone()).or_insert(0) += 1;
            if samples.len() < 40 {
                samples.push(format!(
                    "{}: {shape}",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }
    println!("--- diagnostic shapes (target <- value), top 30 ---");
    let mut by_count: Vec<_> = shapes.into_iter().collect();
    by_count.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (shape, count) in by_count.into_iter().take(30) {
        println!("{count:5}  {shape}");
    }
    println!("--- sample sites ---");
    for s in &samples {
        println!("{s}");
    }
}
