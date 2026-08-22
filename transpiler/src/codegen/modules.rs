//! Phase 3 Step 1: Module Graph
//!
//! Finishes `docs/03_TRANSPILER.md`'s module-structure bullets: enumerates
//! every Rust module the corpus maps to -- one per `.c` file (with or
//! without a matching `.h`; Step 0's `visibility.rs` already answers
//! `pub`-vs-private for its items regardless), plus one per header with no
//! matching `.c` at all (13 of them: `p_local.h`, `r_state.h`, ...,
//! confirmed by direct corpus scan, not just trusted from the doc) -- and
//! which of those 13 header-only modules each `.c`-file module needs to
//! `use` items from.
//!
//! **Transitive, not just direct, `#include`s**: a header-only header
//! doesn't have to be named in a `.c` file's own `#include` list to be
//! needed by it -- if `p_map.c` includes `p_local.h`'s own includer chain
//! indirectly (or `p_local.h` itself, most commonly), the items it declares
//! are visible to `p_map.c` in real C regardless of how many hops away. The
//! transitive closure this step walks is exactly the one
//! `system_headers::read_resolved_chunks_and_includes` already exposes for
//! every other resolver in this codebase (`ExportResolver`,
//! `DeclaredTypesResolver`, `ImportResolver`, ...); reused here rather than
//! reimplemented.
//!
//! **Deliberately not attempted here**: `.c`-to-`.c` module `use` edges
//! (module A calling a function Step 0 found public in module B). Unlike
//! header-only imports, that needs real identifier-usage resolution --
//! which function bodies in A actually reference which cross-module
//! symbol -- not just `#include`-graph reachability; a `.c` file doesn't
//! `#include` another `.c` file's module at all in the C source, so there's
//! no graph to walk here the way there is for headers. Left for a
//! follow-up step.

use crate::parser::system_headers::{read_resolved_chunks_and_includes, resolve_include_path};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleKind {
    /// A `.c` file's module -- has private items (Step 0's `visibility.rs`
    /// decides which of its definitions are `pub`).
    Source,
    /// A header with no matching `.c` -- everything it declares is `pub`.
    HeaderOnly,
}

#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub kind: ModuleKind,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct ModuleGraph {
    pub modules: Vec<Module>,
    /// Source module name -> the header-only module names it needs to
    /// `use` items from.
    pub header_only_imports: HashMap<String, HashSet<String>>,
}

fn module_name(path: &Path) -> String {
    path.file_stem().unwrap().to_string_lossy().to_string()
}

/// Every file `path` `#include`s, directly or transitively (never `path`
/// itself). Cycle-guarded the same way every other resolver in this
/// codebase is.
fn transitive_includes(path: &Path, seen: &mut HashSet<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if !seen.insert(key) {
        return out;
    }
    let Some((_, includes)) = read_resolved_chunks_and_includes(path) else {
        return out;
    };
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    for inc in includes {
        if let Some(resolved) = resolve_include_path(&inc, dir) {
            out.push(resolved.clone());
            out.extend(transitive_includes(&resolved, seen));
        }
    }
    out
}

fn corpus_files(corpus_dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(ext))
        .collect();
    files.sort();
    files
}

/// Builds the full module graph for every `.c`/`.h` file under
/// `corpus_dir`.
pub fn build_module_graph(corpus_dir: &Path) -> ModuleGraph {
    let c_files = corpus_files(corpus_dir, "c");
    let h_files = corpus_files(corpus_dir, "h");
    let c_stems: HashSet<String> = c_files.iter().map(|p| module_name(p)).collect();

    let mut modules: Vec<Module> = c_files
        .iter()
        .map(|c| Module {
            name: module_name(c),
            kind: ModuleKind::Source,
            path: c.clone(),
        })
        .collect();

    let mut header_only_stems = HashSet::new();
    for h in &h_files {
        let stem = module_name(h);
        if c_stems.contains(&stem) {
            continue; // merges into that .c file's own module instead
        }
        header_only_stems.insert(stem.clone());
        modules.push(Module {
            name: stem,
            kind: ModuleKind::HeaderOnly,
            path: h.clone(),
        });
    }

    let mut header_only_imports: HashMap<String, HashSet<String>> = HashMap::new();
    for c in &c_files {
        let mut seen = HashSet::new();
        let needed: HashSet<String> = transitive_includes(c, &mut seen)
            .iter()
            .map(|p| module_name(p))
            .filter(|stem| header_only_stems.contains(stem))
            .collect();
        if !needed.is_empty() {
            header_only_imports.insert(module_name(c), needed);
        }
    }

    ModuleGraph {
        modules,
        header_only_imports,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_module_count_matches_corpus_scan() {
        let graph = build_module_graph(&corpus_dir());
        let source_count = graph
            .modules
            .iter()
            .filter(|m| m.kind == ModuleKind::Source)
            .count();
        let header_only_count = graph
            .modules
            .iter()
            .filter(|m| m.kind == ModuleKind::HeaderOnly)
            .count();
        assert_eq!(source_count, 62, "expected one Source module per .c file");
        assert_eq!(
            header_only_count, 13,
            "expected the 13 no-matching-.c headers docs/03_TRANSPILER.md names"
        );
    }

    #[test]
    fn test_header_only_modules_match_documented_list() {
        let graph = build_module_graph(&corpus_dir());
        let mut header_only: Vec<&str> = graph
            .modules
            .iter()
            .filter(|m| m.kind == ModuleKind::HeaderOnly)
            .map(|m| m.name.as_str())
            .collect();
        header_only.sort();
        assert_eq!(
            header_only,
            vec![
                "d_englsh", "d_event", "d_french", "d_player", "d_textur", "d_think", "d_ticcmd",
                "doomdata", "doomtype", "p_local", "r_defs", "r_local", "r_state",
            ]
        );
    }

    #[test]
    fn test_p_map_needs_p_local_import() {
        // p_map.c has no matching header; every symbol it exposes (Step 0)
        // is declared via p_local.h instead, so its module must import it.
        let graph = build_module_graph(&corpus_dir());
        let needed = graph
            .header_only_imports
            .get("p_map")
            .expect("p_map.c should need at least one header-only import");
        assert!(needed.contains("p_local"));
    }

    #[test]
    fn test_corpus_module_graph_reports_coverage() {
        // Not a pass/fail assertion (matching this project's "measure,
        // don't assume" methodology) -- reports how many Source modules
        // need a header-only import, and which header-only modules see the
        // most use, for cross-checking against docs/03_TRANSPILER.md.
        let graph = build_module_graph(&corpus_dir());
        let mut usage: HashMap<&str, usize> = HashMap::new();
        for needed in graph.header_only_imports.values() {
            for name in needed {
                *usage.entry(name.as_str()).or_default() += 1;
            }
        }
        let mut usage: Vec<_> = usage.into_iter().collect();
        usage.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        eprintln!(
            "module graph: {} modules ({} Source, {} HeaderOnly), {} Source \
             modules need at least one header-only import",
            graph.modules.len(),
            graph
                .modules
                .iter()
                .filter(|m| m.kind == ModuleKind::Source)
                .count(),
            graph
                .modules
                .iter()
                .filter(|m| m.kind == ModuleKind::HeaderOnly)
                .count(),
            graph.header_only_imports.len(),
        );
        eprintln!("header-only modules by number of Source-module importers: {usage:?}");
    }
}
