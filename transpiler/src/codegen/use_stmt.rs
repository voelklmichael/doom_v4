//! Phase 3 Step 3: `use` Block Rendering
//!
//! Turns `ModuleGraph`'s two import maps (`modules.rs`) into the actual
//! Rust text each module's own `use` block needs, per the split that
//! module already establishes: `pub use crate::x::*;` for a header-only
//! import -- safe as a glob because everything such a header declares is
//! genuinely defined in that Rust module, so "declared, not used" costs
//! nothing -- and a grouped `use crate::owner::{a, b};` for exact-path
//! imports, kept private (not `pub`) since these represent this module's
//! own internal calls into another's public API, not something it means to
//! re-export further.
//!
//! Sorted throughout: both import maps are hash-based and iterate in
//! unspecified order, but generated source needs to be deterministic
//! (stable diffs, reproducible builds) regardless of hashing.

use crate::codegen::modules::ModuleGraph;

/// Renders `module_name`'s own `use` block: one `pub use` glob line per
/// needed header-only module (sorted), then one grouped `use` line per
/// Source module it needs exact-path symbols from (owners sorted, symbol
/// names within each line sorted). Empty string if the module needs
/// neither.
pub fn render_use_block(graph: &ModuleGraph, module_name: &str) -> String {
    let mut lines = Vec::new();

    if let Some(header_only) = graph.header_only_imports.get(module_name) {
        let mut modules: Vec<&str> = header_only.iter().map(String::as_str).collect();
        modules.sort_unstable();
        for m in modules {
            lines.push(format!("pub use crate::{m}::*;"));
        }
    }

    if let Some(owners) = graph.source_symbol_imports.get(module_name) {
        let mut owners: Vec<(&str, Vec<&str>)> = owners
            .iter()
            .map(|(owner, names)| {
                let mut names: Vec<&str> = names.iter().map(String::as_str).collect();
                names.sort_unstable();
                (owner.as_str(), names)
            })
            .collect();
        owners.sort_unstable_by_key(|(owner, _)| *owner);
        for (owner, names) in owners {
            lines.push(format!("use crate::{owner}::{{{}}};", names.join(", ")));
        }
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::modules::build_module_graph;
    use std::path::{Path, PathBuf};

    fn corpus_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_empty_for_module_with_no_imports() {
        let graph = ModuleGraph::default();
        assert_eq!(render_use_block(&graph, "nonexistent"), "");
    }

    #[test]
    fn test_header_only_glob_line_shape() {
        let graph = build_module_graph(&corpus_dir());
        let block = render_use_block(&graph, "p_map");
        assert!(
            block.lines().any(|l| l == "pub use crate::p_local::*;"),
            "expected a pub use glob line for p_local in:\n{block}"
        );
    }

    #[test]
    fn test_exact_path_line_shape() {
        let graph = build_module_graph(&corpus_dir());
        let block = render_use_block(&graph, "p_map");
        let line = block
            .lines()
            .find(|l| l.starts_with("use crate::p_maputl::"))
            .unwrap_or_else(|| panic!("expected an exact-path use line for p_maputl in:\n{block}"));
        assert!(
            line.contains("P_SetThingPosition"),
            "expected P_SetThingPosition in: {line}"
        );
        assert!(
            !line.starts_with("pub "),
            "exact-path imports must not be pub: {line}"
        );
    }

    #[test]
    fn test_symbol_names_within_a_line_are_sorted() {
        let graph = build_module_graph(&corpus_dir());
        let block = render_use_block(&graph, "p_map");
        for line in block.lines() {
            let Some(inner) = line
                .strip_prefix("use crate::")
                .and_then(|rest| rest.split_once('{'))
                .map(|(_, rest)| rest.trim_end_matches("};"))
            else {
                continue;
            };
            let names: Vec<&str> = inner.split(", ").collect();
            let mut sorted = names.clone();
            sorted.sort_unstable();
            assert_eq!(names, sorted, "names not sorted in: {line}");
        }
    }

    #[test]
    fn test_rendering_is_deterministic_across_runs() {
        let graph = build_module_graph(&corpus_dir());
        let first = render_use_block(&graph, "p_map");
        let second = render_use_block(&graph, "p_map");
        assert_eq!(first, second);
    }
}
