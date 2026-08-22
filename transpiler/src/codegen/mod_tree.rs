//! Phase 3 Step 4: Crate Module Tree
//!
//! The other half of the boilerplate `use_stmt.rs` doesn't cover: every one
//! of `ModuleGraph`'s 75 modules (`modules.rs`) -- 62 Source, 13 HeaderOnly
//! -- needs a `pub mod name;` declaration somewhere for the crate to see it
//! at all, regardless of who imports what from whom. `pub`, not private:
//! Source and HeaderOnly modules alike are reached from other modules'
//! `use crate::name::...` paths (`use_stmt.rs`'s own output), which
//! requires the module itself to be publicly reachable from the crate
//! root, on top of whatever `resolve_module_visibility` already decided
//! about its individual items.
//!
//! Sorted for the same reason `use_stmt.rs` sorts its own output:
//! `ModuleGraph::modules` is built by walking two separately-sorted file
//! lists (`.c` then `.h`) and concatenating them, not one globally-sorted
//! pass, so deterministic output still needs an explicit sort here.

use crate::codegen::modules::ModuleGraph;

/// Renders every module's `pub mod name;` declaration, one per line,
/// sorted by name.
pub fn render_mod_declarations(graph: &ModuleGraph) -> String {
    let mut names: Vec<&str> = graph.modules.iter().map(|m| m.name.as_str()).collect();
    names.sort_unstable();
    names
        .into_iter()
        .map(|n| format!("pub mod {n};"))
        .collect::<Vec<_>>()
        .join("\n")
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
    fn test_empty_graph_renders_nothing() {
        let graph = ModuleGraph::default();
        assert_eq!(render_mod_declarations(&graph), "");
    }

    #[test]
    fn test_one_line_per_module_all_pub() {
        let graph = build_module_graph(&corpus_dir());
        let rendered = render_mod_declarations(&graph);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), graph.modules.len());
        assert!(
            lines
                .iter()
                .all(|l| l.starts_with("pub mod ") && l.ends_with(';'))
        );
    }

    #[test]
    fn test_covers_both_source_and_header_only_modules() {
        let graph = build_module_graph(&corpus_dir());
        let rendered = render_mod_declarations(&graph);
        assert!(
            rendered.contains("pub mod p_map;"),
            "missing Source module p_map"
        );
        assert!(
            rendered.contains("pub mod p_local;"),
            "missing HeaderOnly module p_local"
        );
    }

    #[test]
    fn test_lines_are_sorted() {
        let graph = build_module_graph(&corpus_dir());
        let rendered = render_mod_declarations(&graph);
        let lines: Vec<&str> = rendered.lines().collect();
        let mut sorted = lines.clone();
        sorted.sort_unstable();
        assert_eq!(lines, sorted);
    }
}
