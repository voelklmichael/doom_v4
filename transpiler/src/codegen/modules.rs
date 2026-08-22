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
//! ## Step 2: exact-path imports for cross-module symbols
//!
//! The piece Step 1 deliberately left out: a `.c` file's module can only
//! `pub use header_only_module::*;` for a header-only header -- everything
//! such a header declares really is defined in that Rust module, so a glob
//! is safe and mirrors C's own "an `#include` brings everything into scope,
//! used or not" semantics. That safety doesn't extend to a header (of
//! either kind) that merely `extern`-declares a function or global actually
//! *defined* in some other `.c` file: `p_local.h` declares `P_TryMove`, but
//! the item genuinely lives in the `p_map` module, not `p_local`'s -- no
//! glob of `p_local` would ever bring it in, because `p_local`'s own module
//! never contains it. Those names need an exact `use owner::name;` instead,
//! pointed at wherever the symbol is actually defined.
//!
//! Unlike the header-only glob, this tier is *usage-pruned*, not
//! "declared, not used": measured against the real corpus, "every name
//! reachable through a file's transitive `#include` closure" is a much
//! weaker filter than it sounds -- central headers like `doomstat.h`-style
//! ones fan out to 24-53 of the 62 `.c` files, so `p_map.c` (player
//! movement/collision) would otherwise pick up exact-path imports for
//! symbols from `r_main`/`r_draw` (rendering) it never calls. A glob import
//! is one line regardless of how much it covers, so "declared, not used"
//! costs nothing there; a *list* of hundreds of mostly-dead exact-path
//! imports is a real quality problem for generated code, so this tier
//! keeps only what's actually referenced.
//!
//! Two pieces already in this codebase give exactly this, no new resolver
//! needed: `resolve_module_visibility` (Step 0) gives, per `.c` file, the
//! externally-linked symbols it *defines* and exposes -- corpus-wide,
//! that's a `name -> defining module` map (`build_symbol_owner`, unique per
//! C's one-definition rule for external linkage). And `typecheck::resolve`'s
//! Step 1 symbol resolver, run *unseeded* (no header exports fed in), gives
//! every identifier a file references that its own declarations don't
//! explain -- locals, params, the file's own top-level decls, and enum
//! constants/typedefs are all still resolved correctly by that walk, so
//! nothing genuinely local or shadowed ever shows up here by construction.
//! Intersecting that unresolved set against `symbol_owner`, and dropping
//! misses (libc calls, unexpanded macros, anything no `.c` file defines),
//! is exactly the cross-module edge this step needs.
//!
//! One pattern the resolver can't help with: `RawDeclarationIndex`'s bare
//! top-level redeclaration with no header at all (e.g. `m_misc.c`'s own
//! `extern int key_right;`, sourced from `g_game.c`). The resolver
//! satisfies a reference to `key_right` *locally*, against that file's own
//! redeclaration -- it would never show up as unresolved even when the
//! file's code does call it. Since this pattern is narrow (`visibility.rs`
//! measured just two corpus-wide symbols needing it), its mere existence is
//! treated as need rather than chased for actual use -- matching that same
//! module's own "not worth more machinery for a two-symbol gap" call.

use crate::codegen::visibility::{RawDeclarationIndex, resolve_module_visibility};
use crate::parser::parse_full;
use crate::parser::system_headers::{read_resolved_chunks_and_includes, resolve_include_path};
use crate::typecheck::exports::ExportResolver;
use crate::typecheck::resolve::resolve_translation_unit;
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
    /// Source module name -> owning Source module name -> the exact symbol
    /// names it needs a `use owner::name;` for (Step 2: a function/global
    /// declared in some header this module includes, or bare-redeclared at
    /// its own top level, but actually *defined* in a different `.c`
    /// file's module -- never safe to reach via a header-only glob).
    pub source_symbol_imports: HashMap<String, HashMap<String, HashSet<String>>>,
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

/// Corpus-wide `name -> defining module` map: every externally-linked
/// symbol any `.c` file's module exposes (Step 0's `resolve_module_visibility`
/// public set), keyed by name. A name two different modules both define
/// with external linkage would be a genuine one-definition-rule violation
/// in the original C, not a case this pipeline needs to handle -- the first
/// module found wins.
fn build_symbol_owner(c_files: &[PathBuf]) -> HashMap<String, String> {
    let mut exports = ExportResolver::new();
    let raw_decls = RawDeclarationIndex::build(c_files);
    let mut owner = HashMap::new();
    for c in c_files {
        let Some(vis) = resolve_module_visibility(c, &mut exports, &raw_decls) else {
            continue;
        };
        let name = module_name(c);
        for sym in vis.public {
            owner.entry(sym.name).or_insert_with(|| name.clone());
        }
    }
    owner
}

fn file_key(path: &Path) -> String {
    path.file_name().unwrap().to_string_lossy().to_string()
}

fn record_import(
    result: &mut HashMap<String, HashMap<String, HashSet<String>>>,
    symbol_owner: &HashMap<String, String>,
    own_name: &str,
    name: &str,
) {
    let Some(owner) = symbol_owner.get(name) else {
        return;
    };
    if owner != own_name {
        result
            .entry(own_name.to_string())
            .or_default()
            .entry(owner.clone())
            .or_default()
            .insert(name.to_string());
    }
}

/// For every `.c` file, the exact-path imports it needs from other Source
/// modules: identifiers Step 1's symbol resolver (run unseeded) can't
/// explain from the file's own declarations, plus names it bare-redeclares
/// at its own top level with no header at all, that resolve via
/// `symbol_owner` to a *different* `.c` file's module.
fn build_source_symbol_imports(
    c_files: &[PathBuf],
    symbol_owner: &HashMap<String, String>,
) -> HashMap<String, HashMap<String, HashSet<String>>> {
    let raw_decls = RawDeclarationIndex::build(c_files);
    let mut result: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();

    for c in c_files {
        let own_name = module_name(c);
        for name in raw_decls.names_declared_by(&file_key(c)) {
            record_import(&mut result, symbol_owner, &own_name, &name);
        }
        let Ok((_, unit)) = parse_full(c.to_str().unwrap_or_default()) else {
            continue;
        };
        let resolved = resolve_translation_unit(&unit);
        for u in &resolved.unresolved {
            record_import(&mut result, symbol_owner, &own_name, &u.name);
        }
    }
    result
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

    let symbol_owner = build_symbol_owner(&c_files);
    let source_symbol_imports = build_source_symbol_imports(&c_files, &symbol_owner);

    ModuleGraph {
        modules,
        header_only_imports,
        source_symbol_imports,
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
    fn test_p_map_needs_exact_path_import_for_p_maputl_symbol() {
        // p_map.c calls P_SetThingPosition, declared (only) via the shared
        // p_local.h but actually defined in p_maputl.c -- p_local itself
        // never defines it, so this must be an exact-path import pointed
        // at p_maputl, not just a p_local glob.
        let graph = build_module_graph(&corpus_dir());
        let needed = graph
            .source_symbol_imports
            .get("p_map")
            .expect("p_map.c should need at least one exact-path import");
        let from_p_maputl = needed
            .get("p_maputl")
            .expect("p_map.c should need something from p_maputl");
        assert!(
            from_p_maputl.contains("P_SetThingPosition"),
            "expected P_SetThingPosition among p_map's imports from p_maputl, got {from_p_maputl:?}"
        );
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

    #[test]
    fn test_corpus_source_symbol_imports_reports_coverage() {
        // Not a pass/fail assertion (same "measure, don't assume"
        // methodology as the header-only coverage test) -- reports how
        // many Source modules need at least one exact-path import from
        // another Source module, and the total symbol count, so this can
        // be cross-checked against docs/03_TRANSPILER.md going forward.
        let graph = build_module_graph(&corpus_dir());
        let modules_needing_imports = graph.source_symbol_imports.len();
        let total_symbols: usize = graph
            .source_symbol_imports
            .values()
            .flat_map(|owners| owners.values())
            .map(|names| names.len())
            .sum();
        let total_edges: usize = graph
            .source_symbol_imports
            .values()
            .map(|owners| owners.len())
            .sum();
        eprintln!(
            "source-to-source exact-path imports: {modules_needing_imports} of {} Source \
             modules need at least one, {total_edges} module-pair edges, \
             {total_symbols} exact symbol imports total",
            graph
                .modules
                .iter()
                .filter(|m| m.kind == ModuleKind::Source)
                .count(),
        );
        assert!(modules_needing_imports > 0);
    }
}
