//! Phase 3 Step 0: Module Visibility Resolution
//!
//! `docs/03_TRANSPILER.md`'s module-structure section originally proposed
//! "declared in this `.c` file's own matching header => `pub`, else
//! private" for deciding what a `.c`/`.h` pair's Rust module exposes.
//! Measured (`docs/KNOWN_LIMITATIONS.md`) to be wrong for a large share of
//! the corpus's real definitions: `linuxdoom-1.10` often declares a `.c`
//! file's externally-linked symbols in a *different* header --
//! `doomstat.h` centralizes `extern` declarations for globals defined
//! across many unrelated `.c` files -- or, for 13 `.c` files, in a shared
//! header with no matching `.c` of its own at all (`p_local.h` for most of
//! the `p_*.c` cases).
//!
//! This step replaces "matching header" with the question that's actually
//! correct for C linkage: is this symbol declared, with external linkage,
//! by *any* header the defining `.c` file's own `#include` graph reaches?
//! That's exactly `ExportResolver::resolve_via_includes` -- symbol
//! visibility falls straight out of reusing it, no per-file special-casing
//! needed for the `doomstat.h`/`p_local.h`-style cases the investigation
//! found, since both are just "some header in the graph declares it".
//!
//! **A third pattern, found while cross-checking this step's own corpus
//! numbers against the investigation's "used elsewhere" ground truth**:
//! `linuxdoom-1.10` sometimes skips headers for a global entirely, and puts
//! a bare `extern`/prototype redeclaration directly at another `.c` file's
//! own top level instead -- e.g. `m_misc.c` itself declares
//! `extern int key_right;`, sourced from `g_game.c`, with no header
//! involved at either end. `resolve_via_includes` can never see this (it
//! only walks `#include`s), so `RawDeclarationIndex` covers it separately:
//! a corpus-wide index of which `.c` files redeclare which names at their
//! own top level, independent of any header.
//!
//! **A fourth pattern, same cross-check**: several `.c` files (`r_bsp.c`,
//! `f_finale.c`, `m_random.c`, `p_saveg.c`, `p_tick.c`, `i_video.c`,
//! `m_argv.c`, ...) never `#include` their *own* matching header at all
//! (`f_finale.c` even has it commented out: `//#include "f_finale.h"`) --
//! the header is still that file's real public API (every symbol it
//! declares really is defined there and really is called from elsewhere),
//! it's just that the defining file itself never needed its own
//! declarations. `resolve_via_includes` walks `c_path`'s `#include`s, so a
//! header `c_path` doesn't include is invisible to it by construction;
//! `own_matching_header_names` checks the matching header directly,
//! self-included or not.
//!
//! **Residual gap, measured and left as-is**: with all four signals
//! combined, corpus-wide cross-checking against
//! `extern_linkage_survey.rs`'s independent "used elsewhere" ground truth
//! finds only 2 of 1360 externally-linked definitions still wrongly private
//! -- `rndindex` (`m_random.c`) and `skyflatnum` (`r_sky.c`), both declared
//! only in `doomstat.h`, which neither defining file `#include`s, nor is
//! either name declared in that file's own matching header (`m_random.h`
//! only declares the three functions, not `rndindex`). The only way to
//! find these would be "declared in *any* header anywhere in the corpus,
//! whether or not the defining file has any relationship to it" -- a much
//! weaker signal than the other four, and not worth adding machinery for
//! a two-symbol gap (matching this project's "measure, document, don't
//! chase to 100%" approach throughout).

use crate::parser::ast::{Declaration, Declarator, DirectDeclarator, ExternalDecl, StorageClass};
use crate::parser::grammar::{declarator_name, extract_top_level_decls};
use crate::parser::{attach_comments, lex_chunks, parse, parse_full};
use crate::typecheck::exports::ExportResolver;
use crate::typecheck::scope::SymbolKind;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// A symbol actually *defined* (not merely declared) at a file's own top
/// level, paired with its coarse kind -- matching `exports.rs`'s own
/// `Symbol`/`SymbolKind` shape rather than introducing a parallel one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefinedSymbol {
    pub name: String,
    pub kind: SymbolKind,
}

/// True for a plain function declarator (`name(...)`), false for a
/// variable of any other shape, including a function *pointer* -- same
/// shape check `exports.rs`'s own (private) `is_function_declarator` uses.
fn is_function_declarator(d: &Declarator) -> bool {
    matches!(&d.direct, DirectDeclarator::Function(base, _) if matches!(**base, DirectDeclarator::Ident(_)))
}

fn own_defined_from_declaration(decl: &Declaration) -> Vec<DefinedSymbol> {
    if matches!(
        decl.specifiers.storage,
        Some(StorageClass::Typedef) | Some(StorageClass::Extern) | Some(StorageClass::Static)
    ) {
        return Vec::new(); // not a definition in this file, or internal linkage
    }
    decl.declarators
        .iter()
        .filter_map(|init_decl| {
            let name = declarator_name(&init_decl.declarator)?;
            if is_function_declarator(&init_decl.declarator) {
                return None; // a bare top-level prototype, not a definition
            }
            Some(DefinedSymbol {
                name,
                kind: SymbolKind::Variable,
            })
        })
        .collect()
}

/// Every externally-linked (non-`static`) symbol *defined* -- not merely
/// declared -- at `items`'s own top level: a function with a body, or a
/// top-level variable declaration that isn't itself an `extern`
/// re-declaration or a bare prototype.
pub fn own_defined_symbols(items: &[ExternalDecl]) -> Vec<DefinedSymbol> {
    let mut out = Vec::new();
    for item in items {
        match item {
            ExternalDecl::FunctionDef(f) => {
                if f.specifiers.storage == Some(StorageClass::Static) {
                    continue;
                }
                if let Some(name) = declarator_name(&f.declarator) {
                    out.push(DefinedSymbol {
                        name,
                        kind: SymbolKind::Function,
                    });
                }
            }
            ExternalDecl::Declaration(decl) => out.extend(own_defined_from_declaration(decl)),
        }
    }
    out
}

/// Corpus-wide index of which `.c` files redeclare which names at their own
/// top level (a bare prototype, or an `extern` variable) -- independent of
/// any header, covering the "no header at either end" pattern this
/// module's docs describe. Only `Declaration` items count, not
/// `FunctionDef`s: a second *definition* of the same name elsewhere would
/// be a real duplicate-definition bug, not a mere forward declaration.
pub struct RawDeclarationIndex {
    declared_in: HashMap<String, HashSet<String>>,
    /// Reverse of `declared_in`: which names each file itself bare-declares
    /// at its own top level, independent of who else also declares them.
    by_file: HashMap<String, HashSet<String>>,
}

fn file_key(path: &Path) -> String {
    path.file_name().unwrap().to_string_lossy().to_string()
}

impl RawDeclarationIndex {
    /// Builds the index from every `.c` file's own top level, via the same
    /// rough, typedef-table-free scan `ExportResolver` itself relies on
    /// (`parse` + `extract_top_level_decls`) rather than `parse_full` --
    /// no full grammar parse needed just to read off top-level names.
    pub fn build(c_files: &[std::path::PathBuf]) -> Self {
        let mut declared_in: HashMap<String, HashSet<String>> = HashMap::new();
        let mut by_file: HashMap<String, HashSet<String>> = HashMap::new();
        for path in c_files {
            let Ok((_, resolved)) = parse(path.to_str().unwrap_or_default()) else {
                continue;
            };
            let Ok(entries) = lex_chunks(&resolved) else {
                continue;
            };
            let stream = attach_comments(entries);
            let fname = file_key(path);
            for item in extract_top_level_decls(&stream) {
                if let ExternalDecl::Declaration(decl) = item {
                    for init_decl in &decl.declarators {
                        if let Some(name) = declarator_name(&init_decl.declarator) {
                            declared_in
                                .entry(name.clone())
                                .or_default()
                                .insert(fname.clone());
                            by_file.entry(fname.clone()).or_default().insert(name);
                        }
                    }
                }
            }
        }
        Self {
            declared_in,
            by_file,
        }
    }

    /// True if some `.c` file other than `defining_file` redeclares `name`
    /// at its own top level.
    fn declared_elsewhere(&self, name: &str, defining_file: &str) -> bool {
        self.declared_in
            .get(name)
            .is_some_and(|files| files.iter().any(|f| f != defining_file))
    }

    /// Names `file` (its bare filename, e.g. `"m_misc.c"`) itself
    /// bare-redeclares at its own top level -- e.g. `m_misc.c`'s own
    /// `extern int key_right;`, with no header on either end. Empty if
    /// `file` declares nothing this way.
    pub fn names_declared_by(&self, file: &str) -> HashSet<String> {
        self.by_file.get(file).cloned().unwrap_or_default()
    }
}

/// Names declared (any storage class) at `header_path`'s own top level
/// only -- via the same rough, typedef-table-free scan as
/// `RawDeclarationIndex::build`, since a header routinely references a
/// typedef (e.g. `boolean`) it never `#include`s itself, relying on
/// whatever `.c` file includes it to have pulled that in first; a real
/// grammar parse (`parse_full`) needs a typedef table and fails standalone
/// on exactly this shape.
fn own_matching_header_names(header_path: &Path) -> Option<HashSet<String>> {
    let (_, resolved) = parse(header_path.to_str()?).ok()?;
    let entries = lex_chunks(&resolved).ok()?;
    let stream = attach_comments(entries);
    let mut names = HashSet::new();
    for item in extract_top_level_decls(&stream) {
        match item {
            ExternalDecl::FunctionDef(f) => {
                if let Some(name) = declarator_name(&f.declarator) {
                    names.insert(name);
                }
            }
            ExternalDecl::Declaration(decl) => {
                for init_decl in &decl.declarators {
                    if let Some(name) = declarator_name(&init_decl.declarator) {
                        names.insert(name);
                    }
                }
            }
        }
    }
    Some(names)
}

/// A `.c` file's externally-linked definitions, split into what's visible
/// to other modules (`pub`) and what isn't.
#[derive(Debug, Clone, Default)]
pub struct ModuleVisibility {
    pub public: Vec<DefinedSymbol>,
    pub private: Vec<DefinedSymbol>,
}

/// Resolves `c_path`'s module visibility: every externally-linked symbol it
/// defines is `pub` iff some header reachable through `c_path`'s own
/// `#include` graph declares the same name (`resolve_via_includes`), or
/// its own matching header declares it even if `c_path` never `#include`s
/// that header itself (`own_matching_header_names`), or some other `.c`
/// file redeclares it at its own top level without going through a header
/// at all (`raw_decls`) -- private otherwise. `None` if `c_path` fails to
/// parse.
pub fn resolve_module_visibility(
    c_path: &Path,
    exports: &mut ExportResolver,
    raw_decls: &RawDeclarationIndex,
) -> Option<ModuleVisibility> {
    let (_, unit) = parse_full(c_path.to_str()?).ok()?;
    let header_declared = exports.resolve_via_includes(c_path);
    let matching_header_declared =
        own_matching_header_names(&c_path.with_extension("h")).unwrap_or_default();
    let fname = file_key(c_path);

    let mut result = ModuleVisibility::default();
    for sym in own_defined_symbols(&unit.items) {
        if header_declared.symbols.contains_key(&sym.name)
            || matching_header_declared.contains(&sym.name)
            || raw_decls.declared_elsewhere(&sym.name, &fname)
        {
            result.public.push(sym);
        } else {
            result.private.push(sym);
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn corpus_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    fn corpus_c_files() -> Vec<PathBuf> {
        let mut files: Vec<_> = std::fs::read_dir(corpus_dir())
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        files.sort();
        files
    }

    #[test]
    fn test_doomstat_style_global_is_public_despite_wrong_matching_header() {
        // `devparm` is defined in d_main.c but declared `extern` only in
        // doomstat.h, not d_main.h -- exactly the case the matching-header
        // heuristic gets wrong (see docs/KNOWN_LIMITATIONS.md).
        let mut exports = ExportResolver::new();
        let raw_decls = RawDeclarationIndex::build(&[]);
        let vis =
            resolve_module_visibility(&corpus_dir().join("d_main.c"), &mut exports, &raw_decls)
                .expect("d_main.c should parse");
        assert!(
            vis.public.iter().any(|s| s.name == "devparm"),
            "devparm should be public: declared in doomstat.h, which d_main.c includes"
        );
    }

    #[test]
    fn test_headerless_c_file_symbol_declared_via_shared_header_is_public() {
        // p_map.c has no matching p_map.h at all; P_TryMove is declared in
        // the shared p_local.h instead and called from many other .c files.
        let mut exports = ExportResolver::new();
        let raw_decls = RawDeclarationIndex::build(&[]);
        let vis =
            resolve_module_visibility(&corpus_dir().join("p_map.c"), &mut exports, &raw_decls)
                .expect("p_map.c should parse");
        assert!(
            vis.public.iter().any(|s| s.name == "P_TryMove"),
            "P_TryMove should be public: declared in p_local.h, which p_map.c includes"
        );
    }

    #[test]
    fn test_symbol_used_only_internally_stays_private() {
        // PIT_CheckLine is a callback used only within p_map.c itself --
        // never declared in any header p_map.c includes, nor redeclared at
        // any other .c file's own top level.
        let mut exports = ExportResolver::new();
        let raw_decls = RawDeclarationIndex::build(&corpus_c_files());
        let vis =
            resolve_module_visibility(&corpus_dir().join("p_map.c"), &mut exports, &raw_decls)
                .expect("p_map.c should parse");
        assert!(
            vis.private.iter().any(|s| s.name == "PIT_CheckLine"),
            "PIT_CheckLine should stay private"
        );
    }

    #[test]
    fn test_symbol_declared_directly_in_another_c_file_with_no_header_is_public() {
        // key_right is defined in g_game.c but never declared in any
        // header -- m_misc.c instead redeclares it directly at its own top
        // level (`extern int key_right;`), bypassing headers on both ends.
        let mut exports = ExportResolver::new();
        let raw_decls = RawDeclarationIndex::build(&corpus_c_files());
        let vis =
            resolve_module_visibility(&corpus_dir().join("g_game.c"), &mut exports, &raw_decls)
                .expect("g_game.c should parse");
        assert!(
            vis.public.iter().any(|s| s.name == "key_right"),
            "key_right should be public: redeclared directly in m_misc.c's own top level"
        );
    }

    #[test]
    fn test_symbol_declared_in_own_matching_header_is_public_even_if_uncincluded() {
        // r_bsp.c never #includes its own r_bsp.h, but curline is declared
        // `extern` there and genuinely used from many other .c files.
        let mut exports = ExportResolver::new();
        let raw_decls = RawDeclarationIndex::build(&[]);
        let vis =
            resolve_module_visibility(&corpus_dir().join("r_bsp.c"), &mut exports, &raw_decls)
                .expect("r_bsp.c should parse");
        assert!(
            vis.public.iter().any(|s| s.name == "curline"),
            "curline should be public: declared in r_bsp.h even though r_bsp.c never includes it"
        );
    }

    #[test]
    fn test_corpus_visibility_resolution_measures_coverage() {
        // Not a pass/fail assertion (matching this project's "measure,
        // don't assume" methodology) -- reports how many externally-linked
        // definitions resolve public vs. private across the corpus, for
        // comparison against the extern_linkage_survey findings in
        // docs/KNOWN_LIMITATIONS.md this step is meant to fix.
        let files = corpus_c_files();
        assert!(files.len() > 50, "expected the full Doom .c corpus");

        let mut exports = ExportResolver::new();
        let raw_decls = RawDeclarationIndex::build(&files);
        let mut total_public = 0;
        let mut total_private = 0;
        for path in &files {
            if let Some(vis) = resolve_module_visibility(path, &mut exports, &raw_decls) {
                total_public += vis.public.len();
                total_private += vis.private.len();
            }
        }
        eprintln!(
            "module visibility over {} files: {total_public} public, \
             {total_private} private (was 344 correctly covered under the \
             matching-header-only heuristic; independently cross-checked \
             against extern_linkage_survey's ground truth at 2 residual \
             mismatches -- see docs/KNOWN_LIMITATIONS.md)",
            files.len()
        );
        assert!(total_public > 0 && total_private > 0);
    }
}
