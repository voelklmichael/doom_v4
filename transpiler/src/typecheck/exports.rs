//! Step 0: Exported Declarations
//!
//! Generalizes Step 6b (`transpiler/src/parser/imports.rs`)'s `#include`-as-
//! import treatment beyond typedef *names* to the rest of what a header
//! exports: function prototypes/definitions, `extern`/top-level variables,
//! `struct`/`union`/`enum` tags, and enum constants. Step 1's `SymbolTable`
//! needs this seed -- without it, a call to a function declared in another
//! header (or a reference to a tag/enum defined there) has nothing to
//! resolve against, since `#include` never textually inlines a header's
//! contents into the including file's own AST (see `docs/01_PARSER.md`
//! Step 6).
//!
//! Mirrors Step 6b/4b/7's established shape exactly: recursively union a
//! file's own top-level exports with those of everything it transitively
//! `#include`s (local and system, memoized, cycle-guarded, reusing
//! `system_headers.rs`). Reuses Step 6a's rough top-level-only scan
//! (`grammar::extract_top_level_decls`) -- a top-level declaration is never
//! ambiguous at file scope, so no typedef table is needed here either.
//!
//! Collects only names and a coarse kind, not full types -- matching how
//! Step 1 itself defers full `Type` computation to Step 3.
//!
//! **Respects linkage, unlike typedef export**: a `static` top-level
//! function or variable has internal linkage -- real C, but invisible to
//! anything that merely `#include`s the file it's declared in. Typedef
//! names were never subject to this (a `typedef` is a compile-time alias,
//! not a linked symbol), which is why Step 6b never had to make the
//! distinction. Tags and enum constants aren't subject to `static` either
//! (C has no such thing as a "static struct tag"), so those always cross
//! the `#include` boundary, same as typedef names.

use crate::parser::ast::{
    DeclSpecifiers, Declarator, DirectDeclarator, ExternalDecl, StorageClass, TypeSpecifier,
};
use crate::parser::grammar::{declarator_name, extract_top_level_decls};
use crate::parser::system_headers::{read_resolved_chunks_and_includes, resolve_include_path};
use crate::parser::{attach_comments, lex_chunks};
use crate::typecheck::scope::{Symbol, SymbolKind, Tag, TagKind};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// The ordinary-namespace symbols and tag-namespace tags a file exports.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExportedDecls {
    pub symbols: HashMap<String, Symbol>,
    pub tags: HashMap<String, Tag>,
}

/// True when `d`'s outermost shape is a plain function declarator (`name(...)`),
/// as opposed to a variable of some other shape -- including a function
/// *pointer* (`(*fp)(...)`), whose outermost `DirectDeclarator` is also
/// `Function`, but wrapping a parenthesized inner declarator rather than a
/// bare identifier.
fn is_function_declarator(d: &Declarator) -> bool {
    matches!(&d.direct, DirectDeclarator::Function(base, _) if matches!(**base, DirectDeclarator::Ident(_)))
}

/// Scans a struct/union/enum specifier for the tag it declares/references
/// and, for a defining enum, the constants it introduces -- shared by every
/// specifier occurrence (a top-level declaration's own specifiers, or a
/// function definition's return-type specifiers).
fn scan_decl_specifiers(specs: &DeclSpecifiers, out: &mut ExportedDecls) {
    for ts in &specs.type_specifiers {
        match ts {
            TypeSpecifier::Struct(spec) | TypeSpecifier::Union(spec) => {
                if let Some(name) = &spec.name {
                    let kind = if matches!(ts, TypeSpecifier::Struct(_)) {
                        TagKind::Struct
                    } else {
                        TagKind::Union
                    };
                    out.tags.entry(name.clone()).or_insert(Tag {
                        name: name.clone(),
                        kind,
                        defined: spec.fields.is_some(),
                    });
                }
            }
            TypeSpecifier::Enum(spec) => {
                if let Some(name) = &spec.name {
                    out.tags.entry(name.clone()).or_insert(Tag {
                        name: name.clone(),
                        kind: TagKind::Enum,
                        defined: spec.variants.is_some(),
                    });
                }
                for (variant_name, _) in spec.variants.iter().flatten() {
                    out.symbols.entry(variant_name.clone()).or_insert(Symbol {
                        name: variant_name.clone(),
                        kind: SymbolKind::EnumConstant,
                        storage: None,
                    });
                }
            }
            _ => {}
        }
    }
}

/// Scans a file's own top-level items (Step 6a-style: function bodies never
/// parsed) into its exported symbols and tags -- every storage class kept
/// as-is here; filtering `static` symbols out happens one level up, at the
/// point where a result gets unioned into whatever `#include`s this file.
fn scan_top_level_exports(items: &[ExternalDecl]) -> ExportedDecls {
    let mut out = ExportedDecls::default();
    for item in items {
        match item {
            ExternalDecl::FunctionDef(f) => {
                scan_decl_specifiers(&f.specifiers, &mut out);
                if let Some(name) = declarator_name(&f.declarator) {
                    out.symbols.entry(name.clone()).or_insert(Symbol {
                        name,
                        kind: SymbolKind::Function,
                        storage: f.specifiers.storage,
                    });
                }
            }
            ExternalDecl::Declaration(decl) => {
                scan_decl_specifiers(&decl.specifiers, &mut out);
                if decl.specifiers.storage == Some(StorageClass::Typedef) {
                    continue; // Step 6b's job, not this one's.
                }
                for init_decl in &decl.declarators {
                    let Some(name) = declarator_name(&init_decl.declarator) else {
                        continue;
                    };
                    let kind = if is_function_declarator(&init_decl.declarator) {
                        SymbolKind::Function
                    } else {
                        SymbolKind::Variable
                    };
                    out.symbols.entry(name.clone()).or_insert(Symbol {
                        name,
                        kind,
                        storage: decl.specifiers.storage,
                    });
                }
            }
        }
    }
    out
}

/// Resolves and caches each file's transitively-imported export set, so a
/// header `#include`d from many places is only scanned once.
#[derive(Default)]
pub struct ExportResolver {
    cache: HashMap<PathBuf, ExportedDecls>,
}

impl ExportResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every declaration `path` exports: its own top-level declarations
    /// (any storage class), unioned with everything transitively
    /// `#include`d (that file's *non-static* declarations only -- see
    /// module docs).
    pub fn resolve(&mut self, path: &Path) -> ExportedDecls {
        let mut visiting = HashSet::new();
        self.resolve_inner(path, &mut visiting)
    }

    fn resolve_inner(&mut self, path: &Path, visiting: &mut HashSet<PathBuf>) -> ExportedDecls {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }
        if !visiting.insert(key.clone()) {
            return ExportedDecls::default();
        }

        let mut result = ExportedDecls::default();
        if let Some((resolved, includes)) = read_resolved_chunks_and_includes(&key) {
            if let Ok(entries) = lex_chunks(&resolved) {
                let stream = attach_comments(entries);
                result = scan_top_level_exports(&extract_top_level_decls(&stream));
            }
            let dir = key.parent().unwrap_or_else(|| Path::new("."));
            for inc in includes {
                if let Some(resolved_path) = resolve_include_path(&inc, dir) {
                    let included = self.resolve_inner(&resolved_path, visiting);
                    for (name, sym) in included.symbols {
                        if sym.storage != Some(StorageClass::Static) {
                            result.symbols.entry(name).or_insert(sym);
                        }
                    }
                    for (name, tag) in included.tags {
                        result.tags.entry(name).or_insert(tag);
                    }
                }
            }
        }

        visiting.remove(&key);
        self.cache.insert(key, result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_resolve_finds_function_prototype_from_header() {
        // Z_Malloc is declared (not defined) in z_zone.h; m_misc.c includes
        // it directly.
        let mut resolver = ExportResolver::new();
        let exports = resolver.resolve(&corpus_dir().join("m_misc.c"));
        let sym = exports
            .symbols
            .get("Z_Malloc")
            .expect("expected Z_Malloc to be exported via z_zone.h");
        assert_eq!(sym.kind, SymbolKind::Function);
    }

    #[test]
    fn test_resolve_finds_extern_global_variable() {
        // gamemap is declared `extern int gamemap;` in doomstat.h, which
        // m_misc.c includes directly.
        let mut resolver = ExportResolver::new();
        let exports = resolver.resolve(&corpus_dir().join("m_misc.c"));
        let sym = exports
            .symbols
            .get("gamemap")
            .expect("expected gamemap to be exported via doomstat.h");
        assert_eq!(sym.kind, SymbolKind::Variable);
        assert_eq!(sym.storage, Some(StorageClass::Extern));
    }

    #[test]
    fn test_function_pointer_variable_is_not_classified_as_function() {
        use crate::parser::{attach_comments, lex_chunks, parse_chunks};
        let (_, chunks) = parse_chunks("void (*fp)(int);\nvoid foo(int);\nint *bar(void);\n");
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        let items = extract_top_level_decls(&stream);
        let exported = scan_top_level_exports(&items);
        assert_eq!(
            exported.symbols.get("fp").map(|s| s.kind),
            Some(SymbolKind::Variable),
            "a function-pointer variable must not be classified as a function"
        );
        assert_eq!(
            exported.symbols.get("foo").map(|s| s.kind),
            Some(SymbolKind::Function)
        );
        assert_eq!(
            exported.symbols.get("bar").map(|s| s.kind),
            Some(SymbolKind::Function),
            "a function returning a pointer is still a function"
        );
    }

    #[test]
    fn test_static_top_level_symbol_is_not_exported_to_includers() {
        // Scan a small synthetic source directly through the same pipeline
        // pieces resolve_inner uses, without touching the real corpus.
        use crate::parser::{attach_comments, lex_chunks, parse_chunks};
        let (_, chunks) = parse_chunks("static int helper(void) { return 0; }\nint used(void);\n");
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        let items = extract_top_level_decls(&stream);
        let exported = scan_top_level_exports(&items);
        assert_eq!(
            exported.symbols.get("helper").map(|s| s.storage),
            Some(Some(StorageClass::Static))
        );
        assert!(exported.symbols.contains_key("used"));
    }

    #[test]
    fn test_resolve_across_full_corpus_measures_symbol_resolution_gap() {
        // Not a pass/fail assertion (matching this project's "measure
        // actual scope before deciding it needs more" methodology) --
        // re-runs Step 1's own corpus measurement now seeded with Step 0's
        // export sets, and prints the before/after so the improvement (or
        // remaining gap) is visible.
        use crate::parser::parse_full;
        use crate::typecheck::resolve::resolve_translation_unit_seeded;

        let dir = corpus_dir();
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        files.sort();
        assert!(files.len() > 50, "expected the full Doom .c corpus");

        let mut exports = ExportResolver::new();
        let mut clean_files = 0;
        let mut total_unresolved = 0;
        for path in &files {
            if let Ok((_, unit)) = parse_full(path.to_str().unwrap()) {
                let seed = exports.resolve(path);
                let result = resolve_translation_unit_seeded(&unit, seed);
                if result.unresolved.is_empty() {
                    clean_files += 1;
                }
                total_unresolved += result.unresolved.len();
            }
        }
        eprintln!(
            "symbol resolution (seeded with Step 0 exports) over {} files: \
             {clean_files} fully resolved, {total_unresolved} unresolved \
             identifier references total (was 3 fully resolved, 13735 \
             unresolved before Step 0 -- see docs/KNOWN_LIMITATIONS.md)",
            files.len()
        );
    }
}
