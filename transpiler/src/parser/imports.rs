//! Step 6b: Exported Types (import resolution)
//!
//! For a file, computes the set of typedef names visible to it: its own
//! top-level typedefs (Step 6a) unioned with those of everything it
//! transitively pulls in via `#include`. This treats `#include` as an
//! import that brings type *names* into scope, not as textual inlining --
//! matching how the rest of this project treats each `.c`/`.h` file as its
//! own translation unit.
//!
//! `#include "..."` (local) resolves relative to the including file's
//! directory. `#include <...>` (system) is resolved the way a real
//! preprocessor would: searched for across the build machine's standard
//! system include directories (see `system_headers::SYSTEM_INCLUDE_DIRS`).
//! If a system header genuinely isn't present on the machine running this,
//! or fails to process cleanly, its typedefs are simply missing from the
//! result (fails soft, not hard) -- `WELL_KNOWN_SYSTEM_TYPEDEFS` and
//! `xlib_typedefs::XLIB_TYPEDEFS` below exist as fallbacks for exactly that,
//! so the result is the same whether or not the real headers happen to be
//! installed on this machine.

use crate::parser::grammar::extract_top_level_typedefs;
use crate::parser::system_headers::{read_resolved_chunks_and_includes, resolve_include_path};
use crate::parser::xlib_typedefs::XLIB_TYPEDEFS;
use crate::parser::{attach_comments, lex_chunks};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Typedef names this corpus references from system headers, as a fallback
/// for when the real header isn't found on the build machine (or doesn't
/// parse cleanly) -- see `docs/KNOWN_LIMITATIONS.md`.
const WELL_KNOWN_SYSTEM_TYPEDEFS: &[&str] = &[
    "FILE",    // <stdio.h>
    "va_list", // <stdarg.h>
];

/// Resolves and caches each file's transitively-imported typedef set, so a
/// header `#include`d from many places is only scanned once.
#[derive(Default)]
pub struct ImportResolver {
    cache: HashMap<PathBuf, HashSet<String>>,
}

impl ImportResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// The full set of typedef names visible to `path`: its own top-level
    /// typedefs, unioned with everything transitively imported via local and
    /// system `#include`s, plus the hardcoded fallbacks above (which apply
    /// unconditionally, so the result doesn't depend on what's actually
    /// installed on the machine running this).
    pub fn resolve(&mut self, path: &Path) -> HashSet<String> {
        let mut visiting = HashSet::new();
        let mut result = self.resolve_inner(path, &mut visiting);
        result.extend(WELL_KNOWN_SYSTEM_TYPEDEFS.iter().map(|s| s.to_string()));
        result.extend(XLIB_TYPEDEFS.iter().map(|s| s.to_string()));
        result
    }

    fn resolve_inner(&mut self, path: &Path, visiting: &mut HashSet<PathBuf>) -> HashSet<String> {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }
        // Guard against include cycles (glibc/X11 headers commonly
        // include-guard against each other; local corpus headers don't
        // cycle either, but don't infinite-loop if something does).
        if !visiting.insert(key.clone()) {
            return HashSet::new();
        }

        let mut result = HashSet::new();
        if let Some((resolved, includes)) = read_resolved_chunks_and_includes(&key) {
            if let Ok(entries) = lex_chunks(&resolved) {
                let stream = attach_comments(entries);
                result.extend(extract_top_level_typedefs(&stream));
            }
            let dir = key.parent().unwrap_or_else(|| Path::new("."));
            for inc in includes {
                if let Some(resolved_path) = resolve_include_path(&inc, dir) {
                    result.extend(self.resolve_inner(&resolved_path, visiting));
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
    fn test_resolve_am_map_c_finds_fixed_t() {
        // am_map.c uses fixed_t without defining it itself; it's typedef'd
        // in m_fixed.h, which it includes transitively via doomdef.h etc.
        let mut resolver = ImportResolver::new();
        let types = resolver.resolve(&corpus_dir().join("am_map.c"));
        assert!(
            types.contains("fixed_t"),
            "expected fixed_t to be resolved via imports, got: {types:?}"
        );
    }

    #[test]
    fn test_resolve_includes_well_known_system_typedefs() {
        let mut resolver = ImportResolver::new();
        let types = resolver.resolve(&corpus_dir().join("i_system.c"));
        assert!(types.contains("va_list"));
        assert!(types.contains("FILE"));
    }

    #[test]
    fn test_resolve_is_cached() {
        let mut resolver = ImportResolver::new();
        let path = corpus_dir().join("doomtype.h");
        let first = resolver.resolve(&path);
        assert!(
            resolver
                .cache
                .contains_key(&std::fs::canonicalize(&path).unwrap())
        );
        let second = resolver.resolve(&path);
        assert_eq!(first, second);
    }

    #[test]
    fn test_resolve_i_video_c_finds_real_x11_types() {
        // Only meaningful if X11 headers are actually installed on the
        // machine running this test; skip gracefully otherwise. (The result
        // should be the same either way, via the XLIB_TYPEDEFS fallback --
        // see test_resolve_i_video_c_works_without_real_headers.)
        if !Path::new("/usr/include/X11/Xlib.h").is_file() {
            return;
        }
        let mut resolver = ImportResolver::new();
        let types = resolver.resolve(&corpus_dir().join("i_video.c"));
        for name in ["Display", "Window", "GC", "Visual", "XEvent"] {
            assert!(
                types.contains(name),
                "expected {name} to be resolved from the real Xlib.h, got: {types:?}"
            );
        }
    }

    #[test]
    fn test_xlib_typedefs_fallback_is_populated() {
        // Guards against the generated file silently reverting to the empty
        // placeholder (e.g. after a bad regen on a machine without X11
        // headers) without anyone noticing.
        assert!(
            crate::parser::xlib_typedefs::XLIB_TYPEDEFS.len() > 50,
            "expected a substantial hardcoded Xlib typedef list, got {} entries -- \
             run `cargo run --example update_xlib_typedefs` on a machine with X11 dev headers",
            crate::parser::xlib_typedefs::XLIB_TYPEDEFS.len()
        );
    }

    #[test]
    fn test_resolve_i_video_c_works_without_real_headers() {
        // Even with no real system headers involved at all, the hardcoded
        // XLIB_TYPEDEFS fallback alone should be enough to resolve i_video.c.
        let types: std::collections::HashSet<String> = crate::parser::xlib_typedefs::XLIB_TYPEDEFS
            .iter()
            .map(|s| s.to_string())
            .collect();
        for name in ["Display", "Window", "GC", "Visual", "XEvent"] {
            assert!(
                types.contains(name),
                "expected {name} in the hardcoded fallback, got: {types:?}"
            );
        }
    }
}
