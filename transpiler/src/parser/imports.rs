//! Step 6b: Exported Types (import resolution)
//!
//! For a file, computes the set of typedef names visible to it: its own
//! top-level typedefs (Step 6a) unioned with those of everything it
//! transitively pulls in via local `#include "..."`s. This treats
//! `#include` as an import that brings type *names* into scope, not as
//! textual inlining -- matching how the rest of this project treats each
//! `.c`/`.h` file as its own translation unit.
//!
//! System includes (`#include <...>`) can't be resolved against this
//! corpus and are simply left out of the set (e.g. `i_video.c`'s X11 types
//! stay unresolved -- see `docs/KNOWN_LIMITATIONS.md`).

use crate::parser::grammar::extract_top_level_typedefs;
use crate::parser::partitioner::PreprocessorDirective;
use crate::parser::{
    PreprocessorEnv, SourceChunk, attach_comments, lex_chunks, parse_chunks, resolve_conditionals,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

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
    /// typedefs, unioned with everything transitively imported via local
    /// `#include`s.
    pub fn resolve(&mut self, path: &Path) -> HashSet<String> {
        let mut visiting = HashSet::new();
        self.resolve_inner(path, &mut visiting)
    }

    fn resolve_inner(&mut self, path: &Path, visiting: &mut HashSet<PathBuf>) -> HashSet<String> {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }
        // Guard against include cycles (none expected in this corpus, but
        // don't infinite-loop if one exists).
        if !visiting.insert(key.clone()) {
            return HashSet::new();
        }

        let mut result = HashSet::new();
        if let Some((own, includes)) = own_and_include_paths(&key) {
            result.extend(own);
            let dir = key.parent().unwrap_or_else(|| Path::new("."));
            for inc in includes {
                result.extend(self.resolve_inner(&dir.join(&inc), visiting));
            }
        }

        visiting.remove(&key);
        self.cache.insert(key, result.clone());
        result
    }
}

/// Reads and processes `path` through Steps 1-4, returning its own
/// top-level typedef names (Step 6a) and the paths of its local
/// `#include "..."`s. `None` if the file can't be read or fails Steps 1-3.
fn own_and_include_paths(path: &Path) -> Option<(HashSet<String>, Vec<String>)> {
    let content = std::fs::read_to_string(path).ok()?;
    let (_, chunks) = parse_chunks(&content);
    let mut env = PreprocessorEnv::linux_doom_defaults();
    let resolved = resolve_conditionals(&chunks, &mut env).ok()?;

    let includes = resolved
        .iter()
        .filter_map(|c| match c {
            SourceChunk::Preprocessor {
                directive:
                    PreprocessorDirective::Include {
                        path,
                        is_system: false,
                    },
                ..
            } => Some(path.clone()),
            _ => None,
        })
        .collect();

    let entries = lex_chunks(&resolved).ok()?;
    let stream = attach_comments(entries);
    let own = extract_top_level_typedefs(&stream).into_iter().collect();

    Some((own, includes))
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
}
