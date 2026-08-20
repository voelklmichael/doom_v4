//! Literal-macro resolution.
//!
//! Computes, for a file, every object-like `#define` whose body is *just* a
//! single string or char literal (e.g. `#define SAVEGAMENAME "doomsav"`),
//! transitively through everything it `#include`s -- mirroring Step 6b
//! (`imports.rs`)'s treatment of `#include` as an import, just for macro
//! bodies instead of typedef names. Used by `macro_literal_subst.rs` to
//! substitute these macros wherever they sit immediately next to a real
//! string/char literal in code, which is the one place a macro's absence
//! actually breaks parsing (see `docs/KNOWN_LIMITATIONS.md`).
//!
//! Deliberately narrow: multi-token bodies, function-like macros, and
//! macros used anywhere *other* than directly touching a literal are left
//! alone -- this is not a general preprocessor macro expander.

use crate::parser::SourceChunk;
use crate::parser::lexer::{LexItem, Token, TokenKind, lex_chunks};
use crate::parser::partitioner::{PreprocessorDirective, partition_source};
use crate::parser::system_headers::{read_resolved_chunks_and_includes, resolve_include_path};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// If `body` is *just* a single string- or char-literal (with nothing else
/// around it), that literal's token; otherwise `None` (multi-token,
/// references another macro, isn't a literal at all, etc. -- anything not
/// directly substitutable). Goes through Step 2 (`partition_source`) first,
/// same as any other source text, since raw quote characters only mean
/// anything to the partitioner -- `lex_code` alone never sees them (Step 2
/// already splits string/char literals into their own chunks before Step 4
/// lexes what's left).
fn as_single_literal(body: &str) -> Option<Token> {
    let chunks = partition_source(body);
    let entries = lex_chunks(&chunks).ok()?;
    let mut tokens = entries.into_iter().filter_map(|e| match e.item {
        LexItem::Token(t) => Some(t),
        _ => None,
    });
    let first = tokens.next()?;
    if tokens.next().is_some() {
        return None;
    }
    matches!(
        first.kind,
        TokenKind::StringLiteral | TokenKind::CharLiteral
    )
    .then_some(first)
}

/// Resolves and caches each file's transitively-imported literal-macro map,
/// so a header `#include`d from many places is only scanned once.
#[derive(Default)]
pub struct LiteralMacroResolver {
    cache: HashMap<PathBuf, HashMap<String, Token>>,
}

impl LiteralMacroResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every literal-bodied object-like macro visible to `path`: its own
    /// top-level `#define`s of this shape, unioned with everything
    /// transitively imported via local and system `#include`s.
    pub fn resolve(&mut self, path: &Path) -> HashMap<String, Token> {
        let mut visiting = HashSet::new();
        self.resolve_inner(path, &mut visiting)
    }

    fn resolve_inner(
        &mut self,
        path: &Path,
        visiting: &mut HashSet<PathBuf>,
    ) -> HashMap<String, Token> {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }
        if !visiting.insert(key.clone()) {
            return HashMap::new();
        }

        let mut result = HashMap::new();
        if let Some((resolved, includes)) = read_resolved_chunks_and_includes(&key) {
            for chunk in &resolved {
                if let SourceChunk::Preprocessor {
                    directive:
                        PreprocessorDirective::Define {
                            name,
                            params: None,
                            body,
                        },
                    ..
                } = chunk
                    && let Some(tok) = as_single_literal(body)
                {
                    result.insert(name.clone(), tok);
                }
            }
            let dir = key.parent().unwrap_or_else(|| Path::new("."));
            for inc in includes {
                if let Some(resolved_path) = resolve_include_path(&inc, dir) {
                    for (name, tok) in self.resolve_inner(&resolved_path, visiting) {
                        result.entry(name).or_insert(tok);
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
    fn test_as_single_literal() {
        assert!(as_single_literal("\"doomsav\"").is_some());
        assert!(as_single_literal("'\\n'").is_some());
        assert!(as_single_literal("(1<<16)").is_none());
        assert!(as_single_literal("\"a\" PRESSKEY").is_none());
        assert!(as_single_literal("").is_none());
    }

    #[test]
    fn test_resolve_g_game_c_finds_savegamename() {
        // SAVEGAMENAME is defined in dstrings.h, not g_game.c itself.
        let mut resolver = LiteralMacroResolver::new();
        let macros = resolver.resolve(&corpus_dir().join("g_game.c"));
        let tok = macros
            .get("SAVEGAMENAME")
            .expect("expected SAVEGAMENAME to resolve via imports");
        assert_eq!(tok.kind, TokenKind::StringLiteral);
        assert_eq!(tok.text, "\"doomsav\"");
    }

    #[test]
    fn test_resolve_d_main_c_finds_own_macros() {
        let mut resolver = LiteralMacroResolver::new();
        let macros = resolver.resolve(&corpus_dir().join("d_main.c"));
        assert!(macros.contains_key("DEVDATA"));
        assert!(macros.contains_key("DEVMAPS"));
    }

    #[test]
    fn test_resolve_excludes_multi_token_macros() {
        // FRACUNIT is a numeric expression macro, not a literal -- must not
        // show up here.
        let mut resolver = LiteralMacroResolver::new();
        let macros = resolver.resolve(&corpus_dir().join("m_fixed.c"));
        assert!(!macros.contains_key("FRACUNIT"));
    }
}
