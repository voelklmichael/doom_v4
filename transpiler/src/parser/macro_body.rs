//! Step 7: Macro Body Parsing
//!
//! Parses each `#define`'s raw replacement text (Step 2's
//! `PreprocessorDirective::Define`) into a structured `MacroBody`: a single
//! `Expr` for an object-like macro, or a parameter list plus body `Expr` for
//! a function-like one. Purely syntactic -- assigns no type (that's the
//! typechecker's job, see `docs/02_TYPECHECKER.md` Step 2).
//!
//! Reuses Step 6c's expression grammar directly
//! (`grammar::parse_expr_from_tokens`), since disambiguating a macro body's
//! expression grammar (e.g. `(fixed_t)(x)` as cast-vs-call) needs the same
//! typedef set Step 6c already carries -- specifically the transitively
//! import-resolved set for the macro's *home translation unit* (Step 6b),
//! matching Step 6c's own single flat per-translation-unit typedef
//! namespace rather than trying to scope it per-header.
//!
//! Collecting which `#define`s are visible to a file (structural, no
//! typedef dependency) is cached across files the same way Step 6b/4b's
//! import resolvers are; parsing each one into an `Expr` (which does depend
//! on the caller's typedef set) is done fresh per top-level file.

use crate::parser::SourceChunk;
use crate::parser::ast::Expr;
use crate::parser::grammar::parse_expr_from_tokens;
use crate::parser::imports::ImportResolver;
use crate::parser::lexer::{LexItem, Token, lex_chunks};
use crate::parser::partitioner::{PreprocessorDirective, partition_source};
use crate::parser::system_headers::{read_resolved_chunks_and_includes, resolve_include_path};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum MacroBody {
    Object(Expr),
    Function {
        params: Vec<String>,
        body: Expr,
    },
    /// Didn't reduce to a single expression (empty body, leftover tokens, or
    /// a non-expression fragment) -- kept for provenance/diagnostics, not a
    /// hard error. See `docs/KNOWN_LIMITATIONS.md`.
    Unparseable(String),
}

/// A `#define`'s raw, unparsed form: `Some(params)` for a function-like
/// macro, plus its replacement text.
type RawMacroDef = (Option<Vec<String>>, String);

/// Every `#define`'s raw `(params, body text)`, keyed by name, visible to
/// `path`: its own top-level macros unioned with everything transitively
/// `#include`d. Structural only -- no typedef set needed, so cacheable
/// across files regardless of which translation unit ends up parsing the
/// bodies.
#[derive(Default)]
struct RawMacroCollector {
    cache: HashMap<PathBuf, HashMap<String, RawMacroDef>>,
}

impl RawMacroCollector {
    fn collect(&mut self, path: &Path) -> HashMap<String, RawMacroDef> {
        let mut visiting = HashSet::new();
        self.collect_inner(path, &mut visiting)
    }

    fn collect_inner(
        &mut self,
        path: &Path,
        visiting: &mut HashSet<PathBuf>,
    ) -> HashMap<String, RawMacroDef> {
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
                    directive: PreprocessorDirective::Define { name, params, body },
                    ..
                } = chunk
                {
                    result.insert(name.clone(), (params.clone(), body.clone()));
                }
            }
            let dir = key.parent().unwrap_or_else(|| Path::new("."));
            for inc in includes {
                if let Some(resolved_path) = resolve_include_path(&inc, dir) {
                    for (name, def) in self.collect_inner(&resolved_path, visiting) {
                        result.entry(name).or_insert(def);
                    }
                }
            }
        }

        visiting.remove(&key);
        self.cache.insert(key, result.clone());
        result
    }
}

/// Lexes a `#define` body's raw text into tokens, going through Step 2
/// partitioning first (same as `macro_literals.rs`'s `as_single_literal`)
/// since raw quote characters only mean anything to the partitioner.
fn lex_body_tokens(body: &str) -> Option<Vec<Token>> {
    let chunks = partition_source(body);
    let entries = lex_chunks(&chunks).ok()?;
    Some(
        entries
            .into_iter()
            .filter_map(|e| match e.item {
                LexItem::Token(t) => Some(t),
                _ => None,
            })
            .collect(),
    )
}

/// Parses one macro's `(params, body text)` into a `MacroBody`, given the
/// typedef set visible at its home translation unit.
fn parse_macro_body(
    params: &Option<Vec<String>>,
    body_text: &str,
    typedefs: &HashSet<String>,
) -> MacroBody {
    let Some(tokens) = lex_body_tokens(body_text).filter(|t| !t.is_empty()) else {
        return MacroBody::Unparseable(body_text.to_string());
    };
    match params {
        None => match parse_expr_from_tokens(tokens, typedefs.clone()) {
            Ok(expr) => MacroBody::Object(expr),
            Err(_) => MacroBody::Unparseable(body_text.to_string()),
        },
        Some(params) => {
            // A macro parameter is just a plain identifier within the body
            // -- if its name happens to collide with a typedef name (e.g. a
            // parameter called `boolean`), it must not be misread as a type
            // in a cast position.
            let mut scoped_typedefs = typedefs.clone();
            for p in params {
                scoped_typedefs.remove(p);
            }
            match parse_expr_from_tokens(tokens, scoped_typedefs) {
                Ok(expr) => MacroBody::Function {
                    params: params.clone(),
                    body: expr,
                },
                Err(_) => MacroBody::Unparseable(body_text.to_string()),
            }
        }
    }
}

/// Resolves every `#define` visible to a file into a parsed `MacroBody`,
/// caching the structural (typedef-independent) part of that work across
/// files the same way `ImportResolver`/`LiteralMacroResolver` do.
#[derive(Default)]
pub struct MacroBodyResolver {
    raw: RawMacroCollector,
    imports: ImportResolver,
}

impl MacroBodyResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every macro visible to `path` (its own `#define`s plus everything
    /// transitively `#include`d), each parsed into a `MacroBody` using
    /// `path`'s own Step 6b import-resolved typedef set -- matching Step
    /// 6c's single flat per-translation-unit typedef namespace.
    pub fn resolve(&mut self, path: &Path) -> HashMap<String, MacroBody> {
        let raw = self.raw.collect(path);
        let typedefs = self.imports.resolve(path);
        raw.into_iter()
            .map(|(name, (params, body))| (name, parse_macro_body(&params, &body, &typedefs)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{BinaryOp, Expr};

    fn corpus_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_object_like_macro_parses_to_single_expr() {
        let typedefs = HashSet::new();
        let body = parse_macro_body(&None, "(1<<16)", &typedefs);
        match body {
            MacroBody::Object(Expr::Binary {
                op: BinaryOp::Shl, ..
            }) => {}
            other => panic!("expected a parsed shift expr, got {other:?}"),
        }
    }

    #[test]
    fn test_function_like_macro_parses_params_and_body() {
        let typedefs = HashSet::new();
        let body = parse_macro_body(
            &Some(vec!["a".to_string(), "b".to_string()]),
            "((a) + (b))",
            &typedefs,
        );
        match body {
            MacroBody::Function { params, body } => {
                assert_eq!(params, vec!["a", "b"]);
                assert!(matches!(
                    body,
                    Expr::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            other => panic!("expected a parsed function-like macro, got {other:?}"),
        }
    }

    #[test]
    fn test_empty_body_is_unparseable() {
        let typedefs = HashSet::new();
        assert!(matches!(
            parse_macro_body(&None, "", &typedefs),
            MacroBody::Unparseable(_)
        ));
    }

    #[test]
    fn test_trailing_tokens_are_unparseable() {
        // Not a single expression -- two statements' worth of tokens.
        let typedefs = HashSet::new();
        assert!(matches!(
            parse_macro_body(&None, "1; 2", &typedefs),
            MacroBody::Unparseable(_)
        ));
    }

    #[test]
    fn test_resolve_fracunit_from_m_fixed_h() {
        // FRACUNIT is an object-like macro defined in m_fixed.h; m_fixed.c
        // pulls it in transitively.
        let mut resolver = MacroBodyResolver::new();
        let macros = resolver.resolve(&corpus_dir().join("m_fixed.c"));
        let body = macros
            .get("FRACUNIT")
            .expect("expected FRACUNIT to resolve via imports");
        assert!(
            matches!(body, MacroBody::Object(_)),
            "expected FRACUNIT to parse as a single object-like expression, got {body:?}"
        );
    }

    #[test]
    fn test_resolve_across_full_corpus_reports_coverage() {
        // Not a pass/fail assertion (matching Step 4b's "measure actual
        // scope before deciding it needs more" methodology) -- just proves
        // the resolver runs cleanly over the whole corpus and prints the
        // Object/Function/Unparseable split for follow-up.
        let dir = corpus_dir();
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        files.sort();
        assert!(files.len() > 50, "expected the full Doom .c corpus");

        let mut resolver = MacroBodyResolver::new();
        let (mut objects, mut functions, mut unparseable) = (0, 0, 0);
        for path in &files {
            for body in resolver.resolve(path).values() {
                match body {
                    MacroBody::Object(_) => objects += 1,
                    MacroBody::Function { .. } => functions += 1,
                    MacroBody::Unparseable(_) => unparseable += 1,
                }
            }
        }
        eprintln!(
            "macro body parsing over {} files: {objects} object, {functions} function, {unparseable} unparseable",
            files.len()
        );
        assert!(
            objects + functions > 0,
            "expected at least some macros to parse"
        );
    }
}
