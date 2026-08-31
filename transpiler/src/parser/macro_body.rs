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
//!
//! Before parsing, a body's tokens also go through Step 4b's literal-macro
//! substitution (`macro_literal_subst::substitute_adjacent_literal_macros`)
//! -- the exact same "a macro identifier sits immediately next to a
//! string/char literal" case Step 4b already handles for code, just applied
//! to a `#define` body's own tokens instead. This is what turns e.g.
//! `#define NETEND "...\n\n"PRESSKEY` (a string literal directly followed
//! by another macro that itself expands to one) into two adjacent string
//! literal tokens, which `grammar.rs`'s primary-expression parser already
//! merges into one `Expr::StringLiteral` -- no separate concatenation logic
//! needed here either.

use crate::parser::SourceChunk;
use crate::parser::ast::{BlockItem, Expr};
use crate::parser::grammar::{parse_block_items_from_tokens, parse_expr_from_tokens};
use crate::parser::imports::ImportResolver;
use crate::parser::lexer::{LexItem, Token, lex_chunks};
use crate::parser::macro_literal_subst::substitute_adjacent_literal_macros;
use crate::parser::macro_literals::LiteralMacroResolver;
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
    /// No replacement text at all (`#define FOO`, or `#define FOO(x)` with
    /// nothing after the parameter list) -- a pure flag macro, meaningful
    /// only to `#ifdef`/`defined()`, never substituted as a value. Kept
    /// distinct from `Unparseable`: this isn't a body that failed to parse,
    /// there's no body to parse in the first place.
    Empty {
        params: Option<Vec<String>>,
    },
    /// Not a single expression, but a real sequence of declarations and/or
    /// statements -- what you'd get if the body were pasted directly into a
    /// function body (e.g. `Z_ChangeTag`'s `{ if (...) I_Error(...);
    /// Z_ChangeTag2(p,t); };`, or a flat `(oc) = 0; if (...) ...;` with no
    /// enclosing braces at all). `params: None` for an object-like macro,
    /// `Some(params)` for a function-like one.
    Statements {
        params: Option<Vec<String>>,
        body: Vec<BlockItem>,
    },
    /// Had replacement text, but it didn't reduce to a single expression or
    /// a parseable statement sequence (a bare type name, a lone storage
    /// class keyword, ...) -- kept for provenance/diagnostics, not a hard
    /// error. See `docs/KNOWN_LIMITATIONS.md`.
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
/// since raw quote characters only mean anything to the partitioner, then
/// applying Step 4b's adjacent-literal-macro substitution so a body like
/// `"..."PRESSKEY` comes out as two literal tokens instead of a literal
/// followed by a bare, unresolvable identifier.
fn lex_body_tokens(body: &str, literal_macros: &HashMap<String, Token>) -> Option<Vec<Token>> {
    let chunks = partition_source(body);
    let entries = lex_chunks(&chunks).ok()?;
    let entries = substitute_adjacent_literal_macros(entries, literal_macros);
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
/// typedef set and literal-macro map visible at its home translation unit.
fn parse_macro_body(
    params: &Option<Vec<String>>,
    body_text: &str,
    typedefs: &HashSet<String>,
    literal_macros: &HashMap<String, Token>,
) -> MacroBody {
    if body_text.trim().is_empty() {
        return MacroBody::Empty {
            params: params.clone(),
        };
    }
    let Some(tokens) = lex_body_tokens(body_text, literal_macros).filter(|t| !t.is_empty()) else {
        return MacroBody::Unparseable(body_text.to_string());
    };
    // A macro parameter is just a plain identifier within the body -- if
    // its name happens to collide with a typedef name (e.g. a parameter
    // called `boolean`), it must not be misread as a type in a cast/decl
    // position, for either attempt below.
    let mut scoped_typedefs = typedefs.clone();
    for p in params.iter().flatten() {
        scoped_typedefs.remove(p);
    }

    if let Ok(expr) = parse_expr_from_tokens(tokens.clone(), scoped_typedefs.clone()) {
        return match params {
            None => MacroBody::Object(expr),
            Some(params) => MacroBody::Function {
                params: params.clone(),
                body: expr,
            },
        };
    }
    if let Ok(items) = parse_block_items_from_tokens(tokens, scoped_typedefs) {
        return MacroBody::Statements {
            params: params.clone(),
            body: items,
        };
    }
    MacroBody::Unparseable(body_text.to_string())
}

/// Resolves every `#define` visible to a file into a parsed `MacroBody`,
/// caching the structural (typedef-independent) part of that work across
/// files the same way `ImportResolver`/`LiteralMacroResolver` do.
#[derive(Default)]
pub struct MacroBodyResolver {
    raw: RawMacroCollector,
    imports: ImportResolver,
    literals: LiteralMacroResolver,
}

impl MacroBodyResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every macro visible to `path` (its own `#define`s plus everything
    /// transitively `#include`d), each parsed into a `MacroBody` using
    /// `path`'s own Step 6b import-resolved typedef set -- matching Step
    /// 6c's single flat per-translation-unit typedef namespace -- and Step
    /// 4b's import-resolved literal-macro map, for adjacent-literal
    /// substitution within the body itself.
    pub fn resolve(&mut self, path: &Path) -> HashMap<String, MacroBody> {
        let raw = self.raw.collect(path);
        let typedefs = self.imports.resolve(path);
        let literal_macros = self.literals.resolve(path);
        raw.into_iter()
            .map(|(name, (params, body))| {
                (
                    name,
                    parse_macro_body(&params, &body, &typedefs, &literal_macros),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::TokenKind;
    use crate::parser::ast::{BinaryOp, Expr, Stmt};

    fn corpus_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_object_like_macro_parses_to_single_expr() {
        let typedefs = HashSet::new();
        let body = parse_macro_body(&None, "(1<<16)", &typedefs, &HashMap::new());
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
            &HashMap::new(),
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
    fn test_empty_body_is_empty_not_unparseable() {
        let typedefs = HashSet::new();
        assert_eq!(
            parse_macro_body(&None, "", &typedefs, &HashMap::new()),
            MacroBody::Empty { params: None }
        );
        assert_eq!(
            parse_macro_body(&None, "   ", &typedefs, &HashMap::new()),
            MacroBody::Empty { params: None }
        );
    }

    #[test]
    fn test_empty_function_like_body_keeps_params() {
        let typedefs = HashSet::new();
        let params = Some(vec!["x".to_string()]);
        assert_eq!(
            parse_macro_body(&params, "", &typedefs, &HashMap::new()),
            MacroBody::Empty {
                params: Some(vec!["x".to_string()])
            }
        );
    }

    #[test]
    fn test_dangling_trailing_tokens_are_unparseable() {
        // Not a single expression, and not a valid statement sequence
        // either -- `2` has no terminating `;`, so it can't close out as a
        // statement.
        let typedefs = HashSet::new();
        assert!(matches!(
            parse_macro_body(&None, "1; 2", &typedefs, &HashMap::new()),
            MacroBody::Unparseable(_)
        ));
    }

    #[test]
    fn test_multi_statement_body_parses_as_statements() {
        // Mirrors am_map.c's DOOUTCODE(oc,mx,my): not a single expression,
        // but a valid sequence of statements.
        let typedefs = HashSet::new();
        let params = Some(vec!["oc".to_string()]);
        let body = parse_macro_body(
            &params,
            "(oc) = 0; if ((oc) < 0) (oc) = 1;",
            &typedefs,
            &HashMap::new(),
        );
        match body {
            MacroBody::Statements { params, body } => {
                assert_eq!(params, Some(vec!["oc".to_string()]));
                assert_eq!(body.len(), 2);
                assert!(matches!(body[0], BlockItem::Stmt(Stmt::Expr(Some(_)))));
                assert!(matches!(body[1], BlockItem::Stmt(Stmt::If { .. })));
            }
            other => panic!("expected a parsed statement sequence, got {other:?}"),
        }
    }

    #[test]
    fn test_braced_compound_body_parses_as_statements() {
        // Mirrors z_zone.h's Z_ChangeTag(p,t): a single `{ ... }` block,
        // plus the macro author's own trailing `;` outside it.
        let typedefs = HashSet::new();
        let params = Some(vec!["p".to_string(), "t".to_string()]);
        let body = parse_macro_body(&params, "{ f(p,t); };", &typedefs, &HashMap::new());
        match body {
            MacroBody::Statements { body, .. } => {
                assert!(matches!(body[0], BlockItem::Stmt(Stmt::Compound(_))));
            }
            other => panic!("expected a parsed statement sequence, got {other:?}"),
        }
    }

    #[test]
    fn test_declaration_body_parses_as_statements() {
        // Mirrors i_sound.h's SEQ_DEFINEBUF(len): declarations, not
        // expressions or "plain" statements.
        let typedefs = HashSet::new();
        let body = parse_macro_body(
            &None,
            "unsigned char buf[4]; int n = 0;",
            &typedefs,
            &HashMap::new(),
        );
        match body {
            MacroBody::Statements { body, .. } => {
                assert_eq!(body.len(), 2);
                assert!(matches!(body[0], BlockItem::Decl(_)));
                assert!(matches!(body[1], BlockItem::Decl(_)));
            }
            other => panic!("expected a parsed statement sequence, got {other:?}"),
        }
    }

    #[test]
    fn test_string_literal_adjacent_to_another_literal_macro_concatenates() {
        // Mirrors d_englsh.h: #define NETEND "...\n\n"PRESSKEY, where
        // PRESSKEY is itself a plain literal-bodied macro (#define PRESSKEY
        // "press a key."). Without Step 4b's substitution this is a string
        // literal directly followed by a bare, unresolvable identifier --
        // not a single expression. With it, both become literal tokens and
        // grammar.rs's own adjacent-string-literal handling merges them.
        let typedefs = HashSet::new();
        let mut literal_macros = HashMap::new();
        literal_macros.insert(
            "PRESSKEY".to_string(),
            Token {
                kind: TokenKind::StringLiteral,
                text: "\"press a key.\"".to_string(),
            },
        );
        let body = parse_macro_body(
            &None,
            "\"you can't end a netgame!\\n\\n\"PRESSKEY",
            &typedefs,
            &literal_macros,
        );
        match body {
            MacroBody::Object(Expr::StringLiteral(s)) => {
                // Real content concatenation (`grammar.rs`'s own
                // adjacent-string-literal handling now merges the two
                // tokens' *content*, dropping the closing/opening quote
                // pair at the seam, rather than naively juxtaposing both
                // already-quoted texts -- see that fix's own doc comment,
                // built against `P_PlayerInSpecialSector`'s real `I_Error`
                // call needing a genuine merged Rust string literal).
                assert_eq!(s, "\"you can't end a netgame!\\n\\npress a key.\"");
            }
            other => panic!("expected a merged string literal, got {other:?}"),
        }
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
    fn test_resolve_z_changetag_with_predefined_file_and_line() {
        // z_zone.h's Z_ChangeTag(p,t) is a braced-block macro whose body
        // includes "Z_CT at "__FILE__":%i" -- a string literal directly
        // followed by __FILE__ directly followed by another string
        // literal. Without treating __FILE__/__LINE__ as predefined
        // literals, this is unparseable (a bare, unresolvable identifier
        // wedged between two literals with no operator).
        let mut resolver = MacroBodyResolver::new();
        let macros = resolver.resolve(&corpus_dir().join("z_zone.c"));
        let body = macros
            .get("Z_ChangeTag")
            .expect("expected Z_ChangeTag to resolve via imports");
        assert!(
            matches!(body, MacroBody::Statements { .. }),
            "expected Z_ChangeTag to parse as a statement sequence, got {body:?}"
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
        let (mut objects, mut functions, mut empty, mut statements, mut unparseable) =
            (0, 0, 0, 0, 0);
        for path in &files {
            for body in resolver.resolve(path).values() {
                match body {
                    MacroBody::Object(_) => objects += 1,
                    MacroBody::Function { .. } => functions += 1,
                    MacroBody::Empty { .. } => empty += 1,
                    MacroBody::Statements { .. } => statements += 1,
                    MacroBody::Unparseable(_) => unparseable += 1,
                }
            }
        }
        eprintln!(
            "macro body parsing over {} files: {objects} object, {functions} function, \
             {empty} empty, {statements} statements, {unparseable} unparseable",
            files.len()
        );
        assert!(
            objects + functions > 0,
            "expected at least some macros to parse"
        );
    }
}
