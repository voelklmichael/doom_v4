//! Substitutes literal-bodied macros (resolved by `macro_literals.rs`)
//! wherever they sit immediately next to a string or char literal in the
//! Step 4 token stream -- the one place a missing macro definition actually
//! breaks parsing (see `docs/KNOWN_LIMITATIONS.md`). After substitution, the
//! macro identifier becomes a literal token itself, so the existing
//! adjacent-string-literal-concatenation handling in `grammar.rs`'s
//! primary-expression parser picks it up automatically -- no separate
//! concatenation logic needed here.
//!
//! Deliberately narrow: a macro identifier anywhere *other* than directly
//! touching a literal token is left untouched. This is not a general
//! preprocessor macro expander.

use crate::parser::lexer::{LexEntry, LexItem, Token, TokenKind};
use std::collections::HashMap;

/// Replaces every `Identifier` token in `entries` that both (a) names a
/// macro in `macros` and (b) has a string/char literal token immediately
/// before or after it (ignoring any comments/directives interleaved between
/// them, which carry no grammatical weight) with that macro's literal
/// token. Every other occurrence of the same identifier is left alone.
pub fn substitute_adjacent_literal_macros(
    mut entries: Vec<LexEntry>,
    macros: &HashMap<String, Token>,
) -> Vec<LexEntry> {
    if macros.is_empty() {
        return entries;
    }

    // Indices of entries that are actual tokens, in stream order -- lets us
    // find each token's true token-neighbors without being thrown off by
    // Comment/Directive entries sitting between them in `entries`.
    let token_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter_map(|(i, e)| matches!(e.item, LexItem::Token(_)).then_some(i))
        .collect();

    let mut replacements = Vec::new();
    for (pos, &idx) in token_indices.iter().enumerate() {
        let LexItem::Token(tok) = &entries[idx].item else {
            unreachable!("token_indices only contains Token entries")
        };
        if tok.kind != TokenKind::Identifier {
            continue;
        }
        let Some(replacement) = macros.get(&tok.text) else {
            continue;
        };

        let prev_is_literal = pos > 0 && is_literal_token(&entries[token_indices[pos - 1]]);
        let next_is_literal =
            pos + 1 < token_indices.len() && is_literal_token(&entries[token_indices[pos + 1]]);
        if prev_is_literal || next_is_literal {
            replacements.push((idx, replacement.clone()));
        }
    }

    for (idx, replacement) in replacements {
        entries[idx].item = LexItem::Token(replacement);
    }
    entries
}

fn is_literal_token(entry: &LexEntry) -> bool {
    matches!(&entry.item, LexItem::Token(t) if matches!(t.kind, TokenKind::StringLiteral | TokenKind::CharLiteral))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::lexer::lex_chunks;
    use crate::parser::partitioner::partition_source;

    fn lex(code: &str) -> Vec<LexEntry> {
        let chunks = partition_source(code);
        lex_chunks(&chunks).unwrap()
    }

    /// Builds a raw literal `Token` directly, avoiding any pipeline call --
    /// `text` is the literal's exact text including quotes (e.g. `"\"x\""`).
    fn literal_token(text: &str) -> Token {
        let kind = if text.starts_with('\'') {
            TokenKind::CharLiteral
        } else {
            TokenKind::StringLiteral
        };
        Token {
            kind,
            text: text.to_string(),
        }
    }

    fn macro_map(pairs: &[(&str, &str)]) -> HashMap<String, Token> {
        pairs
            .iter()
            .map(|(name, literal)| (name.to_string(), literal_token(literal)))
            .collect()
    }

    fn token_texts(entries: &[LexEntry]) -> Vec<(TokenKind, String)> {
        entries
            .iter()
            .filter_map(|e| match &e.item {
                LexItem::Token(t) => Some((t.kind, t.text.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_macro_immediately_before_string_literal_is_substituted() {
        let entries = lex("D_AddFile (DEVDATA\"doom1.wad\");\n");
        let macros = macro_map(&[("DEVDATA", "\"devdata\"")]);
        let out = substitute_adjacent_literal_macros(entries, &macros);
        let texts = token_texts(&out);
        assert!(
            !texts
                .iter()
                .any(|(k, t)| *k == TokenKind::Identifier && t == "DEVDATA"),
            "DEVDATA identifier should have been substituted, got: {texts:?}"
        );
        assert!(
            texts
                .iter()
                .any(|(k, t)| *k == TokenKind::StringLiteral && t == "\"devdata\"")
        );
    }

    #[test]
    fn test_macro_immediately_after_string_literal_is_substituted() {
        let entries = lex("sprintf(s, \"%s\\n\\n\"DOSY, x);\n");
        let macros = macro_map(&[("DOSY", "\"(press y to quit)\"")]);
        let out = substitute_adjacent_literal_macros(entries, &macros);
        let texts = token_texts(&out);
        assert!(
            !texts
                .iter()
                .any(|(k, t)| *k == TokenKind::Identifier && t == "DOSY")
        );
    }

    #[test]
    fn test_macro_sandwiched_between_two_literals_is_substituted() {
        let entries = lex("sprintf(f, \"~\"DEVMAPS\"E%cM%c.wad\");\n");
        let macros = macro_map(&[("DEVMAPS", "\"devmaps\"")]);
        let out = substitute_adjacent_literal_macros(entries, &macros);
        let texts = token_texts(&out);
        assert!(
            !texts
                .iter()
                .any(|(k, t)| *k == TokenKind::Identifier && t == "DEVMAPS")
        );
        // Three consecutive string literals now, ready for the parser's
        // existing adjacent-string-literal merge.
        let string_run: Vec<_> = texts
            .iter()
            .filter(|(k, _)| *k == TokenKind::StringLiteral)
            .collect();
        assert_eq!(string_run.len(), 3);
    }

    #[test]
    fn test_macro_not_adjacent_to_a_literal_is_left_alone() {
        let entries = lex("printf(DEVDATA);\n");
        let macros = macro_map(&[("DEVDATA", "\"devdata\"")]);
        let out = substitute_adjacent_literal_macros(entries, &macros);
        let texts = token_texts(&out);
        assert!(
            texts
                .iter()
                .any(|(k, t)| *k == TokenKind::Identifier && t == "DEVDATA")
        );
    }

    #[test]
    fn test_unrelated_identifier_is_left_alone() {
        let entries = lex("char *s = SOMETHING \"literal\";\n");
        let macros = macro_map(&[("DEVDATA", "\"devdata\"")]);
        let out = substitute_adjacent_literal_macros(entries, &macros);
        let texts = token_texts(&out);
        assert!(
            texts
                .iter()
                .any(|(k, t)| *k == TokenKind::Identifier && t == "SOMETHING")
        );
    }

    #[test]
    fn test_empty_macro_map_is_a_no_op() {
        let entries = lex("DEVDATA\"x\";\n");
        let out = substitute_adjacent_literal_macros(entries, &HashMap::new());
        let texts = token_texts(&out);
        assert!(
            texts
                .iter()
                .any(|(k, t)| *k == TokenKind::Identifier && t == "DEVDATA")
        );
    }
}
