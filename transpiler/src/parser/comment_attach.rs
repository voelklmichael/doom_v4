//! Step 5: Comment Attaching
//!
//! Collapses the Step 4 token/comment/directive stream into tokens (and
//! surviving directives) only, by attaching every comment to exactly one
//! neighbor:
//! - if a token/directive precedes the comment on the same source line,
//!   the comment attaches there (a trailing/inline comment);
//! - otherwise the comment attaches to the token/directive that follows it
//!   (a leading comment).

use crate::parser::lexer::{LexEntry, LexItem, Token};
use crate::parser::partitioner::{CommentChunk, PreprocessorDirective};

/// A value paired with the comments attached to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commented<T> {
    pub t: T,
    pub comments: Vec<CommentChunk>,
}

/// Everything in the Step 4 stream a comment can attach to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Anchor {
    Token(Token),
    Directive(PreprocessorDirective),
}

/// Output of Step 5. `trailing_comments` holds any comments at the very end
/// of the stream with no following anchor to attach to (e.g. a comment after
/// the last token in a file) -- kept separately rather than dropped, since
/// there is no token left to own them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommentedStream {
    pub items: Vec<Commented<Anchor>>,
    pub trailing_comments: Vec<CommentChunk>,
}

/// Attaches each comment in `entries` to a single anchor per the rule above.
pub fn attach_comments(entries: Vec<LexEntry>) -> CommentedStream {
    let mut items: Vec<Commented<Anchor>> = Vec::new();
    let mut pending: Vec<CommentChunk> = Vec::new();
    let mut last_anchor_line: Option<usize> = None;

    for entry in entries {
        match entry.item {
            LexItem::Comment(c) => {
                if last_anchor_line == Some(entry.start_line) {
                    items
                        .last_mut()
                        .expect("last_anchor_line is only set once an anchor has been pushed")
                        .comments
                        .push(c);
                } else {
                    pending.push(c);
                }
            }
            LexItem::Token(tok) => {
                items.push(Commented {
                    t: Anchor::Token(tok),
                    comments: std::mem::take(&mut pending),
                });
                last_anchor_line = Some(entry.start_line);
            }
            LexItem::Directive(dir) => {
                items.push(Commented {
                    t: Anchor::Directive(dir),
                    comments: std::mem::take(&mut pending),
                });
                last_anchor_line = Some(entry.start_line);
            }
        }
    }

    CommentedStream {
        items,
        trailing_comments: pending,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::lexer::lex_chunks;
    use crate::parser::partitioner::partition_source;
    use crate::parser::{PreprocessorEnv, parse_chunks, resolve_conditionals};

    fn attach(code: &str) -> CommentedStream {
        let chunks = partition_source(code);
        let entries = lex_chunks(&chunks).unwrap();
        attach_comments(entries)
    }

    fn comments_of(item: &Commented<Anchor>) -> Vec<&str> {
        item.comments
            .iter()
            .map(|c| match c {
                CommentChunk::Line(s) | CommentChunk::Block(s) => s.as_str(),
            })
            .collect()
    }

    #[test]
    fn test_trailing_comment_attaches_to_preceding_token() {
        let stream = attach("int x; // trailing\n");
        let semi = stream
            .items
            .iter()
            .find(|c| matches!(&c.t, Anchor::Token(t) if t.text == ";"))
            .unwrap();
        assert_eq!(comments_of(semi), vec!["// trailing"]);
        assert!(stream.trailing_comments.is_empty());
    }

    #[test]
    fn test_leading_comment_attaches_to_following_token() {
        let stream = attach("// leading\nint x;\n");
        let int_tok = stream
            .items
            .iter()
            .find(|c| matches!(&c.t, Anchor::Token(t) if t.text == "int"))
            .unwrap();
        assert_eq!(comments_of(int_tok), vec!["// leading"]);
    }

    #[test]
    fn test_multiple_trailing_comments_same_line() {
        let stream = attach("int x; /* c1 */ // c2\n");
        let semi = stream
            .items
            .iter()
            .find(|c| matches!(&c.t, Anchor::Token(t) if t.text == ";"))
            .unwrap();
        assert_eq!(comments_of(semi), vec!["/* c1 */", "// c2"]);
    }

    #[test]
    fn test_block_comment_spanning_lines_is_still_trailing() {
        let stream = attach("foo(); /* long\nmultiline\ncomment */ bar();\n");
        let semi = stream
            .items
            .iter()
            .find(|c| matches!(&c.t, Anchor::Token(t) if t.text == ";" ) && !c.comments.is_empty())
            .unwrap();
        assert_eq!(comments_of(semi), vec!["/* long\nmultiline\ncomment */"]);
        let bar = stream
            .items
            .iter()
            .find(|c| matches!(&c.t, Anchor::Token(t) if t.text == "bar"))
            .unwrap();
        assert!(comments_of(bar).is_empty());
    }

    #[test]
    fn test_trailing_comment_at_eof_has_no_anchor() {
        let stream = attach("int x;\n// eof comment");
        assert_eq!(stream.trailing_comments.len(), 1);
        assert!(stream.items.iter().all(|c| c.comments.is_empty()));
    }

    #[test]
    fn test_comment_between_tokens_on_different_lines_attaches_forward() {
        let stream = attach("int x;\n// for y\nint y;\n");
        // The comment attaches to the *next* token, which is the second "int".
        let second_int = stream
            .items
            .iter()
            .filter(|c| matches!(&c.t, Anchor::Token(t) if t.text == "int"))
            .nth(1)
            .unwrap();
        assert_eq!(comments_of(second_int), vec!["// for y"]);
    }

    #[test]
    fn test_directive_can_be_anchor() {
        // Note: a comment trailing on the *same* line as a directive (e.g.
        // "#define FOO 1 // x") is swallowed into the directive's own raw
        // text by the Step 2 partitioner (it reads to end-of-line) and never
        // becomes a separate Comment item, so it can't reach Step 5 at all.
        // A leading comment on its own line isn't affected by that and
        // exercises directives as attachment anchors just as well.
        let stream = attach("// leading for define\n#define FOO 1\n");
        let define = stream
            .items
            .iter()
            .find(|c| {
                matches!(
                    &c.t,
                    Anchor::Directive(PreprocessorDirective::Define { .. })
                )
            })
            .unwrap();
        assert_eq!(comments_of(define), vec!["// leading for define"]);
    }

    #[test]
    fn test_full_corpus_attaches_every_comment() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("linuxdoom-1.10 directory should exist") {
            let path = entry.unwrap().path();
            let is_source = matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("c") | Some("h")
            );
            if !path.is_file() || !is_source {
                continue;
            }
            let content =
                std::fs::read_to_string(&path).expect("source file should be valid UTF-8");
            let (_, chunks) = parse_chunks(&content);
            let mut env = PreprocessorEnv::linux_doom_defaults();
            let resolved = resolve_conditionals(&chunks, &mut env)
                .unwrap_or_else(|e| panic!("preprocessor error in {}: {e}", path.display()));
            let lex_entries = lex_chunks(&resolved)
                .unwrap_or_else(|e| panic!("lex error in {}: {e}", path.display()));

            let expected_comments = lex_entries
                .iter()
                .filter(|e| matches!(e.item, LexItem::Comment(_)))
                .count();

            let stream = attach_comments(lex_entries);
            let attached: usize = stream.items.iter().map(|c| c.comments.len()).sum();
            let total = attached + stream.trailing_comments.len();

            assert_eq!(
                total,
                expected_comments,
                "comment count mismatch for {}",
                path.display()
            );
            checked += 1;
        }
        assert!(
            checked > 100,
            "expected to check the full Doom corpus, only checked {checked}"
        );
    }
}
