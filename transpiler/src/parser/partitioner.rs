//! Step 2: High-Level Source Code Partitioning
//!
//! Partitions spliced C source into comments, string/char literals,
//! preprocessor directives, and unparsed C code chunks.

/// Classification of a chunk of (already line-spliced) source text.
///
/// Concatenating `raw_text()` of every chunk, in order, reproduces the
/// spliced source exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceChunk {
    /// C/C++ style comments
    Comment(CommentChunk),
    /// String literal, e.g. `"hello world"` (including the quotes)
    StringLiteral(String),
    /// Character literal, e.g. `'a'`, `'\n'` (including the quotes)
    CharLiteral(String),
    /// Preprocessor directive, e.g. `#include <stdio.h>`, `#define FOO 1`.
    /// `raw` retains the exact source text; `directive` is the parsed form.
    Preprocessor {
        raw: String,
        directive: PreprocessorDirective,
    },
    /// Unparsed plain C code chunk
    Code(String),
}

impl SourceChunk {
    /// The exact original source text this chunk was parsed from.
    pub fn raw_text(&self) -> &str {
        match self {
            SourceChunk::Comment(CommentChunk::Line(s)) => s,
            SourceChunk::Comment(CommentChunk::Block(s)) => s,
            SourceChunk::StringLiteral(s) => s,
            SourceChunk::CharLiteral(s) => s,
            SourceChunk::Preprocessor { raw, .. } => raw,
            SourceChunk::Code(s) => s,
        }
    }
}

/// Comment variant: line (`//`) or block (`/* ... */`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommentChunk {
    Line(String),
    Block(String),
}

/// Preprocessor directive variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessorDirective {
    Include {
        path: String,
        /// true for `<...>`, false for `"..."`
        is_system: bool,
    },
    Define {
        name: String,
        /// `Some(params)` for a function-like macro, e.g. `#define FOO(a, b)`
        params: Option<Vec<String>>,
        body: String,
    },
    Undef(String),
    If(String),
    Ifdef(String),
    Ifndef(String),
    Elif(String),
    Else,
    Endif,
    Pragma(String),
    Error(String),
    Other { directive: String, rest: String },
}

/// Partitions spliced source text into high-level chunks.
pub fn partition_source(source: &str) -> Vec<SourceChunk> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut chunks = Vec::new();

    let mut code_start = 0;
    let mut i = 0;
    let mut at_line_start = true;

    macro_rules! flush_code {
        () => {
            if code_start < i {
                chunks.push(SourceChunk::Code(source[code_start..i].to_string()));
            }
        };
    }

    while i < len {
        let b = bytes[i];

        // Leading horizontal whitespace doesn't end a line-start context.
        if at_line_start && (b == b' ' || b == b'\t') {
            i += 1;
            continue;
        }

        // Preprocessor directives must start at the beginning of a line
        // (line continuations were already resolved in Step 1).
        if at_line_start && b == b'#' {
            flush_code!();
            let start = i;
            while i < len && bytes[i] != b'\n' && bytes[i] != b'\r' {
                i += 1;
            }
            let raw = source[start..i].to_string();
            let directive = parse_preprocessor_directive(&raw);
            chunks.push(SourceChunk::Preprocessor { raw, directive });
            code_start = i;
            at_line_start = false;
            continue;
        }

        // Line comment: //
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            flush_code!();
            let start = i;
            i += 2;
            while i < len && bytes[i] != b'\n' && bytes[i] != b'\r' {
                i += 1;
            }
            chunks.push(SourceChunk::Comment(CommentChunk::Line(
                source[start..i].to_string(),
            )));
            code_start = i;
            at_line_start = false;
            continue;
        }

        // Block comment: /* ... */
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            flush_code!();
            let start = i;
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = if i + 1 < len { i + 2 } else { len };
            chunks.push(SourceChunk::Comment(CommentChunk::Block(
                source[start..i].to_string(),
            )));
            code_start = i;
            at_line_start = false;
            continue;
        }

        // String literal: "..."
        if b == b'"' {
            flush_code!();
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'"' {
                i += if bytes[i] == b'\\' && i + 1 < len { 2 } else { 1 };
            }
            if i < len {
                i += 1; // closing quote
            }
            chunks.push(SourceChunk::StringLiteral(source[start..i].to_string()));
            code_start = i;
            at_line_start = false;
            continue;
        }

        // Character literal: '...'
        if b == b'\'' {
            flush_code!();
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'\'' {
                i += if bytes[i] == b'\\' && i + 1 < len { 2 } else { 1 };
            }
            if i < len {
                i += 1; // closing quote
            }
            chunks.push(SourceChunk::CharLiteral(source[start..i].to_string()));
            code_start = i;
            at_line_start = false;
            continue;
        }

        // Plain code byte.
        if b == b'\n' {
            at_line_start = true;
        } else if b != b' ' && b != b'\t' && b != b'\r' {
            at_line_start = false;
        }
        i += 1;
    }

    flush_code!();
    chunks
}

/// Parses a raw preprocessor directive line (e.g. `#include "doomdef.h"`).
pub fn parse_preprocessor_directive(line: &str) -> PreprocessorDirective {
    let trimmed = line.trim();
    let without_hash = trimmed.strip_prefix('#').unwrap_or(trimmed).trim_start();

    let mut parts = without_hash.splitn(2, |c: char| c.is_whitespace());
    let directive_name = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    match directive_name {
        "include" => {
            if rest.starts_with('<') && rest.ends_with('>') && rest.len() >= 2 {
                PreprocessorDirective::Include {
                    path: rest[1..rest.len() - 1].to_string(),
                    is_system: true,
                }
            } else if rest.starts_with('"') && rest.ends_with('"') && rest.len() >= 2 {
                PreprocessorDirective::Include {
                    path: rest[1..rest.len() - 1].to_string(),
                    is_system: false,
                }
            } else {
                PreprocessorDirective::Other {
                    directive: "include".to_string(),
                    rest: rest.to_string(),
                }
            }
        }
        "define" => parse_define(rest),
        "undef" => PreprocessorDirective::Undef(rest.to_string()),
        "if" => PreprocessorDirective::If(rest.to_string()),
        "ifdef" => PreprocessorDirective::Ifdef(rest.to_string()),
        "ifndef" => PreprocessorDirective::Ifndef(rest.to_string()),
        "elif" => PreprocessorDirective::Elif(rest.to_string()),
        "else" => PreprocessorDirective::Else,
        "endif" => PreprocessorDirective::Endif,
        "pragma" => PreprocessorDirective::Pragma(rest.to_string()),
        "error" => PreprocessorDirective::Error(rest.to_string()),
        _ => PreprocessorDirective::Other {
            directive: directive_name.to_string(),
            rest: rest.to_string(),
        },
    }
}

/// Parses `#define FOO ...` or `#define FOO(x, y) ...`.
fn parse_define(rest: &str) -> PreprocessorDirective {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return PreprocessorDirective::Define {
            name: String::new(),
            params: None,
            body: String::new(),
        };
    }

    // Function-like macro: NAME(...), with no whitespace between name and '(' (ISO C).
    if let Some(open_paren) = trimmed.find('(') {
        let name_part = &trimmed[..open_paren];
        if !name_part.is_empty() && name_part.chars().all(|c| c.is_alphanumeric() || c == '_') {
            if let Some(close_offset) = trimmed[open_paren..].find(')') {
                let close_paren_idx = open_paren + close_offset;
                let params_str = &trimmed[open_paren + 1..close_paren_idx];
                let params = if params_str.trim().is_empty() {
                    Vec::new()
                } else {
                    params_str.split(',').map(|s| s.trim().to_string()).collect()
                };
                return PreprocessorDirective::Define {
                    name: name_part.to_string(),
                    params: Some(params),
                    body: trimmed[close_paren_idx + 1..].trim().to_string(),
                };
            }
        }
    }

    // Object-like macro: #define NAME BODY
    let mut parts = trimmed.splitn(2, |c: char| c.is_whitespace());
    let name = parts.next().unwrap_or("").to_string();
    let body = parts.next().unwrap_or("").trim().to_string();
    PreprocessorDirective::Define {
        name,
        params: None,
        body,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::splice;

    #[test]
    fn test_partition_includes_and_defines() {
        let code = r#"
#include <stdio.h>
#include "doomdef.h"

#define FRACBITS 16
#define FRACUNIT (1<<FRACBITS)
#define FIXEDMUL(a,b) (((a)*(b))>>FRACBITS)

// Main function
int main(int argc, char **argv) {
    char *msg = "Hello /* not a comment */ \"world\"";
    char c = '\n';
    /* Block comment
       spanning lines */
    return 0;
}
"#;

        let chunks = partition_source(code);

        let mut include_count = 0;
        let mut define_count = 0;
        let mut comment_count = 0;
        let mut str_lit_count = 0;
        let mut char_lit_count = 0;

        for chunk in &chunks {
            match chunk {
                SourceChunk::Preprocessor {
                    directive: PreprocessorDirective::Include { is_system, path },
                    ..
                } => {
                    include_count += 1;
                    if *is_system {
                        assert_eq!(path, "stdio.h");
                    } else {
                        assert_eq!(path, "doomdef.h");
                    }
                }
                SourceChunk::Preprocessor {
                    directive: PreprocessorDirective::Define { name, params, body },
                    ..
                } => {
                    define_count += 1;
                    if name == "FIXEDMUL" {
                        assert_eq!(params.as_ref().unwrap(), &vec!["a".to_string(), "b".to_string()]);
                        assert_eq!(body, "(((a)*(b))>>FRACBITS)");
                    }
                }
                SourceChunk::Comment(_) => comment_count += 1,
                SourceChunk::StringLiteral(s) => {
                    str_lit_count += 1;
                    assert_eq!(s, "\"Hello /* not a comment */ \\\"world\\\"\"");
                }
                SourceChunk::CharLiteral(c) => {
                    char_lit_count += 1;
                    assert_eq!(c, "'\\n'");
                }
                SourceChunk::Code(_) => {}
                _ => {}
            }
        }

        assert_eq!(include_count, 2);
        assert_eq!(define_count, 3);
        assert_eq!(comment_count, 2);
        assert_eq!(str_lit_count, 1);
        assert_eq!(char_lit_count, 1);
    }

    #[test]
    fn test_round_trip_reconstructs_source() {
        let code = "#include <a.h>\n// comment\nint x = 1; /* c */ char *s = \"a\\\"b\"; char c = '\\'';\n";
        let chunks = partition_source(code);
        let reconstructed: String = chunks.iter().map(SourceChunk::raw_text).collect();
        assert_eq!(reconstructed, code);
    }

    #[test]
    fn test_round_trip_across_full_corpus() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("linuxdoom-1.10 directory should exist") {
            let path = entry.unwrap().path();
            let is_source = matches!(path.extension().and_then(|e| e.to_str()), Some("c") | Some("h"));
            if !path.is_file() || !is_source {
                continue;
            }
            let content = std::fs::read_to_string(&path).expect("source file should be valid UTF-8");
            let spliced = splice(&content);
            let chunks = partition_source(&spliced.text);
            let reconstructed: String = chunks.iter().map(SourceChunk::raw_text).collect();
            assert_eq!(reconstructed, spliced.text, "round-trip mismatch for {}", path.display());
            checked += 1;
        }
        assert!(checked > 100, "expected to check the full Doom corpus, only checked {checked}");
    }
}
