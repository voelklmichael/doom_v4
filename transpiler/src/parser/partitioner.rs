//! Step 1.2: High-Level Source Code Partitioning
//!
//! Partitions spliced C source into comments, string/char literals,
//! preprocessor directives, and unparsed C code chunks.

use crate::parser::splicer::{splice, SplicedSource};

/// Represents the classification of a source code chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceChunk {
    /// C/C++ style comments
    Comment(CommentChunk),
    /// String literals, e.g. "hello world"
    StringLiteral(String),
    /// Character literals, e.g. 'a', '\n'
    CharLiteral(String),
    /// Preprocessor directive, e.g. #include <stdio.h>, #define FOO 1
    Preprocessor(PreprocessorDirective),
    /// Unparsed plain C code chunk
    Code(String),
}

/// Comment variant: line (//) or block (/* ... */).
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
        is_system: bool, // true if <...>, false if "..."
    },
    Define {
        name: String,
        params: Option<Vec<String>>, // Some(...) if function-like macro e.g. #define FOO(a, b)
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
    Other {
        directive: String,
        rest: String,
    },
}

/// Partitions a spliced source string into high-level chunks.
pub fn partition_source(source: &str) -> Vec<SourceChunk> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut chunks = Vec::new();
    let mut current_code = String::new();

    let mut i = 0;
    let mut at_line_start = true;

    while i < len {
        let b = bytes[i];

        // Check for line start (ignoring leading horizontal whitespace)
        if at_line_start && (b == b' ' || b == b'\t') {
            current_code.push(b as char);
            i += 1;
            continue;
        }

        // 1. Check for Preprocessor Directives (must be at line start after optional whitespace)
        if at_line_start && b == b'#' {
            if !current_code.is_empty() {
                chunks.push(SourceChunk::Code(current_code.clone()));
                current_code.clear();
            }

            let start = i;
            // Read until end of line (since line continuations were already spliced in Step 1.1)
            while i < len && bytes[i] != b'\n' && bytes[i] != b'\r' {
                i += 1;
            }
            let directive_raw = &source[start..i];
            let directive = parse_preprocessor_directive(directive_raw);
            chunks.push(SourceChunk::Preprocessor(directive));
            at_line_start = false;
            continue;
        }

        // 2. Check for Line Comment: //
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            if !current_code.is_empty() {
                chunks.push(SourceChunk::Code(current_code.clone()));
                current_code.clear();
            }

            let start = i;
            i += 2;
            while i < len && bytes[i] != b'\n' && bytes[i] != b'\r' {
                i += 1;
            }
            let comment_text = &source[start..i];
            chunks.push(SourceChunk::Comment(CommentChunk::Line(comment_text.to_string())));
            at_line_start = false;
            continue;
        }

        // 3. Check for Block Comment: /* ... */
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            if !current_code.is_empty() {
                chunks.push(SourceChunk::Code(current_code.clone()));
                current_code.clear();
            }

            let start = i;
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 < len {
                i += 2; // include */
            } else {
                i = len;
            }
            let comment_text = &source[start..i];
            chunks.push(SourceChunk::Comment(CommentChunk::Block(comment_text.to_string())));
            at_line_start = false;
            continue;
        }

        // 4. Check for String Literal: "..."
        if b == b'"' {
            if !current_code.is_empty() {
                chunks.push(SourceChunk::Code(current_code.clone()));
                current_code.clear();
            }

            let start = i;
            i += 1; // skip opening quote
            while i < len && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < len {
                    i += 2; // skip escape sequence like \" or \\
                } else {
                    i += 1;
                }
            }
            if i < len && bytes[i] == b'"' {
                i += 1; // skip closing quote
            }
            let str_lit = &source[start..i];
            chunks.push(SourceChunk::StringLiteral(str_lit.to_string()));
            at_line_start = false;
            continue;
        }

        // 5. Check for Character Literal: '...'
        if b == b'\'' {
            if !current_code.is_empty() {
                chunks.push(SourceChunk::Code(current_code.clone()));
                current_code.clear();
            }

            let start = i;
            i += 1; // skip opening quote
            while i < len && bytes[i] != b'\'' {
                if bytes[i] == b'\\' && i + 1 < len {
                    i += 2; // skip escape sequence like \' or \\
                } else {
                    i += 1;
                }
            }
            if i < len && bytes[i] == b'\'' {
                i += 1; // skip closing quote
            }
            let char_lit = &source[start..i];
            chunks.push(SourceChunk::CharLiteral(char_lit.to_string()));
            at_line_start = false;
            continue;
        }

        // 6. Regular Code characters
        current_code.push(b as char);
        if b == b'\n' {
            at_line_start = true;
        } else if b != b' ' && b != b'\t' && b != b'\r' {
            at_line_start = false;
        }
        i += 1;
    }

    if !current_code.is_empty() {
        chunks.push(SourceChunk::Code(current_code));
    }

    chunks
}

/// Convenience function that splices lines first, then partitions the source.
pub fn parse_chunks(source: &str) -> (SplicedSource, Vec<SourceChunk>) {
    let spliced = splice(source);
    let chunks = partition_source(&spliced.text);
    (spliced, chunks)
}

/// Parse a raw preprocessor directive line (e.g. `#include "doomdef.h"`)
pub fn parse_preprocessor_directive(line: &str) -> PreprocessorDirective {
    let trimmed = line.trim();
    let without_hash = if let Some(stripped) = trimmed.strip_prefix('#') {
        stripped.trim_start()
    } else {
        trimmed
    };

    // Extract directive name
    let mut parts = without_hash.splitn(2, |c: char| c.is_whitespace());
    let directive_name = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    match directive_name {
        "include" => {
            if rest.starts_with('<') && rest.ends_with('>') {
                PreprocessorDirective::Include {
                    path: rest[1..rest.len() - 1].to_string(),
                    is_system: true,
                }
            } else if rest.starts_with('"') && rest.ends_with('"') {
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

/// Parse `#define FOO ...` or `#define FOO(x, y) ...`
fn parse_define(rest: &str) -> PreprocessorDirective {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return PreprocessorDirective::Define {
            name: String::new(),
            params: None,
            body: String::new(),
        };
    }

    // Check if function-like: NAME(...)
    if let Some(open_paren) = trimmed.find('(') {
        let name_part = &trimmed[..open_paren];
        // Ensure no whitespace between name and '(' for function-like macro (ISO C standard)
        if !name_part.is_empty() && name_part.chars().all(|c| c.is_alphanumeric() || c == '_') {
            if let Some(close_paren) = trimmed[open_paren..].find(')') {
                let close_paren_idx = open_paren + close_paren;
                let name = name_part.to_string();
                let params_str = &trimmed[open_paren + 1..close_paren_idx];
                let params = if params_str.trim().is_empty() {
                    Vec::new()
                } else {
                    params_str.split(',').map(|s| s.trim().to_string()).collect()
                };
                let body = trimmed[close_paren_idx + 1..].trim().to_string();
                return PreprocessorDirective::Define {
                    name,
                    params: Some(params),
                    body,
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
                SourceChunk::Preprocessor(PreprocessorDirective::Include { is_system, path }) => {
                    include_count += 1;
                    if *is_system {
                        assert_eq!(path, "stdio.h");
                    } else {
                        assert_eq!(path, "doomdef.h");
                    }
                }
                SourceChunk::Preprocessor(PreprocessorDirective::Define { name, params, body }) => {
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
}
