//! Step 4: Lexing
//!
//! Lexes the active (Step 3-resolved) chunks into a flat, ordered stream of
//! C89 tokens, comments, and surviving preprocessor directives.

use crate::parser::partitioner::{CommentChunk, PreprocessorDirective, SourceChunk};

/// A single lexeme with its classification. `text` retains the exact source
/// text the token was lexed from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Keyword(Keyword),
    Identifier,
    IntegerConstant,
    FloatConstant,
    StringLiteral,
    CharLiteral,
    Punct(Punct),
}

/// The 32 C89 reserved keywords.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Auto,
    Break,
    Case,
    Char,
    Const,
    Continue,
    Default,
    Do,
    Double,
    Else,
    Enum,
    Extern,
    Float,
    For,
    Goto,
    If,
    Int,
    Long,
    Register,
    Return,
    Short,
    Signed,
    Sizeof,
    Static,
    Struct,
    Switch,
    Typedef,
    Union,
    Unsigned,
    Void,
    Volatile,
    While,
}

impl Keyword {
    fn from_str(s: &str) -> Option<Keyword> {
        Some(match s {
            "auto" => Keyword::Auto,
            "break" => Keyword::Break,
            "case" => Keyword::Case,
            "char" => Keyword::Char,
            "const" => Keyword::Const,
            "continue" => Keyword::Continue,
            "default" => Keyword::Default,
            "do" => Keyword::Do,
            "double" => Keyword::Double,
            "else" => Keyword::Else,
            "enum" => Keyword::Enum,
            "extern" => Keyword::Extern,
            "float" => Keyword::Float,
            "for" => Keyword::For,
            "goto" => Keyword::Goto,
            "if" => Keyword::If,
            "int" => Keyword::Int,
            "long" => Keyword::Long,
            "register" => Keyword::Register,
            "return" => Keyword::Return,
            "short" => Keyword::Short,
            "signed" => Keyword::Signed,
            "sizeof" => Keyword::Sizeof,
            "static" => Keyword::Static,
            "struct" => Keyword::Struct,
            "switch" => Keyword::Switch,
            "typedef" => Keyword::Typedef,
            "union" => Keyword::Union,
            "unsigned" => Keyword::Unsigned,
            "void" => Keyword::Void,
            "volatile" => Keyword::Volatile,
            "while" => Keyword::While,
            _ => return None,
        })
    }
}

/// C89 punctuators and operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Punct {
    LBracket,
    RBracket,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Dot,
    Arrow,
    PlusPlus,
    MinusMinus,
    Amp,
    Star,
    Plus,
    Minus,
    Tilde,
    Bang,
    Slash,
    Percent,
    ShiftLeft,
    ShiftRight,
    Lt,
    Gt,
    Le,
    Ge,
    EqEq,
    NotEq,
    Caret,
    Pipe,
    AmpAmp,
    PipePipe,
    Question,
    Colon,
    Semicolon,
    Ellipsis,
    Eq,
    StarEq,
    SlashEq,
    PercentEq,
    PlusEq,
    MinusEq,
    ShiftLeftEq,
    ShiftRightEq,
    AmpEq,
    CaretEq,
    PipeEq,
    Comma,
    Hash,
    HashHash,
}

const PUNCTS_3: &[(&str, Punct)] = &[
    ("<<=", Punct::ShiftLeftEq),
    (">>=", Punct::ShiftRightEq),
    ("...", Punct::Ellipsis),
];

const PUNCTS_2: &[(&str, Punct)] = &[
    ("->", Punct::Arrow),
    ("++", Punct::PlusPlus),
    ("--", Punct::MinusMinus),
    ("<<", Punct::ShiftLeft),
    (">>", Punct::ShiftRight),
    ("<=", Punct::Le),
    (">=", Punct::Ge),
    ("==", Punct::EqEq),
    ("!=", Punct::NotEq),
    ("&&", Punct::AmpAmp),
    ("||", Punct::PipePipe),
    ("*=", Punct::StarEq),
    ("/=", Punct::SlashEq),
    ("%=", Punct::PercentEq),
    ("+=", Punct::PlusEq),
    ("-=", Punct::MinusEq),
    ("&=", Punct::AmpEq),
    ("^=", Punct::CaretEq),
    ("|=", Punct::PipeEq),
    ("##", Punct::HashHash),
];

fn punct_1(c: char) -> Option<Punct> {
    Some(match c {
        '[' => Punct::LBracket,
        ']' => Punct::RBracket,
        '(' => Punct::LParen,
        ')' => Punct::RParen,
        '{' => Punct::LBrace,
        '}' => Punct::RBrace,
        '.' => Punct::Dot,
        '&' => Punct::Amp,
        '*' => Punct::Star,
        '+' => Punct::Plus,
        '-' => Punct::Minus,
        '~' => Punct::Tilde,
        '!' => Punct::Bang,
        '/' => Punct::Slash,
        '%' => Punct::Percent,
        '<' => Punct::Lt,
        '>' => Punct::Gt,
        '^' => Punct::Caret,
        '|' => Punct::Pipe,
        '?' => Punct::Question,
        ':' => Punct::Colon,
        ';' => Punct::Semicolon,
        '=' => Punct::Eq,
        ',' => Punct::Comma,
        '#' => Punct::Hash,
        _ => return None,
    })
}

/// An item in the Step 4 output stream: a token, a passed-through comment,
/// or a preprocessor directive that survived Step 3 (e.g. `#include`, `#define`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexItem {
    Token(Token),
    Comment(CommentChunk),
    Directive(PreprocessorDirective),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexError {
    UnrecognizedChar { ch: char, context: String },
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexError::UnrecognizedChar { ch, context } => {
                write!(f, "unrecognized character '{ch}' while lexing: {context}")
            }
        }
    }
}

impl std::error::Error for LexError {}

/// Lexes Step 3-resolved chunks into an ordered stream of tokens/comments/directives.
pub fn lex_chunks(chunks: &[SourceChunk]) -> Result<Vec<LexItem>, LexError> {
    let mut items = Vec::new();
    for chunk in chunks {
        match chunk {
            SourceChunk::Code(text) => {
                for tok in lex_code(text)? {
                    items.push(LexItem::Token(tok));
                }
            }
            SourceChunk::StringLiteral(s) => items.push(LexItem::Token(Token {
                kind: TokenKind::StringLiteral,
                text: s.clone(),
            })),
            SourceChunk::CharLiteral(s) => items.push(LexItem::Token(Token {
                kind: TokenKind::CharLiteral,
                text: s.clone(),
            })),
            SourceChunk::Comment(c) => items.push(LexItem::Comment(c.clone())),
            SourceChunk::Preprocessor { directive, .. } => {
                items.push(LexItem::Directive(directive.clone()))
            }
        }
    }
    Ok(items)
}

/// Lexes a single `Code` chunk's text into C89 tokens, skipping whitespace.
pub fn lex_code(text: &str) -> Result<Vec<Token>, LexError> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Identifier or keyword.
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let kind = match Keyword::from_str(&word) {
                Some(kw) => TokenKind::Keyword(kw),
                None => TokenKind::Identifier,
            };
            tokens.push(Token { kind, text: word });
            continue;
        }

        // Numeric constant: integer or floating, decimal/octal/hex.
        if c.is_ascii_digit() || (c == '.' && i + 1 < len && chars[i + 1].is_ascii_digit()) {
            let start = i;
            let mut is_float = false;

            if c == '0' && i + 1 < len && (chars[i + 1] == 'x' || chars[i + 1] == 'X') {
                i += 2;
                while i < len && chars[i].is_ascii_hexdigit() {
                    i += 1;
                }
            } else {
                while i < len && chars[i].is_ascii_digit() {
                    i += 1;
                }
                if i < len && chars[i] == '.' {
                    is_float = true;
                    i += 1;
                    while i < len && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < len && (chars[i] == 'e' || chars[i] == 'E') {
                    let mark = i;
                    let mut j = i + 1;
                    if j < len && (chars[j] == '+' || chars[j] == '-') {
                        j += 1;
                    }
                    if j < len && chars[j].is_ascii_digit() {
                        is_float = true;
                        i = j;
                        while i < len && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    } else {
                        i = mark;
                    }
                }
            }

            // Integer/float suffix letters (u/U/l/L/f/F in any combination).
            while i < len && matches!(chars[i], 'u' | 'U' | 'l' | 'L' | 'f' | 'F') {
                i += 1;
            }

            let word: String = chars[start..i].iter().collect();
            let kind = if is_float {
                TokenKind::FloatConstant
            } else {
                TokenKind::IntegerConstant
            };
            tokens.push(Token { kind, text: word });
            continue;
        }

        // Punctuators: maximal munch, longest match first.
        let remaining: String = chars[i..(i + 3).min(len)].iter().collect();
        if let Some(&(s, p)) = PUNCTS_3.iter().find(|(s, _)| remaining.starts_with(s)) {
            tokens.push(Token {
                kind: TokenKind::Punct(p),
                text: s.to_string(),
            });
            i += s.chars().count();
            continue;
        }
        if let Some(&(s, p)) = PUNCTS_2.iter().find(|(s, _)| remaining.starts_with(s)) {
            tokens.push(Token {
                kind: TokenKind::Punct(p),
                text: s.to_string(),
            });
            i += s.chars().count();
            continue;
        }
        if let Some(p) = punct_1(c) {
            tokens.push(Token {
                kind: TokenKind::Punct(p),
                text: c.to_string(),
            });
            i += 1;
            continue;
        }

        let context: String = chars[i.saturating_sub(10)..(i + 10).min(len)]
            .iter()
            .collect();
        return Err(LexError::UnrecognizedChar { ch: c, context });
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{PreprocessorEnv, parse_chunks, resolve_conditionals};

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex_code(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_identifiers_and_keywords() {
        assert_eq!(
            kinds("int foo_bar123 return_value"),
            vec![
                TokenKind::Keyword(Keyword::Int),
                TokenKind::Identifier,
                TokenKind::Identifier,
            ]
        );
    }

    #[test]
    fn test_integer_constants() {
        let toks = lex_code("42 0x1F 017 123UL 0XFFu").unwrap();
        assert!(toks.iter().all(|t| t.kind == TokenKind::IntegerConstant));
        assert_eq!(toks[0].text, "42");
        assert_eq!(toks[1].text, "0x1F");
        assert_eq!(toks[3].text, "123UL");
    }

    #[test]
    fn test_float_constants() {
        let toks = lex_code("3.14 1. .5 1e10 2.5e-3f").unwrap();
        assert!(toks.iter().all(|t| t.kind == TokenKind::FloatConstant));
        assert_eq!(toks.len(), 5);
    }

    #[test]
    fn test_punctuators_maximal_munch() {
        let toks = lex_code("a<<=b>>c...d->e++f--g&&h||i").unwrap();
        let puncts: Vec<Punct> = toks
            .iter()
            .filter_map(|t| match t.kind {
                TokenKind::Punct(p) => Some(p),
                _ => None,
            })
            .collect();
        assert_eq!(
            puncts,
            vec![
                Punct::ShiftLeftEq,
                Punct::ShiftRight,
                Punct::Ellipsis,
                Punct::Arrow,
                Punct::PlusPlus,
                Punct::MinusMinus,
                Punct::AmpAmp,
                Punct::PipePipe,
            ]
        );
    }

    #[test]
    fn test_unrecognized_char_errors() {
        let err = lex_code("int a = 1 @ 2;").unwrap_err();
        assert!(matches!(err, LexError::UnrecognizedChar { ch: '@', .. }));
    }

    #[test]
    fn test_lex_chunks_preserves_strings_comments_and_directives() {
        let code = "#define FOO 1\n// hi\nchar *s = \"hello\"; /* c */\n";
        let chunks = crate::parser::partitioner::partition_source(code);
        let items = lex_chunks(&chunks).unwrap();

        assert!(items.iter().any(|i| matches!(i, LexItem::Directive(PreprocessorDirective::Define { name, .. }) if name == "FOO")));
        assert!(
            items
                .iter()
                .any(|i| matches!(i, LexItem::Comment(CommentChunk::Line(s)) if s == "// hi"))
        );
        assert!(
            items
                .iter()
                .any(|i| matches!(i, LexItem::Comment(CommentChunk::Block(s)) if s == "/* c */"))
        );
        assert!(items.iter().any(|i| matches!(i, LexItem::Token(t) if t.kind == TokenKind::StringLiteral && t.text == "\"hello\"")));
    }

    #[test]
    fn test_full_corpus_lexes_cleanly() {
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
            lex_chunks(&resolved)
                .unwrap_or_else(|e| panic!("lex error in {}: {e}", path.display()));
            checked += 1;
        }
        assert!(
            checked > 100,
            "expected to check the full Doom corpus, only checked {checked}"
        );
    }
}
