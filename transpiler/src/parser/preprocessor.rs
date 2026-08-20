//! Step 3: Preprocessor Conditional Resolution
//!
//! Evaluates `#if` / `#ifdef` / `#ifndef` / `#elif` / `#else` / `#endif` blocks
//! and filters out chunks from inactive conditional branches.

use crate::parser::partitioner::{PreprocessorDirective, SourceChunk};
use std::collections::HashMap;

/// Preprocessor compilation environment: defined macros and their values.
#[derive(Debug, Clone, Default)]
pub struct PreprocessorEnv {
    pub macros: HashMap<String, Option<String>>,
}

impl PreprocessorEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// The environment implied by `linuxdoom-1.10/Makefile`'s `CFLAGS`
    /// (`-DNORMALUNIX -DLINUX`; `-DUSEASM` is commented out).
    pub fn linux_doom_defaults() -> Self {
        let mut env = Self::new();
        env.define("NORMALUNIX", None);
        env.define("LINUX", None);
        env
    }

    pub fn define(&mut self, name: &str, value: Option<&str>) {
        self.macros.insert(name.to_string(), value.map(|s| s.to_string()));
    }

    pub fn undef(&mut self, name: &str) {
        self.macros.remove(name);
    }

    pub fn is_defined(&self, name: &str) -> bool {
        self.macros.contains_key(name)
    }

    pub fn get_value(&self, name: &str) -> Option<&str> {
        self.macros.get(name).and_then(|opt| opt.as_deref())
    }
}

/// Errors that can occur while resolving conditional compilation blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessorError {
    UnmatchedEndif,
    UnmatchedElse,
    UnmatchedElif,
    UnterminatedIf,
    ExpressionError(String),
}

impl std::fmt::Display for PreprocessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreprocessorError::UnmatchedEndif => write!(f, "unmatched #endif without a corresponding #if/#ifdef"),
            PreprocessorError::UnmatchedElse => write!(f, "unmatched #else without a corresponding #if/#ifdef"),
            PreprocessorError::UnmatchedElif => write!(f, "unmatched #elif without a corresponding #if/#ifdef"),
            PreprocessorError::UnterminatedIf => write!(f, "unterminated #if/#ifdef block at end of file"),
            PreprocessorError::ExpressionError(msg) => write!(f, "invalid preprocessor expression: {msg}"),
        }
    }
}

impl std::error::Error for PreprocessorError {}

struct CondFrame {
    /// Has any branch in this if/elif/else chain already been taken?
    branch_taken: bool,
    /// Is the current branch active?
    is_active: bool,
    /// Was the enclosing scope active when this frame was pushed?
    parent_active: bool,
}

/// Resolves conditional compilation blocks, returning only the chunks from
/// branches that are active under `env`. Macro definitions/undefs encountered
/// in active branches update `env` as they're processed, in source order.
pub fn resolve_conditionals(
    chunks: &[SourceChunk],
    env: &mut PreprocessorEnv,
) -> Result<Vec<SourceChunk>, PreprocessorError> {
    let mut resolved = Vec::new();
    let mut cond_stack: Vec<CondFrame> = Vec::new();

    let is_currently_active = |stack: &[CondFrame]| stack.last().map(|f| f.is_active).unwrap_or(true);

    for chunk in chunks {
        let directive = match chunk {
            SourceChunk::Preprocessor { directive, .. } => Some(directive),
            _ => None,
        };

        match directive {
            Some(PreprocessorDirective::Ifdef(name)) => {
                let macro_name = extract_first_ident(name);
                let parent_active = is_currently_active(&cond_stack);
                let cond_true = parent_active && env.is_defined(&macro_name);
                cond_stack.push(CondFrame {
                    branch_taken: cond_true,
                    is_active: cond_true,
                    parent_active,
                });
            }
            Some(PreprocessorDirective::Ifndef(name)) => {
                let macro_name = extract_first_ident(name);
                let parent_active = is_currently_active(&cond_stack);
                let cond_true = parent_active && !env.is_defined(&macro_name);
                cond_stack.push(CondFrame {
                    branch_taken: cond_true,
                    is_active: cond_true,
                    parent_active,
                });
            }
            Some(PreprocessorDirective::If(expr)) => {
                let parent_active = is_currently_active(&cond_stack);
                let cond_true = parent_active && evaluate_expr(expr, env)? != 0;
                cond_stack.push(CondFrame {
                    branch_taken: cond_true,
                    is_active: cond_true,
                    parent_active,
                });
            }
            Some(PreprocessorDirective::Elif(expr)) => {
                let frame = cond_stack.last_mut().ok_or(PreprocessorError::UnmatchedElif)?;
                if frame.parent_active && !frame.branch_taken {
                    let cond_true = evaluate_expr(expr, env)? != 0;
                    frame.is_active = cond_true;
                    frame.branch_taken |= cond_true;
                } else {
                    frame.is_active = false;
                }
            }
            Some(PreprocessorDirective::Else) => {
                let frame = cond_stack.last_mut().ok_or(PreprocessorError::UnmatchedElse)?;
                if frame.parent_active && !frame.branch_taken {
                    frame.is_active = true;
                    frame.branch_taken = true;
                } else {
                    frame.is_active = false;
                }
            }
            Some(PreprocessorDirective::Endif) => {
                cond_stack.pop().ok_or(PreprocessorError::UnmatchedEndif)?;
            }
            Some(PreprocessorDirective::Define { name, body, .. }) => {
                if is_currently_active(&cond_stack) {
                    let value = (!body.trim().is_empty()).then(|| body.trim());
                    env.define(name, value);
                    resolved.push(chunk.clone());
                }
            }
            Some(PreprocessorDirective::Undef(name)) => {
                if is_currently_active(&cond_stack) {
                    env.undef(name);
                    resolved.push(chunk.clone());
                }
            }
            _ => {
                if is_currently_active(&cond_stack) {
                    resolved.push(chunk.clone());
                }
            }
        }
    }

    if !cond_stack.is_empty() {
        return Err(PreprocessorError::UnterminatedIf);
    }

    Ok(resolved)
}

/// Extracts the first identifier from a directive argument, ignoring trailing comments
/// (e.g. `LINUX // comment` -> `LINUX`).
fn extract_first_ident(s: &str) -> String {
    strip_trailing_comments(s)
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .find(|ident| !ident.is_empty())
        .unwrap_or("")
        .to_string()
}

fn strip_trailing_comments(s: &str) -> &str {
    let mut end = s.len();
    if let Some(pos) = s.find("//") {
        end = end.min(pos);
    }
    if let Some(pos) = s.find("/*") {
        end = end.min(pos);
    }
    s[..end].trim()
}

/// Evaluates a preprocessor integer constant expression (e.g. `1`, `defined(LINUX)`, `FOO && BAR`).
pub fn evaluate_expr(expr: &str, env: &PreprocessorEnv) -> Result<i64, PreprocessorError> {
    let tokens = tokenize_expr(strip_trailing_comments(expr), env)?;
    ExprParser { tokens: &tokens, pos: 0 }.parse_or()
}

#[derive(Debug, Clone, PartialEq)]
enum ExprTok {
    Num(i64),
    OpenParen,
    CloseParen,
    Plus,
    Minus,
    Bang,
    Tilde,
    Star,
    Slash,
    Percent,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitXor,
    BitOr,
    And,
    Or,
}

fn tokenize_expr(s: &str, env: &PreprocessorEnv) -> Result<Vec<ExprTok>, PreprocessorError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        if c.is_ascii_digit() {
            let start = i;
            if c == '0' && i + 1 < len && (chars[i + 1] == 'x' || chars[i + 1] == 'X') {
                i += 2;
                while i < len && chars[i].is_ascii_hexdigit() {
                    i += 1;
                }
                let num_str: String = chars[start + 2..i].iter().collect();
                let val = i64::from_str_radix(&num_str, 16)
                    .map_err(|e| PreprocessorError::ExpressionError(e.to_string()))?;
                tokens.push(ExprTok::Num(val));
            } else {
                while i < len && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                // Tolerate integer suffixes (L, UL, ...), which C constant-expressions allow.
                while i < len && matches!(chars[i], 'u' | 'U' | 'l' | 'L') {
                    i += 1;
                }
                let val = num_str
                    .parse::<i64>()
                    .map_err(|e| PreprocessorError::ExpressionError(e.to_string()))?;
                tokens.push(ExprTok::Num(val));
            }
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let ident: String = chars[start..i].iter().collect();
            if ident == "defined" {
                while i < len && chars[i].is_whitespace() {
                    i += 1;
                }
                let has_paren = i < len && chars[i] == '(';
                if has_paren {
                    i += 1;
                }
                while i < len && chars[i].is_whitespace() {
                    i += 1;
                }
                let id_start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let target_ident: String = chars[id_start..i].iter().collect();
                if has_paren {
                    while i < len && chars[i].is_whitespace() {
                        i += 1;
                    }
                    if i < len && chars[i] == ')' {
                        i += 1;
                    }
                }
                tokens.push(ExprTok::Num(env.is_defined(&target_ident) as i64));
            } else {
                // Undefined identifiers evaluate to 0, per the C preprocessor spec.
                let val = match env.get_value(&ident) {
                    Some(val_str) => val_str.parse::<i64>().unwrap_or(1),
                    None => env.is_defined(&ident) as i64,
                };
                tokens.push(ExprTok::Num(val));
            }
            continue;
        }

        if i + 1 < len {
            let pair = (chars[i], chars[i + 1]);
            let two_char = match pair {
                ('&', '&') => Some(ExprTok::And),
                ('|', '|') => Some(ExprTok::Or),
                ('=', '=') => Some(ExprTok::Eq),
                ('!', '=') => Some(ExprTok::Ne),
                ('<', '=') => Some(ExprTok::Le),
                ('>', '=') => Some(ExprTok::Ge),
                ('<', '<') => Some(ExprTok::Shl),
                ('>', '>') => Some(ExprTok::Shr),
                _ => None,
            };
            if let Some(tok) = two_char {
                tokens.push(tok);
                i += 2;
                continue;
            }
        }

        let tok = match c {
            '(' => ExprTok::OpenParen,
            ')' => ExprTok::CloseParen,
            '+' => ExprTok::Plus,
            '-' => ExprTok::Minus,
            '!' => ExprTok::Bang,
            '~' => ExprTok::Tilde,
            '*' => ExprTok::Star,
            '/' => ExprTok::Slash,
            '%' => ExprTok::Percent,
            '<' => ExprTok::Lt,
            '>' => ExprTok::Gt,
            '&' => ExprTok::BitAnd,
            '^' => ExprTok::BitXor,
            '|' => ExprTok::BitOr,
            _ => return Err(PreprocessorError::ExpressionError(format!("unexpected character: {c}"))),
        };
        tokens.push(tok);
        i += 1;
    }

    Ok(tokens)
}

struct ExprParser<'a> {
    tokens: &'a [ExprTok],
    pos: usize,
}

impl ExprParser<'_> {
    fn peek(&self) -> Option<&ExprTok> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&ExprTok> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn parse_or(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_and()?;
        while let Some(ExprTok::Or) = self.peek() {
            self.advance();
            let right = self.parse_and()?;
            left = (left != 0 || right != 0) as i64;
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_bitor()?;
        while let Some(ExprTok::And) = self.peek() {
            self.advance();
            let right = self.parse_bitor()?;
            left = (left != 0 && right != 0) as i64;
        }
        Ok(left)
    }

    fn parse_bitor(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_bitxor()?;
        while let Some(ExprTok::BitOr) = self.peek() {
            self.advance();
            left |= self.parse_bitxor()?;
        }
        Ok(left)
    }

    fn parse_bitxor(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_bitand()?;
        while let Some(ExprTok::BitXor) = self.peek() {
            self.advance();
            left ^= self.parse_bitand()?;
        }
        Ok(left)
    }

    fn parse_bitand(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_equality()?;
        while let Some(ExprTok::BitAnd) = self.peek() {
            self.advance();
            left &= self.parse_equality()?;
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_relational()?;
        loop {
            match self.peek() {
                Some(ExprTok::Eq) => {
                    self.advance();
                    left = (left == self.parse_relational()?) as i64;
                }
                Some(ExprTok::Ne) => {
                    self.advance();
                    left = (left != self.parse_relational()?) as i64;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_relational(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_shift()?;
        loop {
            match self.peek() {
                Some(ExprTok::Lt) => {
                    self.advance();
                    left = (left < self.parse_shift()?) as i64;
                }
                Some(ExprTok::Le) => {
                    self.advance();
                    left = (left <= self.parse_shift()?) as i64;
                }
                Some(ExprTok::Gt) => {
                    self.advance();
                    left = (left > self.parse_shift()?) as i64;
                }
                Some(ExprTok::Ge) => {
                    self.advance();
                    left = (left >= self.parse_shift()?) as i64;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_add_sub()?;
        loop {
            match self.peek() {
                Some(ExprTok::Shl) => {
                    self.advance();
                    left <<= self.parse_add_sub()?;
                }
                Some(ExprTok::Shr) => {
                    self.advance();
                    left >>= self.parse_add_sub()?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_add_sub(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_mul_div()?;
        loop {
            match self.peek() {
                Some(ExprTok::Plus) => {
                    self.advance();
                    left += self.parse_mul_div()?;
                }
                Some(ExprTok::Minus) => {
                    self.advance();
                    left -= self.parse_mul_div()?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(ExprTok::Star) => {
                    self.advance();
                    left *= self.parse_unary()?;
                }
                Some(ExprTok::Slash) => {
                    self.advance();
                    let right = self.parse_unary()?;
                    if right == 0 {
                        return Err(PreprocessorError::ExpressionError("division by zero".into()));
                    }
                    left /= right;
                }
                Some(ExprTok::Percent) => {
                    self.advance();
                    let right = self.parse_unary()?;
                    if right == 0 {
                        return Err(PreprocessorError::ExpressionError("modulo by zero".into()));
                    }
                    left %= right;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<i64, PreprocessorError> {
        match self.peek() {
            Some(ExprTok::Plus) => {
                self.advance();
                self.parse_unary()
            }
            Some(ExprTok::Minus) => {
                self.advance();
                Ok(-self.parse_unary()?)
            }
            Some(ExprTok::Bang) => {
                self.advance();
                Ok((self.parse_unary()? == 0) as i64)
            }
            Some(ExprTok::Tilde) => {
                self.advance();
                Ok(!self.parse_unary()?)
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<i64, PreprocessorError> {
        match self.advance() {
            Some(ExprTok::Num(n)) => Ok(*n),
            Some(ExprTok::OpenParen) => {
                let val = self.parse_or()?;
                match self.advance() {
                    Some(ExprTok::CloseParen) => Ok(val),
                    _ => Err(PreprocessorError::ExpressionError("unclosed parenthesis".into())),
                }
            }
            Some(tok) => Err(PreprocessorError::ExpressionError(format!("unexpected token: {tok:?}"))),
            None => Err(PreprocessorError::ExpressionError("unexpected end of expression".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse_chunks, splice};

    fn resolved_code_lines(code: &str, env: &mut PreprocessorEnv) -> Vec<String> {
        let chunks = crate::parser::partitioner::partition_source(code);
        let resolved = resolve_conditionals(&chunks, env).unwrap();
        resolved
            .iter()
            .filter_map(|c| match c {
                SourceChunk::Code(s) => Some(s.trim().to_string()),
                _ => None,
            })
            .filter(|s| !s.is_empty())
            .collect()
    }

    #[test]
    fn test_conditional_if_elif_else() {
        let code = r#"
#if 1
int active_one = 1;
#else
int inactive_one = 0;
#endif

#if 0
int inactive_two = 0;
#elif 1
int active_two = 2;
#else
int inactive_three = 0;
#endif
"#;
        let mut env = PreprocessorEnv::new();
        assert_eq!(
            resolved_code_lines(code, &mut env),
            vec!["int active_one = 1;", "int active_two = 2;"]
        );
    }

    #[test]
    fn test_ifdef_ifndef_with_env() {
        let code = r#"
#ifdef LINUX
int platform = 1;
#else
int platform = 2;
#endif
#ifndef WINDOWS
int not_windows = 1;
#endif
"#;
        let mut env = PreprocessorEnv::linux_doom_defaults();
        assert_eq!(
            resolved_code_lines(code, &mut env),
            vec!["int platform = 1;", "int not_windows = 1;"]
        );
    }

    #[test]
    fn test_nested_conditionals() {
        let code = r#"
#if 1
#if 0
int a = 1;
#else
int b = 2;
#endif
#endif
"#;
        let mut env = PreprocessorEnv::new();
        assert_eq!(resolved_code_lines(code, &mut env), vec!["int b = 2;"]);
    }

    #[test]
    fn test_define_value_used_in_later_if() {
        let code = r#"
#define FOO 1
#if FOO
int a = 1;
#endif
#undef FOO
#if FOO
int b = 2;
#endif
"#;
        let mut env = PreprocessorEnv::new();
        assert_eq!(resolved_code_lines(code, &mut env), vec!["int a = 1;"]);
    }

    #[test]
    fn test_unmatched_endif_errors() {
        let chunks = crate::parser::partitioner::partition_source("#endif\n");
        let mut env = PreprocessorEnv::new();
        assert_eq!(resolve_conditionals(&chunks, &mut env), Err(PreprocessorError::UnmatchedEndif));
    }

    #[test]
    fn test_unterminated_if_errors() {
        let chunks = crate::parser::partitioner::partition_source("#if 1\nint a;\n");
        let mut env = PreprocessorEnv::new();
        assert_eq!(resolve_conditionals(&chunks, &mut env), Err(PreprocessorError::UnterminatedIf));
    }

    #[test]
    fn test_expr_operators() {
        let env = PreprocessorEnv::new();
        assert_eq!(evaluate_expr("1 + 2 * 3", &env).unwrap(), 7);
        assert_eq!(evaluate_expr("(1 + 2) * 3", &env).unwrap(), 9);
        assert_eq!(evaluate_expr("1 << 4", &env).unwrap(), 16);
        assert_eq!(evaluate_expr("!0 && (1 || 0)", &env).unwrap(), 1);
        assert_eq!(evaluate_expr("0x10 == 16", &env).unwrap(), 1);
        assert_eq!(evaluate_expr("defined(UNDEFINED_MACRO)", &env).unwrap(), 0);
    }

    #[test]
    fn test_full_corpus_resolves_cleanly() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("linuxdoom-1.10 directory should exist") {
            let path = entry.unwrap().path();
            let is_source = matches!(path.extension().and_then(|e| e.to_str()), Some("c") | Some("h"));
            if !path.is_file() || !is_source {
                continue;
            }
            let content = std::fs::read_to_string(&path).expect("source file should be valid UTF-8");
            let (_, chunks) = parse_chunks(&content);
            let mut env = PreprocessorEnv::linux_doom_defaults();
            resolve_conditionals(&chunks, &mut env)
                .unwrap_or_else(|e| panic!("preprocessor error in {}: {e}", path.display()));
            checked += 1;
        }
        assert!(checked > 100, "expected to check the full Doom corpus, only checked {checked}");
    }

    #[test]
    fn test_splice_still_reachable() {
        // Sanity check that the re-export path used by main.rs still compiles/works.
        assert_eq!(splice("a\\\nb").text, "ab");
    }
}
