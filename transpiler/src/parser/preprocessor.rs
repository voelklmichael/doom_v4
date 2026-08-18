//! Step 1.3: Preprocessor Conditional Resolution (#if, #ifdef, #ifndef, #elif, #else, #endif)
//!
//! Evaluates conditional compilation directives and filters inactive chunks.

use std::collections::HashMap;
use crate::parser::partitioner::{PreprocessorDirective, SourceChunk};

/// Preprocessor compilation environment defining macros and flags.
#[derive(Debug, Clone)]
pub struct PreprocessorEnv {
    pub macros: HashMap<String, Option<String>>,
}

impl Default for PreprocessorEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl PreprocessorEnv {
    /// Create an empty preprocessor environment.
    pub fn new() -> Self {
        Self {
            macros: HashMap::new(),
        }
    }

    /// Create a standard environment for Linux Doom compilation.
    pub fn linux_doom_defaults() -> Self {
        let mut env = Self::new();
        // Common Linux Doom flags
        env.define("LINUX", None);
        env.define("NORMALUNIX", None);
        env.define("__BYTEBOOL__", None);
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

/// Errors that can occur during preprocessor evaluation.
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
            PreprocessorError::UnmatchedEndif => write!(f, "Unmatched #endif without corresponding #if/#ifdef"),
            PreprocessorError::UnmatchedElse => write!(f, "Unmatched #else without corresponding #if/#ifdef"),
            PreprocessorError::UnmatchedElif => write!(f, "Unmatched #elif without corresponding #if/#ifdef"),
            PreprocessorError::UnterminatedIf => write!(f, "Unterminated #if/#ifdef block at end of file"),
            PreprocessorError::ExpressionError(msg) => write!(f, "Invalid preprocessor expression: {}", msg),
        }
    }
}

impl std::error::Error for PreprocessorError {}

#[derive(Debug, Clone)]
struct CondFrame {
    /// Was any branch in this if/elif/else chain already taken and executed?
    branch_taken: bool,
    /// Is the current branch active?
    is_active: bool,
    /// Was the enclosing parent scope active?
    parent_active: bool,
}

/// Resolves `#if` / `#ifdef` / `#ifndef` / `#elif` / `#else` / `#endif` blocks
/// by filtering out inactive chunks and tracking defined macros in active branches.
pub fn resolve_conditionals(
    chunks: &[SourceChunk],
    env: &mut PreprocessorEnv,
) -> Result<Vec<SourceChunk>, PreprocessorError> {
    let mut resolved = Vec::new();
    let mut cond_stack: Vec<CondFrame> = Vec::new();

    let is_currently_active = |stack: &[CondFrame]| -> bool {
        stack.last().map(|f| f.is_active).unwrap_or(true)
    };

    for chunk in chunks {
        match chunk {
            SourceChunk::Preprocessor(directive) => match directive {
                PreprocessorDirective::Ifdef(name) => {
                    let macro_name = extract_first_ident(name);
                    let parent_active = is_currently_active(&cond_stack);
                    let cond_true = parent_active && env.is_defined(&macro_name);
                    cond_stack.push(CondFrame {
                        branch_taken: cond_true,
                        is_active: cond_true,
                        parent_active,
                    });
                }
                PreprocessorDirective::Ifndef(name) => {
                    let macro_name = extract_first_ident(name);
                    let parent_active = is_currently_active(&cond_stack);
                    let cond_true = parent_active && !env.is_defined(&macro_name);
                    cond_stack.push(CondFrame {
                        branch_taken: cond_true,
                        is_active: cond_true,
                        parent_active,
                    });
                }
                PreprocessorDirective::If(expr) => {
                    let parent_active = is_currently_active(&cond_stack);
                    let cond_true = if parent_active {
                        evaluate_expr(expr, env)? != 0
                    } else {
                        false
                    };
                    cond_stack.push(CondFrame {
                        branch_taken: cond_true,
                        is_active: cond_true,
                        parent_active,
                    });
                }
                PreprocessorDirective::Elif(expr) => {
                    let frame = cond_stack.last_mut().ok_or(PreprocessorError::UnmatchedElif)?;
                    if frame.parent_active && !frame.branch_taken {
                        let cond_true = evaluate_expr(expr, env)? != 0;
                        frame.is_active = cond_true;
                        if cond_true {
                            frame.branch_taken = true;
                        }
                    } else {
                        frame.is_active = false;
                    }
                }
                PreprocessorDirective::Else => {
                    let frame = cond_stack.last_mut().ok_or(PreprocessorError::UnmatchedElse)?;
                    if frame.parent_active && !frame.branch_taken {
                        frame.is_active = true;
                        frame.branch_taken = true;
                    } else {
                        frame.is_active = false;
                    }
                }
                PreprocessorDirective::Endif => {
                    if cond_stack.pop().is_none() {
                        return Err(PreprocessorError::UnmatchedEndif);
                    }
                }
                PreprocessorDirective::Define { name, .. } => {
                    if is_currently_active(&cond_stack) {
                        env.define(name, None);
                        resolved.push(chunk.clone());
                    }
                }
                PreprocessorDirective::Undef(name) => {
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
            },
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

/// Strip any comments and extract the first identifier from a string (e.g. `LINUX // comment` -> `LINUX`)
fn extract_first_ident(s: &str) -> String {
    let clean = strip_trailing_comments(s);
    clean
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

/// Evaluates a preprocessor integer expression (e.g. `1`, `0`, `defined(LINUX)`, `SNDINTR`)
pub fn evaluate_expr(expr: &str, env: &PreprocessorEnv) -> Result<i64, PreprocessorError> {
    let clean_expr = strip_trailing_comments(expr);
    let tokens = tokenize_expr(clean_expr, env)?;
    let mut parser = ExprParser { tokens: &tokens, pos: 0 };
    parser.parse_or()
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

        // Numbers: decimal, hex (0x...) or octal
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
                let val = num_str.parse::<i64>()
                    .map_err(|e| PreprocessorError::ExpressionError(e.to_string()))?;
                tokens.push(ExprTok::Num(val));
            }
            continue;
        }

        // Identifiers or `defined(...)`
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let ident: String = chars[start..i].iter().collect();
            if ident == "defined" {
                // handle defined(NAME) or defined NAME
                while i < len && chars[i].is_whitespace() {
                    i += 1;
                }
                let has_paren = if i < len && chars[i] == '(' {
                    i += 1;
                    true
                } else {
                    false
                };

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
                let is_def = if env.is_defined(&target_ident) { 1 } else { 0 };
                tokens.push(ExprTok::Num(is_def));
            } else {
                // In C preprocessor, undefined identifiers evaluate to 0
                let val = if env.is_defined(&ident) {
                    if let Some(val_str) = env.get_value(&ident) {
                        val_str.parse::<i64>().unwrap_or(1)
                    } else {
                        1
                    }
                } else {
                    0
                };
                tokens.push(ExprTok::Num(val));
            }
            continue;
        }

        // Two-character operators
        if i + 1 < len {
            let pair = (chars[i], chars[i + 1]);
            match pair {
                ('&', '&') => { tokens.push(ExprTok::And); i += 2; continue; }
                ('|', '|') => { tokens.push(ExprTok::Or); i += 2; continue; }
                ('=', '=') => { tokens.push(ExprTok::Eq); i += 2; continue; }
                ('!', '=') => { tokens.push(ExprTok::Ne); i += 2; continue; }
                ('<', '=') => { tokens.push(ExprTok::Le); i += 2; continue; }
                ('>', '=') => { tokens.push(ExprTok::Ge); i += 2; continue; }
                ('<', '<') => { tokens.push(ExprTok::Shl); i += 2; continue; }
                ('>', '>') => { tokens.push(ExprTok::Shr); i += 2; continue; }
                _ => {}
            }
        }

        // Single-character operators
        match c {
            '(' => tokens.push(ExprTok::OpenParen),
            ')' => tokens.push(ExprTok::CloseParen),
            '+' => tokens.push(ExprTok::Plus),
            '-' => tokens.push(ExprTok::Minus),
            '!' => tokens.push(ExprTok::Bang),
            '~' => tokens.push(ExprTok::Tilde),
            '*' => tokens.push(ExprTok::Star),
            '/' => tokens.push(ExprTok::Slash),
            '%' => tokens.push(ExprTok::Percent),
            '<' => tokens.push(ExprTok::Lt),
            '>' => tokens.push(ExprTok::Gt),
            '&' => tokens.push(ExprTok::BitAnd),
            '^' => tokens.push(ExprTok::BitXor),
            '|' => tokens.push(ExprTok::BitOr),
            _ => {
                return Err(PreprocessorError::ExpressionError(format!("Unexpected character: {}", c)));
            }
        }
        i += 1;
    }

    Ok(tokens)
}

struct ExprParser<'a> {
    tokens: &'a [ExprTok],
    pos: usize,
}

impl<'a> ExprParser<'a> {
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
            left = if left != 0 || right != 0 { 1 } else { 0 };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_bitor()?;
        while let Some(ExprTok::And) = self.peek() {
            self.advance();
            let right = self.parse_bitor()?;
            left = if left != 0 && right != 0 { 1 } else { 0 };
        }
        Ok(left)
    }

    fn parse_bitor(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_bitxor()?;
        while let Some(ExprTok::BitOr) = self.peek() {
            self.advance();
            let right = self.parse_bitxor()?;
            left |= right;
        }
        Ok(left)
    }

    fn parse_bitxor(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_bitand()?;
        while let Some(ExprTok::BitXor) = self.peek() {
            self.advance();
            let right = self.parse_bitand()?;
            left ^= right;
        }
        Ok(left)
    }

    fn parse_bitand(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_equality()?;
        while let Some(ExprTok::BitAnd) = self.peek() {
            self.advance();
            let right = self.parse_equality()?;
            left &= right;
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_relational()?;
        while let Some(tok) = self.peek() {
            match tok {
                ExprTok::Eq => { self.advance(); let right = self.parse_relational()?; left = if left == right { 1 } else { 0 }; }
                ExprTok::Ne => { self.advance(); let right = self.parse_relational()?; left = if left != right { 1 } else { 0 }; }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_relational(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_shift()?;
        while let Some(tok) = self.peek() {
            match tok {
                ExprTok::Lt => { self.advance(); let right = self.parse_shift()?; left = if left < right { 1 } else { 0 }; }
                ExprTok::Le => { self.advance(); let right = self.parse_shift()?; left = if left <= right { 1 } else { 0 }; }
                ExprTok::Gt => { self.advance(); let right = self.parse_shift()?; left = if left > right { 1 } else { 0 }; }
                ExprTok::Ge => { self.advance(); let right = self.parse_shift()?; left = if left >= right { 1 } else { 0 }; }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_add_sub()?;
        while let Some(tok) = self.peek() {
            match tok {
                ExprTok::Shl => { self.advance(); let right = self.parse_add_sub()?; left <<= right; }
                ExprTok::Shr => { self.advance(); let right = self.parse_add_sub()?; left >>= right; }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_add_sub(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_mul_div()?;
        while let Some(tok) = self.peek() {
            match tok {
                ExprTok::Plus => { self.advance(); let right = self.parse_mul_div()?; left += right; }
                ExprTok::Minus => { self.advance(); let right = self.parse_mul_div()?; left -= right; }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<i64, PreprocessorError> {
        let mut left = self.parse_unary()?;
        while let Some(tok) = self.peek() {
            match tok {
                ExprTok::Star => { self.advance(); let right = self.parse_unary()?; left *= right; }
                ExprTok::Slash => {
                    self.advance();
                    let right = self.parse_unary()?;
                    if right == 0 {
                        return Err(PreprocessorError::ExpressionError("Division by zero in preprocessor expr".into()));
                    }
                    left /= right;
                }
                ExprTok::Percent => {
                    self.advance();
                    let right = self.parse_unary()?;
                    if right == 0 {
                        return Err(PreprocessorError::ExpressionError("Modulo by zero in preprocessor expr".into()));
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
            Some(ExprTok::Plus) => { self.advance(); self.parse_unary() }
            Some(ExprTok::Minus) => { self.advance(); Ok(-self.parse_unary()?) }
            Some(ExprTok::Bang) => { self.advance(); Ok(if self.parse_unary()? == 0 { 1 } else { 0 }) }
            Some(ExprTok::Tilde) => { self.advance(); Ok(!self.parse_unary()?) }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<i64, PreprocessorError> {
        match self.advance() {
            Some(ExprTok::Num(n)) => Ok(*n),
            Some(ExprTok::OpenParen) => {
                let val = self.parse_or()?;
                if let Some(ExprTok::CloseParen) = self.advance() {
                    Ok(val)
                } else {
                    Err(PreprocessorError::ExpressionError("Unclosed parenthesis in expression".into()))
                }
            }
            Some(tok) => Err(PreprocessorError::ExpressionError(format!("Unexpected token: {:?}", tok))),
            None => Err(PreprocessorError::ExpressionError("Unexpected end of expression".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::partitioner::partition_source;

    #[test]
    fn test_conditional_if_else() {
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
        let chunks = partition_source(code);
        let mut env = PreprocessorEnv::new();
        let resolved = resolve_conditionals(&chunks, &mut env).unwrap();

        let code_texts: Vec<String> = resolved
            .iter()
            .filter_map(|c| match c {
                SourceChunk::Code(s) => Some(s.trim().to_string()),
                _ => None,
            })
            .filter(|s| !s.is_empty())
            .collect();

        assert_eq!(code_texts, vec!["int active_one = 1;", "int active_two = 2;"]);
    }

    #[test]
    fn test_ifdef_with_env() {
        let code = r#"
#ifdef LINUX
int platform = 1;
#else
int platform = 2;
#endif
"#;
        let chunks = partition_source(code);
        let mut env = PreprocessorEnv::linux_doom_defaults();
        let resolved = resolve_conditionals(&chunks, &mut env).unwrap();

        let code_texts: Vec<String> = resolved
            .iter()
            .filter_map(|c| match c {
                SourceChunk::Code(s) => Some(s.trim().to_string()),
                _ => None,
            })
            .filter(|s| !s.is_empty())
            .collect();

        assert_eq!(code_texts, vec!["int platform = 1;"]);
    }
}
