//! Step 6: C89 AST Grammar Parser
//!
//! A recursive-descent parser over the Step 5 `Commented<Anchor>` stream,
//! producing the AST defined in `ast.rs`. Surviving preprocessor directives
//! (`Anchor::Directive`) carry no C grammar and are dropped up front; only
//! `Anchor::Token` matters here.
//!
//! Handles the classic typedef-vs-identifier ambiguity with a running
//! symbol table of typedef names, updated as `typedef` declarations are
//! parsed (single global namespace -- doesn't model block scoping of
//! typedefs, which `linuxdoom-1.10` doesn't rely on).
//!
//! Runs in one of two modes (see `TypedefCheck`):
//! - **Final** (`parse_translation_unit_seeded`): typedef names are looked
//!   up in a pre-seeded set (Step 6b's transitively-imported names union'd
//!   with this file's own), extended as further `typedef`s are parsed.
//!   Function bodies are parsed in full.
//! - **Rough** (`extract_top_level_typedefs`, Step 6a): no typedef set
//!   exists yet -- a bare leading identifier in a declaration-specifier
//!   position is judged solely by whether a declarator-shaped continuation
//!   follows it, which is always correct at file/struct-field/parameter
//!   scope (there's no other valid C89 production it could be). Function
//!   bodies are skipped (brace-balanced, not parsed) since that's the one
//!   place this judgment call would actually be ambiguous, and bodies don't
//!   affect what a header exports anyway.

use crate::parser::ast::*;
use crate::parser::comment_attach::{Anchor, CommentedStream};
use crate::parser::lexer::{Keyword, Punct, Token, TokenKind};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub near: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (near '{}')", self.message, self.near)
    }
}

impl std::error::Error for ParseError {}

/// Parses a full Step 5 stream into a `TranslationUnit`, with no typedef
/// names known ahead of time (only useful for snippets that don't reference
/// externally-defined types). Prefer `parse_translation_unit_seeded` with
/// Step 6b's resolved import set for real source files.
pub fn parse_translation_unit(stream: &CommentedStream) -> Result<TranslationUnit, ParseError> {
    parse_translation_unit_seeded(stream, HashSet::new())
}

/// Parses a full Step 5 stream into a `TranslationUnit`, seeded with a set
/// of known typedef names (this file's own top-level typedefs, unioned with
/// everything transitively imported via local `#include`s -- see Step 6b).
pub fn parse_translation_unit_seeded(
    stream: &CommentedStream,
    seed_typedefs: HashSet<String>,
) -> Result<TranslationUnit, ParseError> {
    Parser::new(stream, TypedefCheck::Lookup(seed_typedefs), false).run()
}

/// Step 6a: extracts every name this file declares via a top-level
/// `typedef` (function bodies are skipped, not scanned -- see module docs).
pub fn extract_top_level_typedefs(stream: &CommentedStream) -> Vec<String> {
    let mut parser = Parser::new(stream, TypedefCheck::Heuristic, true);
    let _ = parser.run();
    parser.discovered_typedefs
}

/// Step 7: parses a raw token stream (already lexed, no comments/directives)
/// as a single expression -- reusing this module's own expression grammar,
/// seeded with `typedefs` for cast-vs-call disambiguation (e.g. `(fixed_t)(x)`).
/// Fails if the tokens are empty or don't reduce to exactly one expression
/// with nothing left over; see `docs/01_PARSER.md` Step 7.
pub fn parse_expr_from_tokens(
    tokens: Vec<Token>,
    typedefs: HashSet<String>,
) -> Result<Expr, ParseError> {
    if tokens.is_empty() {
        return Err(ParseError {
            message: "empty token stream".to_string(),
            near: String::new(),
        });
    }
    let mut parser = Parser {
        tokens,
        pos: 0,
        typedef_check: TypedefCheck::Lookup(typedefs),
        discovered_typedefs: Vec::new(),
        skip_bodies: false,
    };
    let expr = parser.parse_expr()?;
    if let Some(tok) = parser.peek() {
        return Err(ParseError {
            message: "unexpected trailing tokens after expression".to_string(),
            near: tok.text.clone(),
        });
    }
    Ok(expr)
}

/// Step 7: parses a raw token stream as a sequence of block items
/// (declarations and/or statements, freely mixed, in any order -- exactly
/// what's allowed inside a `{ ... }` body) rather than a single expression.
/// For a macro body that reads like `parse_expr_from_tokens` failed on it
/// because it's several statements, not one expression -- e.g.
/// `(oc) = 0; if ((my) < 0) (oc) |= TOP; ...` -- this parses it the same
/// way it would parse if pasted directly into a function body: repeated
/// block items until the tokens run out. An empty token stream is *not*
/// treated as zero items here (that's `MacroBody::Empty`'s case, checked
/// before either of these entry points is reached).
pub fn parse_block_items_from_tokens(
    tokens: Vec<Token>,
    typedefs: HashSet<String>,
) -> Result<Vec<BlockItem>, ParseError> {
    if tokens.is_empty() {
        return Err(ParseError {
            message: "empty token stream".to_string(),
            near: String::new(),
        });
    }
    let mut parser = Parser {
        tokens,
        pos: 0,
        typedef_check: TypedefCheck::Lookup(typedefs),
        discovered_typedefs: Vec::new(),
        skip_bodies: false,
    };
    let mut items = Vec::new();
    while parser.peek().is_some() {
        if parser.looks_like_decl_start() {
            items.push(BlockItem::Decl(parser.parse_declaration()?));
        } else {
            items.push(BlockItem::Stmt(parser.parse_stmt()?));
        }
    }
    Ok(items)
}

/// Typechecker Step 0: like Step 6a, but keeps the full top-level
/// `TranslationUnit` (function definitions with their bodies brace-skipped,
/// declarations, struct/union/enum specifiers) instead of discarding
/// everything but typedef names -- Step 0 needs every kind of top-level
/// export, not just typedefs. Same rough/heuristic mode as Step 6a: no
/// typedef table exists yet, and none is needed, since a bare leading
/// identifier at file scope is never ambiguous (see module docs).
pub fn extract_top_level_decls(stream: &CommentedStream) -> Vec<ExternalDecl> {
    let mut parser = Parser::new(stream, TypedefCheck::Heuristic, true);
    parser.run().map(|tu| tu.items).unwrap_or_default()
}

/// How the parser decides whether a bare identifier at a declaration-
/// specifier position is a type name.
enum TypedefCheck {
    /// Step 6c (final): authoritative set, extended as `typedef`s are parsed.
    Lookup(HashSet<String>),
    /// Step 6a (rough): no set exists yet -- judged by lookahead instead.
    Heuristic,
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    typedef_check: TypedefCheck,
    /// Every name registered via a `typedef` declaration, in parse order,
    /// regardless of mode. Step 6a's public result.
    discovered_typedefs: Vec<String>,
    /// Rough mode (Step 6a): don't parse into function bodies, just
    /// brace-balance past them.
    skip_bodies: bool,
}

impl Parser {
    fn new(stream: &CommentedStream, typedef_check: TypedefCheck, skip_bodies: bool) -> Self {
        let tokens: Vec<Token> = stream
            .items
            .iter()
            .filter_map(|c| match &c.t {
                Anchor::Token(t) => Some(t.clone()),
                Anchor::Directive(_) => None,
            })
            .collect();
        Parser {
            tokens,
            pos: 0,
            typedef_check,
            discovered_typedefs: Vec::new(),
            skip_bodies,
        }
    }
}

/// A flexible declarator tree used while parsing: its leaf may or may not
/// carry an identifier, since the same pointer/array/function/paren
/// structure is shared between named declarators, abstract declarators, and
/// parameter declarators (which can be either). Resolved to the right final
/// AST type once parsing finishes.
struct FlexDeclarator {
    pointer_quals: Vec<Vec<TypeQualifier>>,
    direct: FlexDirect,
}

enum FlexDirect {
    Leaf(Option<String>),
    Paren(Box<FlexDeclarator>),
    Array(Box<FlexDirect>, Option<Expr>),
    Function(Box<FlexDirect>, ParamList),
}

fn flex_has_ident(d: &FlexDirect) -> bool {
    match d {
        FlexDirect::Leaf(name) => name.is_some(),
        FlexDirect::Paren(inner) => flex_has_ident(&inner.direct),
        FlexDirect::Array(base, _) => flex_has_ident(base),
        FlexDirect::Function(base, _) => flex_has_ident(base),
    }
}

fn flex_to_named_direct(d: FlexDirect) -> DirectDeclarator {
    match d {
        FlexDirect::Leaf(Some(name)) => DirectDeclarator::Ident(name),
        FlexDirect::Leaf(None) => unreachable!("caller checked flex_has_ident"),
        FlexDirect::Paren(inner) => DirectDeclarator::Paren(Box::new(Declarator {
            pointer_quals: inner.pointer_quals,
            direct: flex_to_named_direct(inner.direct),
        })),
        FlexDirect::Array(base, size) => {
            DirectDeclarator::Array(Box::new(flex_to_named_direct(*base)), size)
        }
        FlexDirect::Function(base, params) => {
            DirectDeclarator::Function(Box::new(flex_to_named_direct(*base)), params)
        }
    }
}

fn flex_to_abstract_direct(d: FlexDirect) -> Option<DirectAbstractDeclarator> {
    match d {
        FlexDirect::Leaf(None) => None,
        FlexDirect::Leaf(Some(_)) => unreachable!("caller checked !flex_has_ident"),
        FlexDirect::Paren(inner) => Some(DirectAbstractDeclarator::Paren(Box::new(
            AbstractDeclarator {
                pointer_quals: inner.pointer_quals,
                direct: flex_to_abstract_direct(inner.direct),
            },
        ))),
        FlexDirect::Array(base, size) => Some(DirectAbstractDeclarator::Array(
            flex_to_abstract_direct(*base).map(Box::new),
            size,
        )),
        FlexDirect::Function(base, params) => Some(DirectAbstractDeclarator::Function(
            flex_to_abstract_direct(*base).map(Box::new),
            params,
        )),
    }
}

pub(crate) fn direct_declarator_name(d: &DirectDeclarator) -> Option<String> {
    match d {
        DirectDeclarator::Ident(s) => Some(s.clone()),
        DirectDeclarator::Paren(inner) => direct_declarator_name(&inner.direct),
        DirectDeclarator::Array(inner, _) => direct_declarator_name(inner),
        DirectDeclarator::Function(inner, _) => direct_declarator_name(inner),
    }
}

/// The name a (non-abstract) declarator introduces, e.g. `x` in `int *x[3]`.
pub(crate) fn declarator_name(d: &Declarator) -> Option<String> {
    direct_declarator_name(&d.direct)
}

fn binop_info(kind: TokenKind) -> Option<(BinaryOp, u8)> {
    match kind {
        TokenKind::Punct(Punct::PipePipe) => Some((BinaryOp::LogOr, 1)),
        TokenKind::Punct(Punct::AmpAmp) => Some((BinaryOp::LogAnd, 2)),
        TokenKind::Punct(Punct::Pipe) => Some((BinaryOp::BitOr, 3)),
        TokenKind::Punct(Punct::Caret) => Some((BinaryOp::BitXor, 4)),
        TokenKind::Punct(Punct::Amp) => Some((BinaryOp::BitAnd, 5)),
        TokenKind::Punct(Punct::EqEq) => Some((BinaryOp::Eq, 6)),
        TokenKind::Punct(Punct::NotEq) => Some((BinaryOp::Ne, 6)),
        TokenKind::Punct(Punct::Lt) => Some((BinaryOp::Lt, 7)),
        TokenKind::Punct(Punct::Le) => Some((BinaryOp::Le, 7)),
        TokenKind::Punct(Punct::Gt) => Some((BinaryOp::Gt, 7)),
        TokenKind::Punct(Punct::Ge) => Some((BinaryOp::Ge, 7)),
        TokenKind::Punct(Punct::ShiftLeft) => Some((BinaryOp::Shl, 8)),
        TokenKind::Punct(Punct::ShiftRight) => Some((BinaryOp::Shr, 8)),
        TokenKind::Punct(Punct::Plus) => Some((BinaryOp::Add, 9)),
        TokenKind::Punct(Punct::Minus) => Some((BinaryOp::Sub, 9)),
        TokenKind::Punct(Punct::Star) => Some((BinaryOp::Mul, 10)),
        TokenKind::Punct(Punct::Slash) => Some((BinaryOp::Div, 10)),
        TokenKind::Punct(Punct::Percent) => Some((BinaryOp::Mod, 10)),
        _ => None,
    }
}

fn assign_op(kind: TokenKind) -> Option<AssignOp> {
    match kind {
        TokenKind::Punct(Punct::Eq) => Some(AssignOp::Assign),
        TokenKind::Punct(Punct::StarEq) => Some(AssignOp::MulAssign),
        TokenKind::Punct(Punct::SlashEq) => Some(AssignOp::DivAssign),
        TokenKind::Punct(Punct::PercentEq) => Some(AssignOp::ModAssign),
        TokenKind::Punct(Punct::PlusEq) => Some(AssignOp::AddAssign),
        TokenKind::Punct(Punct::MinusEq) => Some(AssignOp::SubAssign),
        TokenKind::Punct(Punct::ShiftLeftEq) => Some(AssignOp::ShlAssign),
        TokenKind::Punct(Punct::ShiftRightEq) => Some(AssignOp::ShrAssign),
        TokenKind::Punct(Punct::AmpEq) => Some(AssignOp::AndAssign),
        TokenKind::Punct(Punct::CaretEq) => Some(AssignOp::XorAssign),
        TokenKind::Punct(Punct::PipeEq) => Some(AssignOp::OrAssign),
        _ => None,
    }
}

fn is_decl_start_keyword(k: Keyword) -> bool {
    matches!(
        k,
        Keyword::Typedef
            | Keyword::Extern
            | Keyword::Static
            | Keyword::Auto
            | Keyword::Register
            | Keyword::Const
            | Keyword::Volatile
            | Keyword::Void
            | Keyword::Char
            | Keyword::Short
            | Keyword::Int
            | Keyword::Long
            | Keyword::Float
            | Keyword::Double
            | Keyword::Signed
            | Keyword::Unsigned
            | Keyword::Struct
            | Keyword::Union
            | Keyword::Enum
    )
}

fn is_type_start_keyword(k: Keyword) -> bool {
    matches!(
        k,
        Keyword::Const
            | Keyword::Volatile
            | Keyword::Void
            | Keyword::Char
            | Keyword::Short
            | Keyword::Int
            | Keyword::Long
            | Keyword::Float
            | Keyword::Double
            | Keyword::Signed
            | Keyword::Unsigned
            | Keyword::Struct
            | Keyword::Union
            | Keyword::Enum
    )
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn peek_at(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.pos + n)
    }

    fn advance(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn err(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            message: msg.into(),
            near: self
                .peek()
                .map(|t| t.text.clone())
                .unwrap_or_else(|| "<eof>".to_string()),
        }
    }

    fn is_punct(&self, p: Punct) -> bool {
        matches!(self.peek(), Some(t) if t.kind == TokenKind::Punct(p))
    }

    fn eat_punct(&mut self, p: Punct) -> bool {
        if self.is_punct(p) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consumes a `{` and everything up to and including its matching `}`.
    fn skip_balanced_braces(&mut self) {
        debug_assert!(self.is_punct(Punct::LBrace));
        let mut depth = 0i32;
        loop {
            match self.advance() {
                Some(t) if t.kind == TokenKind::Punct(Punct::LBrace) => depth += 1,
                Some(t) if t.kind == TokenKind::Punct(Punct::RBrace) => {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                Some(_) => {}
                None => return,
            }
        }
    }

    /// Consumes a `(` and everything up to and including its matching `)`.
    fn skip_balanced_parens(&mut self) {
        debug_assert!(self.is_punct(Punct::LParen));
        let mut depth = 0i32;
        loop {
            match self.advance() {
                Some(t) if t.kind == TokenKind::Punct(Punct::LParen) => depth += 1,
                Some(t) if t.kind == TokenKind::Punct(Punct::RParen) => {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                Some(_) => {}
                None => return,
            }
        }
    }

    /// Rough-scan only (Step 6a/Step 0): skips a run of trailing GNU
    /// declaration decorations real system headers rely on pervasively
    /// (`__THROW`, `__THROWNL`, `__nonnull ((1))`, `__wur`,
    /// `__attribute__((...))`, ...) between a declarator and its
    /// terminating `;`/`{`/`,`. Our C89 grammar has no model for these, and
    /// the rough scan only needs a declaration's name and shape, not these
    /// attributes' actual meaning -- so any bare identifier (optionally
    /// followed by a parenthesized argument list) found in a position
    /// where only `;`/`{`/`,` would otherwise be valid is, by definition,
    /// one of these, not a real declaration. Gated behind `skip_bodies` so
    /// Step 6c's strict, already-100%-passing corpus parse is untouched --
    /// `linuxdoom-1.10` never uses any of this.
    fn skip_gnu_decorations(&mut self) {
        if !self.skip_bodies {
            return;
        }
        while matches!(self.peek().map(|t| t.kind), Some(TokenKind::Identifier)) {
            self.advance();
            if self.is_punct(Punct::LParen) {
                self.skip_balanced_parens();
            }
        }
    }

    /// Rough-scan error recovery (Step 6a/Step 0 only, `skip_bodies` mode):
    /// a real system header uses GNU/C99 syntax our grammar doesn't cover
    /// (`__restrict`, `__attribute__((...))`, `__THROW`, ...), so one
    /// top-level construct failing to parse is expected, not exceptional.
    /// Rather than lose every declaration already collected from the same
    /// file (the caller would otherwise have to discard the whole
    /// `TranslationUnit` on any single `Err`), skip forward from wherever
    /// parsing broke down to the next top-level boundary: the next `;` seen
    /// at bracket/paren/brace depth 0 (consumed, e.g. a broken prototype),
    /// or the next `}` that closes back to depth 0 (consumed, e.g. a broken
    /// `static inline` function's body -- no trailing `;` to look for
    /// there). Leaves the parser positioned to resume with the next
    /// external declaration, same as `skip_balanced_braces` does for a
    /// body it never tries to parse in the first place.
    fn recover_to_next_top_level_boundary(&mut self) {
        let mut depth = 0i32;
        loop {
            match self.peek().map(|t| t.kind) {
                Some(TokenKind::Punct(Punct::LParen | Punct::LBrace | Punct::LBracket)) => {
                    depth += 1;
                    self.advance();
                }
                Some(TokenKind::Punct(Punct::RBrace)) if depth > 0 => {
                    depth -= 1;
                    self.advance();
                    if depth == 0 {
                        return;
                    }
                }
                Some(TokenKind::Punct(Punct::RParen | Punct::RBracket)) if depth > 0 => {
                    depth -= 1;
                    self.advance();
                }
                Some(TokenKind::Punct(Punct::RParen | Punct::RBrace | Punct::RBracket)) => {
                    // An unmatched closer at depth 0 belongs to whatever
                    // scope called us -- stop without consuming it.
                    return;
                }
                Some(TokenKind::Punct(Punct::Semicolon)) if depth == 0 => {
                    self.advance();
                    return;
                }
                Some(_) => {
                    self.advance();
                }
                None => return,
            }
        }
    }

    fn expect_punct(&mut self, p: Punct) -> Result<(), ParseError> {
        if self.eat_punct(p) {
            Ok(())
        } else {
            Err(self.err(format!("expected '{p:?}'")))
        }
    }

    fn expect_keyword(&mut self, k: Keyword) -> Result<(), ParseError> {
        if matches!(self.peek(), Some(t) if t.kind == TokenKind::Keyword(k)) {
            self.advance();
            Ok(())
        } else {
            Err(self.err(format!("expected keyword '{k:?}'")))
        }
    }

    fn eat_ident(&mut self) -> Option<String> {
        if matches!(self.peek(), Some(t) if t.kind == TokenKind::Identifier) {
            self.advance().map(|t| t.text)
        } else {
            None
        }
    }

    /// Whether the identifier at `offset` is a type name. In `Lookup` mode
    /// this is authoritative set membership; in `Heuristic` mode (Step 6a,
    /// no set exists yet) it's judged by whether a declarator-shaped
    /// continuation follows -- always correct at file/field/parameter scope
    /// (see module docs), which is the only scope Step 6a ever looks at.
    fn is_typedef_name_at(&self, offset: usize) -> bool {
        let Some(tok) = self.peek_at(offset) else {
            return false;
        };
        if tok.kind != TokenKind::Identifier {
            return false;
        }
        match &self.typedef_check {
            TypedefCheck::Lookup(set) => set.contains(&tok.text),
            TypedefCheck::Heuristic => matches!(
                self.peek_at(offset + 1).map(|t| &t.kind),
                Some(TokenKind::Identifier | TokenKind::Punct(Punct::Star | Punct::LParen))
            ),
        }
    }

    fn looks_like_decl_start_at(&self, offset: usize) -> bool {
        match self.peek_at(offset) {
            Some(tok) => match &tok.kind {
                TokenKind::Keyword(k) => is_decl_start_keyword(*k),
                TokenKind::Identifier => self.is_typedef_name_at(offset),
                _ => false,
            },
            None => false,
        }
    }

    fn looks_like_decl_start(&self) -> bool {
        self.looks_like_decl_start_at(0)
    }

    fn peek_is_type_start_at(&self, offset: usize) -> bool {
        match self.peek_at(offset) {
            Some(tok) => match &tok.kind {
                TokenKind::Keyword(k) => is_type_start_keyword(*k),
                TokenKind::Identifier => self.is_typedef_name_at(offset),
                _ => false,
            },
            None => false,
        }
    }

    fn run(&mut self) -> Result<TranslationUnit, ParseError> {
        let mut items = Vec::new();
        while self.peek().is_some() {
            if self.eat_punct(Punct::Semicolon) {
                continue;
            }
            match self.parse_external_decl() {
                Ok(item) => items.push(item),
                Err(e) => {
                    // Rough-scan mode (Step 6a/Step 0): a real system
                    // header's GNU/C99 syntax our grammar doesn't cover is
                    // expected, not exceptional -- recover instead of
                    // discarding every declaration already collected.
                    // Step 6c's real, final parse (`skip_bodies == false`)
                    // stays strict; it should never hit this.
                    if !self.skip_bodies {
                        return Err(e);
                    }
                    let before = self.pos;
                    self.recover_to_next_top_level_boundary();
                    if self.pos == before {
                        // Guarantee forward progress even if recovery
                        // couldn't consume anything on its own.
                        self.advance();
                    }
                }
            }
        }
        Ok(TranslationUnit { items })
    }

    // ---- declarations ----

    fn parse_decl_specifiers(&mut self) -> Result<DeclSpecifiers, ParseError> {
        let mut spec = DeclSpecifiers::default();
        while let Some(kind) = self.peek().map(|t| t.kind) {
            match kind {
                TokenKind::Keyword(Keyword::Typedef) => {
                    spec.storage = Some(StorageClass::Typedef);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Extern) => {
                    spec.storage = Some(StorageClass::Extern);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Static) => {
                    spec.storage = Some(StorageClass::Static);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Auto) => {
                    spec.storage = Some(StorageClass::Auto);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Register) => {
                    spec.storage = Some(StorageClass::Register);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Const) => {
                    spec.qualifiers.push(TypeQualifier::Const);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Volatile) => {
                    spec.qualifiers.push(TypeQualifier::Volatile);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Void) => {
                    spec.type_specifiers.push(TypeSpecifier::Void);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Char) => {
                    spec.type_specifiers.push(TypeSpecifier::Char);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Short) => {
                    spec.type_specifiers.push(TypeSpecifier::Short);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Int) => {
                    spec.type_specifiers.push(TypeSpecifier::Int);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Long) => {
                    spec.type_specifiers.push(TypeSpecifier::Long);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Float) => {
                    spec.type_specifiers.push(TypeSpecifier::Float);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Double) => {
                    spec.type_specifiers.push(TypeSpecifier::Double);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Signed) => {
                    spec.type_specifiers.push(TypeSpecifier::Signed);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Unsigned) => {
                    spec.type_specifiers.push(TypeSpecifier::Unsigned);
                    self.advance();
                }
                TokenKind::Keyword(Keyword::Struct) => {
                    self.advance();
                    let s = self.parse_struct_or_union_body()?;
                    spec.type_specifiers.push(TypeSpecifier::Struct(s));
                }
                TokenKind::Keyword(Keyword::Union) => {
                    self.advance();
                    let s = self.parse_struct_or_union_body()?;
                    spec.type_specifiers.push(TypeSpecifier::Union(s));
                }
                TokenKind::Keyword(Keyword::Enum) => {
                    self.advance();
                    let s = self.parse_enum_body()?;
                    spec.type_specifiers.push(TypeSpecifier::Enum(s));
                }
                TokenKind::Identifier if spec.type_specifiers.is_empty() => {
                    if self.is_typedef_name_at(0) {
                        let name = self.advance().unwrap().text;
                        spec.type_specifiers.push(TypeSpecifier::TypedefName(name));
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        if spec.storage.is_none() && spec.qualifiers.is_empty() && spec.type_specifiers.is_empty() {
            return Err(self.err("expected declaration specifiers"));
        }
        Ok(spec)
    }

    fn parse_struct_or_union_body(&mut self) -> Result<StructOrUnionSpec, ParseError> {
        let name = self.eat_ident();
        let fields = if self.eat_punct(Punct::LBrace) {
            let mut fields = Vec::new();
            while !self.is_punct(Punct::RBrace) {
                fields.push(self.parse_field_decl()?);
            }
            self.expect_punct(Punct::RBrace)?;
            Some(fields)
        } else {
            None
        };
        Ok(StructOrUnionSpec { name, fields })
    }

    fn parse_field_decl(&mut self) -> Result<FieldDecl, ParseError> {
        let specifiers = self.parse_decl_specifiers()?;
        let mut declarators = Vec::new();
        if !self.is_punct(Punct::Semicolon) {
            loop {
                let declarator = if self.is_punct(Punct::Colon) {
                    None
                } else {
                    Some(self.parse_declarator()?)
                };
                let width = if self.eat_punct(Punct::Colon) {
                    Some(self.parse_conditional_expr()?)
                } else {
                    None
                };
                declarators.push((declarator, width));
                if !self.eat_punct(Punct::Comma) {
                    break;
                }
            }
        }
        self.expect_punct(Punct::Semicolon)?;
        Ok(FieldDecl {
            specifiers,
            declarators,
        })
    }

    fn parse_enum_body(&mut self) -> Result<EnumSpec, ParseError> {
        let name = self.eat_ident();
        let variants = if self.eat_punct(Punct::LBrace) {
            let mut variants = Vec::new();
            loop {
                let ident = self
                    .eat_ident()
                    .ok_or_else(|| self.err("expected enumerator identifier"))?;
                let value = if self.eat_punct(Punct::Eq) {
                    Some(self.parse_conditional_expr()?)
                } else {
                    None
                };
                variants.push((ident, value));
                if !self.eat_punct(Punct::Comma) {
                    break;
                }
                if self.is_punct(Punct::RBrace) {
                    break;
                }
            }
            self.expect_punct(Punct::RBrace)?;
            Some(variants)
        } else {
            None
        };
        Ok(EnumSpec { name, variants })
    }

    fn register_typedef(&mut self, d: &Declarator) {
        if let Some(name) = direct_declarator_name(&d.direct) {
            if let TypedefCheck::Lookup(set) = &mut self.typedef_check {
                set.insert(name.clone());
            }
            self.discovered_typedefs.push(name);
        }
    }

    fn parse_declarator(&mut self) -> Result<Declarator, ParseError> {
        let flex = self.parse_flex_declarator()?;
        if !flex_has_ident(&flex.direct) {
            return Err(self.err("expected a named declarator"));
        }
        Ok(Declarator {
            pointer_quals: flex.pointer_quals,
            direct: flex_to_named_direct(flex.direct),
        })
    }

    fn parse_abstract_declarator(&mut self) -> Result<AbstractDeclarator, ParseError> {
        let flex = self.parse_flex_declarator()?;
        Ok(AbstractDeclarator {
            pointer_quals: flex.pointer_quals,
            direct: flex_to_abstract_direct(flex.direct),
        })
    }

    fn parse_flex_declarator(&mut self) -> Result<FlexDeclarator, ParseError> {
        let pointer_quals = self.parse_pointer_quals();
        let direct = self.parse_flex_direct()?;
        Ok(FlexDeclarator {
            pointer_quals,
            direct,
        })
    }

    fn parse_pointer_quals(&mut self) -> Vec<Vec<TypeQualifier>> {
        let mut levels = Vec::new();
        while self.eat_punct(Punct::Star) {
            let mut quals = Vec::new();
            loop {
                match self.peek().map(|t| t.kind) {
                    Some(TokenKind::Keyword(Keyword::Const)) => {
                        quals.push(TypeQualifier::Const);
                        self.advance();
                    }
                    Some(TokenKind::Keyword(Keyword::Volatile)) => {
                        quals.push(TypeQualifier::Volatile);
                        self.advance();
                    }
                    // `restrict` (C99) / `__restrict` / `__restrict__` (GNU,
                    // usable even in C89 mode) -- real system headers use
                    // these pervasively (glibc's stdio.h/string.h alone:
                    // hundreds of occurrences). Not in our `Keyword` enum
                    // (C89-only lexer), so these lex as plain identifiers;
                    // recognized here by spelling and discarded, since
                    // restrict-ness isn't part of `TypeQualifier` and isn't
                    // needed for anything this pipeline currently does.
                    Some(TokenKind::Identifier)
                        if matches!(
                            self.peek().map(|t| t.text.as_str()),
                            Some("restrict" | "__restrict" | "__restrict__")
                        ) =>
                    {
                        self.advance();
                    }
                    _ => break,
                }
            }
            levels.push(quals);
        }
        levels
    }

    /// True when the `(` at the current position starts a grouping around a
    /// nested declarator, as opposed to a parameter-list applied directly to
    /// an abstract (nameless) declarator -- e.g. `(*x)` groups, but `(int)`
    /// or `()` don't.
    fn paren_is_grouping(&self) -> bool {
        match self.peek_at(1).map(|t| &t.kind) {
            Some(TokenKind::Punct(Punct::RParen)) => false,
            Some(TokenKind::Punct(Punct::Ellipsis)) => false,
            _ => !self.looks_like_decl_start_at(1),
        }
    }

    fn parse_flex_direct(&mut self) -> Result<FlexDirect, ParseError> {
        let mut base = if self.is_punct(Punct::LParen) && !self.paren_is_grouping() {
            self.advance();
            let params = self.parse_param_list()?;
            self.expect_punct(Punct::RParen)?;
            FlexDirect::Function(Box::new(FlexDirect::Leaf(None)), params)
        } else if self.eat_punct(Punct::LParen) {
            let inner = self.parse_flex_declarator()?;
            self.expect_punct(Punct::RParen)?;
            FlexDirect::Paren(Box::new(inner))
        } else if let Some(name) = self.eat_ident() {
            FlexDirect::Leaf(Some(name))
        } else {
            FlexDirect::Leaf(None)
        };

        loop {
            if self.eat_punct(Punct::LBracket) {
                let size = if self.is_punct(Punct::RBracket) {
                    None
                } else {
                    Some(self.parse_conditional_expr()?)
                };
                self.expect_punct(Punct::RBracket)?;
                base = FlexDirect::Array(Box::new(base), size);
            } else if self.eat_punct(Punct::LParen) {
                let params = self.parse_param_list()?;
                self.expect_punct(Punct::RParen)?;
                base = FlexDirect::Function(Box::new(base), params);
            } else {
                break;
            }
        }
        Ok(base)
    }

    fn parse_param_list(&mut self) -> Result<ParamList, ParseError> {
        let mut params = Vec::new();
        let mut variadic = false;
        if self.is_punct(Punct::RParen) {
            return Ok(ParamList { params, variadic });
        }
        if matches!(
            self.peek().map(|t| t.kind),
            Some(TokenKind::Keyword(Keyword::Void))
        ) && matches!(
            self.peek_at(1).map(|t| t.kind),
            Some(TokenKind::Punct(Punct::RParen))
        ) {
            self.advance();
            return Ok(ParamList { params, variadic });
        }
        loop {
            if self.eat_punct(Punct::Ellipsis) {
                variadic = true;
                break;
            }
            params.push(self.parse_param_decl()?);
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        Ok(ParamList { params, variadic })
    }

    fn parse_param_decl(&mut self) -> Result<ParamDecl, ParseError> {
        let specifiers = self.parse_decl_specifiers()?;
        let flex = self.parse_flex_declarator()?;
        let declarator = if flex_has_ident(&flex.direct) {
            ParamDeclarator::Named(Declarator {
                pointer_quals: flex.pointer_quals,
                direct: flex_to_named_direct(flex.direct),
            })
        } else if flex.pointer_quals.is_empty() && matches!(flex.direct, FlexDirect::Leaf(None)) {
            ParamDeclarator::Bare
        } else {
            ParamDeclarator::Abstract(AbstractDeclarator {
                pointer_quals: flex.pointer_quals,
                direct: flex_to_abstract_direct(flex.direct),
            })
        };
        Ok(ParamDecl {
            specifiers,
            declarator,
        })
    }

    fn parse_opt_initializer(&mut self) -> Result<Option<Initializer>, ParseError> {
        if self.eat_punct(Punct::Eq) {
            Ok(Some(self.parse_initializer()?))
        } else {
            Ok(None)
        }
    }

    fn parse_initializer(&mut self) -> Result<Initializer, ParseError> {
        if self.eat_punct(Punct::LBrace) {
            let mut items = Vec::new();
            if !self.is_punct(Punct::RBrace) {
                loop {
                    items.push(self.parse_initializer()?);
                    if !self.eat_punct(Punct::Comma) {
                        break;
                    }
                    if self.is_punct(Punct::RBrace) {
                        break;
                    }
                }
            }
            self.expect_punct(Punct::RBrace)?;
            Ok(Initializer::List(items))
        } else {
            Ok(Initializer::Expr(self.parse_assignment_expr()?))
        }
    }

    fn parse_declaration(&mut self) -> Result<Declaration, ParseError> {
        let specifiers = self.parse_decl_specifiers()?;
        let mut declarators = Vec::new();
        if !self.is_punct(Punct::Semicolon) {
            loop {
                let d = self.parse_declarator()?;
                if specifiers.storage == Some(StorageClass::Typedef) {
                    self.register_typedef(&d);
                }
                let init = self.parse_opt_initializer()?;
                declarators.push(InitDeclarator {
                    declarator: d,
                    initializer: init,
                });
                if !self.eat_punct(Punct::Comma) {
                    break;
                }
            }
        }
        self.expect_punct(Punct::Semicolon)?;
        Ok(Declaration {
            specifiers,
            declarators,
        })
    }

    fn parse_external_decl(&mut self) -> Result<ExternalDecl, ParseError> {
        // Rough-scan only: glibc headers occasionally prefix a declaration
        // with `__extension__` (GNU: "don't warn this is an extension"),
        // e.g. stdlib.h's `__extension__ extern long long int atoll (...)`.
        if self.skip_bodies && matches!(self.peek().map(|t| t.text.as_str()), Some("__extension__"))
        {
            self.advance();
        }
        let specifiers = self.parse_decl_specifiers()?;
        if self.eat_punct(Punct::Semicolon) {
            return Ok(ExternalDecl::Declaration(Declaration {
                specifiers,
                declarators: Vec::new(),
            }));
        }
        let declarator = self.parse_declarator()?;
        self.skip_gnu_decorations();
        if self.is_punct(Punct::LBrace) {
            let body = self.parse_compound_stmt()?;
            return Ok(ExternalDecl::FunctionDef(FunctionDef {
                specifiers,
                declarator,
                body,
            }));
        }

        if specifiers.storage == Some(StorageClass::Typedef) {
            self.register_typedef(&declarator);
        }
        let init = self.parse_opt_initializer()?;
        let mut declarators = vec![InitDeclarator {
            declarator,
            initializer: init,
        }];
        while self.eat_punct(Punct::Comma) {
            let d = self.parse_declarator()?;
            if specifiers.storage == Some(StorageClass::Typedef) {
                self.register_typedef(&d);
            }
            let init = self.parse_opt_initializer()?;
            declarators.push(InitDeclarator {
                declarator: d,
                initializer: init,
            });
        }
        self.expect_punct(Punct::Semicolon)?;
        Ok(ExternalDecl::Declaration(Declaration {
            specifiers,
            declarators,
        }))
    }

    // ---- statements ----

    fn parse_compound_stmt(&mut self) -> Result<CompoundStmt, ParseError> {
        if self.skip_bodies {
            self.skip_balanced_braces();
            return Ok(CompoundStmt { items: Vec::new() });
        }
        self.expect_punct(Punct::LBrace)?;
        let mut items = Vec::new();
        while !self.is_punct(Punct::RBrace) {
            if self.looks_like_decl_start() {
                items.push(BlockItem::Decl(self.parse_declaration()?));
            } else {
                items.push(BlockItem::Stmt(self.parse_stmt()?));
            }
        }
        self.expect_punct(Punct::RBrace)?;
        Ok(CompoundStmt { items })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        let kind = self.peek().map(|t| t.kind);
        match kind {
            Some(TokenKind::Punct(Punct::LBrace)) => {
                Ok(Stmt::Compound(self.parse_compound_stmt()?))
            }
            Some(TokenKind::Punct(Punct::Semicolon)) => {
                self.advance();
                Ok(Stmt::Expr(None))
            }
            Some(TokenKind::Keyword(Keyword::If)) => self.parse_if_stmt(),
            Some(TokenKind::Keyword(Keyword::Switch)) => self.parse_switch_stmt(),
            Some(TokenKind::Keyword(Keyword::While)) => self.parse_while_stmt(),
            Some(TokenKind::Keyword(Keyword::Do)) => self.parse_do_while_stmt(),
            Some(TokenKind::Keyword(Keyword::For)) => self.parse_for_stmt(),
            Some(TokenKind::Keyword(Keyword::Goto)) => {
                self.advance();
                let label = self
                    .eat_ident()
                    .ok_or_else(|| self.err("expected label after 'goto'"))?;
                self.expect_punct(Punct::Semicolon)?;
                Ok(Stmt::Goto(label))
            }
            Some(TokenKind::Keyword(Keyword::Continue)) => {
                self.advance();
                self.expect_punct(Punct::Semicolon)?;
                Ok(Stmt::Continue)
            }
            Some(TokenKind::Keyword(Keyword::Break)) => {
                self.advance();
                self.expect_punct(Punct::Semicolon)?;
                Ok(Stmt::Break)
            }
            Some(TokenKind::Keyword(Keyword::Return)) => {
                self.advance();
                let e = if self.is_punct(Punct::Semicolon) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                self.expect_punct(Punct::Semicolon)?;
                Ok(Stmt::Return(e))
            }
            Some(TokenKind::Keyword(Keyword::Case)) => {
                self.advance();
                let expr = self.parse_conditional_expr()?;
                self.expect_punct(Punct::Colon)?;
                let stmt = Box::new(self.parse_stmt()?);
                Ok(Stmt::Case { expr, stmt })
            }
            Some(TokenKind::Keyword(Keyword::Default)) => {
                self.advance();
                self.expect_punct(Punct::Colon)?;
                Ok(Stmt::Default(Box::new(self.parse_stmt()?)))
            }
            Some(TokenKind::Identifier)
                if matches!(
                    self.peek_at(1).map(|t| t.kind),
                    Some(TokenKind::Punct(Punct::Colon))
                ) =>
            {
                let label = self.advance().unwrap().text;
                self.advance();
                let stmt = Box::new(self.parse_stmt()?);
                Ok(Stmt::Labeled { label, stmt })
            }
            _ => {
                let e = self.parse_expr()?;
                self.expect_punct(Punct::Semicolon)?;
                Ok(Stmt::Expr(Some(e)))
            }
        }
    }

    fn parse_if_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.advance();
        self.expect_punct(Punct::LParen)?;
        let cond = self.parse_expr()?;
        self.expect_punct(Punct::RParen)?;
        let then_branch = Box::new(self.parse_stmt()?);
        let else_branch = if matches!(
            self.peek().map(|t| t.kind),
            Some(TokenKind::Keyword(Keyword::Else))
        ) {
            self.advance();
            Some(Box::new(self.parse_stmt()?))
        } else {
            None
        };
        Ok(Stmt::If {
            cond,
            then_branch,
            else_branch,
        })
    }

    fn parse_switch_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.advance();
        self.expect_punct(Punct::LParen)?;
        let cond = self.parse_expr()?;
        self.expect_punct(Punct::RParen)?;
        let body = Box::new(self.parse_stmt()?);
        Ok(Stmt::Switch { cond, body })
    }

    fn parse_while_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.advance();
        self.expect_punct(Punct::LParen)?;
        let cond = self.parse_expr()?;
        self.expect_punct(Punct::RParen)?;
        let body = Box::new(self.parse_stmt()?);
        Ok(Stmt::While { cond, body })
    }

    fn parse_do_while_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.advance();
        let body = Box::new(self.parse_stmt()?);
        self.expect_keyword(Keyword::While)?;
        self.expect_punct(Punct::LParen)?;
        let cond = self.parse_expr()?;
        self.expect_punct(Punct::RParen)?;
        self.expect_punct(Punct::Semicolon)?;
        Ok(Stmt::DoWhile { body, cond })
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.advance();
        self.expect_punct(Punct::LParen)?;
        let init = if self.is_punct(Punct::Semicolon) {
            self.advance();
            None
        } else if self.looks_like_decl_start() {
            Some(ForInit::Decl(self.parse_declaration()?))
        } else {
            let e = self.parse_expr()?;
            self.expect_punct(Punct::Semicolon)?;
            Some(ForInit::Expr(e))
        };
        let cond = if self.is_punct(Punct::Semicolon) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect_punct(Punct::Semicolon)?;
        let step = if self.is_punct(Punct::RParen) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect_punct(Punct::RParen)?;
        let body = Box::new(self.parse_stmt()?);
        Ok(Stmt::For {
            init,
            cond,
            step,
            body,
        })
    }

    // ---- expressions ----

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let mut e = self.parse_assignment_expr()?;
        while self.eat_punct(Punct::Comma) {
            let rhs = self.parse_assignment_expr()?;
            e = Expr::Comma(Box::new(e), Box::new(rhs));
        }
        Ok(e)
    }

    fn parse_assignment_expr(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_conditional_expr()?;
        let op = self.peek().map(|t| t.kind).and_then(assign_op);
        if let Some(op) = op {
            self.advance();
            let rhs = self.parse_assignment_expr()?;
            Ok(Expr::Assign {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        } else {
            Ok(lhs)
        }
    }

    fn parse_conditional_expr(&mut self) -> Result<Expr, ParseError> {
        let cond = self.parse_binary_expr(1)?;
        if self.eat_punct(Punct::Question) {
            let then_expr = self.parse_expr()?;
            self.expect_punct(Punct::Colon)?;
            let else_expr = self.parse_conditional_expr()?;
            Ok(Expr::Conditional {
                cond: Box::new(cond),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
            })
        } else {
            Ok(cond)
        }
    }

    fn parse_binary_expr(&mut self, min_prec: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_cast_expr()?;
        while let Some(kind) = self.peek().map(|t| t.kind) {
            let Some((op, prec)) = binop_info(kind) else {
                break;
            };
            if prec < min_prec {
                break;
            }
            self.advance();
            let rhs = self.parse_binary_expr(prec + 1)?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_cast_expr(&mut self) -> Result<Expr, ParseError> {
        if self.is_punct(Punct::LParen) && self.peek_is_type_start_at(1) {
            self.advance();
            let type_name = self.parse_type_name()?;
            self.expect_punct(Punct::RParen)?;
            let expr = self.parse_cast_expr()?;
            return Ok(Expr::Cast {
                type_name,
                expr: Box::new(expr),
            });
        }
        self.parse_unary_expr()
    }

    fn parse_type_name(&mut self) -> Result<TypeName, ParseError> {
        let specifiers = self.parse_decl_specifiers()?;
        let abstract_declarator = if matches!(
            self.peek().map(|t| &t.kind),
            Some(TokenKind::Punct(
                Punct::Star | Punct::LParen | Punct::LBracket
            ))
        ) {
            Some(Box::new(self.parse_abstract_declarator()?))
        } else {
            None
        };
        Ok(TypeName {
            specifiers,
            abstract_declarator,
        })
    }

    fn parse_unary_expr(&mut self) -> Result<Expr, ParseError> {
        let kind = self.peek().map(|t| t.kind);
        match kind {
            Some(TokenKind::Punct(Punct::PlusPlus)) => {
                self.advance();
                Ok(Expr::PreIncDec {
                    expr: Box::new(self.parse_unary_expr()?),
                    op: IncDecOp::Inc,
                })
            }
            Some(TokenKind::Punct(Punct::MinusMinus)) => {
                self.advance();
                Ok(Expr::PreIncDec {
                    expr: Box::new(self.parse_unary_expr()?),
                    op: IncDecOp::Dec,
                })
            }
            Some(TokenKind::Punct(Punct::Amp)) => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnaryOp::AddrOf,
                    expr: Box::new(self.parse_cast_expr()?),
                })
            }
            Some(TokenKind::Punct(Punct::Star)) => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnaryOp::Deref,
                    expr: Box::new(self.parse_cast_expr()?),
                })
            }
            Some(TokenKind::Punct(Punct::Plus)) => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnaryOp::Plus,
                    expr: Box::new(self.parse_cast_expr()?),
                })
            }
            Some(TokenKind::Punct(Punct::Minus)) => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnaryOp::Minus,
                    expr: Box::new(self.parse_cast_expr()?),
                })
            }
            Some(TokenKind::Punct(Punct::Tilde)) => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnaryOp::BitNot,
                    expr: Box::new(self.parse_cast_expr()?),
                })
            }
            Some(TokenKind::Punct(Punct::Bang)) => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(self.parse_cast_expr()?),
                })
            }
            Some(TokenKind::Keyword(Keyword::Sizeof)) => {
                self.advance();
                if self.is_punct(Punct::LParen) && self.peek_is_type_start_at(1) {
                    self.advance();
                    let type_name = self.parse_type_name()?;
                    self.expect_punct(Punct::RParen)?;
                    Ok(Expr::Sizeof(SizeofArg::Type(type_name)))
                } else {
                    Ok(Expr::Sizeof(SizeofArg::Expr(Box::new(
                        self.parse_unary_expr()?,
                    ))))
                }
            }
            _ => self.parse_postfix_expr(),
        }
    }

    fn parse_postfix_expr(&mut self) -> Result<Expr, ParseError> {
        let mut e = self.parse_primary_expr()?;
        loop {
            let kind = self.peek().map(|t| t.kind);
            match kind {
                Some(TokenKind::Punct(Punct::LBracket)) => {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect_punct(Punct::RBracket)?;
                    e = Expr::Index {
                        base: Box::new(e),
                        index: Box::new(index),
                    };
                }
                Some(TokenKind::Punct(Punct::LParen)) => {
                    self.advance();
                    let mut args = Vec::new();
                    if !self.is_punct(Punct::RParen) {
                        loop {
                            args.push(self.parse_assignment_expr()?);
                            if !self.eat_punct(Punct::Comma) {
                                break;
                            }
                        }
                    }
                    self.expect_punct(Punct::RParen)?;
                    e = Expr::Call {
                        callee: Box::new(e),
                        args,
                    };
                }
                Some(TokenKind::Punct(Punct::Dot)) => {
                    self.advance();
                    let field = self
                        .eat_ident()
                        .ok_or_else(|| self.err("expected field name after '.'"))?;
                    e = Expr::Member {
                        base: Box::new(e),
                        field,
                        arrow: false,
                    };
                }
                Some(TokenKind::Punct(Punct::Arrow)) => {
                    self.advance();
                    let field = self
                        .eat_ident()
                        .ok_or_else(|| self.err("expected field name after '->'"))?;
                    e = Expr::Member {
                        base: Box::new(e),
                        field,
                        arrow: true,
                    };
                }
                Some(TokenKind::Punct(Punct::PlusPlus)) => {
                    self.advance();
                    e = Expr::PostIncDec {
                        expr: Box::new(e),
                        op: IncDecOp::Inc,
                    };
                }
                Some(TokenKind::Punct(Punct::MinusMinus)) => {
                    self.advance();
                    e = Expr::PostIncDec {
                        expr: Box::new(e),
                        op: IncDecOp::Dec,
                    };
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, ParseError> {
        let kind = self.peek().map(|t| t.kind);
        match kind {
            Some(TokenKind::Identifier) => Ok(Expr::Ident(self.advance().unwrap().text)),
            Some(TokenKind::IntegerConstant) => Ok(Expr::IntLiteral(self.advance().unwrap().text)),
            Some(TokenKind::FloatConstant) => Ok(Expr::FloatLiteral(self.advance().unwrap().text)),
            Some(TokenKind::CharLiteral) => Ok(Expr::CharLiteral(self.advance().unwrap().text)),
            Some(TokenKind::StringLiteral) => {
                // C's own adjacent-string-literal concatenation
                // (`I_Error ("P_PlayerInSpecialSector: "\n"unknown special
                // %i", ..)`, a message split across source lines) --
                // each token's own `text` already includes its
                // surrounding quotes verbatim (the lexer's own choice,
                // confirmed by `lexer.rs`'s own test asserting `text ==
                // "\"hello\""`), so a second literal can't just be
                // appended after the first's closing quote (that would
                // produce `"a""b"`, two adjacent Rust string literals --
                // not valid Rust syntax at all, confirmed by a real
                // `rustc` rejection when this was tried) -- the closing
                // quote of each literal but the last, and the opening
                // quote of each but the first, are dropped instead, so
                // the merged result is one real quoted string whose
                // content is the concatenation of both, matching what C
                // itself does semantically.
                let mut s = self.advance().unwrap().text;
                while matches!(self.peek().map(|t| t.kind), Some(TokenKind::StringLiteral)) {
                    let next = self.advance().unwrap().text;
                    s.pop();
                    s.push_str(next.strip_prefix('"').unwrap_or(&next));
                }
                Ok(Expr::StringLiteral(s))
            }
            Some(TokenKind::Punct(Punct::LParen)) => {
                self.advance();
                let e = self.parse_expr()?;
                self.expect_punct(Punct::RParen)?;
                Ok(e)
            }
            _ => Err(self.err("expected expression")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{attach_comments, lex_chunks, partition_source};

    fn parse(code: &str) -> TranslationUnit {
        let chunks = partition_source(code);
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        parse_translation_unit(&stream).unwrap_or_else(|e| panic!("parse error: {e}\nin: {code}"))
    }

    fn rough_decls(code: &str) -> Vec<ExternalDecl> {
        let chunks = partition_source(code);
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        extract_top_level_decls(&stream)
    }

    fn parse_err(code: &str) -> ParseError {
        let chunks = partition_source(code);
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        parse_translation_unit(&stream).unwrap_err()
    }

    #[test]
    fn test_simple_declaration() {
        let unit = parse("int x;\n");
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0] {
            ExternalDecl::Declaration(d) => {
                assert_eq!(d.specifiers.type_specifiers, vec![TypeSpecifier::Int]);
                assert_eq!(d.declarators.len(), 1);
                assert_eq!(
                    d.declarators[0].declarator.direct,
                    DirectDeclarator::Ident("x".into())
                );
            }
            other => panic!("expected declaration, got {other:?}"),
        }
    }

    #[test]
    fn test_pointer_and_array_declarators() {
        let unit = parse("int *p; int arr[10]; int *parr[5];\n");
        assert_eq!(unit.items.len(), 3);
        let ExternalDecl::Declaration(d0) = &unit.items[0] else {
            panic!()
        };
        assert_eq!(d0.declarators[0].declarator.pointer_quals.len(), 1);
        let ExternalDecl::Declaration(d1) = &unit.items[1] else {
            panic!()
        };
        assert!(matches!(
            &d1.declarators[0].declarator.direct,
            DirectDeclarator::Array(_, Some(_))
        ));
        let ExternalDecl::Declaration(d2) = &unit.items[2] else {
            panic!()
        };
        // int *parr[5]: array of pointers, i.e. Array(Ident, ...) with one pointer level.
        assert_eq!(d2.declarators[0].declarator.pointer_quals.len(), 1);
        assert!(matches!(
            &d2.declarators[0].declarator.direct,
            DirectDeclarator::Array(_, Some(_))
        ));
    }

    #[test]
    fn test_typedef_and_struct_self_reference() {
        let unit =
            parse("typedef struct mobj_s { struct mobj_s *next; int x; } mobj_t;\nmobj_t *m;\n");
        assert_eq!(unit.items.len(), 2);
        let ExternalDecl::Declaration(d1) = &unit.items[1] else {
            panic!()
        };
        assert_eq!(
            d1.specifiers.type_specifiers,
            vec![TypeSpecifier::TypedefName("mobj_t".into())]
        );
    }

    #[test]
    fn test_function_pointer_typedef() {
        let unit = parse("typedef int (*traverser_t) (int x);\n");
        let ExternalDecl::Declaration(d) = &unit.items[0] else {
            panic!()
        };
        let DirectDeclarator::Function(inner, params) = &d.declarators[0].declarator.direct else {
            panic!()
        };
        assert!(matches!(**inner, DirectDeclarator::Paren(_)));
        assert_eq!(params.params.len(), 1);
    }

    #[test]
    fn test_function_definition_with_control_flow() {
        let unit = parse(
            r#"
int add(int a, int b) {
    int result = a + b;
    if (result > 0) {
        return result;
    } else {
        return -result;
    }
}
"#,
        );
        let ExternalDecl::FunctionDef(f) = &unit.items[0] else {
            panic!("expected function def")
        };
        assert_eq!(f.body.items.len(), 2);
        assert!(matches!(&f.body.items[0], BlockItem::Decl(_)));
        match &f.body.items[1] {
            BlockItem::Stmt(Stmt::If { .. }) => {}
            other => panic!("expected if statement, got {other:?}"),
        }
    }

    #[test]
    fn test_loops_switch_goto() {
        let unit = parse(
            r#"
void f(void) {
    int i;
    for (i = 0; i < 10; i++) {
        switch (i) {
            case 0: break;
            default: continue;
        }
    }
    while (i) { i--; }
    do { i++; } while (i < 5);
    goto done;
done:
    return;
}
"#,
        );
        let ExternalDecl::FunctionDef(f) = &unit.items[0] else {
            panic!()
        };
        assert!(f.body.items.len() >= 5);
    }

    #[test]
    fn test_expression_precedence() {
        let unit = parse("int x = 1 + 2 * 3;\n");
        let ExternalDecl::Declaration(d) = &unit.items[0] else {
            panic!()
        };
        let Some(Initializer::Expr(e)) = &d.declarators[0].initializer else {
            panic!()
        };
        // 1 + (2 * 3): top node is Add, rhs is Mul.
        let Expr::Binary {
            op: BinaryOp::Add,
            rhs,
            ..
        } = e
        else {
            panic!("expected Add, got {e:?}")
        };
        assert!(matches!(
            **rhs,
            Expr::Binary {
                op: BinaryOp::Mul,
                ..
            }
        ));
    }

    #[test]
    fn test_cast_and_sizeof() {
        let unit = parse("int a = (int) 3.5; int b = sizeof(int); int c = sizeof x;\n");
        let ExternalDecl::Declaration(d0) = &unit.items[0] else {
            panic!()
        };
        assert!(matches!(
            &d0.declarators[0].initializer,
            Some(Initializer::Expr(Expr::Cast { .. }))
        ));
        let ExternalDecl::Declaration(d1) = &unit.items[1] else {
            panic!()
        };
        assert!(matches!(
            &d1.declarators[0].initializer,
            Some(Initializer::Expr(Expr::Sizeof(SizeofArg::Type(_))))
        ));
        let ExternalDecl::Declaration(d2) = &unit.items[2] else {
            panic!()
        };
        assert!(matches!(
            &d2.declarators[0].initializer,
            Some(Initializer::Expr(Expr::Sizeof(SizeofArg::Expr(_))))
        ));
    }

    #[test]
    fn test_adjacent_string_literal_concatenation() {
        // `I_Error ("P_PlayerInSpecialSector: "\n"unknown special %i",
        // ..)` (`p_spec.c`) is the real corpus source this was built
        // against: two adjacent string-literal tokens must merge into one
        // *real* quoted string whose content is their concatenation
        // (`"hello world"`), not the raw juxtaposition of both tokens'
        // own already-quoted text (`"hello ""world"`, not valid Rust
        // syntax at all -- confirmed by a real `rustc` rejection when
        // `function_body.rs`'s `Expr::StringLiteral` render passed that
        // through unchanged for `P_PlayerInSpecialSector`'s own `I_Error`
        // call). Previously asserted the wrong (juxtaposed) value here,
        // undetected until a real corpus function actually needed a
        // multi-line message.
        let unit = parse(r#"char *s = "hello " "world";"#);
        let ExternalDecl::Declaration(d) = &unit.items[0] else {
            panic!()
        };
        assert_eq!(
            d.declarators[0].initializer,
            Some(Initializer::Expr(Expr::StringLiteral(
                "\"hello world\"".into()
            )))
        );
    }

    #[test]
    fn test_unresolved_type_reports_error() {
        let err = parse_err("FILE *f;\n");
        assert!(err.message.contains("declaration specifiers") || err.near == "FILE");
    }

    #[test]
    fn test_seeded_typedef_resolves_unknown_type() {
        let chunks = partition_source("fixed_t x;\n");
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        let mut seed = HashSet::new();
        seed.insert("fixed_t".to_string());
        let unit = parse_translation_unit_seeded(&stream, seed).unwrap();
        let ExternalDecl::Declaration(d) = &unit.items[0] else {
            panic!()
        };
        assert_eq!(
            d.specifiers.type_specifiers,
            vec![TypeSpecifier::TypedefName("fixed_t".into())]
        );
    }

    #[test]
    fn test_extract_top_level_typedefs_skips_function_bodies() {
        let code = "typedef int foo_t;\nvoid f(void) { typedef int local_t; local_t x; }\n";
        let chunks = partition_source(code);
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        assert_eq!(
            extract_top_level_typedefs(&stream),
            vec!["foo_t".to_string()]
        );
    }

    #[test]
    fn test_extract_top_level_typedefs_handles_forward_reference() {
        // "boolean" is unknown at scan time, but the heuristic still
        // correctly identifies "flag_t" as the declared name, not "boolean".
        let code = "typedef boolean flag_t;\n";
        let chunks = partition_source(code);
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        assert_eq!(
            extract_top_level_typedefs(&stream),
            vec!["flag_t".to_string()]
        );
    }

    fn decl_name(item: &ExternalDecl) -> Option<String> {
        match item {
            ExternalDecl::FunctionDef(f) => declarator_name(&f.declarator),
            ExternalDecl::Declaration(d) => d
                .declarators
                .first()
                .and_then(|id| declarator_name(&id.declarator)),
        }
    }

    #[test]
    fn test_restrict_qualifier_is_ignored_in_strict_mode() {
        // Real system headers (glibc's stdio.h/string.h) use this
        // pervasively; must parse (and still just be a plain pointer) in
        // Step 6c's strict mode too, not just the rough scan.
        let unit = parse("void f(char *__restrict a, const char *restrict b);\n");
        assert_eq!(unit.items.len(), 1);
    }

    #[test]
    fn test_gnu_trailing_decorations_skipped_in_rough_mode() {
        // Mirrors glibc string.h's real strcpy/strlen declarations.
        let code = "extern char *strcpy (char *__restrict d, const char *__restrict s) __THROW __nonnull ((1, 2));\nextern size_t strlen (const char *s) __THROW __attribute_pure__ __nonnull ((1));\n";
        let items = rough_decls(code);
        let names: Vec<_> = items.iter().filter_map(decl_name).collect();
        assert_eq!(names, vec!["strcpy".to_string(), "strlen".to_string()]);
    }

    #[test]
    fn test_extension_prefix_skipped_in_rough_mode() {
        let code = "__extension__ extern long long int atoll (const char *s);\n";
        let items = rough_decls(code);
        let names: Vec<_> = items.iter().filter_map(decl_name).collect();
        assert_eq!(names, vec!["atoll".to_string()]);
    }

    #[test]
    fn test_rough_scan_recovers_after_unparseable_top_level_construct() {
        // A construct our grammar has no model for at all (inline asm)
        // must not wipe out declarations before *and* after it.
        let code = "extern int before (void);\n__asm__(\".text\");\nextern int after (void);\n";
        let items = rough_decls(code);
        let names: Vec<_> = items.iter().filter_map(decl_name).collect();
        assert!(
            names.contains(&"before".to_string()),
            "expected 'before' to survive recovery, got {names:?}"
        );
        assert!(
            names.contains(&"after".to_string()),
            "expected 'after' to survive recovery, got {names:?}"
        );
    }

    #[test]
    fn test_strict_mode_still_fails_on_unparseable_construct() {
        // Recovery is rough-scan only -- Step 6c's real, final parse must
        // still surface a genuine error rather than silently skip it.
        let err = parse_err("__asm__(\".text\");\n");
        assert!(!err.message.is_empty());
    }
}
