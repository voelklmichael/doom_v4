//! Step 6: C89 AST types.
//!
//! A fairly direct transcription of the C89 grammar (K&R App. A13), minus
//! constructs `linuxdoom-1.10` doesn't use (no K&R-style old-style function
//! definitions, no designated initializers, no GNU extensions).

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranslationUnit {
    pub items: Vec<ExternalDecl>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExternalDecl {
    FunctionDef(FunctionDef),
    Declaration(Declaration),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub specifiers: DeclSpecifiers,
    pub declarator: Declarator,
    pub body: CompoundStmt,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeclSpecifiers {
    pub storage: Option<StorageClass>,
    pub qualifiers: Vec<TypeQualifier>,
    pub type_specifiers: Vec<TypeSpecifier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Typedef,
    Extern,
    Static,
    Auto,
    Register,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeQualifier {
    Const,
    Volatile,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeSpecifier {
    Void,
    Char,
    Short,
    Int,
    Long,
    Float,
    Double,
    Signed,
    Unsigned,
    Struct(StructOrUnionSpec),
    Union(StructOrUnionSpec),
    Enum(EnumSpec),
    TypedefName(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructOrUnionSpec {
    pub name: Option<String>,
    /// `None` for a forward reference / bare use (`struct Foo *p;`);
    /// `Some` when this occurrence defines the members (`struct Foo { ... }`).
    pub fields: Option<Vec<FieldDecl>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldDecl {
    pub specifiers: DeclSpecifiers,
    /// `(declarator, bit-field width)`. The declarator is absent for an
    /// anonymous bit-field (`int : 3;`).
    pub declarators: Vec<(Option<Declarator>, Option<Expr>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumSpec {
    pub name: Option<String>,
    pub variants: Option<Vec<(String, Option<Expr>)>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub specifiers: DeclSpecifiers,
    pub declarators: Vec<InitDeclarator>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitDeclarator {
    pub declarator: Declarator,
    pub initializer: Option<Initializer>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Initializer {
    Expr(Expr),
    List(Vec<Initializer>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Declarator {
    /// One entry per `*`, holding that pointer level's own qualifiers
    /// (`* const * x` -> `[[], [Const]]`).
    pub pointer_quals: Vec<Vec<TypeQualifier>>,
    pub direct: DirectDeclarator,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DirectDeclarator {
    Ident(String),
    Paren(Box<Declarator>),
    Array(Box<DirectDeclarator>, Option<Expr>),
    Function(Box<DirectDeclarator>, ParamList),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParamList {
    pub params: Vec<ParamDecl>,
    pub variadic: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParamDecl {
    pub specifiers: DeclSpecifiers,
    pub declarator: ParamDeclarator,
}

/// A parameter's declarator: a named one (`int x`), an unnamed/abstract one
/// with pointer/array/function structure (`char *`, `int []`), or nothing at
/// all beyond the type (a bare `int`).
#[derive(Debug, Clone, PartialEq)]
pub enum ParamDeclarator {
    Named(Declarator),
    Abstract(AbstractDeclarator),
    Bare,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeName {
    pub specifiers: DeclSpecifiers,
    pub abstract_declarator: Option<Box<AbstractDeclarator>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractDeclarator {
    pub pointer_quals: Vec<Vec<TypeQualifier>>,
    pub direct: Option<DirectAbstractDeclarator>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DirectAbstractDeclarator {
    Paren(Box<AbstractDeclarator>),
    Array(Option<Box<DirectAbstractDeclarator>>, Option<Expr>),
    Function(Option<Box<DirectAbstractDeclarator>>, ParamList),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompoundStmt {
    pub items: Vec<BlockItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockItem {
    Decl(Declaration),
    Stmt(Stmt),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expr(Option<Expr>),
    Compound(CompoundStmt),
    If {
        cond: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    Switch {
        cond: Expr,
        body: Box<Stmt>,
    },
    Case {
        expr: Expr,
        stmt: Box<Stmt>,
    },
    Default(Box<Stmt>),
    While {
        cond: Expr,
        body: Box<Stmt>,
    },
    DoWhile {
        body: Box<Stmt>,
        cond: Expr,
    },
    For {
        init: Option<ForInit>,
        cond: Option<Expr>,
        step: Option<Expr>,
        body: Box<Stmt>,
    },
    Goto(String),
    Continue,
    Break,
    Return(Option<Expr>),
    Labeled {
        label: String,
        stmt: Box<Stmt>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    Decl(Declaration),
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Ident(String),
    IntLiteral(String),
    FloatLiteral(String),
    /// Already merged from any adjacent string-literal tokens (C89 translation phase 6).
    StringLiteral(String),
    CharLiteral(String),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Assign {
        op: AssignOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Conditional {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    Comma(Box<Expr>, Box<Expr>),
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    Member {
        base: Box<Expr>,
        field: String,
        arrow: bool,
    },
    PostIncDec {
        expr: Box<Expr>,
        op: IncDecOp,
    },
    PreIncDec {
        expr: Box<Expr>,
        op: IncDecOp,
    },
    Cast {
        type_name: TypeName,
        expr: Box<Expr>,
    },
    Sizeof(SizeofArg),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SizeofArg {
    Expr(Box<Expr>),
    Type(TypeName),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Plus,
    Minus,
    Not,
    BitNot,
    Deref,
    AddrOf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Mul,
    Div,
    Mod,
    Add,
    Sub,
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
    LogAnd,
    LogOr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    MulAssign,
    DivAssign,
    ModAssign,
    AddAssign,
    SubAssign,
    ShlAssign,
    ShrAssign,
    AndAssign,
    XorAssign,
    OrAssign,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncDecOp {
    Inc,
    Dec,
}
