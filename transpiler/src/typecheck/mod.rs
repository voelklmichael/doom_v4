//! Phase 2: Semantic Analysis & Typechecking (docs/02_TYPECHECKER.md)

pub mod exports;
pub mod resolve;
pub mod scope;

pub use exports::{ExportResolver, ExportedDecls};
pub use resolve::{
    ResolveResult, UnresolvedIdent, resolve_translation_unit, resolve_translation_unit_seeded,
};
pub use scope::{Symbol, SymbolKind, SymbolTable, Tag, TagKind};
