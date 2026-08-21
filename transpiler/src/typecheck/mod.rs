//! Phase 2: Semantic Analysis & Typechecking (docs/02_TYPECHECKER.md)

pub mod resolve;
pub mod scope;

pub use resolve::{ResolveResult, UnresolvedIdent, resolve_translation_unit};
pub use scope::{Symbol, SymbolKind, SymbolTable, Tag, TagKind};
