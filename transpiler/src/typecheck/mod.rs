//! Phase 2: Semantic Analysis & Typechecking (docs/02_TYPECHECKER.md)

pub mod array_shape;
pub mod check;
pub mod declared_types;
pub mod exports;
pub mod macro_types;
pub mod mutability;
pub mod nullability;
pub mod resolve;
pub mod scope;
pub mod types;

pub use array_shape::{
    ArrayShape, ArrayShapeAnalysis, Evidence, ParamKey, analyze, collect_body_evidence,
    collect_call_evidence, functions_with_bodies,
};
pub use check::{DiagnosticKind, TypeCheckResult, TypeDiagnostic, check_translation_unit};
pub use declared_types::{DeclaredTypes, DeclaredTypesResolver};
pub use exports::{ExportResolver, ExportedDecls};
pub use macro_types::{MacroTyper, MacroUse, collect_macro_uses, substitute};
pub use resolve::{
    ResolveResult, UnresolvedIdent, resolve_translation_unit, resolve_translation_unit_seeded,
};
pub use scope::{Symbol, SymbolKind, SymbolTable, Tag, TagKind};
pub use types::{FunctionSignature, Type};

// `mutability`'s and `nullability`'s own Evidence/analyze/
// collect_*_evidence intentionally aren't re-exported flat here -- they'd
// collide with `array_shape`'s same-named counterparts above. Reach them
// via `typecheck::mutability::`/`typecheck::nullability::`.
