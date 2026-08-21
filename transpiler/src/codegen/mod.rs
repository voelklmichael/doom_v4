//! Phase 3: Transpilation & Code Generation (docs/03_TRANSPILER.md)

pub mod modules;
pub mod visibility;

pub use modules::{Module, ModuleGraph, ModuleKind, build_module_graph};
pub use visibility::{
    DefinedSymbol, ModuleVisibility, RawDeclarationIndex, own_defined_symbols,
    resolve_module_visibility,
};
