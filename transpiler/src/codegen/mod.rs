//! Phase 3: Transpilation & Code Generation (docs/03_TRANSPILER.md)

pub mod visibility;

pub use visibility::{
    DefinedSymbol, ModuleVisibility, RawDeclarationIndex, own_defined_symbols,
    resolve_module_visibility,
};
