//! Phase 3: Transpilation & Code Generation (docs/03_TRANSPILER.md)

pub mod mod_tree;
pub mod modules;
pub mod runtime;
pub mod use_stmt;
pub mod visibility;

pub use mod_tree::render_mod_declarations;
pub use modules::{Module, ModuleGraph, ModuleKind, build_module_graph};
pub use runtime::FixedT;
pub use use_stmt::render_use_block;
pub use visibility::{
    DefinedSymbol, ModuleVisibility, RawDeclarationIndex, own_defined_symbols,
    resolve_module_visibility,
};
