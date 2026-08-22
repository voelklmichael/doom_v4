//! Phase 3: Transpilation & Code Generation (docs/03_TRANSPILER.md)

pub mod enum_values;
pub mod mod_tree;
pub mod modules;
pub mod runtime;
pub mod struct_fields;
pub mod thinkers;
pub mod use_stmt;
pub mod visibility;

pub use enum_values::{compute_enum_values, fold_const_int, render_enum_consts};
pub use mod_tree::render_mod_declarations;
pub use modules::{Module, ModuleGraph, ModuleKind, build_module_graph};
pub use runtime::{
    Arena, FixedT, Handle, LineId, PlayerId, SectorId, SideId, SubsectorId, VertexId,
};
pub use struct_fields::{MappedField, find_typedef_struct, map_struct_fields, render_struct};
pub use thinkers::{render_thinker_dispatch_stub, render_thinker_enum, render_thinker_structs};
pub use use_stmt::render_use_block;
pub use visibility::{
    DefinedSymbol, ModuleVisibility, RawDeclarationIndex, own_defined_symbols,
    resolve_module_visibility,
};
