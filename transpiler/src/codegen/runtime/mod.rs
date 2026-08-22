//! Phase 3 Rust Runtime Support
//!
//! Unlike the rest of `codegen/`, this isn't analysis over the corpus --
//! it's literal Rust source that the transpiled crate will need alongside
//! its generated modules (its own copy of these files, not something
//! `ModuleGraph` has an entry for). Kept here, compiled and tested as part
//! of this crate too, so its behavior is verified once rather than trusted
//! by inspection wherever it eventually gets copied.

pub mod arena;
pub mod fixed;
pub mod geometry;
pub mod player;

pub use arena::{Arena, Handle};
pub use fixed::{FRACBITS, FRACUNIT, FixedT};
pub use geometry::{LineId, SectorId, SideId, SubsectorId, VertexId};
pub use player::PlayerId;
