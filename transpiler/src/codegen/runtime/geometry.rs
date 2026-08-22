//! Level-geometry index newtypes (`docs/03_TRANSPILER.md`'s Memory Model
//! section): `sector_t`/`line_t`/`side_t`/... are bulk-allocated once per
//! level and never individually resized or freed mid-level, so a plain
//! index is enough -- no generation counter, unlike `Arena<T>`'s handles.
//!
//! `SectorId`/`SubsectorId` (`p_spec.h`'s lighting-effect thinkers,
//! `mobj_t.subsector`) and `VertexId`/`SideId`/`LineId` (`r_defs.h`'s
//! `vertex_t`/`side_t`/`line_t`, cross-referenced from `seg_t`/`sector_t`)
//! exist so far. More types get added the same way, as real translated
//! fields actually need them, not spun up speculatively ahead of that.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SectorId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubsectorId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SideId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineId(pub u32);
