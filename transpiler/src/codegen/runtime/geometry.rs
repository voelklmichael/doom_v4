//! Level-geometry index newtypes (`docs/03_TRANSPILER.md`'s Memory Model
//! section): `sector_t`/`line_t`/`side_t`/... are bulk-allocated once per
//! level and never individually resized or freed mid-level, so a plain
//! index is enough -- no generation counter, unlike `Arena<T>`'s handles.
//!
//! `SectorId` and `SubsectorId` exist so far -- `SectorId` for the
//! `p_spec.h` lighting-effect thinkers, `SubsectorId` for `mobj_t`'s own
//! `subsector` field (`p_maputl.c`'s `P_SetThingPosition` sets it
//! unconditionally for a live thing, and every read dereferences it
//! without a null check, so non-`Option` like `SectorId`). More types
//! (`LineId`, `SideId`, `VertexId`, and so on) get added the same way, as
//! real translated fields actually need them, not spun up speculatively
//! ahead of that.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SectorId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubsectorId(pub u32);
