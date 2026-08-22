//! Level-geometry index newtypes (`docs/03_TRANSPILER.md`'s Memory Model
//! section): `sector_t`/`line_t`/`side_t`/... are bulk-allocated once per
//! level and never individually resized or freed mid-level, so a plain
//! index is enough -- no generation counter, unlike `Arena<T>`'s handles.
//!
//! Only `SectorId` exists so far, since it's the only one any translated
//! struct references yet (the four `p_spec.h` lighting-effect thinkers).
//! More (`LineId`, `SideId`, `VertexId`, ...) get added the same way, as
//! real translated fields actually need them -- not spun up speculatively
//! ahead of that.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SectorId(pub u32);
