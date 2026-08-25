//! Phase 3: `World` -- the Tick Loop's Level Context
//!
//! `T_FireFlicker` (`docs/03_TRANSPILER.md`, function-body transpilation
//! -- see `function_body.rs`) is the first real tick-function body
//! translated, and it immediately needs to resolve
//! `flick->sector->lightlevel`: a cross-reference, since `FireFlicker.
//! sector` is a `SectorId` (a plain index, per the memory-model
//! decision), not a real pointer. Getting from an index back to a real
//! `&mut Sector` needs somewhere to look it up -- `World` is that
//! somewhere, a level's actual storage, indexed by the same `*Id`
//! newtypes `runtime::geometry` already defines.
//!
//! Minimal on purpose: only `sectors`/`sides` exist so far, since no
//! translated function body has needed anything else yet. More fields
//! join the same way every other table in this codebase grew -- a real
//! body needs it, not speculatively ahead of that. `sides` joined once
//! `EV_DoPlat`'s own `sides[line->sidenum[0]].sector` needed somewhere to
//! resolve a bare (non-`&`) `sides[i]` index read into.
//!
//! Lives in `p_tick` (`type_placement.rs`), alongside `Thinker`: both are
//! new, invented infrastructure with no direct corpus counterpart, and
//! both exist to replace the same C mechanism -- the tick loop's own
//! bookkeeping (`P_RunThinkers`'s intrusive list, and the raw pointers
//! its called functions dereference).

/// Rendered as literal text, the same way `thinkers.rs` renders `Thinker`
/// -- not mapped from any single corpus struct, so there's no field list
/// to drive a mechanical `render_struct` call.
pub fn render_world_struct() -> String {
    "\
pub struct World {
    pub sectors: Vec<Sector>,
    pub sides: Vec<Side>,
}

impl std::ops::Index<SectorId> for World {
    type Output = Sector;
    fn index(&self, id: SectorId) -> &Sector {
        &self.sectors[id.0 as usize]
    }
}

impl std::ops::IndexMut<SectorId> for World {
    fn index_mut(&mut self, id: SectorId) -> &mut Sector {
        &mut self.sectors[id.0 as usize]
    }
}

impl std::ops::Index<SideId> for World {
    type Output = Side;
    fn index(&self, id: SideId) -> &Side {
        &self.sides[id.0 as usize]
    }
}

impl std::ops::IndexMut<SideId> for World {
    fn index_mut(&mut self, id: SideId) -> &mut Side {
        &mut self.sides[id.0 as usize]
    }
}"
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renders_expected_shape() {
        let rendered = render_world_struct();
        assert!(rendered.starts_with("pub struct World {"));
        assert!(rendered.contains("pub sectors: Vec<Sector>,"));
        assert!(rendered.contains("pub sides: Vec<Side>,"));
        assert!(rendered.contains("impl std::ops::Index<SectorId> for World"));
        assert!(rendered.contains("impl std::ops::IndexMut<SectorId> for World"));
        assert!(rendered.contains("impl std::ops::Index<SideId> for World"));
        assert!(rendered.contains("impl std::ops::IndexMut<SideId> for World"));
    }
}
