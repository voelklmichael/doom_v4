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
//! Minimal on purpose: only `sectors`/`sides`/`players` exist so far,
//! since no translated function body has needed anything else yet. More
//! fields join the same way every other table in this codebase grew -- a
//! real body needs it, not speculatively ahead of that. `sides` joined
//! once `EV_DoPlat`'s own `sides[line->sidenum[0]].sector` needed
//! somewhere to resolve a bare (non-`&`) `sides[i]` index read into.
//! `players` joined once `EV_DoLockedDoor`'s own `p->cards[..]`/`p->
//! message` (a `player_t*` local, unwrapped at the point of use from its
//! `Option<PlayerId>`) needed somewhere to resolve a `PlayerId` into --
//! `[Player; MAXPLAYERS]`, not `Vec`, matching `runtime/player.rs`'s own
//! already-documented design (a plain, fixed-size, never-resized array,
//! unlike `sectors`/`sides`' genuinely per-level-sized tables).
//! `linetarget` joined once `A_Punch`/`A_Saw` (`p_pspr.c`) needed
//! somewhere to hold `p_local.h`'s own `extern mobj_t* linetarget;` --
//! genuine file-scope mutable game state (the last thing
//! `P_AimLineAttack` hit, corpus-wide, not a per-level table like
//! `sectors`/`sides` or a program-lifetime fixed array like `players`),
//! but still just one more piece of state a tick/action function's own
//! body needs to resolve through, so it lives here rather than as some
//! new kind of parameter -- a plain `Option<Handle<Thinker>>` field,
//! the same type `Mobj.target`/`.tracer` already use for "no target"
//! vs. a real live thinker.
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
    pub players: [Player; MAXPLAYERS],
    pub linetarget: Option<Handle<Thinker>>,
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
}

impl std::ops::Index<PlayerId> for World {
    type Output = Player;
    fn index(&self, id: PlayerId) -> &Player {
        &self.players[id.0 as usize]
    }
}

impl std::ops::IndexMut<PlayerId> for World {
    fn index_mut(&mut self, id: PlayerId) -> &mut Player {
        &mut self.players[id.0 as usize]
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
        assert!(rendered.contains("pub players: [Player; MAXPLAYERS],"));
        assert!(rendered.contains("pub linetarget: Option<Handle<Thinker>>,"));
        assert!(rendered.contains("impl std::ops::Index<SectorId> for World"));
        assert!(rendered.contains("impl std::ops::IndexMut<SectorId> for World"));
        assert!(rendered.contains("impl std::ops::Index<SideId> for World"));
        assert!(rendered.contains("impl std::ops::IndexMut<SideId> for World"));
        assert!(rendered.contains("impl std::ops::Index<PlayerId> for World"));
        assert!(rendered.contains("impl std::ops::IndexMut<PlayerId> for World"));
    }
}
