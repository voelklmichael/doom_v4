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
//! `subsectors` joined once `A_Look`'s own `actor->subsector->sector->
//! soundtarget` needed somewhere to resolve `Mobj.subsector`
//! (`SubsectorId`) into, the same per-level `Vec`-backed shape as
//! `sectors`/`sides` (`r_defs.h`'s `subsector_t` is bulk-allocated once
//! per level, same as every other geometry struct).
//! `braintargets`/`numbraintargets`/`braintargeton` joined once
//! `A_BrainSpit` (`p_enemy.c`) needed somewhere to hold its own file-
//! scope `mobj_t* braintargets[32]; int numbraintargets; int
//! braintargeton;` -- the same "genuine mutable game state, not a per-
//! level table or a fixed-size player array" category `linetarget`
//! already established, just a fixed-size array of targets (`A_BrainAwake`
//! populates it; `A_BrainSpit` only ever reads/advances it) instead of a
//! single value.
//! `a_brain_spit_easy` joined once `A_BrainSpit`'s own `static int easy =
//! 0;` needed somewhere to live: a C function-local `static` persists
//! across every call, unlike an ordinary local (which a Rust `let`
//! re-initializes fresh each time this function's own body runs) --
//! genuinely the same *kind* of state `linetarget`/`braintargeton` are
//! (mutable, outliving any one call), just scoped to one function's own
//! textual visibility in the original C rather than visible file-wide, so
//! it gets the same treatment (a `World` field) with the function's own
//! name folded into the field name to keep it from colliding with some
//! future unrelated function's own same-named `static` local (`docs/
//! 03_TRANSPILER.md`'s `render_fn`/`FnBodyContext::static_locals`, in
//! `function_body.rs`, computes this exact name and skips ever rendering
//! the original declaration's own `let`). Its `= 0` initializer isn't
//! wired up here -- nothing in this codebase constructs a real `World`
//! value yet (disk emission/level loading don't exist), so there's
//! nothing to initialize *to* yet; whichever future piece builds that
//! constructor is responsible for seeding this at `0`, matching the C
//! initializer's own "runs once, at program start" semantics.
//! `corpsehit`/`vileobj`/`viletryx`/`viletryy` joined once `PIT_VileCheck`/
//! `A_VileChase` (`p_enemy.c`) needed somewhere to hold their own shared
//! file-scope `mobj_t* corpsehit; mobj_t* vileobj; fixed_t viletryx;
//! fixed_t viletryy;` -- the same "genuine mutable game state a callback
//! and its caller communicate through" category `linetarget`/
//! `braintargets` already established, just a pair of single values
//! (`corpsehit`/`vileobj`, `Option<Handle<Thinker>>`) and a pair of
//! coordinates (`viletryx`/`viletryy`, `FixedT`) instead of one value or
//! an array.
//! `crushchange`/`nofit` joined once `PIT_ChangeSector` (`p_map.c`)
//! needed somewhere to hold `P_ChangeSector`'s own file-scope `boolean
//! crushchange; boolean nofit;` -- the same "genuine mutable game state a
//! callback and its caller communicate through" category `corpsehit`/
//! `vileobj` already established, just plain `bool` (the corpus's own
//! `boolean` typedef, not `Option<Handle<Thinker>>` or `FixedT`) since
//! neither is ever null or fixed-point.
//! `itemrespawnque`/`itemrespawntime`/`iquehead`/`iquetail` joined once
//! `P_RemoveMobj` (`p_mobj.c`) needed somewhere to hold its own file-scope
//! `mapthing_t itemrespawnque[ITEMQUESIZE]; int itemrespawntime[ITEMQUESIZE];
//! int iquehead; int iquetail;` (`ITEMQUESIZE` is a `#define`d `128`,
//! `p_local.h`, confirmed by direct read -- this parser never expands
//! macros, so the array length is written out literally here the same
//! way every other hand-rendered `World` field already is) -- a circular
//! queue of respawn points for special items, the same "genuine mutable
//! game state a callback and its caller communicate through" category
//! `corpsehit`/`vileobj`/`crushchange`/`nofit` already established, just
//! a pair of fixed-size arrays (one of `MapThing` values, one of `i32`
//! timestamps) alongside their own head/tail counters instead of a
//! single value or a shorter special-purpose array.
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
    pub subsectors: Vec<Subsector>,
    pub players: [Player; MAXPLAYERS],
    pub linetarget: Option<Handle<Thinker>>,
    pub braintargets: [Option<Handle<Thinker>>; 32],
    pub numbraintargets: i32,
    pub braintargeton: i32,
    pub a_brain_spit_easy: i32,
    pub corpsehit: Option<Handle<Thinker>>,
    pub vileobj: Option<Handle<Thinker>>,
    pub viletryx: FixedT,
    pub viletryy: FixedT,
    pub crushchange: bool,
    pub nofit: bool,
    pub itemrespawnque: [MapThing; 128],
    pub itemrespawntime: [i32; 128],
    pub iquehead: i32,
    pub iquetail: i32,
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

impl std::ops::Index<SubsectorId> for World {
    type Output = Subsector;
    fn index(&self, id: SubsectorId) -> &Subsector {
        &self.subsectors[id.0 as usize]
    }
}

impl std::ops::IndexMut<SubsectorId> for World {
    fn index_mut(&mut self, id: SubsectorId) -> &mut Subsector {
        &mut self.subsectors[id.0 as usize]
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
        assert!(rendered.contains("pub subsectors: Vec<Subsector>,"));
        assert!(rendered.contains("pub players: [Player; MAXPLAYERS],"));
        assert!(rendered.contains("pub linetarget: Option<Handle<Thinker>>,"));
        assert!(rendered.contains("pub braintargets: [Option<Handle<Thinker>>; 32],"));
        assert!(rendered.contains("pub numbraintargets: i32,"));
        assert!(rendered.contains("pub braintargeton: i32,"));
        assert!(rendered.contains("pub a_brain_spit_easy: i32,"));
        assert!(rendered.contains("pub corpsehit: Option<Handle<Thinker>>,"));
        assert!(rendered.contains("pub vileobj: Option<Handle<Thinker>>,"));
        assert!(rendered.contains("pub viletryx: FixedT,"));
        assert!(rendered.contains("pub viletryy: FixedT,"));
        assert!(rendered.contains("pub crushchange: bool,"));
        assert!(rendered.contains("pub nofit: bool,"));
        assert!(rendered.contains("pub itemrespawnque: [MapThing; 128],"));
        assert!(rendered.contains("pub itemrespawntime: [i32; 128],"));
        assert!(rendered.contains("pub iquehead: i32,"));
        assert!(rendered.contains("pub iquetail: i32,"));
        assert!(rendered.contains("impl std::ops::Index<SectorId> for World"));
        assert!(rendered.contains("impl std::ops::IndexMut<SectorId> for World"));
        assert!(rendered.contains("impl std::ops::Index<SideId> for World"));
        assert!(rendered.contains("impl std::ops::IndexMut<SideId> for World"));
        assert!(rendered.contains("impl std::ops::Index<SubsectorId> for World"));
        assert!(rendered.contains("impl std::ops::IndexMut<SubsectorId> for World"));
        assert!(rendered.contains("impl std::ops::Index<PlayerId> for World"));
        assert!(rendered.contains("impl std::ops::IndexMut<PlayerId> for World"));
    }
}
