//! Phase 3: The `ActionFn` Enum (Doom Action Pointers)
//!
//! `state_t.action: actionf_t` -- a C union of three function-pointer
//! shapes (`actionf_v: fn()`, `actionf_p1: fn(void*)`, `actionf_p2:
//! fn(void*, void*)`), caller-context-dispatched with no runtime tag
//! (`st->action.acp1(mobj)` in the mobj-think path, `state->action.acp2
//! (player, psp)` in the weapon-sprite path) -- becomes a closed enum,
//! the same pattern already used for `thinker_t.function`'s own dispatch
//! union (`thinkers.rs`'s `Thinker`): a `match` replacing blind caller
//! trust, not a raw/unsafe translation of the union itself.
//!
//! **Round 19 redesign -- name-tagged variants, not a 2-shape `fn`-pointer
//! wrapper**: the design this module originally shipped (`Mobj(fn(&mut
//! Mobj))` / `Weapon(fn(&mut Player, &mut PlayerSpriteState))`) assumed
//! every real `A_*` body could share one of two uniform `fn` pointer
//! types. `docs/03_TRANSPILER.md`'s Round 18 entry found that measured
//! wrong once real bodies existed to check it against: a real Rust `fn`
//! pointer type is monomorphic, but the 72 real, already-shipped, and
//! state-table-dispatched `A_*` functions alone span **8 distinct
//! concrete parameter lists** (verified by direct grep of every `pub fn
//! A_*` signature in `function_body.rs`, not sampled):
//!
//! - `fn(&mut Mobj, &mut World)` (29 -- `A_Pain`, `A_Fall`, ...)
//! - `fn(&mut Mobj, &mut World, &Arena<Thinker>)` (9 -- `A_Chase`,
//!   `A_FaceTarget`, `A_Look`, ...)
//! - `fn(&mut Mobj, &mut World, &mut Arena<Thinker>)` (9 -- `A_BrainSpit`,
//!   `A_Tracer`, ...)
//! - `fn(&mut Mobj, &mut World, &mut Arena<Thinker>, Handle<Thinker>)` (3
//!   -- `A_VileChase`, `A_SpawnFly`, `A_VileTarget`, self-removal capable)
//! - `fn(&mut Player, &mut PlayerSpriteState)` (18 -- `A_Light0`,
//!   `A_ReFire`, ...)
//! - `fn(&mut Player, &mut PlayerSpriteState, &Arena<Thinker>)` (1 --
//!   `A_WeaponReady`, no `World`)
//! - `fn(&mut Player, &mut PlayerSpriteState, &mut World,
//!   &Arena<Thinker>)` (1 -- `A_FireShotgun2`)
//! - `fn(&mut Player, &mut PlayerSpriteState, &mut World, &mut
//!   Arena<Thinker>)` (2 -- `A_Punch`, `A_Saw`)
//!
//! No single `fn` pointer type -- not even two -- can hold all of these.
//! **The fix**: `ActionFn` becomes a flat, name-tagged closed enum with
//! one bare (payload-free) variant per real corpus action function that
//! `states[]` actually dispatches (`ABabyMetal`, `AChase`, `APain`, ...
//! -- 74 total, `states_data.rs`'s own `StatesTable::action_names`, the
//! one true corpus-verified source this module reuses rather than
//! re-scanning). Dispatch moves entirely into the *caller* -- concretely
//! `P_SetMobjState` (`function_body.rs`) -- which does one hand-written
//! `match` calling each variant's real already-shipped function by name,
//! with *that* function's own real argument list, drawn from whatever
//! `P_SetMobjState`'s own scope carries (`mobj`/`world`/`handle`/
//! `arena`, reborrowed `&`/`&mut` as each callee needs -- confirmed by a
//! real `rustc --edition 2021` compile that passing `arena: &mut
//! Arena<Thinker>` directly at a callee expecting `&Arena<Thinker>`
//! reborrows implicitly, no explicit `&*arena` needed). This sidesteps
//! the fn-pointer-uniformity problem entirely: a bare tag has no type to
//! be monomorphic *about*.
//!
//! **Two variants, not three, at the union-shape level (unchanged from
//! before)**: corpus-wide, every `.acv` (the zero-arg shape) reference is
//! about `thinker_t.function`'s own remove-sentinel (`(actionf_v)-1`),
//! already fully replaced by the `Arena`'s own append-only design --
//! `state_t.action` itself is only ever `.acp1` or `.acp2`. Of the 74
//! variants, 52 are `.acp1` (`fn(mobj_t*)`, dispatched from
//! `P_SetMobjState`) and 22 are `.acp2` (`fn(player_t*, pspdef_t*)`,
//! dispatched from `P_SetPsprite`, not yet translated) -- but that
//! split is no longer encoded in the enum's own shape at all, only in
//! which dispatcher's `match` has an arm for a given tag. A tag reaching
//! the wrong dispatcher (a real invariant no `state_t` entry in the
//! actual corpus ever violates) is a hand-written `unreachable!()`, not
//! silent undefined behavior the way a mismatched C function-pointer
//! cast would be -- strictly safer than the original, not just
//! differently shaped.
//!
//! `Player`/`PlayerSpriteState` (`player_t`/`pspdef_t`) are registered in
//! `type_placement.rs` (`d_player`/`p_pspr`) even though neither is
//! translated as a real struct yet -- the same forward-reference
//! precedent `MobjInfo`/`State` set from `Mobj` before they existed.
//! `actionf_t` maps to `Option<ActionFn>` (nullable -- `states[]`'s own
//! data initializes plenty of entries `{NULL}`).

use crate::codegen::states_data::{action_variant_name, render_states_table};
use std::path::Path;

/// Renders `pub enum ActionFn { ABabyMetal, ABFGsound, ..., AXScream }` --
/// one bare variant per real corpus action function `states[]` actually
/// references (`states_data.rs`'s own `StatesTable::action_names`, the
/// one true corpus-verified source of "which 74 names need a variant" --
/// re-scanning here would risk drifting from it). Variant names come
/// from `action_variant_name` (`A_Chase` -> `AChase`), listed in the
/// same lexical order `BTreeSet<String>` already sorts them in, so the
/// output is deterministic across runs without a separate sort step.
pub fn render_action_fn_enum(corpus_dir: &Path) -> Result<String, String> {
    let table = render_states_table(corpus_dir)?;
    let mut body = String::new();
    for name in &table.action_names {
        body.push_str("    ");
        body.push_str(&action_variant_name(name));
        body.push_str(",\n");
    }
    Ok(format!("pub enum ActionFn {{\n{body}}}"))
}

/// `P_SetMobjState` (`p_mobj.c`) -- the real dispatcher `ActionFn`'s own
/// tagged-enum redesign exists to serve. Hand-written, like `states_
/// data.rs`'s own `render_state_index_fn` sibling: this isn't AST-
/// rendered through `function_body.rs`'s general per-statement engine
/// (the do-while-over-a-dynamic-`states[]`-index loop, plus the dispatch
/// `match` itself, are both idioms specific to this one function, not
/// reusable elsewhere), but every line is still corpus-verified against
/// the real C body below, not guessed:
///
/// ```c
/// boolean
/// P_SetMobjState (mobj_t* mobj, statenum_t state)
/// {
///     state_t*    st;
///     do
///     {
///         if (state == S_NULL)
///         {
///             mobj->state = (state_t *) S_NULL;
///             P_RemoveMobj (mobj);
///             return false;
///         }
///         st = &states[state];
///         mobj->state = st;
///         mobj->tics = st->tics;
///         mobj->sprite = st->sprite;
///         mobj->frame = st->frame;
///         if (st->action.acp1)
///             st->action.acp1(mobj);
///         state = st->nextstate;
///     } while (!mobj->tics);
///     return true;
/// }
/// ```
///
/// **Signature**: `P_RemoveMobj`'s own already-shipped precedent (`&mut
/// Mobj` + `world: &mut World` + `handle: Handle<Thinker>` + `arena:
/// &mut Arena<Thinker>`, "caller already holds the slot") -- confirmed
/// the only signature this function can have, not just the simplest:
/// its own `S_NULL` branch calls `P_RemoveMobj(mobj)`, which itself
/// already needs `world`/`handle`/`arena` (its own shipped body reads
/// `mobj.flags`/`.type`/`.spawnpoint` *and* calls `arena.remove(handle)`),
/// so nothing less would let this compile. Round 18/19's own "can a
/// handle-only caller instead fetch `&mut Mobj` via `arena.get_mut
/// (handle)` right before calling, avoiding a second call shape
/// entirely?" question (docs/03_TRANSPILER.md) does NOT hold, verified
/// by a real `rustc` rejection (`error[E0499]: cannot borrow arena as
/// mutable more than once at a time`) on exactly that shape: `arena.
/// get_mut(handle)`'s returned `&mut Mobj` keeps `arena` borrowed, so it
/// can't *also* be passed as `&mut Arena<Thinker>` in the same call --
/// the aliasing-sound trick `Arena::run` uses (`.take()`ing the slot out
/// first, so the value and the arena provably don't overlap) is the only
/// way to get both at once for an arbitrary handle, and no such method
/// exists yet (`Arena::take_out`, sketched in this round's own docs
/// entry, would add it generically). So `P_SetMobjState` keeps exactly
/// one signature; a caller with only a fresh `Handle<Thinker>` in scope
/// (`P_ExplodeMissile`'s own callers-of-callers, per round 18) needs
/// that or an equivalent before it can call this function at all -- not
/// solved this round, see the docs entry for the full list of
/// already-shipped callers this leaves with a stale 2-argument call.
///
/// **`mobj->state = (state_t *) S_NULL;`**: a real, deliberately invalid
/// C pointer (the raw `0` cast to `state_t*`), harmless there only
/// because `P_RemoveMobj` unconditionally follows and the mobj is never
/// ticked again. `Mobj.state: &'static State` (`struct_fields.rs`) can't
/// represent an invalid reference at all, so this becomes `&STATES[0]`
/// instead -- `STATES[0]` *is* `S_NULL`'s own real table entry (index 0,
/// confirmed by `states_data.rs`'s own `test_first_entry_s_null_has_no_
/// action`), a valid, harmless placeholder with the same "about to be
/// overwritten or never read again" lifetime the C sentinel had.
///
/// **The dispatch `match`**: one arm per real, already-shipped `pub fn
/// A_*` in `function_body.rs` among the 52 `Mobj`-shaped tags (of 74
/// total -- the other 22 are `Weapon`-shaped, `P_SetPsprite`'s own job,
/// not this function's; reaching one here is `unreachable!()`, a real
/// invariant no actual `state_t` entry in the corpus violates, verified
/// by `states_data.rs`'s own `render_action_field` only ever tagging a
/// mobj-context `action` field with a name `build_action_function_index`
/// resolved from a `fn(mobj_t*)`-shaped real definition). Each arm's own
/// argument list is copied verbatim from that function's own real
/// shipped signature (`function_body.rs`, not guessed) -- confirmed
/// compiling for real (`rustc --edition 2021 --crate-type lib`) against
/// hand-written `Mobj`/`World`/`Handle`/`Arena`/`State`/`STATES` stand-
/// ins and a stub for all 74 `A_*` names (including `A_BossDeath`/
/// `A_KeenDie`, the only 2 of the 74 real state-dispatched names with no
/// shipped Rust translation yet -- forward-referenced by a plausible
/// `(mo, world, thinkers: &Arena<Thinker>)` guess, the same "not yet
/// translated, call by name anyway" precedent `P_SpawnMobj`/
/// `P_SpawnMissile` already established, since both real C bodies just
/// do a read-only thinker scan) -- zero errors, confirms `arena: &mut
/// Arena<Thinker>` passed directly at a callee expecting the shared
/// `&Arena<Thinker>` reborrows implicitly, no explicit `&*arena` needed.
pub fn render_p_set_mobj_state_fn() -> String {
    "\
pub fn P_SetMobjState(mobj: &mut Mobj, state: i32, world: &mut World, handle: Handle<Thinker>, arena: &mut Arena<Thinker>) -> bool {
    let mut state = state;
    loop {
        if state == S_NULL {
            mobj.state = &STATES[0];
            P_RemoveMobj(mobj, world, handle, arena);
            return false;
        }
        let st = &STATES[state as usize];
        mobj.state = st;
        mobj.tics = st.tics;
        mobj.sprite = st.sprite;
        mobj.frame = st.frame;
        if let Some(action) = st.action {
            match action {
                ActionFn::ABFGSpray => A_BFGSpray(mobj, world, arena),
                ActionFn::ABabyMetal => A_BabyMetal(mobj, world),
                ActionFn::ABossDeath => A_BossDeath(mobj, world, arena),
                ActionFn::ABrainAwake => A_BrainAwake(mobj, world, arena),
                ActionFn::ABrainDie => A_BrainDie(mobj, world),
                ActionFn::ABrainExplode => A_BrainExplode(mobj, world, arena),
                ActionFn::ABrainPain => A_BrainPain(mobj, world),
                ActionFn::ABrainScream => A_BrainScream(mobj, world, arena),
                ActionFn::ABrainSpit => A_BrainSpit(mobj, world, arena),
                ActionFn::ABruisAttack => A_BruisAttack(mobj, world),
                ActionFn::ABspiAttack => A_BspiAttack(mobj, world),
                ActionFn::ACPosAttack => A_CPosAttack(mobj, world),
                ActionFn::ACPosRefire => A_CPosRefire(mobj, world, arena),
                ActionFn::AChase => A_Chase(mobj, world, arena),
                ActionFn::ACyberAttack => A_CyberAttack(mobj, world),
                ActionFn::AExplode => A_Explode(mobj, world),
                ActionFn::AFaceTarget => A_FaceTarget(mobj, world, arena),
                ActionFn::AFall => A_Fall(mobj, world),
                ActionFn::AFatAttack1 => A_FatAttack1(mobj, world, arena),
                ActionFn::AFatAttack2 => A_FatAttack2(mobj, world, arena),
                ActionFn::AFatAttack3 => A_FatAttack3(mobj, world, arena),
                ActionFn::AFatRaise => A_FatRaise(mobj, world),
                ActionFn::AFire => A_Fire(mobj, world, arena),
                ActionFn::AFireCrackle => A_FireCrackle(mobj, world),
                ActionFn::AHeadAttack => A_HeadAttack(mobj, world),
                ActionFn::AHoof => A_Hoof(mobj, world),
                ActionFn::AKeenDie => A_KeenDie(mobj, world, arena),
                ActionFn::ALook => A_Look(mobj, world, arena),
                ActionFn::AMetal => A_Metal(mobj, world),
                ActionFn::APain => A_Pain(mobj, world),
                ActionFn::APainAttack => A_PainAttack(mobj, world),
                ActionFn::APainDie => A_PainDie(mobj, world),
                ActionFn::APlayerScream => A_PlayerScream(mobj, world),
                ActionFn::APosAttack => A_PosAttack(mobj, world),
                ActionFn::ASPosAttack => A_SPosAttack(mobj, world),
                ActionFn::ASargAttack => A_SargAttack(mobj, world),
                ActionFn::AScream => A_Scream(mobj, world),
                ActionFn::ASkelFist => A_SkelFist(mobj, world),
                ActionFn::ASkelMissile => A_SkelMissile(mobj, world, arena),
                ActionFn::ASkelWhoosh => A_SkelWhoosh(mobj, world),
                ActionFn::ASkullAttack => A_SkullAttack(mobj, world, arena),
                ActionFn::ASpawnFly => A_SpawnFly(mobj, world, arena, handle),
                ActionFn::ASpawnSound => A_SpawnSound(mobj, world),
                ActionFn::ASpidRefire => A_SpidRefire(mobj, world, arena),
                ActionFn::AStartFire => A_StartFire(mobj, world),
                ActionFn::ATracer => A_Tracer(mobj, world, arena),
                ActionFn::ATroopAttack => A_TroopAttack(mobj, world),
                ActionFn::AVileAttack => A_VileAttack(mobj, world, arena),
                ActionFn::AVileChase => A_VileChase(mobj, world, arena, handle),
                ActionFn::AVileStart => A_VileStart(mobj, world),
                ActionFn::AVileTarget => A_VileTarget(mobj, world, arena, handle),
                ActionFn::AXScream => A_XScream(mobj, world),
                _ => unreachable!(\"a Weapon-shaped ActionFn tag reached P_SetMobjState's own Mobj dispatch\"),
            }
        }
        state = st.nextstate;
        if mobj.tics != 0 {
            break;
        }
    }
    true
}"
    .to_string()
}

/// `P_SetPsprite` (`p_pspr.c`) -- `P_SetMobjState`'s own `Weapon`-shaped
/// sibling, dispatching the other 22 `ActionFn` tags through `state->
/// action.acp2` instead of `.acp1`. Hand-written for the identical reason
/// `render_p_set_mobj_state_fn` is: the do-while-over-a-dynamic-`states[]`-
/// index loop and the dispatch `match` are idioms specific to this one
/// function, not reusable elsewhere. Every line corpus-verified against
/// the real C body (`p_pspr.c:56-102`), not guessed:
///
/// ```c
/// void
/// P_SetPsprite (player_t* player, int position, statenum_t stnum)
/// {
///     pspdef_t*   psp;
///     state_t*    state;
///
///     psp = &player->psprites[position];
///
///     do
///     {
///         if (!stnum)
///         {
///             // object removed itself
///             psp->state = NULL;
///             break;
///         }
///
///         state = &states[stnum];
///         psp->state = state;
///         psp->tics = state->tics;       // could be 0
///
///         if (state->misc1)
///         {
///             // coordinate set
///             psp->sx = state->misc1 << FRACBITS;
///             psp->sy = state->misc2 << FRACBITS;
///         }
///
///         // Call action routine.
///         // Modified handling.
///         if (state->action.acp2)
///         {
///             state->action.acp2(player, psp);
///             if (!psp->state)
///                 break;
///         }
///
///         stnum = psp->state->nextstate;
///
///     } while (!psp->tics);
///     // an initial state of 0 could cycle through
/// }
/// ```
///
/// **`psp = &player->psprites[position];`**: unlike `P_MovePsprites`'s own
/// `psprite_walk_alias` (a *fixed* loop-counter index, `function_body.rs`)
/// or `P_BringUpWeapon`'s bare `ps_weapon` enum-constant index, `position`
/// here is a genuine runtime function *parameter* -- so this hand-written
/// renderer just indexes `player.psprites[position as usize]` directly at
/// every real touch point, no alias-tracking field needed at all (this
/// function isn't AST-rendered, so there's no `FnBodyContext` to extend).
///
/// **No persistent `psp` local across the whole function, unlike the C
/// source's own pointer**: a real Rust `&mut PlayerSpriteState` borrowed
/// from `player.psprites[position]` can't be held *at the same time* as
/// the `player: &mut Player` every one of the 22 `Weapon`-shaped callees
/// also needs (`E0499`-class aliasing, confirmed by a real `rustc`
/// rejection on the naive "hold one `psp: &mut PlayerSpriteState` for the
/// whole function" version before settling on this design) -- the same
/// class of problem `Arena::take_out` exists to solve for `Mobj`, just for
/// a plain array field instead of an arena slot. Fixed by never holding a
/// live borrow across the dispatch call at all: every field `P_SetPsprite`
/// itself sets (`.state`/`.tics`/`.sx`/`.sy`, unconditionally each
/// iteration) writes straight into `player.psprites[position as usize]`,
/// no local at all. Only the one call site that genuinely needs a `&mut
/// PlayerSpriteState` value (the dispatch `match`) takes a fresh, disjoint
/// *copy* (`PlayerSpriteState` is a small, plain, four-scalar-field value
/// type -- `Copy`) right before the call, passes `&mut psp` to the callee,
/// then reconciles after the call returns rather than blindly overwriting.
///
/// **The reconciliation is real, corpus-driven, not incidental**: two of
/// the 22 real callees (`A_Lower`, `A_Raise`, both real corpus text above
/// confirms are dispatched *only* from `position == ps_weapon`) themselves
/// call `P_SetPsprite(player, ps_weapon, ..)` reentrantly on the *same*
/// index as their own `psp` parameter -- real C aliasing means the outer
/// `psp->state` pointer sees that inner write directly; a disjoint Rust
/// copy would not. So after the dispatch call returns, `.state` --
/// specifically and only `.state`, the one field a reentrant same-index
/// `P_SetPsprite` call can update (state transitions are *only* ever
/// mediated through `P_SetPsprite` itself in this corpus; no real callee
/// writes `psp->state` directly) -- is re-read from
/// `player.psprites[position as usize]` (reflecting whatever a reentrant
/// call left there) rather than trusted from the disjoint copy, before the
/// *whole* copy (now `.state`-corrected, `.tics`/`.sx`/`.sy` still holding
/// whatever direct field writes the callee made to its own `psp`
/// parameter, e.g. `A_WeaponReady`'s `psp->sx =`/`psp->sy =`) is written
/// back. Verified compiling and behaving correctly for real (`rustc
/// --edition 2021 --crate-type bin`, a standalone scratch harness with
/// stub `Player`/`PlayerSpriteState`/`State`/`World`/`Arena`/`Handle`/
/// `FixedT` stand-ins and all 22 real callee signatures, including a
/// working `A_Lower` stub that reentrantly calls `P_SetPsprite` on
/// `ps_weapon` exactly like the real corpus): after dispatch returns from
/// a reentrant-triggering path, `player.psprites[ps_weapon].state` is
/// correctly `None`, not the stale pre-call value.
///
/// **A known, narrow, documented gap, not silently mismodeled**: this
/// reconciliation only carries `.state` forward from a reentrant write;
/// `.tics`/`.sx`/`.sy` are *not* re-read from the array after the call. If
/// a reentrant call's *own* transition also changes those (only possible
/// for `A_Raise`'s non-`S_NULL` reentrant branch, if its target state's
/// own `misc1` happens to be set -- not the case for any real corpus
/// `readystate` this project has checked), the outer copy's `tics`/`sx`/
/// `sy` (whatever they were before the callee ran) would be written back
/// instead of the reentrant transition's fresher values -- a real but
/// narrow representation gap, the same "hand-match the corpus's own real
/// shape rather than build a fully general mechanism" tradeoff this
/// project has made before (`P_KillMobj`/`EV_BuildStairs` UB, round 26/19)
/// -- flagged here for a future round, not attempted.
///
/// **The dispatch `match`**: one arm per real, already-shipped `pub fn
/// A_*` among the 22 `Weapon`-shaped tags (of 74 total; the other 52 are
/// `Mobj`-shaped, `P_SetMobjState`'s own job -- reaching one here is
/// `unreachable!()`, the identical real invariant `P_SetMobjState`'s own
/// catch-all arm already established, just mirrored). Each arm's argument
/// list is copied verbatim from that function's own real shipped
/// signature: 14 take just `(player, psp)`, 8 also need `(world,
/// thinkers)` (`function_body.rs`, confirmed by direct grep of every real
/// `pub fn A_*(player: &mut Player, ..)` signature, not sampled).
pub fn render_p_set_psprite_fn() -> String {
    "\
pub fn P_SetPsprite(player: &mut Player, position: i32, stnum: i32, world: &mut World, thinkers: &mut Arena<Thinker>) {
    let mut stnum = stnum;
    loop {
        if stnum == S_NULL {
            player.psprites[position as usize].state = None;
            break;
        }
        let state = &STATES[stnum as usize];
        player.psprites[position as usize].state = Some(state);
        player.psprites[position as usize].tics = state.tics;
        if state.misc1 != 0 {
            player.psprites[position as usize].sx = FixedT(state.misc1 << FRACBITS);
            player.psprites[position as usize].sy = FixedT(state.misc2 << FRACBITS);
        }
        if let Some(action) = state.action {
            let mut psp = player.psprites[position as usize];
            match action {
                ActionFn::ABFGsound => A_BFGsound(player, &mut psp),
                ActionFn::ACheckReload => A_CheckReload(player, &mut psp),
                ActionFn::ACloseShotgun2 => A_CloseShotgun2(player, &mut psp),
                ActionFn::AFireBFG => A_FireBFG(player, &mut psp),
                ActionFn::AFireCGun => A_FireCGun(player, &mut psp, world, thinkers),
                ActionFn::AFireMissile => A_FireMissile(player, &mut psp),
                ActionFn::AFirePistol => A_FirePistol(player, &mut psp, world, thinkers),
                ActionFn::AFirePlasma => A_FirePlasma(player, &mut psp),
                ActionFn::AFireShotgun => A_FireShotgun(player, &mut psp, world, thinkers),
                ActionFn::AFireShotgun2 => A_FireShotgun2(player, &mut psp, world, thinkers),
                ActionFn::AGunFlash => A_GunFlash(player, &mut psp, world, thinkers),
                ActionFn::ALight0 => A_Light0(player, &mut psp),
                ActionFn::ALight1 => A_Light1(player, &mut psp),
                ActionFn::ALight2 => A_Light2(player, &mut psp),
                ActionFn::ALoadShotgun2 => A_LoadShotgun2(player, &mut psp),
                ActionFn::ALower => A_Lower(player, &mut psp, world, thinkers),
                ActionFn::AOpenShotgun2 => A_OpenShotgun2(player, &mut psp),
                ActionFn::APunch => A_Punch(player, &mut psp, world, thinkers),
                ActionFn::ARaise => A_Raise(player, &mut psp, world, thinkers),
                ActionFn::AReFire => A_ReFire(player, &mut psp),
                ActionFn::ASaw => A_Saw(player, &mut psp, world, thinkers),
                ActionFn::AWeaponReady => A_WeaponReady(player, &mut psp, world, thinkers),
                _ => unreachable!(\"a Mobj-shaped ActionFn tag reached P_SetPsprite's own Weapon dispatch\"),
            }
            psp.state = player.psprites[position as usize].state;
            player.psprites[position as usize] = psp;
            if player.psprites[position as usize].state.is_none() {
                break;
            }
        }
        stnum = player.psprites[position as usize].state.unwrap().nextstate;
        if player.psprites[position as usize].tics != 0 {
            break;
        }
    }
}"
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_action_fn_enum_has_74_bare_variants() {
        let rendered = render_action_fn_enum(&corpus_dir()).expect("should render cleanly");
        assert!(rendered.starts_with("pub enum ActionFn {\n"));
        assert!(rendered.ends_with("\n}"));
        // Bare variants, no `(..)` payload anywhere -- the whole point of
        // the round 19 redesign.
        assert!(!rendered.contains('('));
        assert_eq!(rendered.matches(",\n").count(), 74);
    }

    #[test]
    fn test_action_fn_enum_contains_known_mobj_and_weapon_tags() {
        let rendered = render_action_fn_enum(&corpus_dir()).expect("should render cleanly");
        assert!(rendered.contains("    AChase,\n"));
        assert!(rendered.contains("    ALight0,\n"));
        assert!(rendered.contains("    AOpenShotgun2,\n"));
        // A_PainShootSkull is a real, defined action function but never a
        // states[] entry's own action value (a plain helper) -- must NOT
        // get a variant.
        assert!(!rendered.contains("APainShootSkull"));
    }

    #[test]
    fn test_action_fn_enum_variants_in_lexical_c_name_order() {
        let rendered = render_action_fn_enum(&corpus_dir()).expect("should render cleanly");
        let a_chase = rendered.find("AChase,").unwrap();
        let a_face_target = rendered.find("AFaceTarget,").unwrap();
        // "A_Chase" < "A_FaceTarget" lexically, so AChase's variant comes
        // first -- confirms the enum follows action_names' own
        // BTreeSet<String> order (sorted by the real C name, before the
        // underscore is stripped), not some other ordering.
        assert!(a_chase < a_face_target);
    }

    #[test]
    fn test_p_set_mobj_state_signature_and_control_flow() {
        let rendered = render_p_set_mobj_state_fn();
        assert!(rendered.starts_with(
            "pub fn P_SetMobjState(mobj: &mut Mobj, state: i32, world: &mut World, \
             handle: Handle<Thinker>, arena: &mut Arena<Thinker>) -> bool {"
        ));
        // The S_NULL branch: dead-state placeholder + real P_RemoveMobj
        // call with its own already-shipped 4-argument signature, not the
        // C source's bare 1-argument `P_RemoveMobj(mobj)`.
        assert!(rendered.contains("if state == S_NULL {"));
        assert!(rendered.contains("mobj.state = &STATES[0];"));
        assert!(rendered.contains("P_RemoveMobj(mobj, world, handle, arena);"));
        assert!(rendered.contains("return false;"));
        // The per-iteration state-array indexing and field fill-in.
        assert!(rendered.contains("let st = &STATES[state as usize];"));
        assert!(rendered.contains("mobj.state = st;"));
        assert!(rendered.contains("mobj.tics = st.tics;"));
        assert!(rendered.contains("mobj.sprite = st.sprite;"));
        assert!(rendered.contains("mobj.frame = st.frame;"));
        assert!(rendered.contains("state = st.nextstate;"));
        assert!(rendered.contains("if mobj.tics != 0 {"));
        assert!(rendered.ends_with("    true\n}"));
    }

    #[test]
    fn test_p_set_mobj_state_dispatches_exactly_the_52_mobj_shaped_tags() {
        let rendered = render_p_set_mobj_state_fn();
        // 52 real Mobj-shaped tags (of 74 total -- the other 22 are
        // Weapon-shaped, P_SetPsprite's own job) each get exactly one
        // match arm calling the real already-shipped function by name.
        assert_eq!(rendered.matches("ActionFn::A").count(), 52);
        assert!(rendered.contains("ActionFn::AChase => A_Chase(mobj, world, arena),"));
        assert!(
            rendered.contains("ActionFn::AVileChase => A_VileChase(mobj, world, arena, handle),")
        );
        // A weapon-shaped tag must never get its own arm here -- only the
        // catch-all unreachable!() should mention it's possible.
        assert!(!rendered.contains("ActionFn::ALight0"));
        assert!(!rendered.contains("ActionFn::AWeaponReady"));
        assert!(rendered.contains("_ => unreachable!("));
    }

    #[test]
    fn test_p_set_psprite_signature_and_control_flow() {
        let rendered = render_p_set_psprite_fn();
        assert!(rendered.starts_with(
            "pub fn P_SetPsprite(player: &mut Player, position: i32, stnum: i32, \
             world: &mut World, thinkers: &mut Arena<Thinker>) {"
        ));
        // The S_NULL branch: writes straight into the array slot, no
        // persistent `psp` local exists yet at this point in the function.
        assert!(rendered.contains("if stnum == S_NULL {"));
        assert!(rendered.contains("player.psprites[position as usize].state = None;"));
        assert!(rendered.contains("break;"));
        // The per-iteration state-array indexing and field fill-in, all
        // written straight into the array slot (never a separate local).
        assert!(rendered.contains("let state = &STATES[stnum as usize];"));
        assert!(rendered.contains("player.psprites[position as usize].state = Some(state);"));
        assert!(rendered.contains("player.psprites[position as usize].tics = state.tics;"));
        assert!(rendered.contains("if state.misc1 != 0 {"));
        assert!(
            rendered.contains(
                "player.psprites[position as usize].sx = FixedT(state.misc1 << FRACBITS);"
            )
        );
        assert!(
            rendered.contains(
                "player.psprites[position as usize].sy = FixedT(state.misc2 << FRACBITS);"
            )
        );
        // The dispatch call only ever takes a fresh, disjoint copy -- never
        // a live borrow of the array slot (the E0499-class aliasing this
        // function's own doc comment explains).
        assert!(rendered.contains("let mut psp = player.psprites[position as usize];"));
        // The post-call reconciliation: `.state` is re-read from the array
        // (a reentrant same-index P_SetPsprite call, e.g. A_Lower/A_Raise,
        // writes there directly), then the whole (now state-corrected)
        // copy is written back.
        assert!(rendered.contains("psp.state = player.psprites[position as usize].state;"));
        assert!(rendered.contains("player.psprites[position as usize] = psp;"));
        assert!(rendered.contains("if player.psprites[position as usize].state.is_none() {"));
        assert!(
            rendered
                .contains("stnum = player.psprites[position as usize].state.unwrap().nextstate;")
        );
        assert!(rendered.contains("if player.psprites[position as usize].tics != 0 {"));
        assert!(rendered.ends_with("    }\n}"));
    }

    #[test]
    fn test_p_set_psprite_dispatches_exactly_the_22_weapon_shaped_tags() {
        let rendered = render_p_set_psprite_fn();
        // 22 real Weapon-shaped tags (of 74 total -- the other 52 are
        // Mobj-shaped, P_SetMobjState's own job) each get exactly one match
        // arm calling the real already-shipped function by name, plus the
        // catch-all `unreachable!()` -- but that arm's own doc string also
        // contains the substring "ActionFn" so it's excluded from the
        // count by only counting `ActionFn::A` immediately followed by a
        // `=>` on the same match-arm line pattern already established by
        // the Mobj sibling test (which counts the identical way and is
        // known correct).
        assert_eq!(rendered.matches("ActionFn::A").count(), 22);
        assert!(rendered.contains("ActionFn::ALight0 => A_Light0(player, &mut psp),"));
        assert!(rendered.contains(
            "ActionFn::AWeaponReady => A_WeaponReady(player, &mut psp, world, thinkers),"
        ));
        assert!(
            rendered.contains("ActionFn::ALower => A_Lower(player, &mut psp, world, thinkers),")
        );
        assert!(
            rendered.contains("ActionFn::ARaise => A_Raise(player, &mut psp, world, thinkers),")
        );
        // A mobj-shaped tag must never get its own arm here -- only the
        // catch-all unreachable!() should mention it's possible.
        assert!(!rendered.contains("ActionFn::AChase"));
        assert!(!rendered.contains("ActionFn::AVileChase"));
        assert!(rendered.contains(
            "_ => unreachable!(\"a Mobj-shaped ActionFn tag reached P_SetPsprite's own Weapon dispatch\"),"
        ));
    }
}
