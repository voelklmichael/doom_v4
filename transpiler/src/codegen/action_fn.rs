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
}
