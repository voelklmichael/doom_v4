//! Phase 3: The `ActionFn` Enum (Doom Action Pointers)
//!
//! `state_t.action: actionf_t` -- a C union of three function-pointer
//! shapes (`actionf_v: fn()`, `actionf_p1: fn(void*)`, `actionf_p2:
//! fn(void*, void*)`), caller-context-dispatched with no runtime tag
//! (`st->action.acp1(mobj)` in the mobj-think path, `state->action.acp2
//! (player, psp)` in the weapon-sprite path) -- becomes a closed 2-variant
//! enum, the same pattern already used for `thinker_t.function`'s own
//! dispatch union (`thinkers.rs`'s `Thinker`): a `match` replacing blind
//! caller trust, not a raw/unsafe translation of the union itself.
//!
//! **Two variants, not three**: corpus-wide, every `.acv` (the zero-arg
//! shape) reference is about `thinker_t.function`'s own remove-sentinel
//! (`(actionf_v)-1`), already fully replaced by the `Arena`'s own
//! append-only design -- `state_t.action` itself is only ever `.acp1` or
//! `.acp2`. Confirmed against all 132 real `A_*` action functions
//! (`info.c`/`p_enemy.c`/`p_pspr.c`): every one is either `fn(mobj_t*)`
//! (the large majority -- `A_Chase`, `A_FaceTarget`, ...) or
//! `fn(player_t*, pspdef_t*)` (a handful -- `A_ReFire`, `A_Light0/1/2`,
//! `A_WeaponReady`, `A_Lower`, `A_Raise`, the weapon-sprite state chain).
//!
//! **Deliberately loose on the exact parameter list**, same reasoning as
//! `thinkers.rs`'s own dispatch stub: whether a real `A_Chase` body needs
//! more than `&mut Mobj` (e.g. `&mut Arena<Thinker>`, to look up other
//! mobjs) isn't decidable without a real body to check it against --
//! pinning it down now risks the same "guessed wrong, had to revise"
//! story the arena's own first design had. Starts with the minimal
//! literal translation of the C parameter types.

/// Renders `pub enum ActionFn { Mobj(fn(&mut Mobj)), Weapon(fn(&mut
/// Player, &mut PlayerSpriteState)) }` -- shape only; `Player`/
/// `PlayerSpriteState` aren't translated as real structs yet (see this
/// module's own docs and `type_placement.rs`), the same forward-reference
/// pattern `Mobj`'s own fields already used for `MobjInfo`/`State` before
/// they existed.
pub fn render_action_fn_enum() -> String {
    "pub enum ActionFn {\n    \
     Mobj(fn(&mut Mobj)),\n    \
     Weapon(fn(&mut Player, &mut PlayerSpriteState)),\n\
     }"
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_fn_enum_shape() {
        assert_eq!(
            render_action_fn_enum(),
            "pub enum ActionFn {\n\
             \x20   Mobj(fn(&mut Mobj)),\n\
             \x20   Weapon(fn(&mut Player, &mut PlayerSpriteState)),\n\
             }"
        );
    }
}
