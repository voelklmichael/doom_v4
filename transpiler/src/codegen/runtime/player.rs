//! `PlayerId`: an index into `players: [Player; MAXPLAYERS]` (`g_game.c`'s
//! `player_t players[MAXPLAYERS];`) -- a plain, fixed-size, never-resized
//! array, same "no generation counter needed" reasoning as
//! `geometry.rs`'s index newtypes, just at program lifetime rather than
//! per-level scope: every slot exists for the whole run, "in game" or not
//! tracked separately (`playeringame[MAXPLAYERS]`), never individually
//! freed or reallocated. Kept in its own module rather than folded into
//! `geometry.rs`, since a player isn't level geometry.
//!
//! Nullable wherever referenced from `mobj_t`: most mobjs (monsters,
//! items, decorations) never have a player attached, only the handful of
//! actual player pawns do -- `Option<PlayerId>`, not a bare `PlayerId`.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerId(pub u32);
