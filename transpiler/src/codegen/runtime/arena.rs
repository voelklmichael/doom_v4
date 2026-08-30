//! `Arena<T>`: the dynamic-entity store (`docs/03_TRANSPILER.md`'s Memory
//! Model section) -- what `mobj_t` and the 9 other `p_spec.h` structs that
//! embed `thinker_t` as a hand-rolled C vtable get ported to.
//!
//! **Append-only, no generation counter, no free-list** -- not the
//! generational-slot design that section originally proposed, revised once
//! actually specifying `remove`/`run`'s semantics against the real
//! original: `p_tick.c`'s thinkers were never array-slot-based in C at all
//! (each one its own independent `Z_Malloc`), so slot reuse -- and the ABA
//! hazard a generation counter would defend against -- is something a
//! naive port would have *introduced*, not something the original has.
//! Worse, `P_RunThinkers`' single forward pass visits a thinker spawned
//! mid-tick (always appended at the true tail, ahead of the traversal
//! cursor) in that *same* tick; a free-list that reused an earlier,
//! already-visited index for that new thinker would make it invisible for
//! the rest of the pass, silently diverging from the original. Always
//! appending at the true end (`insert` here) avoids this outright.
//! Removed slots stay permanently dead for the rest of that arena's
//! lifetime -- fine, since the whole arena is discarded and recreated
//! fresh at each level load, matching the original's own `PU_LEVEL`
//! wipe-everything-at-once semantics; a level's total thinker count over a
//! whole playthrough is a few thousand at most.
//!
//! **`remove`'s two cases**: `p_tick.c`'s `P_RemoveThinker` doesn't free
//! immediately -- it sets a sentinel, and the actual unlink+free happens
//! lazily next time `P_RunThinkers`' traversal reaches that node. Here,
//! removing a handle that ISN'T the one currently mid-`run` clears its slot
//! immediately (safe: nothing borrows that slot, and whether the traversal
//! has already passed it or hasn't reached it yet, the *observable* result
//! -- "not ticked, gone by the time this pass ends" -- matches the
//! original either way). Removing the handle that IS currently mid-`run`
//! is different: its slot is empty right now (`run` took the value out to
//! hand the closure both `&mut T` and `&mut Arena<T>` simultaneously,
//! without which Rust's aliasing rules would forbid a thinker from
//! removing itself or another handle during its own tick -- the single
//! most common real pattern in Doom's own gameplay code, e.g. a missile
//! removing itself on impact) -- so removal there is recorded and checked
//! by `run` right after the closure returns, instead of writing to a slot
//! that has nothing in it to write over.

use std::marker::PhantomData;

/// An index into an `Arena<T>`. Only constructible by `Arena::insert`, so a
/// caller can never forge one; still distinguishable across two arenas of
/// different `T` (a `Handle<Thinker>` doesn't type-check where a
/// `Handle<Other>` is expected), at zero runtime cost since `T` here is
/// purely a marker.
#[derive(Debug)]
pub struct Handle<T> {
    index: u32,
    _marker: PhantomData<fn() -> T>,
}

// Hand-written rather than derived: `derive` would require `T: Copy` /
// `T: Eq` / etc. even though `T` never actually appears in this type,
// only in the zero-sized marker.
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<T> Eq for Handle<T> {}
impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> Handle<T> {
    fn new(index: u32) -> Self {
        Self {
            index,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct Arena<T> {
    slots: Vec<Option<T>>,
    /// The index `run` currently has taken out of `slots`, if any --
    /// `remove` checks this to tell "remove the item mid-tick right now"
    /// apart from "remove some other, resident item."
    currently_processing: Option<u32>,
    /// Set by `remove` when its target *is* `currently_processing`; `run`
    /// checks this right after the closure returns to decide whether to
    /// put the value back.
    self_removed: bool,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            currently_processing: None,
            self_removed: false,
        }
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `value`, always at the true end -- mirrors `P_AddThinker`
    /// appending at the list's tail. Safe to call reentrantly from inside
    /// a `run` closure (via the `&mut Arena<T>` it's handed): the new
    /// index is necessarily past every index `run`'s traversal has reached
    /// so far, so it's visited later in that same pass, matching the
    /// original's same-tick visibility for a thinker spawned mid-tick.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        let index = self.slots.len() as u32;
        self.slots.push(Some(value));
        Handle::new(index)
    }

    /// Removes `handle`. If it's the handle `run` currently has taken out
    /// (a thinker removing itself, or another in-progress handle, mid-tick
    /// -- see this module's docs), the value is dropped instead of put
    /// back once the closure returns; otherwise the slot clears
    /// immediately. Calling this on an already-removed handle is a no-op.
    pub fn remove(&mut self, handle: Handle<T>) {
        if self.currently_processing == Some(handle.index) {
            self.self_removed = true;
        } else if let Some(slot) = self.slots.get_mut(handle.index as usize) {
            *slot = None;
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.slots.get(handle.index as usize)?.as_ref()
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        self.slots.get_mut(handle.index as usize)?.as_mut()
    }

    /// A read-only pass over every live slot, in insertion order --
    /// mirrors the original's own raw `thinkercap` linked-list walk
    /// (`for (th = thinkercap.next; th != &thinkercap; th = th->next)`,
    /// `p_enemy.c`'s `A_PainShootSkull`/`A_KeenDie`/`A_BrainAwake`/
    /// `A_BossDeath` all use this exact idiom to scan every live thinker
    /// without ticking or mutating any of them). Since `insert` always
    /// appends at the true end and a tombstoned slot stays permanently
    /// `None` for the rest of this arena's lifetime (this module's own
    /// doc comment), a plain forward `Vec` scan already visits live slots
    /// in the same order the original's list traversal would -- no extra
    /// bookkeeping needed, unlike `run`'s own take-then-put-back dance
    /// (nothing here ever hands a slot's value back to the caller by
    /// `&mut`, so there's no reentrancy hazard to guard against).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|slot| slot.as_ref())
    }

    /// `iter`'s own sibling for the one real corpus shape `iter` alone
    /// can't serve: `A_BrainAwake`'s own scan doesn't just read each live
    /// thinker, it *records* the ones matching `MT_BOSSTARGET` into
    /// `braintargets[]` for `A_BrainSpit` to fire at later -- that needs
    /// each entry's own `Handle<T>`, not just a borrowed reference to its
    /// value, and `iter` never had a reason to hand one out before now.
    /// Additive rather than a breaking change to `iter` itself (every
    /// existing caller keeps compiling against the plain `&T` shape
    /// unchanged) -- indices line up with `iter`'s own since both walk
    /// the same `self.slots`, in the same order, for the same
    /// "tombstoned slots stay permanently `None`" reason documented above.
    pub fn iter_with_handle(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|v| (Handle::new(i as u32), v)))
    }

    /// One pass over every live slot, in insertion order -- mirrors
    /// `P_RunThinkers`' single forward traversal. `f` gets the item's own
    /// value, its handle, and the arena itself, so it can remove or insert
    /// (including removing its own handle) exactly as the original's
    /// tick functions freely do to `thinkercap`'s list.
    pub fn run<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T, Handle<T>, &mut Arena<T>),
    {
        let mut i = 0usize;
        while i < self.slots.len() {
            if let Some(mut value) = self.slots[i].take() {
                let handle = Handle::new(i as u32);
                self.currently_processing = Some(i as u32);
                self.self_removed = false;
                f(&mut value, handle, self);
                if !self.self_removed {
                    self.slots[i] = Some(value);
                }
                self.currently_processing = None;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut arena: Arena<i32> = Arena::new();
        let h = arena.insert(42);
        assert_eq!(arena.get(h), Some(&42));
    }

    #[test]
    fn test_run_visits_in_insertion_order() {
        let mut arena: Arena<i32> = Arena::new();
        arena.insert(1);
        arena.insert(2);
        arena.insert(3);
        let mut seen = Vec::new();
        arena.run(|v, _, _| seen.push(*v));
        assert_eq!(seen, vec![1, 2, 3]);
    }

    #[test]
    fn test_remove_other_handle_takes_effect_immediately() {
        let mut arena: Arena<i32> = Arena::new();
        let a = arena.insert(1);
        let _b = arena.insert(2);
        arena.remove(a);
        assert_eq!(arena.get(a), None);
        let mut seen = Vec::new();
        arena.run(|v, _, _| seen.push(*v));
        assert_eq!(seen, vec![2]);
    }

    #[test]
    fn test_removed_slot_stays_dead_forever() {
        let mut arena: Arena<i32> = Arena::new();
        let a = arena.insert(1);
        arena.remove(a);
        arena.insert(2);
        arena.insert(3);
        // `a`'s old index is never reused by later inserts.
        let mut seen = Vec::new();
        arena.run(|v, _, _| seen.push(*v));
        assert_eq!(seen, vec![2, 3]);
    }

    #[test]
    fn test_self_removal_during_own_tick() {
        // The common real Doom pattern: a thinker removes itself mid-tick
        // (e.g. a missile on impact). Must not panic (aliasing) and must
        // not be visited again afterward.
        let mut arena: Arena<i32> = Arena::new();
        let a = arena.insert(1);
        arena.run(|_, handle, world| {
            world.remove(handle);
        });
        assert_eq!(arena.get(a), None);
        let mut seen = Vec::new();
        arena.run(|v, _, _| seen.push(*v));
        assert!(seen.is_empty());
    }

    #[test]
    fn test_mutation_during_own_tick_persists_if_not_removed() {
        let mut arena: Arena<i32> = Arena::new();
        let a = arena.insert(1);
        arena.run(|v, _, _| *v += 41);
        assert_eq!(arena.get(a), Some(&42));
    }

    #[test]
    fn test_inserted_during_tick_is_visited_same_pass() {
        // Mirrors P_RunThinkers: a thinker spawned mid-tick (appended at
        // the true tail, ahead of the traversal cursor) is visited in
        // that same pass, not deferred to the next one.
        let mut arena: Arena<i32> = Arena::new();
        arena.insert(1);
        let mut seen = Vec::new();
        arena.run(|v, _, world| {
            seen.push(*v);
            if *v == 1 {
                world.insert(2);
            }
        });
        assert_eq!(seen, vec![1, 2]);
    }

    #[test]
    fn test_remove_of_a_not_yet_visited_handle_skips_it_same_pass() {
        let mut arena: Arena<i32> = Arena::new();
        let a = arena.insert(1);
        let b = arena.insert(2);
        let mut seen = Vec::new();
        arena.run(|v, handle, world| {
            seen.push(*v);
            if handle == a {
                world.remove(b);
            }
        });
        assert_eq!(seen, vec![1]);
    }

    #[test]
    fn test_remove_of_an_already_visited_handle_has_no_effect_this_pass() {
        let mut arena: Arena<i32> = Arena::new();
        let a = arena.insert(1);
        let b = arena.insert(2);
        let mut seen = Vec::new();
        arena.run(|v, handle, world| {
            seen.push(*v);
            if handle == b {
                world.remove(a); // already visited earlier this same pass
            }
        });
        assert_eq!(
            seen,
            vec![1, 2],
            "removing an already-visited handle shouldn't un-visit it"
        );
        // But it does take effect for the *next* pass.
        let mut seen_next = Vec::new();
        arena.run(|v, _, _| seen_next.push(*v));
        assert_eq!(seen_next, vec![2]);
    }

    #[test]
    fn test_iter_visits_live_slots_in_insertion_order_skipping_removed() {
        let mut arena: Arena<i32> = Arena::new();
        let a = arena.insert(1);
        arena.insert(2);
        arena.insert(3);
        arena.remove(a);
        let seen: Vec<i32> = arena.iter().copied().collect();
        assert_eq!(seen, vec![2, 3]);
    }

    #[test]
    fn test_iter_with_handle_yields_handles_get_can_look_back_up() {
        let mut arena: Arena<i32> = Arena::new();
        let a = arena.insert(1);
        let b = arena.insert(2);
        arena.remove(a);
        let c = arena.insert(3);
        let seen: Vec<(Handle<i32>, i32)> =
            arena.iter_with_handle().map(|(h, v)| (h, *v)).collect();
        assert_eq!(seen, vec![(b, 2), (c, 3)]);
        for (h, v) in seen {
            assert_eq!(arena.get(h), Some(&v));
        }
    }

    #[test]
    fn test_remove_already_removed_handle_is_a_no_op() {
        let mut arena: Arena<i32> = Arena::new();
        let a = arena.insert(1);
        arena.remove(a);
        arena.remove(a); // must not panic
        assert_eq!(arena.get(a), None);
    }
}
