//! `FixedT`: Doom's 16.16 fixed-point type
//!
//! Mirrors `m_fixed.c`/`m_fixed.h` exactly, including its two odd corners,
//! rather than a "cleaner" reimplementation -- the whole point of a
//! newtype here is type-safety (the compiler now catches accidentally
//! mixing `FixedT` with a plain `i32`), not changed arithmetic, since
//! `docs/03_TRANSPILER.md`'s validation strategy is behavior-identical
//! output against the original:
//! - `fixed_mul`: the original widens to `long long` (`i64` here) before
//!   the shift specifically to avoid intermediate overflow that a 32-bit
//!   multiply would hit; narrows back to `i32` (wrapping, matching the
//!   original's own implicit truncating cast) only after.
//! - `fixed_div`: saturates to `i32::MIN`/`i32::MAX` (by the result's
//!   sign) via a cheap pre-check (`abs(a) >> 14 >= abs(b)`) *before*
//!   attempting the division proper, rather than detecting overflow after
//!   the fact -- `fixed_div2` alone, given the same inputs, would hit its
//!   own overflow panic instead of this tier's saturation.
//! - `fixed_div2`: goes through `f64`, matching the original's own
//!   double-precision implementation (the `long long`-only alternative is
//!   `#if 0`'d out in the original itself, never compiled); panics past a
//!   32-bit range, carrying the original's own message even though the
//!   condition it actually guards is overflow, not a literal zero divisor.
//!
//! Plain arithmetic (`Add`/`Sub`/`Neg`) wraps rather than panics on
//! overflow, matching the original's C `int` semantics (two's-complement
//! wraparound in practice) rather than Rust's own debug-mode default.
//!
//! `Mul<FixedT>`/`Mul<i32>`/`Div<i32>` (plain scalar, not the rescaling
//! `fixed_mul`/`fixed_div`) and `AddAssign`/`SubAssign` are new,
//! genuinely-corpus-needed additions (`A_Tracer`, `p_enemy.c`): C's
//! `fixed_t` is a bare `typedef int`, so an idiom like `40*FRACUNIT`
//! (constructing a fixed value from a plain scale factor) or `dist /
//! actor->info->speed` (a `fixed_t` divided by a plain `int`, e.g. to
//! turn a distance into a tic count) or `actor->momz -= FRACUNIT/8`
//! never goes through `FixedDiv`/`FixedMul` at all in the original --
//! it's just raw `int` arithmetic on the representation, exactly what
//! `Add`/`Sub`/`Neg` already model for `+`/`-`/unary `-`. Kept as their
//! own operators, not folded into `fixed_mul`/`fixed_div`, since those
//! two *do* rescale by `FRACUNIT` (true fixed-point multiply/divide)
//! and mixing the two meanings under one name would be wrong.

use std::ops::{
    Add, AddAssign, BitAnd, BitXor, Div, Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign, Sub,
    SubAssign,
};

pub const FRACBITS: u32 = 16;
pub const FRACUNIT: FixedT = FixedT(1 << FRACBITS);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct FixedT(pub i32);

impl FixedT {
    pub const ZERO: FixedT = FixedT(0);
    pub const MAX: FixedT = FixedT(i32::MAX);
    pub const MIN: FixedT = FixedT(i32::MIN);

    pub fn fixed_mul(self, other: FixedT) -> FixedT {
        FixedT(((self.0 as i64 * other.0 as i64) >> FRACBITS) as i32)
    }

    pub fn fixed_div(self, other: FixedT) -> FixedT {
        if (self.0.unsigned_abs() >> 14) >= other.0.unsigned_abs() {
            if (self.0 ^ other.0) < 0 {
                FixedT::MIN
            } else {
                FixedT::MAX
            }
        } else {
            self.fixed_div2(other)
        }
    }

    pub fn fixed_div2(self, other: FixedT) -> FixedT {
        let c = (self.0 as f64 / other.0 as f64) * (FRACUNIT.0 as f64);
        if !(-2147483648.0..2147483648.0).contains(&c) {
            panic!("FixedDiv: divide by zero");
        }
        FixedT(c as i32)
    }
}

impl Add for FixedT {
    type Output = FixedT;
    fn add(self, rhs: FixedT) -> FixedT {
        FixedT(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for FixedT {
    type Output = FixedT;
    fn sub(self, rhs: FixedT) -> FixedT {
        FixedT(self.0.wrapping_sub(rhs.0))
    }
}

impl Neg for FixedT {
    type Output = FixedT;
    fn neg(self) -> FixedT {
        FixedT(self.0.wrapping_neg())
    }
}

impl AddAssign for FixedT {
    fn add_assign(&mut self, rhs: FixedT) {
        self.0 = self.0.wrapping_add(rhs.0);
    }
}

impl SubAssign for FixedT {
    fn sub_assign(&mut self, rhs: FixedT) {
        self.0 = self.0.wrapping_sub(rhs.0);
    }
}

/// `mo->x + (P_Random()-P_Random())*2048`/`128 + P_Random()*2*FRACUNIT`
/// (`A_BrainExplode`) -- a `fixed_t` field offset by a plain scaled
/// `int` (or the reverse order), the same "no rescaling, just raw
/// representation arithmetic" idiom `Mul<i32>`/`Div<i32>` below already
/// cover for `*`/`/`; needed in both operand orders since the real
/// corpus uses both (`x = mo->x + ...` puts the `FixedT` first, `z =
/// 128 + ...` puts the plain `int` first).
impl Add<i32> for FixedT {
    type Output = FixedT;
    fn add(self, rhs: i32) -> FixedT {
        FixedT(self.0.wrapping_add(rhs))
    }
}

impl Add<FixedT> for i32 {
    type Output = FixedT;
    fn add(self, rhs: FixedT) -> FixedT {
        FixedT(self.wrapping_add(rhs.0))
    }
}

/// `40*FRACUNIT`-style construction (a plain scale factor times
/// `FRACUNIT`, or any other `FixedT` constant/value) -- raw `i32`
/// multiply of the representation, matching what C's `int*int` really
/// computes here (no `FRACUNIT`-rescaling `fixed_mul` involved).
impl Mul<FixedT> for i32 {
    type Output = FixedT;
    fn mul(self, rhs: FixedT) -> FixedT {
        FixedT(self.wrapping_mul(rhs.0))
    }
}

impl Mul<i32> for FixedT {
    type Output = FixedT;
    fn mul(self, rhs: i32) -> FixedT {
        FixedT(self.0.wrapping_mul(rhs))
    }
}

/// `z += ((P_Random()-P_Random())<<10);` (`P_SpawnPuff`/`P_SpawnBlood`,
/// `p_mobj.c`) -- a `fixed_t`-declared parameter compound-assigned from a
/// plain raw `int` expression (no `FixedT` source anywhere in it), the
/// compound-assign sibling of `Add<i32> for FixedT` above: C's `fixed_t`
/// is a bare `typedef int`, so this never goes through `FixedAdd`/any
/// rescaling at all, just raw representation arithmetic, matching every
/// other operator in this file.
impl AddAssign<i32> for FixedT {
    fn add_assign(&mut self, rhs: i32) {
        self.0 = self.0.wrapping_add(rhs);
    }
}

/// `thrust *= 4;` (`P_DamageMobj`, `p_inter.c`) -- the compound-assign
/// sibling of `Mul<i32> for FixedT` above, needed for the same "no
/// rescaling, just raw representation arithmetic" idiom `AddAssign<i32>`
/// already established for `+=`.
impl MulAssign<i32> for FixedT {
    fn mul_assign(&mut self, rhs: i32) {
        self.0 = self.0.wrapping_mul(rhs);
    }
}

/// `dist / actor->info->speed`-style division by a plain scalar `int`
/// (not another `fixed_t`) -- raw `i32` division of the representation,
/// matching what C's `int/int` really computes. The rescaling `fixed_div`/
/// `FixedDiv` is the usual explicitly-named operation for dividing one
/// `fixed_t` by another, but it isn't the *only* one the real corpus
/// uses -- see the bare `Div` impl just below.
impl Div<i32> for FixedT {
    type Output = FixedT;
    fn div(self, rhs: i32) -> FixedT {
        FixedT(self.0 / rhs)
    }
}

/// `slope = (dest->z+40*FRACUNIT - actor->z) / dist;` (`A_Tracer`,
/// `p_enemy.c`) -- a bare `/` genuinely dividing one `fixed_t` value
/// (`dist`, reassigned from `dist / actor->info->speed` a few lines
/// earlier) by *another* `fixed_t` value, both declared `fixed_t` in the
/// real corpus, confirmed by direct read -- not routed through `FixedDiv`
/// at all, just raw `int/int` division of the two representations,
/// matching every other bare arithmetic operator here (`Add`/`Sub`).
/// Surfaced once `P_CheckMissileRange` closed the general "genuinely
/// `fixed_t`-declared local" tracking (`FnBodyContext::fixed_t_locals`)
/// and `A_Tracer`'s own verification harness stopped dodging `dist`'s
/// real type with an `i32`-returning `P_AproxDistance` stub -- confirmed
/// a real `rustc` rejection (no `Div<FixedT> for FixedT` existed before
/// this), not a hypothetical.
impl Div for FixedT {
    type Output = FixedT;
    fn div(self, rhs: FixedT) -> FixedT {
        FixedT(self.0 / rhs.0)
    }
}

/// `dest->height>>1`-style right shift directly on a `fixed_t` value
/// (`A_SkullAttack`, halving a height while staying in the same
/// fixed-point scale) -- a raw bit-shift of the representation, the same
/// "thin wrapping-arithmetic pass-through" idea as every other operator
/// here: shifting a fixed-point number right by `n` divides it by `2^n`
/// while preserving its `FRACUNIT` scale exactly (unlike `Div<i32>`,
/// which is also scale-preserving but via true division rather than a
/// bit shift), so the result stays a `FixedT`, not a plain `i32`.
impl Shr<i32> for FixedT {
    type Output = FixedT;
    fn shr(self, rhs: i32) -> FixedT {
        FixedT(self.0 >> rhs)
    }
}

/// `corpsehit->height <<= 2; ... corpsehit->height >>= 2;`
/// (`PIT_VileCheck`, temporarily inflating a corpse's collision height
/// to test-fit a resurrection, then restoring it) -- the compound-assign
/// siblings of `Shr<i32>` above, needed together since C pairs a
/// `<<=`/`>>=` by the same shift amount as a matched save/restore idiom.
/// Same "thin wrapping-arithmetic pass-through, stays `FixedT`" reasoning
/// throughout.
impl Shl<i32> for FixedT {
    type Output = FixedT;
    fn shl(self, rhs: i32) -> FixedT {
        FixedT(self.0 << rhs)
    }
}

impl ShlAssign<i32> for FixedT {
    fn shl_assign(&mut self, rhs: i32) {
        self.0 <<= rhs;
    }
}

impl ShrAssign<i32> for FixedT {
    fn shr_assign(&mut self, rhs: i32) {
        self.0 >>= rhs;
    }
}

/// `(dist - thing->radius) >> FRACBITS` (`PIT_RadiusAttack`) -- the same
/// raw-bit-shift idiom as `Shr<i32>` above, just shifted by `FRACBITS`
/// itself (`runtime::FRACBITS: u32`, not `i32` -- this parser never
/// macro-expands `#define`s, so the rendered output always references the
/// real corpus name bare, and needs the shift amount's own real declared
/// Rust type to type-check). Kept as its own impl rather than widening
/// `Shr<i32>`'s own signature: both shift-amount types appear in the real
/// corpus (a literal shift count like `dest->height>>1` is `i32`-inferred,
/// while `FRACBITS` itself is `u32`), so both need their own impl, the
/// same "both operand types get impl'd" precedent `Add<i32>`/`Mul<i32>`
/// already established for other operators.
impl Shr<u32> for FixedT {
    type Output = FixedT;
    fn shr(self, rhs: u32) -> FixedT {
        FixedT(self.0 >> rhs)
    }
}

/// `line->dy ^ line->dx ^ dx ^ dy` (`P_PointOnDivlineSide`, `p_maputl.c`)
/// -- a raw sign-bit fast-path check directly on `fixed_t` values (`x^y`
/// on the representation, not a rescaling operation -- `fixed_t` really
/// is just `int` here, the same idiom every other bitwise/arithmetic
/// operator in this file already models). Round 13's own corpus survey
/// flagged this exact gap (`FixedT` had no `BitXor` impl at all) as the
/// one new mechanism this function's sign-bit shortcut needed.
impl BitXor for FixedT {
    type Output = FixedT;
    fn bitxor(self, rhs: FixedT) -> FixedT {
        FixedT(self.0 ^ rhs.0)
    }
}

/// `(...) & 0x80000000` (`P_PointOnDivlineSide`) -- the `BitAnd` sibling
/// of `BitXor` just above, needed for the same sign-bit fast-path check
/// (masking down to just the top bit to compare signs without a full
/// `FixedMul`). Same raw-representation-`&`, not a rescaling operation.
impl BitAnd for FixedT {
    type Output = FixedT;
    fn bitand(self, rhs: FixedT) -> FixedT {
        FixedT(self.0 & rhs.0)
    }
}

/// `(node->dy>>FRACBITS) * (dx>>FRACBITS)` (`P_DivlineSide`, `p_sight.c`)
/// -- a plain, non-`FixedMul` integer multiply of two pre-shifted
/// `fixed_t` operands, deliberately abusing C's "`fixed_t` really is just
/// `int`" idiom the same way `Mul<i32>`/`Mul<FixedT> for i32` already do
/// for a scale-factor multiply -- this is the missing same-type case
/// (`FixedT * FixedT`, raw representation, no `FRACUNIT` rescaling),
/// round 13's own second flagged gap for this function.
impl Mul for FixedT {
    type Output = FixedT;
    fn mul(self, rhs: FixedT) -> FixedT {
        FixedT(self.0.wrapping_mul(rhs.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fracunit_is_one_shifted_by_fracbits() {
        assert_eq!(FRACUNIT.0, 1 << 16);
        assert_eq!(FRACUNIT.0, 65536);
    }

    #[test]
    fn test_fixed_mul_two_and_a_half_times_two() {
        // 2.5 * 2 = 5.0, in 16.16 fixed point.
        let two_point_five = FixedT((2.5 * FRACUNIT.0 as f64) as i32);
        let two = FixedT(2 * FRACUNIT.0);
        let five = FixedT(5 * FRACUNIT.0);
        assert_eq!(two_point_five.fixed_mul(two), five);
    }

    #[test]
    fn test_fixed_mul_by_fracunit_is_identity() {
        let x = FixedT(12345);
        assert_eq!(x.fixed_mul(FRACUNIT), x);
    }

    #[test]
    fn test_fixed_div_ten_by_two_is_five() {
        let ten = FixedT(10 * FRACUNIT.0);
        let two = FixedT(2 * FRACUNIT.0);
        let five = FixedT(5 * FRACUNIT.0);
        assert_eq!(ten.fixed_div(two), five);
    }

    #[test]
    fn test_fixed_div_saturates_on_overflow() {
        // A huge positive divided by a tiny positive would overflow
        // fixed_t's 32-bit range -- same sign, so saturates to MAX.
        let huge = FixedT(i32::MAX);
        let tiny = FixedT(1);
        assert_eq!(huge.fixed_div(tiny), FixedT::MAX);
    }

    #[test]
    fn test_fixed_div_saturates_to_min_on_opposite_sign_overflow() {
        let huge = FixedT(i32::MAX);
        let tiny_negative = FixedT(-1);
        assert_eq!(huge.fixed_div(tiny_negative), FixedT::MIN);
    }

    #[test]
    #[should_panic(expected = "FixedDiv: divide by zero")]
    fn test_fixed_div2_panics_past_32_bit_range() {
        // Calling fixed_div2 directly bypasses fixed_div's own saturating
        // pre-check, so this large-over-tiny division overflows for real.
        FixedT(i32::MAX).fixed_div2(FixedT(1));
    }

    #[test]
    fn test_add_sub_wrap_on_overflow_matching_c_int() {
        assert_eq!(FixedT(i32::MAX) + FixedT(1), FixedT(i32::MIN));
        assert_eq!(FixedT(i32::MIN) - FixedT(1), FixedT(i32::MAX));
    }

    #[test]
    fn test_neg_matches_c_unary_minus() {
        assert_eq!(-FixedT(5 * FRACUNIT.0), FixedT(-5 * FRACUNIT.0));
    }

    #[test]
    fn test_add_assign_sub_assign_match_add_sub() {
        let mut x = FixedT(3 * FRACUNIT.0);
        x += FixedT(FRACUNIT.0);
        assert_eq!(x, FixedT(4 * FRACUNIT.0));
        x -= FixedT(FRACUNIT.0);
        assert_eq!(x, FixedT(3 * FRACUNIT.0));
    }

    #[test]
    fn test_int_times_fixed_matches_40_times_fracunit_idiom() {
        // `40*FRACUNIT` (`A_Tracer`'s own `dest->z+40*FRACUNIT`) -- a
        // plain scale factor times FRACUNIT, not a rescaling fixed_mul.
        assert_eq!(40 * FRACUNIT, FixedT(40 * FRACUNIT.0));
    }

    #[test]
    fn test_fixed_times_int_is_symmetric_with_int_times_fixed() {
        assert_eq!(FRACUNIT * 8, 8 * FRACUNIT);
    }

    #[test]
    fn test_fixed_div_by_plain_int_is_raw_division() {
        // `FRACUNIT/8` (`A_Tracer`'s own `actor->momz -= FRACUNIT/8`) --
        // raw division of the representation, not FixedDiv's rescaling.
        assert_eq!(FRACUNIT / 8, FixedT(FRACUNIT.0 / 8));
    }

    #[test]
    fn test_fixed_plus_int_is_symmetric_with_int_plus_fixed() {
        // `A_BrainExplode`'s own `mo->x + (...)*2048` and `128 +
        // P_Random()*2*FRACUNIT` -- both operand orders appear in the
        // real corpus, so both need to agree.
        assert_eq!(FRACUNIT + 2048, 2048 + FRACUNIT);
        assert_eq!(FRACUNIT + 2048, FixedT(FRACUNIT.0 + 2048));
    }

    #[test]
    fn test_fixed_shr_by_one_halves_while_staying_fixed_point() {
        // `dest->height>>1` (`A_SkullAttack`) -- a raw bit-shift of the
        // representation, staying `FixedT` (unlike `Div<i32>`, which is
        // also scale-preserving but via true division).
        let height = FixedT(56 * FRACUNIT.0);
        assert_eq!(height >> 1, FixedT(height.0 >> 1));
        assert_eq!(height >> 1, FixedT(28 * FRACUNIT.0));
    }

    #[test]
    fn test_fixed_shr_by_fracbits_like_pit_radius_attack() {
        // `(dist - thing->radius) >> FRACBITS` (`PIT_RadiusAttack`) --
        // shifting by the real `u32`-typed `FRACBITS` constant, not a
        // bare `i32` literal shift count.
        let dist = FixedT(56 * FRACUNIT.0);
        assert_eq!(dist >> FRACBITS, FixedT(dist.0 >> FRACBITS));
        assert_eq!(dist >> FRACBITS, FixedT(56));
    }

    #[test]
    fn test_shl_shr_assign_round_trip_like_pit_vile_check() {
        // `corpsehit->height <<= 2; ... corpsehit->height >>= 2;`
        // (`PIT_VileCheck`) -- inflate then restore, ending back at the
        // original value.
        let mut height = FixedT(56 * FRACUNIT.0);
        let original = height;
        height <<= 2;
        assert_eq!(height, FixedT(original.0 << 2));
        height >>= 2;
        assert_eq!(height, original);
    }

    #[test]
    fn test_add_assign_i32_matches_add_i32() {
        // `z += ((P_Random()-P_Random())<<10);` (`P_SpawnPuff`) -- the
        // compound-assign sibling of `Add<i32>`, staying consistent with
        // it.
        let mut z = FixedT(100);
        z += 5;
        assert_eq!(z, FixedT(100) + 5);
    }

    #[test]
    fn test_mul_assign_i32_matches_mul_i32() {
        // `thrust *= 4;` (`P_DamageMobj`) -- the compound-assign sibling
        // of `Mul<i32>`, staying consistent with it.
        let mut thrust = FixedT(100);
        thrust *= 4;
        assert_eq!(thrust, FixedT(100) * 4);
    }

    #[test]
    fn test_bare_div_like_a_tracer_slope() {
        // `slope = (dest->z+40*FRACUNIT - actor->z) / dist;` (`A_Tracer`)
        // -- a bare `/` genuinely dividing one `fixed_t` value by
        // another, both declared `fixed_t` in the real corpus, never
        // routed through `FixedDiv` at all -- raw `int/int` division of
        // the two representations, matching every other bare arithmetic
        // operator here.
        let numerator = FixedT(100);
        let divisor = FixedT(4);
        assert_eq!(numerator / divisor, FixedT(25));
    }

    #[test]
    fn test_bitxor_matches_raw_representation_xor() {
        // `line->dy ^ line->dx ^ dx ^ dy` (`P_PointOnDivlineSide`) -- raw
        // bitwise XOR of the representation, not a rescaling operation.
        assert_eq!(FixedT(0b1010) ^ FixedT(0b0110), FixedT(0b1100));
    }

    #[test]
    fn test_bitand_matches_raw_representation_and() {
        // `(...) & 0x80000000` (`P_PointOnDivlineSide`) -- masking down to
        // the sign bit for the sign-bit fast-path check.
        let neg = FixedT(-1_i32); // all bits set
        let sign_mask = FixedT(0x80000000_u32 as i32);
        assert_eq!(neg & sign_mask, sign_mask);
        assert_eq!(FixedT(1) & sign_mask, FixedT(0));
    }

    #[test]
    fn test_mul_fixed_by_fixed_is_raw_representation_multiply() {
        // `(node->dy>>FRACBITS) * (dx>>FRACBITS)` (`P_DivlineSide`) -- a
        // plain `int*int` of two already-shifted-down operands, not a
        // rescaling `fixed_mul`.
        assert_eq!(FixedT(5) * FixedT(4), FixedT(20));
    }
}
