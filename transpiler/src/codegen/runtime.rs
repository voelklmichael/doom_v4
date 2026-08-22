//! Phase 3 Rust Runtime Support
//!
//! Unlike the rest of `codegen/`, this isn't analysis over the corpus --
//! it's literal Rust source that the transpiled crate will need alongside
//! its generated modules (its own copy of this file, not something
//! `ModuleGraph` has an entry for). Kept here, compiled and tested as part
//! of this crate too, so its behavior is verified once rather than trusted
//! by inspection wherever it eventually gets copied.
//!
//! ## `FixedT`: Doom's 16.16 fixed-point type
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

use std::ops::{Add, Neg, Sub};

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
}
