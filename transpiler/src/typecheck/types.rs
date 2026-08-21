//! Type Representation (docs/02_TYPECHECKER.md Goal 2), introduced here as
//! Step 2's own prerequisite -- Step 2 is "the first consumer that actually
//! needs a type" (see `macro_types.rs`'s module docs), so this is where a
//! `Type` first has to exist. Deliberately scoped to what Step 2 needs:
//! enough to type literals, casts, and the usual arithmetic conversions.
//! `const`/`volatile` qualifiers and full struct/union field layouts are
//! *not* modeled here -- Step 0 only ever collected coarse symbol/tag
//! *kinds*, not full types (see `exports.rs`), so a struct/union/enum tag
//! reference resolves to an opaque `Type::Struct`/`Union`/`Enum(name)`
//! rather than its member list. That, and resolving a typedef name (e.g.
//! `fixed_t`) down to its underlying representation, are Step 3's job.
//!
//! **ILP32 assumption**: the usual arithmetic conversions below assume the
//! actual target ABI `linuxdoom-1.10` was built for (`int`/`long`/pointer
//! all 32-bit), not portable worst-case C89. This matters concretely: on
//! ILP32, `long` cannot represent every `unsigned int` value (same width),
//! so a `long`/`unsigned int` mix converts to `unsigned long` per the C89
//! table -- a machine-dependent outcome, not a `long` win as it would be on
//! an LP64 target. Deliberate, matching this project's "we're transpiling
//! this specific binary's actual semantics" stance elsewhere.

use crate::parser::ast::{
    AbstractDeclarator, BinaryOp, DeclSpecifiers, DirectAbstractDeclarator, TypeName, TypeSpecifier,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Void,
    Char,
    SChar,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    Float,
    Double,
    LongDouble,
    Pointer(Box<Type>),
    Array(Box<Type>),
    /// Return type only -- parameter types aren't modeled (see module docs'
    /// scoping note on `DirectAbstractDeclarator::Function`).
    Function(Box<Type>),
    Struct(String),
    Union(String),
    Enum(String),
    /// A typedef name (e.g. `fixed_t`) whose underlying representation
    /// isn't resolved yet -- Step 3's job. Still a *resolved* type, unlike
    /// `Unknown`: the name itself is real, useful information.
    Named(String),
    /// No usable type information could be determined -- an identifier that
    /// isn't a macro or a known enum constant, a struct/union member access
    /// (field layouts aren't modeled), a function-pointer cast, or an
    /// operand that was itself `Unknown`.
    Unknown,
}

impl Type {
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Type::Char
                | Type::SChar
                | Type::UChar
                | Type::Short
                | Type::UShort
                | Type::Int
                | Type::UInt
                | Type::Long
                | Type::ULong
                | Type::Enum(_)
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::Float | Type::Double | Type::LongDouble)
    }

    pub fn is_arithmetic(&self) -> bool {
        self.is_integer() || self.is_float()
    }
}

/// C89 6.2.1.1 integer promotion: anything narrower than `int` promotes to
/// `int` (every value fits, since ILP32's `int` is 32 bits). Wider types,
/// non-integers, and `Unknown` pass through unchanged.
pub fn integer_promote(t: &Type) -> Type {
    match t {
        Type::Char | Type::SChar | Type::UChar | Type::Short | Type::UShort => Type::Int,
        other => other.clone(),
    }
}

fn float_rank(t: &Type) -> Option<u8> {
    match t {
        Type::Float => Some(1),
        Type::Double => Some(2),
        Type::LongDouble => Some(3),
        _ => None,
    }
}

/// The result type of a binary arithmetic operator's two (already-typed)
/// operands, per C89 6.2.1.1 -- ILP32-specific where the standard's outcome
/// is machine-dependent (see module docs).
pub fn usual_arithmetic_conversions(a: &Type, b: &Type) -> Type {
    if !a.is_arithmetic() || !b.is_arithmetic() {
        return Type::Unknown;
    }
    match (float_rank(a), float_rank(b)) {
        (Some(ra), Some(rb)) => {
            if ra >= rb {
                a.clone()
            } else {
                b.clone()
            }
        }
        (Some(_), None) => a.clone(),
        (None, Some(_)) => b.clone(),
        (None, None) => {
            let (pa, pb) = (integer_promote(a), integer_promote(b));
            if pa == pb {
                return pa;
            }
            use Type::*;
            match (&pa, &pb) {
                (ULong, _) | (_, ULong) => ULong,
                // ILP32: `long` and `unsigned int` share a width, so `long`
                // can't represent every `unsigned int` value -- both
                // convert to `unsigned long` (see module docs).
                (Long, UInt) | (UInt, Long) => ULong,
                (Long, _) | (_, Long) => Long,
                (UInt, _) | (_, UInt) => UInt,
                _ => Int,
            }
        }
    }
}

/// Strips a trailing `u`/`U`/`l`/`L` suffix run and classifies an integer
/// literal's radix, per C89 6.1.3.2's candidate-type table.
pub fn type_of_int_literal(text: &str) -> Type {
    let bytes = text.as_bytes();
    let mut end = bytes.len();
    let mut has_u = false;
    let mut has_l = false;
    while end > 0 {
        match bytes[end - 1] {
            b'u' | b'U' => has_u = true,
            b'l' | b'L' => has_l = true,
            _ => break,
        }
        end -= 1;
    }
    let digits = &text[..end];
    let (radix, digits) = if let Some(hex) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        (16, hex)
    } else if digits.len() > 1 && digits.starts_with('0') {
        (8, &digits[1..])
    } else {
        (10, digits)
    };
    let value = u128::from_str_radix(digits, radix).unwrap_or(u128::MAX);
    classify_int(value, radix == 10, has_u, has_l)
}

fn classify_int(value: u128, is_decimal: bool, has_u: bool, has_l: bool) -> Type {
    let candidates: &[Type] = match (is_decimal, has_u, has_l) {
        (true, false, false) => &[Type::Int, Type::Long],
        (true, false, true) => &[Type::Long],
        (true, true, false) => &[Type::UInt, Type::ULong],
        (true, true, true) => &[Type::ULong],
        (false, false, false) => &[Type::Int, Type::UInt, Type::Long, Type::ULong],
        (false, false, true) => &[Type::Long, Type::ULong],
        (false, true, false) => &[Type::UInt, Type::ULong],
        (false, true, true) => &[Type::ULong],
    };
    for c in candidates {
        // ILP32: `int` and `long` share a 32-bit signed range.
        let max: u128 = match c {
            Type::Int | Type::Long => i32::MAX as u128,
            Type::UInt | Type::ULong => u32::MAX as u128,
            _ => unreachable!(),
        };
        if value <= max {
            return c.clone();
        }
    }
    Type::ULong
}

pub fn type_of_float_literal(text: &str) -> Type {
    match text.chars().next_back() {
        Some('f') | Some('F') => Type::Float,
        Some('l') | Some('L') => Type::LongDouble,
        _ => Type::Double,
    }
}

/// Builds a `Type` from a cast's or `sizeof`'s `TypeName`.
pub fn type_from_type_name(tn: &TypeName) -> Type {
    let base = type_from_specifiers(&tn.specifiers);
    match &tn.abstract_declarator {
        None => base,
        Some(ad) => type_from_abstract_declarator(base, ad),
    }
}

/// Combines a `DeclSpecifiers`' storage-independent type-specifier list
/// into a base `Type`, ignoring qualifiers (not needed for value typing --
/// see module docs).
pub fn type_from_specifiers(specs: &DeclSpecifiers) -> Type {
    for ts in &specs.type_specifiers {
        match ts {
            TypeSpecifier::Struct(s) => return Type::Struct(s.name.clone().unwrap_or_default()),
            TypeSpecifier::Union(s) => return Type::Union(s.name.clone().unwrap_or_default()),
            TypeSpecifier::Enum(s) => return Type::Enum(s.name.clone().unwrap_or_default()),
            TypeSpecifier::TypedefName(n) => return Type::Named(n.clone()),
            _ => {}
        }
    }

    let mut signed = false;
    let mut unsigned = false;
    let mut short = 0u8;
    let mut long = 0u8;
    let mut base: Option<&'static str> = None;
    for ts in &specs.type_specifiers {
        match ts {
            TypeSpecifier::Void => base = Some("void"),
            TypeSpecifier::Char => base = Some("char"),
            TypeSpecifier::Int => base = base.or(Some("int")),
            TypeSpecifier::Float => base = Some("float"),
            TypeSpecifier::Double => base = Some("double"),
            TypeSpecifier::Signed => signed = true,
            TypeSpecifier::Unsigned => unsigned = true,
            TypeSpecifier::Short => short += 1,
            TypeSpecifier::Long => long += 1,
            _ => {}
        }
    }
    match base {
        Some("void") => Type::Void,
        Some("char") => {
            if unsigned {
                Type::UChar
            } else if signed {
                Type::SChar
            } else {
                Type::Char
            }
        }
        Some("float") => Type::Float,
        Some("double") => {
            if long > 0 {
                Type::LongDouble
            } else {
                Type::Double
            }
        }
        // Implicit/K&R `int` and every plain `short`/`long`/`signed`/
        // `unsigned` combination land here.
        _ => {
            if short > 0 {
                if unsigned { Type::UShort } else { Type::Short }
            } else if long > 0 {
                if unsigned { Type::ULong } else { Type::Long }
            } else if unsigned {
                Type::UInt
            } else {
                Type::Int
            }
        }
    }
}

/// Applies one `Pointer` wrap per entry of `pointer_quals` (order doesn't
/// matter since per-level qualifiers aren't tracked -- see module docs).
fn apply_pointers<T>(base: Type, pointer_quals: &[Vec<T>]) -> Type {
    pointer_quals
        .iter()
        .fold(base, |t, _| Type::Pointer(Box::new(t)))
}

/// Threads `base` outward through an abstract declarator exactly like a
/// real declarator-to-type builder: `pointer_quals` wraps `base` first,
/// then the direct part's array/function shape wraps *that* -- except a
/// parenthesized inner declarator resets what "outward" means, which is
/// why the array/function case below passes its own wrapped type down as
/// the *new* base for whatever's nested inside, rather than wrapping
/// around that nested type's independently-built result. This is the
/// standard trick behind e.g. `int *a[3]` (array of pointers) vs.
/// `int (*a)[3]` (pointer to array) coming out differently despite sharing
/// every token but the parens.
fn type_from_abstract_declarator(base: Type, ad: &AbstractDeclarator) -> Type {
    let base = apply_pointers(base, &ad.pointer_quals);
    match &ad.direct {
        None => base,
        Some(d) => type_from_direct_abstract(d, base),
    }
}

fn type_from_direct_abstract(d: &DirectAbstractDeclarator, base: Type) -> Type {
    match d {
        DirectAbstractDeclarator::Paren(inner) => type_from_abstract_declarator(base, inner),
        DirectAbstractDeclarator::Array(sub, _size) => {
            let wrapped = Type::Array(Box::new(base));
            match sub {
                Some(s) => type_from_direct_abstract(s, wrapped),
                None => wrapped,
            }
        }
        DirectAbstractDeclarator::Function(sub, _params) => {
            // Function-pointer-shaped casts (e.g. `(void (*)(int))x`) don't
            // occur in this corpus's macro bodies -- left as `Unknown`
            // rather than modeling parameter types nothing needs yet.
            let _ = sub;
            Type::Unknown
        }
    }
}

pub fn unary_arith_result(t: &Type) -> Type {
    if t.is_float() {
        t.clone()
    } else if t.is_integer() {
        integer_promote(t)
    } else {
        Type::Unknown
    }
}

/// The result type of `lt op rt`, given `op` is one of the operators that
/// can apply to pointers (`Add`/`Sub`); every other operator's result is
/// decided by the caller via `usual_arithmetic_conversions` or a fixed
/// `int` result (relational/logical operators).
pub fn type_of_additive(op: BinaryOp, lt: &Type, rt: &Type) -> Type {
    match (lt, rt, op) {
        (Type::Pointer(inner), other, BinaryOp::Add) if other.is_integer() => {
            Type::Pointer(inner.clone())
        }
        (other, Type::Pointer(inner), BinaryOp::Add) if other.is_integer() => {
            Type::Pointer(inner.clone())
        }
        (Type::Pointer(inner), other, BinaryOp::Sub) if other.is_integer() => {
            Type::Pointer(inner.clone())
        }
        (Type::Pointer(a), Type::Pointer(b), BinaryOp::Sub) if a == b => Type::Int,
        _ if lt.is_arithmetic() && rt.is_arithmetic() => usual_arithmetic_conversions(lt, rt),
        _ => Type::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{DeclSpecifiers, TypeQualifier};

    #[test]
    fn test_decimal_int_literal_defaults_to_int() {
        assert_eq!(type_of_int_literal("16"), Type::Int);
    }

    #[test]
    fn test_unsigned_suffix_forces_unsigned() {
        assert_eq!(type_of_int_literal("4u"), Type::UInt);
        assert_eq!(type_of_int_literal("4U"), Type::UInt);
    }

    #[test]
    fn test_long_suffix_forces_long() {
        assert_eq!(type_of_int_literal("4L"), Type::Long);
        assert_eq!(type_of_int_literal("4ul"), Type::ULong);
    }

    #[test]
    fn test_decimal_too_big_for_int_falls_back_to_unsigned_long() {
        // 2^31 doesn't fit ILP32's `int` *or* `long` (same 32-bit signed
        // range) -- C89's decimal-literal table has no wider candidate,
        // so this falls back to `unsigned long` (see `classify_int`'s
        // post-loop fallback), matching real compilers' behavior here.
        assert_eq!(type_of_int_literal("2147483648"), Type::ULong);
    }

    #[test]
    fn test_hex_too_big_for_int_promotes_to_unsigned_int() {
        // Hex/octal candidates include unsigned int before long, unlike
        // decimal (C89 6.1.3.2's table) -- 0xffffffff is UInt, not Long.
        assert_eq!(type_of_int_literal("0xffffffff"), Type::UInt);
    }

    #[test]
    fn test_octal_literal_parses_in_base_8() {
        // 010 (octal) is 8, well within Int either way, but confirms the
        // leading-zero-implies-octal parse doesn't silently misparse.
        assert_eq!(type_of_int_literal("010"), Type::Int);
        assert_eq!(type_of_int_literal("0"), Type::Int);
    }

    #[test]
    fn test_float_suffixes() {
        assert_eq!(type_of_float_literal("3.14"), Type::Double);
        assert_eq!(type_of_float_literal("3.14f"), Type::Float);
        assert_eq!(type_of_float_literal("3.14L"), Type::LongDouble);
    }

    #[test]
    fn test_usual_arithmetic_conversions_int_and_double_is_double() {
        assert_eq!(
            usual_arithmetic_conversions(&Type::Int, &Type::Double),
            Type::Double
        );
    }

    #[test]
    fn test_usual_arithmetic_conversions_char_promotes_before_combining() {
        assert_eq!(
            usual_arithmetic_conversions(&Type::Char, &Type::Char),
            Type::Int
        );
    }

    #[test]
    fn test_ilp32_long_and_unsigned_int_combine_to_unsigned_long() {
        assert_eq!(
            usual_arithmetic_conversions(&Type::Long, &Type::UInt),
            Type::ULong
        );
    }

    fn specs(ts: Vec<TypeSpecifier>) -> DeclSpecifiers {
        DeclSpecifiers {
            storage: None,
            qualifiers: Vec::new(),
            type_specifiers: ts,
        }
    }

    #[test]
    fn test_plain_specifiers() {
        assert_eq!(
            type_from_specifiers(&specs(vec![TypeSpecifier::Int])),
            Type::Int
        );
        assert_eq!(
            type_from_specifiers(&specs(vec![TypeSpecifier::Unsigned, TypeSpecifier::Char])),
            Type::UChar
        );
        assert_eq!(
            type_from_specifiers(&specs(vec![
                TypeSpecifier::Unsigned,
                TypeSpecifier::Long,
                TypeSpecifier::Int
            ])),
            Type::ULong
        );
        assert_eq!(
            type_from_specifiers(&specs(vec![TypeSpecifier::TypedefName("fixed_t".into())])),
            Type::Named("fixed_t".into())
        );
    }

    #[test]
    fn test_implicit_int_with_no_specifiers() {
        assert_eq!(type_from_specifiers(&specs(vec![])), Type::Int);
    }

    #[test]
    fn test_pointer_cast_type_name() {
        let tn = TypeName {
            specifiers: specs(vec![TypeSpecifier::Void]),
            abstract_declarator: Some(Box::new(AbstractDeclarator {
                pointer_quals: vec![vec![]],
                direct: None,
            })),
        };
        assert_eq!(
            type_from_type_name(&tn),
            Type::Pointer(Box::new(Type::Void))
        );
    }

    #[test]
    fn test_array_of_pointers_vs_pointer_to_array() {
        // `int *a[3]`: array of pointers -- pointer_quals on the outer
        // declarator, Array directly wrapping the (absent) identifier.
        let array_of_pointers = TypeName {
            specifiers: specs(vec![TypeSpecifier::Int]),
            abstract_declarator: Some(Box::new(AbstractDeclarator {
                pointer_quals: vec![vec![]],
                direct: Some(DirectAbstractDeclarator::Array(None, None)),
            })),
        };
        assert_eq!(
            type_from_type_name(&array_of_pointers),
            Type::Array(Box::new(Type::Pointer(Box::new(Type::Int))))
        );

        // `int (*a)[3]`: pointer to array -- the pointer is inside a
        // parenthesized inner abstract declarator that the outer `[3]`
        // wraps around.
        let pointer_to_array = TypeName {
            specifiers: specs(vec![TypeSpecifier::Int]),
            abstract_declarator: Some(Box::new(AbstractDeclarator {
                pointer_quals: vec![],
                direct: Some(DirectAbstractDeclarator::Array(
                    Some(Box::new(DirectAbstractDeclarator::Paren(Box::new(
                        AbstractDeclarator {
                            pointer_quals: vec![vec![]],
                            direct: None,
                        },
                    )))),
                    None,
                )),
            })),
        };
        assert_eq!(
            type_from_type_name(&pointer_to_array),
            Type::Pointer(Box::new(Type::Array(Box::new(Type::Int))))
        );
    }

    #[test]
    fn test_qualified_pointer_still_just_wraps() {
        let tn = TypeName {
            specifiers: specs(vec![TypeSpecifier::Char]),
            abstract_declarator: Some(Box::new(AbstractDeclarator {
                pointer_quals: vec![vec![TypeQualifier::Const]],
                direct: None,
            })),
        };
        assert_eq!(
            type_from_type_name(&tn),
            Type::Pointer(Box::new(Type::Char))
        );
    }
}
