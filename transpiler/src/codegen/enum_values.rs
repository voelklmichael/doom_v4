//! Phase 3 Step 5: Enum Constant Values
//!
//! Implements `docs/03_TRANSPILER.md`'s enum-representation decision
//! (plain integer constants, not real Rust `enum`s -- Doom's state and
//! animation code does arithmetic directly on enum values, which a real
//! Rust `enum` has no direct equivalent for) by computing each variant's
//! actual integer value from `EnumSpec`'s `Option<Expr>`-per-variant AST.
//!
//! Measured first (`examples/enum_value_survey.rs`), matching this
//! project's usual practice before building an evaluator: across all 43
//! defining enum specs / 1642 variants in the corpus, values are either
//! implicit (1592, "previous + 1", 0 for the first) or built purely from
//! integer literals -- bare, unary-negated, or combined with `+`/`*`,
//! never deeper than a couple of levels. A small literal-only constant
//! folder covers 1638 of those 1642 (99.76%); the remaining 4
//! (`doomdef.h`'s `INVULNTICS`/`INVISTICS`/`INFRATICS`/`IRONTICS`, all
//! `N * TICRATE`) reference a `#define`d macro rather than a literal, which
//! this folder deliberately doesn't chase -- narrow enough a gap (four
//! variants, one macro) not to pull in full macro substitution just for
//! them, matching `visibility.rs`'s own "not worth more machinery for a
//! two-symbol gap" precedent.
//!
//! **A second gap, found while starting to actually render struct fields
//! against this module's output**: `doomtype.h`'s `typedef enum {false,
//! true} boolean;` folds its two variants to `0`/`1` cleanly (no macro
//! involved), but rendering them as `pub const false: i32 = 0;` doesn't
//! parse -- `false`/`true` are Rust *strict* keywords, not usable as an
//! identifier even via `r#raw` escaping. `render_enum_consts` now skips
//! any variant whose name collides with one (corpus-wide, only this one
//! enum's variants do); moot for `boolean` specifically anyway, once a
//! `boolean`-typed field maps to Rust's own native `bool` rather than
//! going through this module's `i32`-constant treatment at all.

use crate::parser::ast::{BinaryOp, EnumSpec, Expr, UnaryOp};

/// Parses a C integer literal token's numeric value (decimal/octal/hex,
/// with any trailing `u`/`U`/`l`/`L` suffix stripped) -- same suffix/radix
/// logic as `typecheck::types::type_of_int_literal`, which only exposes
/// the literal's *type*, not its value.
fn parse_int_literal(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    let mut end = bytes.len();
    while end > 0 && matches!(bytes[end - 1], b'u' | b'U' | b'l' | b'L') {
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
    i64::from_str_radix(digits, radix).ok()
}

/// Folds a constant integer expression, if `e`'s shape is one the corpus
/// actually uses for enum values: integer literals, unary `+`/`-`/`~` over
/// a foldable operand, and arithmetic/bitwise binary ops between two
/// foldable operands. `None` for anything else (an identifier reference, a
/// call, ...) -- never seen in the corpus's own enum values (see this
/// module's docs), but callers treat a `None` as "can't render this
/// constant" rather than guessing at one.
pub fn fold_const_int(e: &Expr) -> Option<i64> {
    match e {
        Expr::IntLiteral(text) => parse_int_literal(text),
        Expr::Unary { op, expr } => {
            let v = fold_const_int(expr)?;
            match op {
                UnaryOp::Minus => Some(-v),
                UnaryOp::Plus => Some(v),
                UnaryOp::BitNot => Some(!v),
                UnaryOp::Not | UnaryOp::Deref | UnaryOp::AddrOf => None,
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let a = fold_const_int(lhs)?;
            let b = fold_const_int(rhs)?;
            match op {
                BinaryOp::Add => Some(a + b),
                BinaryOp::Sub => Some(a - b),
                BinaryOp::Mul => Some(a * b),
                BinaryOp::Div if b != 0 => Some(a / b),
                BinaryOp::Mod if b != 0 => Some(a % b),
                BinaryOp::Shl => Some(a << b),
                BinaryOp::Shr => Some(a >> b),
                BinaryOp::BitAnd => Some(a & b),
                BinaryOp::BitOr => Some(a | b),
                BinaryOp::BitXor => Some(a ^ b),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Computes every variant's actual value, per C89's sequencing rule: an
/// explicit value resets the running counter, an absent one continues it
/// (`prev + 1`, 0 for the first variant) -- regardless of whether the
/// *previous* variant's own value was explicit or implicit. `None` for a
/// variant whose expression `fold_const_int` can't fold, paired with
/// `Some` for every other -- partial per-variant, not all-or-nothing,
/// since one unfoldable variant shouldn't hide the rest.
pub fn compute_enum_values(spec: &EnumSpec) -> Vec<(String, Option<i64>)> {
    let mut current = 0i64;
    let mut out = Vec::new();
    let Some(variants) = &spec.variants else {
        return out;
    };
    for (name, value) in variants {
        let resolved = match value {
            Some(e) => fold_const_int(e),
            None => Some(current),
        };
        out.push((name.clone(), resolved));
        current = resolved.map_or(current + 1, |v| v + 1);
    }
    out
}

/// Rust strict keywords: never usable as an identifier, not even via
/// `r#raw` escaping (unlike a contextual/reserved keyword). Corpus-wide,
/// only one enum's variants collide with this list at all --
/// `doomtype.h`'s `typedef enum {false, true} boolean;` -- and that whole
/// enum's constants are moot anyway once a `boolean`-typed field maps to
/// Rust's own native `bool` (its real `false`/`true` literals already
/// exist as language builtins). Kept as a general check here, not a
/// `boolean`-specific special case, as defense in depth for any other enum
/// a future corpus revision might add.
const RUST_STRICT_KEYWORDS: &[&str] = &[
    "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn", "for",
    "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
    "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe", "use", "where",
    "while", "dyn", "async", "await",
];

/// Renders `spec`'s variants as `pub const NAME: i32 = value;` lines, one
/// per successfully-folded, non-keyword-colliding variant. An unfoldable
/// variant is skipped, not panicked on -- never happens on the real corpus
/// (see this module's docs), but the renderer shouldn't crash the whole
/// run over it either; same for a variant whose name collides with a Rust
/// keyword (`pub const false: i32 = 0;` doesn't parse).
pub fn render_enum_consts(spec: &EnumSpec) -> String {
    compute_enum_values(spec)
        .into_iter()
        .filter_map(|(name, value)| {
            if RUST_STRICT_KEYWORDS.contains(&name.as_str()) {
                return None;
            }
            value.map(|v| format!("pub const {name}: i32 = {v};"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{ExternalDecl, TypeSpecifier};
    use crate::parser::parse_full;
    use std::path::{Path, PathBuf};

    fn parse_enum(src: &str) -> EnumSpec {
        let full = format!("{src};");
        let (_, chunks) = crate::parser::parse_chunks(&full);
        let mut env = crate::parser::PreprocessorEnv::linux_doom_defaults();
        let resolved = crate::parser::resolve_conditionals(&chunks, &mut env).unwrap();
        let entries = crate::parser::lex_chunks(&resolved).unwrap();
        let stream = crate::parser::attach_comments(entries);
        let unit = crate::parser::parse_translation_unit(&stream).unwrap();
        for item in &unit.items {
            if let ExternalDecl::Declaration(decl) = item {
                for ts in &decl.specifiers.type_specifiers {
                    if let TypeSpecifier::Enum(spec) = ts {
                        return spec.clone();
                    }
                }
            }
        }
        panic!("no enum spec found in: {src}");
    }

    #[test]
    fn test_implicit_sequencing_starts_at_zero() {
        let spec = parse_enum("enum { A, B, C }");
        assert_eq!(
            compute_enum_values(&spec),
            vec![
                ("A".to_string(), Some(0)),
                ("B".to_string(), Some(1)),
                ("C".to_string(), Some(2)),
            ]
        );
    }

    #[test]
    fn test_explicit_value_resets_the_counter() {
        let spec = parse_enum("enum { A = 5, B, C = 10, D }");
        assert_eq!(
            compute_enum_values(&spec),
            vec![
                ("A".to_string(), Some(5)),
                ("B".to_string(), Some(6)),
                ("C".to_string(), Some(10)),
                ("D".to_string(), Some(11)),
            ]
        );
    }

    #[test]
    fn test_unary_minus_literal() {
        let spec = parse_enum("enum { NEG = -1 }");
        assert_eq!(
            compute_enum_values(&spec),
            vec![("NEG".to_string(), Some(-1))]
        );
    }

    #[test]
    fn test_nested_add_of_literals() {
        // Real corpus shape: BT_WEAPONMASK = 8+16+32 (parses as
        // Binary(Binary(8,16),32), not "simple" but still fully foldable).
        let spec = parse_enum("enum { MASK = 8+16+32 }");
        assert_eq!(
            compute_enum_values(&spec),
            vec![("MASK".to_string(), Some(56))]
        );
    }

    #[test]
    fn test_unfoldable_expression_yields_none_without_panicking() {
        let spec = parse_enum("enum { REF = SOME_OTHER_NAME }");
        assert_eq!(compute_enum_values(&spec), vec![("REF".to_string(), None)]);
        assert_eq!(render_enum_consts(&spec), "");
    }

    #[test]
    fn test_render_shape() {
        let spec = parse_enum("enum { A, B = 5 }");
        assert_eq!(
            render_enum_consts(&spec),
            "pub const A: i32 = 0;\npub const B: i32 = 5;"
        );
    }

    #[test]
    fn test_render_skips_rust_keyword_colliding_variants() {
        // doomtype.h's real shape: typedef enum {false, true} boolean;
        let spec = parse_enum("enum { false, true }");
        assert_eq!(
            render_enum_consts(&spec),
            "",
            "false/true can't be Rust identifiers, even escaped"
        );
    }

    #[test]
    fn test_render_skips_only_the_colliding_variant_not_its_siblings() {
        let spec = parse_enum("enum { A, false, B }");
        assert_eq!(
            render_enum_consts(&spec),
            "pub const A: i32 = 0;\npub const B: i32 = 2;"
        );
    }

    fn corpus_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_corpus_enum_value_coverage() {
        // Not a pass/fail assertion (matching this project's "measure,
        // don't assume" methodology) -- reports how many of the corpus's
        // real enum variants fold successfully, for cross-checking against
        // examples/enum_value_survey.rs's own findings (1638/1642 expected
        // -- the 4 doomdef.h TICRATE-multiple variants reference a macro,
        // not a literal; see this module's own docs).
        let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("c") | Some("h")
                )
            })
            .collect();
        files.sort();
        assert!(files.len() > 50, "expected the full Doom .c/.h corpus");

        let mut total = 0;
        let mut folded = 0;
        for path in &files {
            let Ok((_, unit)) = parse_full(path.to_str().unwrap()) else {
                continue;
            };
            for item in &unit.items {
                if let ExternalDecl::Declaration(decl) = item {
                    for ts in &decl.specifiers.type_specifiers {
                        if let TypeSpecifier::Enum(spec) = ts {
                            if spec.variants.is_some() {
                                for (_, value) in compute_enum_values(spec) {
                                    total += 1;
                                    if value.is_some() {
                                        folded += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        eprintln!("enum constant value folding: {folded} of {total} variants folded successfully");
        assert!(total > 0);
    }
}
