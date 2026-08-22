//! Investigation for `docs/03_TRANSPILER.md`'s decided-but-not-yet-implemented
//! enum representation (plain integer constants, not real Rust `enum`s):
//! before building a constant-expression evaluator to turn each variant's
//! `Option<Expr>` into an actual `i32` value, measure what shapes those
//! expressions actually take across the corpus, matching this project's
//! "measure, don't assume" methodology throughout the rest of Phase 0-3.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use transpiler::parser::ast::{BinaryOp, EnumSpec, Expr, ExternalDecl, TypeSpecifier, UnaryOp};
use transpiler::parser::parse_full;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Shape {
    /// No explicit value -- implicit "previous + 1" (or 0 for the first).
    Implicit,
    /// A bare integer literal, e.g. `= 5`.
    IntLiteral,
    /// Unary minus/plus/bitnot over a literal, e.g. `= -1`.
    UnaryOfLiteral,
    /// A binary op between two "simple" operands (literal, or unary-negated
    /// literal), e.g. `1 << 4` -- **not** counted here if either side is or
    /// contains an `Ident` (see `Shape::references_ident` below instead);
    /// an earlier version of this survey conflated "literal or ident" as
    /// both "simple," which silently hid a real macro-reference case
    /// (`30*TICRATE`) inside this bucket.
    SimpleBinary,
    /// References another named constant/macro directly, e.g. `= FOO`.
    IdentReference,
    /// Anything deeper (nested binary, casts, calls, ...) that doesn't
    /// itself contain an identifier reference.
    Other,
}

fn classify(e: &Expr) -> Shape {
    match e {
        Expr::IntLiteral(_) => Shape::IntLiteral,
        Expr::Ident(_) => Shape::IdentReference,
        Expr::Unary {
            op: UnaryOp::Minus | UnaryOp::Plus | UnaryOp::BitNot,
            expr,
        } if matches!(expr.as_ref(), Expr::IntLiteral(_)) => Shape::UnaryOfLiteral,
        Expr::Binary { lhs, rhs, .. } if is_simple(lhs) && is_simple(rhs) => Shape::SimpleBinary,
        _ => Shape::Other,
    }
}

fn is_simple(e: &Expr) -> bool {
    matches!(e, Expr::IntLiteral(_))
        || matches!(e, Expr::Unary { op: UnaryOp::Minus | UnaryOp::Plus | UnaryOp::BitNot, expr } if matches!(expr.as_ref(), Expr::IntLiteral(_)))
}

/// True if `e` contains an `Ident` node anywhere in its tree -- orthogonal
/// to `classify`'s top-level `Shape` bucket, since an identifier can be
/// buried inside an otherwise-literal-looking binary expression (`30 *
/// TICRATE`, a `#define`d macro, not another enum constant).
fn references_ident(e: &Expr) -> bool {
    match e {
        Expr::Ident(_) => true,
        Expr::IntLiteral(_)
        | Expr::FloatLiteral(_)
        | Expr::StringLiteral(_)
        | Expr::CharLiteral(_) => false,
        Expr::Unary { expr, .. } => references_ident(expr),
        Expr::Binary { lhs, rhs, .. } => references_ident(lhs) || references_ident(rhs),
        _ => true, // anything else (call, member, cast, ...) -- treat conservatively as "not pure"
    }
}

fn corpus_files(corpus_dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(ext))
        .collect();
    files.sort();
    files
}

fn collect_enum_specs(items: &[ExternalDecl], out: &mut Vec<EnumSpec>) {
    use transpiler::parser::ast::{Declaration, ExternalDecl as ED};
    fn from_decl(decl: &Declaration, out: &mut Vec<EnumSpec>) {
        for ts in &decl.specifiers.type_specifiers {
            if let TypeSpecifier::Enum(spec) = ts
                && spec.variants.is_some()
            {
                out.push(spec.clone());
            }
        }
    }
    for item in items {
        match item {
            ED::Declaration(decl) => from_decl(decl, out),
            ED::FunctionDef(_) => {} // enums aren't declared at function-def top level in this corpus
        }
    }
}

fn main() {
    let corpus_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
    let mut files = corpus_files(&corpus_dir, "c");
    files.extend(corpus_files(&corpus_dir, "h"));
    files.sort();

    let mut shape_counts: HashMap<Shape, usize> = HashMap::new();
    let mut binary_op_counts: HashMap<&str, usize> = HashMap::new();
    let mut total_enums = 0usize;
    let mut total_variants = 0usize;
    let mut other_examples: Vec<String> = Vec::new();
    let mut ident_reference_examples: Vec<String> = Vec::new();
    let mut variants_referencing_ident = 0usize;
    let mut unparsed_files = 0usize;

    for path in &files {
        let Ok((_, unit)) = parse_full(path.to_str().unwrap()) else {
            unparsed_files += 1;
            continue;
        };
        let mut specs = Vec::new();
        collect_enum_specs(&unit.items, &mut specs);
        for spec in specs {
            total_enums += 1;
            let variants = spec.variants.unwrap();
            for (name, value) in &variants {
                total_variants += 1;
                let shape = match value {
                    None => Shape::Implicit,
                    Some(e) => classify(e),
                };
                *shape_counts.entry(shape).or_default() += 1;
                if let Some(Expr::Binary { op, .. }) = value {
                    let op_name = match op {
                        BinaryOp::Shl => "Shl",
                        BinaryOp::Shr => "Shr",
                        BinaryOp::BitOr => "BitOr",
                        BinaryOp::BitAnd => "BitAnd",
                        BinaryOp::BitXor => "BitXor",
                        BinaryOp::Add => "Add",
                        BinaryOp::Sub => "Sub",
                        BinaryOp::Mul => "Mul",
                        _ => "other-binop",
                    };
                    *binary_op_counts.entry(op_name).or_default() += 1;
                }
                if shape == Shape::Other && other_examples.len() < 15 {
                    other_examples.push(format!(
                        "{}: {} = {:?}",
                        path.file_name().unwrap().to_string_lossy(),
                        name,
                        value
                    ));
                }
                if let Some(e) = value
                    && references_ident(e)
                {
                    variants_referencing_ident += 1;
                    if ident_reference_examples.len() < 15 {
                        ident_reference_examples.push(format!(
                            "{}: {} = {:?}",
                            path.file_name().unwrap().to_string_lossy(),
                            name,
                            e
                        ));
                    }
                }
            }
        }
    }

    println!(
        "{} files ({} unparsed), {total_enums} defining enum specs, {total_variants} total variants",
        files.len(),
        unparsed_files
    );
    let mut shapes: Vec<_> = shape_counts.into_iter().collect();
    shapes.sort_by_key(|s| std::cmp::Reverse(s.1));
    println!("shape breakdown: {shapes:?}");
    println!(
        "binary op breakdown (SimpleBinary + Other cases with a Binary top node): {binary_op_counts:?}"
    );
    println!("\nOther-shape examples (up to 15):");
    for ex in &other_examples {
        println!("  {ex}");
    }
    println!(
        "\n{variants_referencing_ident} of {total_variants} variants reference an identifier \
         somewhere in their value expression (a #define'd macro or another named constant) -- \
         these can't be literal-folded no matter how deep the surrounding arithmetic is:"
    );
    for ex in &ident_reference_examples {
        println!("  {ex}");
    }
}
