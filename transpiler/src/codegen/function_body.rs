//! Phase 3: Function-Body Transpilation (first slice)
//!
//! Everything else this codebase generates is data *shape*: module
//! structure, enum constants, struct fields, the `mobjinfo[]`/`states[]`
//! tables. This module is the first to render real C statements and
//! expressions as Rust -- starting narrowly, from one concretely verified
//! function (`T_FireFlicker`, `p_lights.c`), not as a general C-to-Rust
//! statement transpiler. `render_stmt`/`render_expr` only recognize the
//! AST shapes actually seen so far; anything else is a loud `Err`, never
//! a guess -- the same discipline every other renderer in this codebase
//! follows.
//!
//! **Cross-reference resolution**: `flick->sector` is a `SectorId` (a
//! plain index, not a real pointer -- see the memory-model decision in
//! docs/03_TRANSPILER.md), so `flick->sector->lightlevel` needs a
//! `World` to resolve the index back to a real `&mut Sector` before
//! continuing the member chain: `world[flick.sector].lightlevel`. This
//! renderer only resolves cross-references at the `self` parameter's own
//! *direct* fields (given by `self_field_types`, the same `MappedField`
//! list `struct_fields.rs` already produces) -- a cross-reference nested
//! inside some other translated struct isn't handled yet, since no real
//! function body has needed it so far.
//!
//! **`if (--flick->count) return;`**: Rust has no prefix `--` operator,
//! so a `PreIncDec` used directly as an `if`'s condition is hoisted into
//! its own statement immediately before the `if`, matching C's own
//! evaluation order (decrement happens, *then* the result is tested).
//!
//! **`switch`/`case` (`T_Glow`)**: C parses `case N: stmt` as a single
//! labeled statement wrapping only the *one* statement right after the
//! colon -- everything else in that case's body is a flat sibling in the
//! same enclosing block, up to the next `case`/`default` or a `break`.
//! `render_switch` re-groups those flat siblings back into one Rust match
//! arm's block per case, and requires each group to end in an explicit
//! `break` before the next label (or run to the end of the switch) --
//! C's implicit fallthrough (a case with no `break`) isn't recognized
//! yet, since no real function has needed it so far, and Rust `match`
//! arms don't fall through the same way regardless. A `switch` with no
//! `default:` gets an implicit trailing `_ => {}` arm, matching C's own
//! "no case matched, do nothing" semantics for a plain integer subject
//! (not a closed enum Rust could check exhaustiveness on directly).

use crate::parser::ast::{
    AssignOp, BinaryOp, BlockItem, Declaration, DirectDeclarator, Expr, ExternalDecl, FunctionDef,
    IncDecOp, ParamDeclarator, Stmt, TypeSpecifier, UnaryOp,
};
use crate::parser::grammar::declarator_name;
use crate::parser::parse_full;
use std::collections::HashMap;
use std::path::Path;

/// Rust types this renderer knows are `World`-indexed cross-references,
/// not plain values -- see module docs.
const CROSS_REF_TYPES: &[&str] = &["SectorId", "VertexId", "SideId", "LineId", "SubsectorId"];

fn is_cross_ref(rust_type: &str) -> bool {
    CROSS_REF_TYPES.contains(&rust_type)
}

struct FnBodyContext<'a> {
    self_param: &'a str,
    self_field_types: &'a HashMap<String, String>,
}

fn indent(depth: usize) -> String {
    "    ".repeat(depth)
}

fn binary_prec(op: BinaryOp) -> u8 {
    use BinaryOp::*;
    match op {
        Mul | Div | Mod => 10,
        Add | Sub => 9,
        Shl | Shr => 8,
        BitAnd => 7,
        BitXor => 6,
        BitOr => 5,
        Lt | Le | Gt | Ge => 4,
        Eq | Ne => 3,
        LogAnd => 2,
        LogOr => 1,
    }
}

fn render_binop(op: BinaryOp) -> &'static str {
    use BinaryOp::*;
    match op {
        Mul => "*",
        Div => "/",
        Mod => "%",
        Add => "+",
        Sub => "-",
        Shl => "<<",
        Shr => ">>",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
        Eq => "==",
        Ne => "!=",
        BitAnd => "&",
        BitXor => "^",
        BitOr => "|",
        LogAnd => "&&",
        LogOr => "||",
    }
}

fn is_comparison_or_logical(op: BinaryOp) -> bool {
    use BinaryOp::*;
    matches!(op, Lt | Le | Gt | Ge | Eq | Ne | LogAnd | LogOr)
}

fn render_assign_op(op: AssignOp) -> &'static str {
    use AssignOp::*;
    match op {
        Assign => "=",
        MulAssign => "*=",
        DivAssign => "/=",
        ModAssign => "%=",
        AddAssign => "+=",
        SubAssign => "-=",
        ShlAssign => "<<=",
        ShrAssign => ">>=",
        AndAssign => "&=",
        XorAssign => "^=",
        OrAssign => "|=",
    }
}

/// Renders `e`, returning `(text, is_unresolved_cross_ref)` -- the second
/// element is only ever `true` for a `Member` result naming a direct
/// cross-reference field of `self_param` (see module docs); every other
/// shape is a plain, already-resolved value.
fn render_expr(e: &Expr, ctx: &FnBodyContext) -> Result<(String, bool), String> {
    match e {
        Expr::IntLiteral(s) => Ok((s.clone(), false)),
        Expr::Ident(name) => Ok((name.clone(), false)),
        Expr::Member { base, field, .. } => {
            let (base_text, base_is_crossref) = render_expr(base, ctx)?;
            let base_text = if base_is_crossref {
                format!("world[{base_text}]")
            } else {
                base_text
            };
            let is_self_field = matches!(base.as_ref(), Expr::Ident(n) if n == ctx.self_param);
            let is_crossref = is_self_field
                && ctx
                    .self_field_types
                    .get(field.as_str())
                    .is_some_and(|t| is_cross_ref(t));
            Ok((format!("{base_text}.{field}"), is_crossref))
        }
        Expr::Binary { op, lhs, rhs } => {
            let prec = binary_prec(*op);
            let (lhs_text, _) = render_expr(lhs, ctx)?;
            let lhs_text = parenthesize_if_needed(lhs, &lhs_text, prec, false);
            let (rhs_text, _) = render_expr(rhs, ctx)?;
            let rhs_text = parenthesize_if_needed(rhs, &rhs_text, prec, true);
            Ok((
                format!("{lhs_text} {} {rhs_text}", render_binop(*op)),
                false,
            ))
        }
        Expr::Unary { op, expr } => {
            let op_text = match op {
                UnaryOp::Minus => "-",
                UnaryOp::Plus => "+",
                UnaryOp::Not | UnaryOp::BitNot => "!",
                UnaryOp::Deref | UnaryOp::AddrOf => {
                    return Err(format!(
                        "render_expr: unary {op:?} isn't supported yet -- translated code has no real pointers"
                    ));
                }
            };
            let (inner_text, _) = render_expr(expr, ctx)?;
            // Unary operators bind tighter than every binary operator this
            // renderer handles, so any binary child always needs parens.
            let inner_text = parenthesize_if_needed(expr, &inner_text, u8::MAX, false);
            Ok((format!("{op_text}{inner_text}"), false))
        }
        Expr::Call { callee, args } => {
            let (callee_text, _) = render_expr(callee, ctx)?;
            let mut rendered_args = Vec::with_capacity(args.len());
            for a in args {
                rendered_args.push(render_expr(a, ctx)?.0);
            }
            Ok((
                format!("{callee_text}({})", rendered_args.join(", ")),
                false,
            ))
        }
        _ => Err(format!("render_expr: unsupported expression shape: {e:?}")),
    }
}

/// Wraps `child_text` in parens if rendering it as an operand of a
/// `parent_prec`-precedence binary operator would otherwise change its
/// meaning -- Rust's operator precedence matches C's exactly for every
/// operator this renderer handles, so this is a standard precedence-
/// climbing pretty-printer rule: lower child precedence always needs
/// parens; equal precedence needs parens only on the right (these are all
/// left-associative operators).
fn parenthesize_if_needed(
    child: &Expr,
    child_text: &str,
    parent_prec: u8,
    is_right: bool,
) -> String {
    let Expr::Binary { op, .. } = child else {
        return child_text.to_string();
    };
    let child_prec = binary_prec(*op);
    if child_prec < parent_prec || (child_prec == parent_prec && is_right) {
        format!("({child_text})")
    } else {
        child_text.to_string()
    }
}

/// Renders `cond` as an `if`'s test, returning any statements that must
/// be hoisted immediately before it (only nonempty for the `--x`-as-
/// condition idiom -- see module docs) plus the condition text itself.
fn render_condition(
    cond: &Expr,
    ctx: &FnBodyContext,
    depth: usize,
) -> Result<(Vec<String>, String), String> {
    match cond {
        Expr::PreIncDec { expr, op } => {
            let (target_text, _) = render_expr(expr, ctx)?;
            let op_text = match op {
                IncDecOp::Inc => "+= 1",
                IncDecOp::Dec => "-= 1",
            };
            let hoisted = vec![format!("{}{target_text} {op_text};", indent(depth))];
            Ok((hoisted, format!("{target_text} != 0")))
        }
        Expr::Binary { op, .. } if is_comparison_or_logical(*op) => {
            Ok((Vec::new(), render_expr(cond, ctx)?.0))
        }
        _ => Err(format!(
            "render_condition: unsupported condition shape: {cond:?}"
        )),
    }
}

/// Renders `s` as the body lines of an `if`/`else` arm (or any other
/// brace-delimited block): a `Compound` renders its items directly at
/// `depth`; any other single statement is treated as a one-item block,
/// since Rust (unlike C) always requires braces here.
fn render_block(s: &Stmt, ctx: &FnBodyContext, depth: usize) -> Result<Vec<String>, String> {
    match s {
        Stmt::Compound(c) => render_compound_items(&c.items, ctx, depth),
        other => render_stmt(other, ctx, depth),
    }
}

fn render_decl(d: &Declaration, ctx: &FnBodyContext, depth: usize) -> Result<Vec<String>, String> {
    if !matches!(
        d.specifiers.type_specifiers.as_slice(),
        [TypeSpecifier::Int]
    ) {
        return Err(format!(
            "render_decl: only a bare `int` declaration is supported so far, got {:?}",
            d.specifiers.type_specifiers
        ));
    }
    let [decl] = d.declarators.as_slice() else {
        return Err("render_decl: only a single declarator is supported so far".to_string());
    };
    if decl.initializer.is_some() {
        return Err("render_decl: an initializer is not supported so far".to_string());
    }
    let name = declarator_name(&decl.declarator)
        .ok_or_else(|| "render_decl: declarator has no plain name".to_string())?;
    let _ = ctx;
    Ok(vec![format!("{}let mut {name};", indent(depth))])
}

fn render_stmt(s: &Stmt, ctx: &FnBodyContext, depth: usize) -> Result<Vec<String>, String> {
    match s {
        Stmt::Expr(Some(e)) => Ok(vec![format!(
            "{}{};",
            indent(depth),
            render_expr_stmt(e, ctx)?
        )]),
        Stmt::Return(None) => Ok(vec![format!("{}return;", indent(depth))]),
        Stmt::Return(Some(e)) => Ok(vec![format!(
            "{}return {};",
            indent(depth),
            render_expr(e, ctx)?.0
        )]),
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let (hoisted, cond_text) = render_condition(cond, ctx, depth)?;
            let mut lines = hoisted;
            lines.push(format!("{}if {cond_text} {{", indent(depth)));
            lines.extend(render_block(then_branch, ctx, depth + 1)?);
            match else_branch {
                None => lines.push(format!("{}}}", indent(depth))),
                Some(eb) => {
                    lines.push(format!("{}}} else {{", indent(depth)));
                    lines.extend(render_block(eb, ctx, depth + 1)?);
                    lines.push(format!("{}}}", indent(depth)));
                }
            }
            Ok(lines)
        }
        Stmt::Switch { cond, body } => render_switch(cond, body, ctx, depth),
        _ => Err(format!("render_stmt: unsupported statement shape: {s:?}")),
    }
}

/// Renders `switch (cond) { ... }` as a Rust `match` -- see module docs
/// for how C's per-statement `case`/`default` labels get re-grouped into
/// one block per arm, and why an implicit `_ => {}` is added when there's
/// no explicit `default:`.
fn render_switch(
    cond: &Expr,
    body: &Stmt,
    ctx: &FnBodyContext,
    depth: usize,
) -> Result<Vec<String>, String> {
    let (cond_text, _) = render_expr(cond, ctx)?;
    let Stmt::Compound(c) = body else {
        return Err("render_switch: only a compound switch body is supported so far".to_string());
    };
    let mut stmts: Vec<&Stmt> = Vec::with_capacity(c.items.len());
    for item in &c.items {
        let BlockItem::Stmt(s) = item else {
            return Err(
                "render_switch: only statements are supported directly inside a switch body, not declarations"
                    .to_string(),
            );
        };
        stmts.push(s);
    }

    let mut arms: Vec<(Option<String>, Vec<&Stmt>)> = Vec::new();
    let mut has_default = false;
    let mut i = 0;
    while i < stmts.len() {
        let (label, first_stmt) = match stmts[i] {
            Stmt::Case { expr, stmt } => (Some(render_expr(expr, ctx)?.0), stmt.as_ref()),
            Stmt::Default(stmt) => {
                has_default = true;
                (None, stmt.as_ref())
            }
            other => {
                return Err(format!(
                    "render_switch: expected a `case`/`default` label here, got {other:?}"
                ));
            }
        };
        i += 1;
        let mut arm_stmts = vec![first_stmt];
        let mut saw_break = false;
        while i < stmts.len() {
            match stmts[i] {
                Stmt::Case { .. } | Stmt::Default(_) => break,
                Stmt::Break => {
                    saw_break = true;
                    i += 1;
                    break;
                }
                other => {
                    arm_stmts.push(other);
                    i += 1;
                }
            }
        }
        if !saw_break && i < stmts.len() {
            return Err(
                "render_switch: fallthrough (a case with no `break` before the next case) isn't supported yet"
                    .to_string(),
            );
        }
        arms.push((label, arm_stmts));
    }

    let mut lines = vec![format!("{}match {cond_text} {{", indent(depth))];
    for (label, arm_stmts) in &arms {
        let pattern = label.clone().unwrap_or_else(|| "_".to_string());
        lines.push(format!("{}{pattern} => {{", indent(depth + 1)));
        for s in arm_stmts {
            lines.extend(render_stmt(s, ctx, depth + 2)?);
        }
        lines.push(format!("{}}}", indent(depth + 1)));
    }
    if !has_default {
        lines.push(format!("{}_ => {{}}", indent(depth + 1)));
    }
    lines.push(format!("{}}}", indent(depth)));
    Ok(lines)
}

/// `Expr::Assign` is rendered here (not in `render_expr`) since Doom's
/// tick functions only ever use it as a bare statement, never nested
/// inside a larger expression -- confirmed for `T_FireFlicker`, not yet
/// generalized.
fn render_expr_stmt(e: &Expr, ctx: &FnBodyContext) -> Result<String, String> {
    if let Expr::Assign { op, lhs, rhs } = e {
        let (lhs_text, _) = render_expr(lhs, ctx)?;
        let (rhs_text, _) = render_expr(rhs, ctx)?;
        Ok(format!("{lhs_text} {} {rhs_text}", render_assign_op(*op)))
    } else {
        Ok(render_expr(e, ctx)?.0)
    }
}

fn render_compound_items(
    items: &[BlockItem],
    ctx: &FnBodyContext,
    depth: usize,
) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for item in items {
        match item {
            BlockItem::Decl(d) => out.extend(render_decl(d, ctx, depth)?),
            BlockItem::Stmt(s) => out.extend(render_stmt(s, ctx, depth)?),
        }
    }
    Ok(out)
}

fn find_function_def<'a>(unit_items: &'a [ExternalDecl], fn_name: &str) -> Option<&'a FunctionDef> {
    unit_items.iter().find_map(|item| {
        let ExternalDecl::FunctionDef(f) = item else {
            return None;
        };
        let DirectDeclarator::Function(base, _) = &f.declarator.direct else {
            return None;
        };
        let DirectDeclarator::Ident(name) = base.as_ref() else {
            return None;
        };
        (name == fn_name).then_some(f)
    })
}

fn first_param_name(f: &FunctionDef) -> Option<String> {
    let DirectDeclarator::Function(_, params) = &f.declarator.direct else {
        return None;
    };
    let param = params.params.first()?;
    let ParamDeclarator::Named(d) = &param.declarator else {
        return None;
    };
    declarator_name(d)
}

/// Renders `fn_name` (found in `corpus_dir.join(file)`) as a real Rust
/// `pub fn`, given `self_rust_type` (the already-translated struct name
/// for its first parameter) and `self_field_types` (that struct's
/// `MappedField` list, for cross-reference resolution -- see module
/// docs). Every tick function needs `world: &mut World` alongside its
/// own struct, since resolving a cross-reference field requires it.
pub fn render_fn(
    corpus_dir: &Path,
    file: &str,
    fn_name: &str,
    self_rust_type: &str,
    self_field_types: &HashMap<String, String>,
) -> Result<String, String> {
    let (_, unit) = parse_full(corpus_dir.join(file).to_str().unwrap())?;
    let f = find_function_def(&unit.items, fn_name)
        .ok_or_else(|| format!("{fn_name} not found in {file}"))?;
    let param_name = first_param_name(f)
        .ok_or_else(|| format!("{fn_name}: first parameter has no plain name"))?;
    let ctx = FnBodyContext {
        self_param: &param_name,
        self_field_types,
    };
    let body_lines = render_compound_items(&f.body.items, &ctx, 1)?;
    Ok(format!(
        "pub fn {fn_name}({param_name}: &mut {self_rust_type}, world: &mut World) {{\n{}\n}}",
        body_lines.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    fn fireflicker_field_types() -> HashMap<String, String> {
        [
            ("sector".to_string(), "SectorId".to_string()),
            ("count".to_string(), "i32".to_string()),
            ("maxlight".to_string(), "i32".to_string()),
            ("minlight".to_string(), "i32".to_string()),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn test_t_fire_flicker_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_lights.c",
            "T_FireFlicker",
            "FireFlicker",
            &fireflicker_field_types(),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn T_FireFlicker(flick: &mut FireFlicker, world: &mut World) {
    let mut amount;
    flick.count -= 1;
    if flick.count != 0 {
        return;
    }
    amount = (P_Random() & 3) * 16;
    if world[flick.sector].lightlevel - amount < flick.minlight {
        world[flick.sector].lightlevel = flick.minlight;
    } else {
        world[flick.sector].lightlevel = flick.maxlight - amount;
    }
    flick.count = 4;
}";
        assert_eq!(rendered, expected);
    }

    fn light_effect_field_types() -> HashMap<String, String> {
        [
            ("sector".to_string(), "SectorId".to_string()),
            ("count".to_string(), "i32".to_string()),
            ("maxlight".to_string(), "i32".to_string()),
            ("minlight".to_string(), "i32".to_string()),
            ("maxtime".to_string(), "i32".to_string()),
            ("mintime".to_string(), "i32".to_string()),
            ("brighttime".to_string(), "i32".to_string()),
            ("darktime".to_string(), "i32".to_string()),
        ]
        .into_iter()
        .collect()
    }

    /// Structurally near-identical to `T_FireFlicker` (same `--x` guard,
    /// same cross-reference field), but the `if`'s own comparison is `==`
    /// rather than `<` -- confirms `is_comparison_or_logical` and the
    /// precedence renderer generalize beyond the one function they were
    /// built against, not hand-tuned to it alone.
    #[test]
    fn test_t_light_flash_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_lights.c",
            "T_LightFlash",
            "LightFlash",
            &light_effect_field_types(),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn T_LightFlash(flash: &mut LightFlash, world: &mut World) {
    flash.count -= 1;
    if flash.count != 0 {
        return;
    }
    if world[flash.sector].lightlevel == flash.maxlight {
        world[flash.sector].lightlevel = flash.minlight;
        flash.count = (P_Random() & flash.mintime) + 1;
    } else {
        world[flash.sector].lightlevel = flash.maxlight;
        flash.count = (P_Random() & flash.maxtime) + 1;
    }
}";
        assert_eq!(rendered, expected);
    }

    /// Same shape again, this time with a plain field-to-field assignment
    /// (`flash->count = flash->brighttime;`, no arithmetic at all) on one
    /// branch -- confirms a bare `Member` rhs needs no special handling.
    #[test]
    fn test_t_strobe_flash_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_lights.c",
            "T_StrobeFlash",
            "Strobe",
            &light_effect_field_types(),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn T_StrobeFlash(flash: &mut Strobe, world: &mut World) {
    flash.count -= 1;
    if flash.count != 0 {
        return;
    }
    if world[flash.sector].lightlevel == flash.minlight {
        world[flash.sector].lightlevel = flash.maxlight;
        flash.count = flash.brighttime;
    } else {
        world[flash.sector].lightlevel = flash.minlight;
        flash.count = flash.darktime;
    }
}";
        assert_eq!(rendered, expected);
    }

    /// A genuinely different control-flow shape from the other three
    /// (`switch`/`case` instead of straight-line `if`/`else`), plus
    /// compound-assignment operators (`+=`/`-=`) and a unary-negative
    /// case label (`case -1:`) -- exercises `render_switch` and
    /// `Expr::Unary` for the first time.
    #[test]
    fn test_t_glow_renders_exactly() {
        let field_types: HashMap<String, String> = [
            ("sector".to_string(), "SectorId".to_string()),
            ("minlight".to_string(), "i32".to_string()),
            ("maxlight".to_string(), "i32".to_string()),
            ("direction".to_string(), "i32".to_string()),
        ]
        .into_iter()
        .collect();
        let rendered = render_fn(&corpus_dir(), "p_lights.c", "T_Glow", "Glow", &field_types)
            .expect("should render cleanly");
        let expected = "\
pub fn T_Glow(g: &mut Glow, world: &mut World) {
    match g.direction {
        -1 => {
            world[g.sector].lightlevel -= GLOWSPEED;
            if world[g.sector].lightlevel <= g.minlight {
                world[g.sector].lightlevel += GLOWSPEED;
                g.direction = 1;
            }
        }
        1 => {
            world[g.sector].lightlevel += GLOWSPEED;
            if world[g.sector].lightlevel >= g.maxlight {
                world[g.sector].lightlevel -= GLOWSPEED;
                g.direction = -1;
            }
        }
        _ => {}
    }
}";
        assert_eq!(rendered, expected);
    }
}
