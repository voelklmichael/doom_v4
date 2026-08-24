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
//!
//! **`P_Spawn*` constructor functions (`render_spawn_fn`)**: a genuinely
//! different idiom from a tick function's straight-line/`if`/`switch`
//! logic, not just new statement/expression shapes -- see its own doc
//! comment for the full reasoning. `Z_Malloc` + `P_AddThinker` + a field-
//! by-field imperative fill-in becomes one `Thinker::Variant(Struct {
//! ... })` literal handed to `Arena::insert`, reordering statements
//! (every field write groups at the `insert` call) in a way that's sound
//! only because C's single-threaded, synchronous execution means nothing
//! ever observes the value mid-construction.

use crate::codegen::struct_fields::rust_field_name;
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
    /// Other identifiers (typically a constructor function's own
    /// parameters, e.g. `sector: SectorId`) whose *own* declared type is
    /// directly a cross-reference index type -- distinct from
    /// `self_field_types`, which is about a *field of* `self_param`.
    /// Empty for ordinary tick functions, whose only cross-references are
    /// through `self`'s own fields.
    extra_cross_ref_idents: &'a HashMap<String, String>,
    /// A constructor function's own not-yet-fully-built local (e.g.
    /// `flash` in `P_SpawnLightFlash`), if any. A field access rooted at
    /// it (`flash->maxtime`, referencing a field `render_spawn_fn` has
    /// already emitted as `let maxtime = ...;`) resolves to that bare
    /// local name rather than `world[...]` indexing or a plain `.field`
    /// access -- unlike `self_param`, this isn't a real existing value to
    /// dereference, just a name for fields being built up one `let` at a
    /// time. Empty for ordinary tick functions.
    ctor_var: &'a str,
    /// When non-empty, a *bare* `ctor_var` reference (not `ctor_var->
    /// field`) resolves to this name instead of erroring -- used only
    /// once the constructed value has actually been inserted into its
    /// `Arena` and bound to a `let handle = ...;` local, for rendering a
    /// back-reference statement (`sec->specialdata = door;`) that comes
    /// *after* the insert in the generated output. Empty everywhere else,
    /// including while a constructor's own field expressions are still
    /// being rendered (where a bare `ctor_var` reference would be a bug,
    /// not a back-reference -- see `render_spawn_fn`).
    ctor_var_handle_name: &'a str,
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
        Expr::Ident(name) => {
            if !ctx.ctor_var_handle_name.is_empty()
                && !ctx.ctor_var.is_empty()
                && name == ctx.ctor_var
            {
                return Ok((ctx.ctor_var_handle_name.to_string(), false));
            }
            let is_crossref = ctx
                .extra_cross_ref_idents
                .get(name.as_str())
                .is_some_and(|t| is_cross_ref(t));
            Ok((name.clone(), is_crossref))
        }
        Expr::Member { base, field, .. } => {
            if !ctx.ctor_var.is_empty()
                && matches!(base.as_ref(), Expr::Ident(n) if n == ctx.ctor_var)
            {
                return Ok((rust_field_name(field)?, false));
            }
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
        Expr::Unary {
            op: UnaryOp::AddrOf,
            expr,
        } if matches!(
            expr.as_ref(),
            Expr::Index { base, .. } if matches!(base.as_ref(), Expr::Ident(n) if n == "sectors")
        ) =>
        {
            // `&sectors[i]` -- the corpus's own idiom for getting a
            // `sector_t*` by index (`sectors` is the level's global
            // sector array). Since `sector_t*` already maps to `SectorId`
            // (a plain index, not a real pointer -- see the memory-model
            // decision), this needs no real address-of or array-index
            // operation at all, just wrapping the index itself.
            let Expr::Index { index, .. } = expr.as_ref() else {
                unreachable!("guarded above")
            };
            let (index_text, _) = render_expr(index, ctx)?;
            Ok((format!("SectorId({index_text} as u32)"), false))
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

/// Renders `cond` as a plain boolean-valued Rust expression -- a
/// comparison/logical `Binary` passes straight through (already `bool`-
/// valued); `!x` on a plain (non-`bool`) C value is C's truthiness test,
/// so it becomes `x == 0`, not Rust's own `!` (which would silently
/// compile as a *bitwise* NOT on an integer, a real, wrong-behavior trap
/// -- this renderer has no per-identifier `bool`-vs-`int` tracking beyond
/// this, so it's applied unconditionally for now; a genuinely `bool`-
/// typed operand would need this revisited, not encountered yet). Used
/// both by `render_condition` (an `if` statement's test) and by the
/// if/else-as-expression field synthesis in `render_spawn_fn` (a
/// condition with no statements to hoist).
fn render_bool_expr(cond: &Expr, ctx: &FnBodyContext) -> Result<String, String> {
    match cond {
        Expr::Binary { op, .. } if is_comparison_or_logical(*op) => Ok(render_expr(cond, ctx)?.0),
        Expr::Unary {
            op: UnaryOp::Not,
            expr,
        } => Ok(format!("{} == 0", render_expr(expr, ctx)?.0)),
        // A bare value used for truthiness (not a comparison/negation) --
        // C's `if (x)` tests non-zero/non-null. `specialdata` is the one
        // corpus field known to be `Option`-typed (`struct_fields.rs`'s
        // own name-based special case, reused by `render_expr_stmt`'s
        // `Some(..)`-wrapping too), so a bare reference to it needs
        // `.is_some()`, not the `== 0` truthiness every other (plain
        // `int`) value gets -- nothing else this renderer handles is
        // `Option`-typed yet.
        Expr::Member { field, .. } if field == "specialdata" => {
            Ok(format!("{}.is_some()", render_expr(cond, ctx)?.0))
        }
        _ => Err(format!(
            "render_bool_expr: unsupported condition shape: {cond:?}"
        )),
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
        _ => Ok((Vec::new(), render_bool_expr(cond, ctx)?)),
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
    // `int amount;` (a plain scalar) and `sector_t* sec;` (a single
    // pointer to an already-known cross-reference type, e.g.
    // `EV_StartLightStrobing`'s own loop variable) both render the same
    // way: Rust infers the type from later use, so no annotation is
    // needed regardless of which C type it was. Anything else (arrays,
    // multiple declarators, an initializer) isn't supported yet.
    if !matches!(
        d.specifiers.type_specifiers.as_slice(),
        [TypeSpecifier::Int] | [TypeSpecifier::TypedefName(_)]
    ) {
        return Err(format!(
            "render_decl: only a bare `int` or single-pointer known-type declaration is supported so far, got {:?}",
            d.specifiers.type_specifiers
        ));
    }
    let [decl] = d.declarators.as_slice() else {
        return Err("render_decl: only a single declarator is supported so far".to_string());
    };
    if decl.initializer.is_some() {
        return Err("render_decl: an initializer is not supported so far".to_string());
    }
    if !matches!(decl.declarator.direct, DirectDeclarator::Ident(_)) {
        return Err(
            "render_decl: only a plain (non-array, non-function) declarator is supported so far"
                .to_string(),
        );
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
        Stmt::While { cond, body } => render_while(cond, body, ctx, depth),
        Stmt::Continue => Ok(vec![format!("{}continue;", indent(depth))]),
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

/// Renders `while (cond) body` -- so far, only the `while ((x = f(..))
/// CMP y) { .. }` idiom Doom's tagged-object iteration uses everywhere
/// (`while ((secnum = P_FindSectorFromLineTag(line,secnum)) >= 0)`).
/// Rust's `while` re-evaluates a *fresh* condition expression each pass
/// with no way to inject a statement first (unlike the `--x`-as-condition
/// idiom `render_condition` hoists ahead of an `if`, which only needs to
/// run *once*), so this becomes `loop { <hoisted assignment>; if !(<test>)
/// { break; } <body> }` instead: the assignment runs every pass, exactly
/// where C's own re-evaluation would run it, and the `loop` gives
/// somewhere to put it.
fn render_while(
    cond: &Expr,
    body: &Stmt,
    ctx: &FnBodyContext,
    depth: usize,
) -> Result<Vec<String>, String> {
    let Expr::Binary { op, lhs, rhs } = cond else {
        return Err(format!(
            "render_while: unsupported condition shape: {cond:?}"
        ));
    };
    if !is_comparison_or_logical(*op) {
        return Err(format!(
            "render_while: unsupported condition shape: {cond:?}"
        ));
    }
    let Expr::Assign {
        op: AssignOp::Assign,
        lhs: assign_lhs,
        rhs: assign_rhs,
    } = lhs.as_ref()
    else {
        return Err(
            "render_while: only the `(x = f()) CMP y` condition idiom is supported so far"
                .to_string(),
        );
    };
    let (target_text, _) = render_expr(assign_lhs, ctx)?;
    let (value_text, _) = render_expr(assign_rhs, ctx)?;
    let (rhs_text, _) = render_expr(rhs, ctx)?;
    let test = format!("{target_text} {} {rhs_text}", render_binop(*op));

    let mut lines = vec![format!("{}loop {{", indent(depth))];
    lines.push(format!(
        "{}{target_text} = {value_text};",
        indent(depth + 1)
    ));
    lines.push(format!("{}if !({test}) {{", indent(depth + 1)));
    lines.push(format!("{}break;", indent(depth + 2)));
    lines.push(format!("{}}}", indent(depth + 1)));
    lines.extend(render_block(body, ctx, depth + 1)?);
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
        // `sector_t.specialdata`/`line_t.specialdata` map to
        // `Option<Handle<Thinker>>` (struct_fields.rs's own name-based
        // special case -- it's checked for truthiness/reset to `NULL`
        // corpus-wide, not dereferenced unconditionally), so a
        // constructor's back-reference assignment to it (`sec->
        // specialdata = door;`, only reachable once `ctor_var_handle_name`
        // is active -- see module docs) needs the same `Some(..)`
        // wrapping every other `Option`-typed field gets from its own
        // corpus initializer, even though this renderer has no general
        // per-field-type awareness beyond this one matching special case.
        let needs_option_wrap = !ctx.ctor_var_handle_name.is_empty()
            && matches!(lhs.as_ref(), Expr::Member { field, .. } if field == "specialdata")
            && matches!(rhs.as_ref(), Expr::Ident(n) if n == ctx.ctor_var);
        let rhs_text = if needs_option_wrap {
            format!("Some({rhs_text})")
        } else {
            rhs_text
        };
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
    let no_extra_cross_refs = HashMap::new();
    let ctx = FnBodyContext {
        self_param: &param_name,
        self_field_types,
        extra_cross_ref_idents: &no_extra_cross_refs,
        ctor_var: "",
        ctor_var_handle_name: "",
    };
    let body_lines = render_compound_items(&f.body.items, &ctx, 1)?;
    Ok(format!(
        "pub fn {fn_name}({param_name}: &mut {self_rust_type}, world: &mut World) {{\n{}\n}}",
        body_lines.join("\n")
    ))
}

/// True for `var = Z_Malloc(...);` -- the allocation call itself, always
/// discarded (see module docs: `render_spawn_fn` replaces it with
/// `Arena::insert`).
fn is_malloc_assign(s: &Stmt, ctor_var: &str) -> bool {
    let Stmt::Expr(Some(Expr::Assign {
        op: AssignOp::Assign,
        lhs,
        rhs,
    })) = s
    else {
        return false;
    };
    matches!(lhs.as_ref(), Expr::Ident(n) if n == ctor_var)
        && matches!(rhs.as_ref(), Expr::Call { callee, .. } if matches!(callee.as_ref(), Expr::Ident(n) if n == "Z_Malloc"))
}

/// True for `P_AddThinker(&var->thinker);` -- always discarded, same
/// reason.
fn is_add_thinker_call(s: &Stmt) -> bool {
    let Stmt::Expr(Some(Expr::Call { callee, .. })) = s else {
        return false;
    };
    matches!(callee.as_ref(), Expr::Ident(n) if n == "P_AddThinker")
}

/// True for `var->thinker.function.acpN = (cast) FnName;` -- the deepest
/// `Member` in the chain (closest to `var`) names field `"thinker"`,
/// regardless of how many `.function`/`.acpN` levels sit on top of it.
/// Always discarded: the enum variant tag already encodes which function
/// this is.
fn is_function_pointer_assign(s: &Stmt, ctor_var: &str) -> bool {
    fn innermost_base_and_field(e: &Expr) -> Option<(&str, &str)> {
        let Expr::Member { base, field, .. } = e else {
            return None;
        };
        match base.as_ref() {
            Expr::Ident(name) => Some((name.as_str(), field.as_str())),
            inner => innermost_base_and_field(inner),
        }
    }
    let Stmt::Expr(Some(Expr::Assign { lhs, .. })) = s else {
        return false;
    };
    innermost_base_and_field(lhs)
        .is_some_and(|(base, field)| base == ctor_var && field == "thinker")
}

/// `var->field = expr;`, split into which field and the right-hand side
/// -- used both to recognize `var->thinker.function.acpN = (cast) FnName;`
/// (the `field == "thinker"` case, discarded -- the enum variant tag
/// already encodes which function this is) and every other field, which
/// becomes one constructor-literal field.
fn ctor_field_assign<'a>(s: &'a Stmt, ctor_var: &str) -> Option<(&'a str, &'a Expr)> {
    let Stmt::Expr(Some(Expr::Assign {
        op: AssignOp::Assign,
        lhs,
        rhs,
    })) = s
    else {
        return None;
    };
    let Expr::Member {
        base,
        field,
        arrow: true,
    } = lhs.as_ref()
    else {
        return None;
    };
    if matches!(base.as_ref(), Expr::Ident(n) if n == ctor_var) {
        Some((field.as_str(), rhs))
    } else {
        None
    }
}

/// `other->field = var;` -- assigning the constructed value itself (not
/// one of *its* fields) to some field of a different, already-existing
/// object, e.g. `p_doors.c`'s `sec->specialdata = door;` (the sector's
/// own back-reference to the mover thinker currently active on it). Only
/// resolvable once `var` has become a real `Handle<Thinker>`, so
/// `render_spawn_fn` renders the whole function in two phases whenever
/// this appears anywhere in the body: every constructor field first, then
/// the `Arena::insert` call bound to `let handle = ...;`, then every
/// "other" statement (this one included) rendered with a bare `var`
/// resolving to `handle` (`FnBodyContext::ctor_var_handle_name`).
fn is_ctor_var_backreference(s: &Stmt, ctor_var: &str) -> bool {
    matches!(s, Stmt::Expr(Some(Expr::Assign { op: AssignOp::Assign, rhs, .. }))
        if matches!(rhs.as_ref(), Expr::Ident(n) if n == ctor_var))
}

fn body_has_backreference(items: &[BlockItem], ctor_var: &str) -> bool {
    items
        .iter()
        .any(|item| matches!(item, BlockItem::Stmt(s) if is_ctor_var_backreference(s, ctor_var)))
}

/// `if (cond) { var->field = then_val; } else { var->field = else_val; }`
/// -- both branches assigning the *same* field, with no `else if` chain
/// (each branch a single plain statement, not a further nested `If`).
/// This is the "field defined entirely by a condition" idiom
/// (`P_SpawnStrobeFlash`'s `count`, which has no unconditional assignment
/// anywhere else in the function) -- distinct from a single-branch
/// conditional *override* of an already-`let`-bound field
/// (`P_SpawnStrobeFlash`'s `minlight`, handled by the ordinary
/// `render_stmt` path once that field's initial `let` is marked `mut`),
/// which this deliberately does not match (no `else`).
fn if_else_ctor_field_assign<'a>(
    s: &'a Stmt,
    ctor_var: &str,
) -> Option<(&'a str, &'a Expr, &'a Expr)> {
    let Stmt::If {
        then_branch,
        else_branch: Some(else_branch),
        ..
    } = s
    else {
        return None;
    };
    let (then_field, then_rhs) = ctor_field_assign(then_branch, ctor_var)?;
    let (else_field, else_rhs) = ctor_field_assign(else_branch, ctor_var)?;
    (then_field == else_field).then_some((then_field, then_rhs, else_rhs))
}

/// Counts, for every field of `ctor_var`, how many statements assign it
/// -- anywhere in the function body, including nested inside `if`/`switch`
/// bodies, not just at the top level. A field assigned more than once
/// needs its initial `let` marked `mut` (`P_SpawnStrobeFlash`'s
/// `minlight`: one unconditional set, then a conditional override).
fn count_ctor_field_assigns(
    items: &[BlockItem],
    ctor_var: &str,
    counts: &mut HashMap<String, usize>,
) {
    for item in items {
        if let BlockItem::Stmt(s) = item {
            count_ctor_field_assigns_stmt(s, ctor_var, counts);
        }
    }
}

/// Like `ctor_field_assign`, but for *any* assignment operator, not just
/// plain `=` -- used only for the `mut` pre-scan, since a field can be
/// unconditionally refined by a compound assignment right after its own
/// initial value (`P_SpawnDoorRaiseIn5Mins`'s `door->topheight =
/// P_FindLowestCeilingSurrounding(sec); door->topheight -= 4*FRACUNIT;`),
/// which still needs `let mut` even though `ctor_field_assign` itself
/// (rightly) never treats a compound assignment as a field's *initial*
/// value.
fn ctor_field_assign_target<'a>(s: &'a Stmt, ctor_var: &str) -> Option<&'a str> {
    let Stmt::Expr(Some(Expr::Assign { lhs, .. })) = s else {
        return None;
    };
    let Expr::Member {
        base,
        field,
        arrow: true,
    } = lhs.as_ref()
    else {
        return None;
    };
    matches!(base.as_ref(), Expr::Ident(n) if n == ctor_var).then_some(field.as_str())
}

fn count_ctor_field_assigns_stmt(s: &Stmt, ctor_var: &str, counts: &mut HashMap<String, usize>) {
    if let Some(field) = ctor_field_assign_target(s, ctor_var) {
        *counts.entry(field.to_string()).or_insert(0) += 1;
        return;
    }
    match s {
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            count_ctor_field_assigns_stmt(then_branch, ctor_var, counts);
            if let Some(eb) = else_branch {
                count_ctor_field_assigns_stmt(eb, ctor_var, counts);
            }
        }
        Stmt::Switch { body, .. } => count_ctor_field_assigns_stmt(body, ctor_var, counts),
        Stmt::Compound(c) => count_ctor_field_assigns(&c.items, ctor_var, counts),
        _ => {}
    }
}

/// A conservative, `render_expr`/`render_stmt`-shaped walk: `true` if
/// `ctor_var` appears anywhere *other than* as the base of a
/// `ctor_var->field` member access (which `FnBodyContext::ctor_var`
/// already resolves to that field's own `let`-bound local, so it's fine
/// wherever `render_stmt`/`render_expr` themselves would accept it) --
/// and `true` (defensively) for any expression/statement shape this
/// walk doesn't specifically recognize, so `render_spawn_fn` errs loudly
/// on a statement it can't prove is safe, rather than silently
/// mistranslating a bare reference to the not-yet-fully-built value
/// (`sec->specialdata = door;`, seen in `p_doors.c`'s door spawners, isn't
/// supported yet -- see `docs/03_TRANSPILER.md`).
fn bare_ctor_ident_used(e: &Expr, ctor_var: &str) -> bool {
    match e {
        Expr::Ident(n) => n == ctor_var,
        Expr::IntLiteral(_) => false,
        Expr::Member { base, .. } => {
            if matches!(base.as_ref(), Expr::Ident(n) if n == ctor_var) {
                false
            } else {
                bare_ctor_ident_used(base, ctor_var)
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            bare_ctor_ident_used(lhs, ctor_var) || bare_ctor_ident_used(rhs, ctor_var)
        }
        Expr::Unary { expr, .. } => bare_ctor_ident_used(expr, ctor_var),
        Expr::Call { callee, args } => {
            bare_ctor_ident_used(callee, ctor_var)
                || args.iter().any(|a| bare_ctor_ident_used(a, ctor_var))
        }
        Expr::Assign { lhs, rhs, .. } => {
            bare_ctor_ident_used(lhs, ctor_var) || bare_ctor_ident_used(rhs, ctor_var)
        }
        _ => true,
    }
}

fn stmt_uses_bare_ctor_ident(s: &Stmt, ctor_var: &str) -> bool {
    match s {
        Stmt::Expr(Some(e)) => bare_ctor_ident_used(e, ctor_var),
        Stmt::Expr(None) => false,
        Stmt::Return(Some(e)) => bare_ctor_ident_used(e, ctor_var),
        Stmt::Return(None) => false,
        Stmt::Break => false,
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => {
            bare_ctor_ident_used(cond, ctor_var)
                || stmt_uses_bare_ctor_ident(then_branch, ctor_var)
                || else_branch
                    .as_ref()
                    .is_some_and(|eb| stmt_uses_bare_ctor_ident(eb, ctor_var))
        }
        Stmt::Switch { cond, body } => {
            bare_ctor_ident_used(cond, ctor_var) || stmt_uses_bare_ctor_ident(body, ctor_var)
        }
        Stmt::Case { expr, stmt } => {
            bare_ctor_ident_used(expr, ctor_var) || stmt_uses_bare_ctor_ident(stmt, ctor_var)
        }
        Stmt::Default(stmt) => stmt_uses_bare_ctor_ident(stmt, ctor_var),
        Stmt::Compound(c) => c.items.iter().any(|item| match item {
            BlockItem::Stmt(s) => stmt_uses_bare_ctor_ident(s, ctor_var),
            BlockItem::Decl(_) => false,
        }),
        _ => true,
    }
}

/// Renders a `P_Spawn*`-shaped constructor function (`fn_name`, found in
/// `corpus_dir.join(file)`) as a real Rust `pub fn` -- a genuinely
/// different idiom from `render_fn`'s tick functions, not just new
/// statement/expression shapes: `Z_Malloc` + `P_AddThinker` + a field-by-
/// field imperative fill-in becomes one `Thinker::Variant(Struct { ... })`
/// literal handed to `Arena::insert` in a single call, since the enum
/// variant tag already replaces the `var->thinker.function.acpN =
/// (cast) FnName;` line entirely (the same substitution `Thinker` itself
/// already makes for the tick-dispatch side). This reorders statements
/// relative to the original (every constructor-literal field is grouped
/// together at the `insert` call, regardless of where its assignment fell
/// in the original's source order) -- sound because C's own single-
/// threaded, synchronous execution means nothing observes the
/// partially-built value between `Z_Malloc` and the constructor function
/// returning, so grouping the field writes changes *when* they happen
/// relative to each other, never *whether* the final value ends up
/// correct.
///
/// `ctor_rust_type` names both the constructed struct and its `Thinker`
/// variant (true for every case seen so far -- `FireFlicker`,
/// `LightFlash`, `Strobe`, `Glow`, `VerticalDoor`). `param_cross_ref_types`
/// gives the function's own parameters' Rust types (e.g. `{"sector":
/// "SectorId"}`), needed both to render the function's own signature and
/// to resolve a parameter used as a cross-reference in a field's own
/// value expression (`sector->lightlevel`, just like a tick function's
/// `self` fields). `ctor_field_types` is the constructed struct's own
/// `MappedField` list (the same one `struct_fields.rs` already produces),
/// used *only* to check every one of its fields actually got a value --
/// C can leave a `Z_Malloc`'d field's value as whatever garbage the
/// allocator happened to return and simply never read it back
/// (`P_SpawnDoorCloseIn30` genuinely never sets `topheight`/`topwait`,
/// unlike its sibling `P_SpawnDoorRaiseIn5Mins`), but Rust's struct
/// literal has no equivalent -- every field needs a real value, so a
/// function missing one errs loudly here rather than emitting an
/// incomplete literal that would only fail much later, confusingly, when
/// the generated output is actually compiled.
///
/// **Scope**: only functions matching this exact shape -- one local
/// `Foo* var;`, allocated via `Z_Malloc`, immediately `P_AddThinker`'d,
/// with every other statement either a recognized discard, a `var->field
/// = expr;` constructor field (possibly reassigned later by a plain
/// conditional override, or -- if it has no unconditional assignment at
/// all -- fully decided by an `if`/`else` where both branches assign it,
/// rendered as one `let field = if cond {..} else {..};`), or an "other"
/// side-effect statement that doesn't itself reference `var` bare (a
/// back-reference like `sec->specialdata = door;`, seen in `p_doors.c`'s
/// door spawners, isn't supported yet -- see `docs/03_TRANSPILER.md`).
/// Renders `f`'s own parameter list as `name: RustType` pairs, using
/// `param_types` for each one's Rust type -- shared by every entry point
/// past `render_fn` (whose own single `self` parameter is handled
/// separately, since it always gets `&mut` and doesn't come from this
/// map).
fn render_params(
    f: &FunctionDef,
    fn_name: &str,
    param_types: &HashMap<String, String>,
) -> Result<Vec<String>, String> {
    let DirectDeclarator::Function(_, params) = &f.declarator.direct else {
        return Err(format!("{fn_name}: not a function declarator"));
    };
    let mut rendered = Vec::with_capacity(params.params.len());
    for p in &params.params {
        let ParamDeclarator::Named(d) = &p.declarator else {
            return Err(format!("{fn_name}: an unnamed parameter isn't supported"));
        };
        let name = declarator_name(d)
            .ok_or_else(|| format!("{fn_name}: a parameter declarator has no plain name"))?;
        let rust_type = param_types
            .get(&name)
            .ok_or_else(|| format!("{fn_name}: parameter `{name}`'s Rust type isn't known"))?;
        rendered.push(format!("{name}: {rust_type}"));
    }
    Ok(rendered)
}

pub fn render_spawn_fn(
    corpus_dir: &Path,
    file: &str,
    fn_name: &str,
    ctor_rust_type: &str,
    param_cross_ref_types: &HashMap<String, String>,
    ctor_field_types: &HashMap<String, String>,
) -> Result<String, String> {
    let (_, unit) = parse_full(corpus_dir.join(file).to_str().unwrap())?;
    let f = find_function_def(&unit.items, fn_name)
        .ok_or_else(|| format!("{fn_name} not found in {file}"))?;

    let rendered_params = render_params(f, fn_name, param_cross_ref_types)?;

    let mut ctor_var: Option<String> = None;
    for item in &f.body.items {
        let BlockItem::Decl(d) = item else { continue };
        if !matches!(
            d.specifiers.type_specifiers.as_slice(),
            [TypeSpecifier::TypedefName(_)]
        ) {
            continue;
        }
        let [decl] = d.declarators.as_slice() else {
            continue;
        };
        if decl.declarator.pointer_quals.len() != 1 || decl.initializer.is_some() {
            continue;
        }
        let name = declarator_name(&decl.declarator)
            .ok_or_else(|| format!("{fn_name}: constructor-shaped local has no plain name"))?;
        if ctor_var.is_some() {
            return Err(format!(
                "{fn_name}: more than one constructor-shaped local declared, not supported yet"
            ));
        }
        ctor_var = Some(name);
    }
    let ctor_var = ctor_var
        .ok_or_else(|| format!("{fn_name}: no constructor-shaped local (`Foo* x;`) found"))?;

    let ctx = FnBodyContext {
        self_param: "",
        self_field_types: &HashMap::new(),
        extra_cross_ref_idents: param_cross_ref_types,
        ctor_var: &ctor_var,
        ctor_var_handle_name: "",
    };

    let mut reassign_counts: HashMap<String, usize> = HashMap::new();
    count_ctor_field_assigns(&f.body.items, &ctor_var, &mut reassign_counts);

    // A back-reference (`sec->specialdata = door;`) needs the constructed
    // value's real `Handle` before it can be rendered at all, so its
    // presence anywhere in the body switches this whole function to a
    // two-phase render: every constructor field first (regardless of
    // where its assignment fell in the original source, same reordering
    // argument as always), then the `Arena::insert` call bound to `let
    // handle = ...;`, then every "other" statement (queued into
    // `pending_other` below) rendered afterward with a bare `ctor_var`
    // resolving to `handle`. Without a back-reference, "other" statements
    // render immediately, interleaved in original source order exactly
    // as before -- this mode is unchanged from every earlier spawn
    // function this module already handles.
    let has_backreference = body_has_backreference(&f.body.items, &ctor_var);

    // Each `var->field = expr;` becomes its own `let field = expr;` (in
    // original source order), not a flat struct-literal entry directly:
    // a later field's own value can legitimately read an earlier one back
    // (`P_SpawnLightFlash`'s `flash->count = (P_Random()&flash->maxtime)
    // +1;`), which only stays correct if that read resolves to an
    // already-`let`-bound local rather than the (nonexistent, in the
    // translated output) original C pointer. This also means every field
    // name already matches its binding, so the final literal is plain
    // shorthand. A field assigned more than once (`reassign_counts`) gets
    // `let mut` -- its later reassignment (e.g. a plain conditional
    // override, `if (flash->minlight == flash->maxlight) flash->minlight
    // = 0;`) then falls out of the *ordinary* `render_stmt` path with no
    // special-casing at all, since `ctx.ctor_var` already resolves every
    // `flash->field` reference to that field's own local either way.
    let mut lines: Vec<String> = Vec::new();
    let mut ctor_field_names: Vec<String> = Vec::new();
    let mut pending_other: Vec<&Stmt> = Vec::new();
    for item in &f.body.items {
        let BlockItem::Stmt(s) = item else { continue };
        if is_malloc_assign(s, &ctor_var)
            || is_add_thinker_call(s)
            || is_function_pointer_assign(s, &ctor_var)
        {
            continue;
        }
        if let Some((field, then_rhs, else_rhs)) = if_else_ctor_field_assign(s, &ctor_var) {
            let field = rust_field_name(field)?;
            if ctor_field_names.contains(&field) {
                return Err(format!(
                    "{fn_name}: field `{field}` already has an unconditional value; an if/else fully re-deciding it too isn't supported yet"
                ));
            }
            let Stmt::If { cond, .. } = s else {
                unreachable!("if_else_ctor_field_assign only matches Stmt::If")
            };
            let cond_text = render_bool_expr(cond, &ctx)?;
            let (then_text, _) = render_expr(then_rhs, &ctx)?;
            let (else_text, _) = render_expr(else_rhs, &ctx)?;
            lines.push(format!(
                "    let {field} = if {cond_text} {{ {then_text} }} else {{ {else_text} }};"
            ));
            ctor_field_names.push(field);
            continue;
        }
        if let Some((field, rhs)) = ctor_field_assign(s, &ctor_var) {
            let field = rust_field_name(field)?;
            let (rhs_text, _) = render_expr(rhs, &ctx)?;
            // `flick->sector = sector;`-style passthroughs need no `let`
            // at all -- the field's shorthand in the final literal
            // already resolves to the same outer binding.
            if rhs_text != field {
                let mutability = if reassign_counts.get(&field).copied().unwrap_or(0) > 1 {
                    "mut "
                } else {
                    ""
                };
                lines.push(format!("    let {mutability}{field} = {rhs_text};"));
            }
            ctor_field_names.push(field);
            continue;
        }
        if ctor_field_assign_target(s, &ctor_var).is_some() {
            // A *compound* assignment refining an already-`let`-bound
            // field right after its own initial value
            // (`P_SpawnDoorRaiseIn5Mins`'s `door->topheight -=
            // 4*FRACUNIT;` -- `ctor_field_assign` above only matches
            // plain `=`, so reaching here means this is exactly that
            // case). Still part of *constructing* the value, so it's
            // rendered inline via the ordinary path right here, never
            // deferred to `pending_other` even when `has_backreference`
            // -- deferring it would insert the pre-refinement value.
            lines.extend(render_stmt(s, &ctx, 1)?);
            continue;
        }
        if stmt_uses_bare_ctor_ident(s, &ctor_var) && !is_ctor_var_backreference(s, &ctor_var) {
            return Err(format!(
                "{fn_name}: a statement referencing the constructed value in an unsupported way: {s:?}"
            ));
        }
        if has_backreference {
            pending_other.push(s);
        } else {
            lines.extend(render_stmt(s, &ctx, 1)?);
        }
    }

    let missing_fields: Vec<&str> = ctor_field_types
        .keys()
        .map(String::as_str)
        .filter(|f| !ctor_field_names.iter().any(|n| n == f))
        .collect();
    if !missing_fields.is_empty() {
        return Err(format!(
            "{fn_name}: never assigns {ctor_rust_type}'s field(s) {}, so the constructed literal would be incomplete",
            missing_fields.join(", ")
        ));
    }

    let insert_expr = format!(
        "Thinker::{ctor_rust_type}({ctor_rust_type} {{ {} }})",
        ctor_field_names.join(", ")
    );
    if has_backreference {
        lines.push(format!("    let handle = thinkers.insert({insert_expr});"));
        let ctx_after = FnBodyContext {
            ctor_var_handle_name: "handle",
            ..ctx
        };
        for s in pending_other {
            lines.extend(render_stmt(s, &ctx_after, 1)?);
        }
    } else {
        lines.push(format!("    thinkers.insert({insert_expr});"));
    }
    Ok(format!(
        "pub fn {fn_name}({}, world: &mut World, thinkers: &mut Arena<Thinker>) {{\n{}\n}}",
        rendered_params.join(", "),
        lines.join("\n")
    ))
}

/// Renders a "trigger" function -- a third kind of body, alongside a
/// tick function's `self`-dispatched logic (`render_fn`) and a
/// constructor's field fill-in (`render_spawn_fn`). A trigger function
/// (e.g. `EV_StartLightStrobing`) has no `self`-like receiver at all: it
/// just takes some already-cross-reference-typed parameters (`line:
/// LineId`) and does something to the level in response, typically
/// iterating tagged sectors and spawning thinkers into them -- so it
/// always needs both `world: &mut World` and `thinkers: &mut
/// Arena<Thinker>`, the same as a constructor, but with no field-
/// synthesis logic at all (`FnBodyContext::ctor_var` stays empty).
///
/// `local_var_types` gives the Rust cross-reference type for any local
/// variable the function declares and later assigns from an already-
/// translated expression (`sector_t* sec;`, assigned via `sec =
/// &sectors[secnum];` -- see `render_expr`'s own special case for that
/// exact idiom) -- merged into `param_types` for `FnBodyContext::
/// extra_cross_ref_idents`, since both are just "this identifier's own
/// declared type," regardless of whether it's a parameter or a local.
pub fn render_trigger_fn(
    corpus_dir: &Path,
    file: &str,
    fn_name: &str,
    param_types: &HashMap<String, String>,
    local_var_types: &HashMap<String, String>,
) -> Result<String, String> {
    let (_, unit) = parse_full(corpus_dir.join(file).to_str().unwrap())?;
    let f = find_function_def(&unit.items, fn_name)
        .ok_or_else(|| format!("{fn_name} not found in {file}"))?;

    let rendered_params = render_params(f, fn_name, param_types)?;

    let mut all_cross_refs = param_types.clone();
    all_cross_refs.extend(local_var_types.iter().map(|(k, v)| (k.clone(), v.clone())));
    let ctx = FnBodyContext {
        self_param: "",
        self_field_types: &HashMap::new(),
        extra_cross_ref_idents: &all_cross_refs,
        ctor_var: "",
        ctor_var_handle_name: "",
    };

    let body_lines = render_compound_items(&f.body.items, &ctx, 1)?;
    Ok(format!(
        "pub fn {fn_name}({}, world: &mut World, thinkers: &mut Arena<Thinker>) {{\n{}\n}}",
        rendered_params.join(", "),
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

    fn sector_param() -> HashMap<String, String> {
        [("sector".to_string(), "SectorId".to_string())]
            .into_iter()
            .collect()
    }

    fn field_types(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_p_spawn_fire_flicker_renders_exactly() {
        let rendered = render_spawn_fn(
            &corpus_dir(),
            "p_lights.c",
            "P_SpawnFireFlicker",
            "FireFlicker",
            &sector_param(),
            &field_types(&[
                ("sector", "SectorId"),
                ("count", "i32"),
                ("maxlight", "i32"),
                ("minlight", "i32"),
            ]),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn P_SpawnFireFlicker(sector: SectorId, world: &mut World, thinkers: &mut Arena<Thinker>) {
    world[sector].special = 0;
    let maxlight = world[sector].lightlevel;
    let minlight = P_FindMinSurroundingLight(sector, world[sector].lightlevel) + 16;
    let count = 4;
    thinkers.insert(Thinker::FireFlicker(FireFlicker { sector, maxlight, minlight, count }));
}";
        assert_eq!(rendered, expected);
    }

    /// Confirms a field's value can legitimately read an *earlier* field
    /// back (`flash->count = (P_Random()&flash->maxtime)+1;`) -- this only
    /// stays correct because each field becomes its own `let` in source
    /// order, so `maxtime` resolves to the already-bound local rather
    /// than a nonexistent `flash` variable in the translated output.
    #[test]
    fn test_p_spawn_light_flash_renders_exactly() {
        let rendered = render_spawn_fn(
            &corpus_dir(),
            "p_lights.c",
            "P_SpawnLightFlash",
            "LightFlash",
            &sector_param(),
            &field_types(&[
                ("sector", "SectorId"),
                ("count", "i32"),
                ("maxlight", "i32"),
                ("minlight", "i32"),
                ("maxtime", "i32"),
                ("mintime", "i32"),
            ]),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn P_SpawnLightFlash(sector: SectorId, world: &mut World, thinkers: &mut Arena<Thinker>) {
    world[sector].special = 0;
    let maxlight = world[sector].lightlevel;
    let minlight = P_FindMinSurroundingLight(sector, world[sector].lightlevel);
    let maxtime = 64;
    let mintime = 7;
    let count = (P_Random() & maxtime) + 1;
    thinkers.insert(Thinker::LightFlash(LightFlash { sector, maxlight, minlight, maxtime, mintime, count }));
}";
        assert_eq!(rendered, expected);
    }

    /// The "other" side-effect statement (`sector->special = 0;`) falls
    /// *after* every constructor field here (unlike the other two, where
    /// it comes first) -- confirms statements interleave in real source
    /// order rather than always being hoisted to one end.
    #[test]
    fn test_p_spawn_glowing_light_renders_exactly() {
        let rendered = render_spawn_fn(
            &corpus_dir(),
            "p_lights.c",
            "P_SpawnGlowingLight",
            "Glow",
            &sector_param(),
            &field_types(&[
                ("sector", "SectorId"),
                ("minlight", "i32"),
                ("maxlight", "i32"),
                ("direction", "i32"),
            ]),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn P_SpawnGlowingLight(sector: SectorId, world: &mut World, thinkers: &mut Arena<Thinker>) {
    let minlight = P_FindMinSurroundingLight(sector, world[sector].lightlevel);
    let maxlight = world[sector].lightlevel;
    let direction = -1;
    world[sector].special = 0;
    thinkers.insert(Thinker::Glow(Glow { sector, minlight, maxlight, direction }));
}";
        assert_eq!(rendered, expected);
    }

    /// Two new field-construction idioms in one function: `minlight` is
    /// unconditionally computed, then conditionally overridden (needs
    /// `let mut` and falls through the *ordinary* `render_stmt`/`if`
    /// path, no special-casing); `count` has *no* unconditional
    /// assignment at all, decided entirely by an `if`/`else` whose
    /// condition is `!inSync` (an `int`, not a real `bool` -- exercises
    /// `render_bool_expr`'s C-truthiness `== 0` rendering).
    #[test]
    fn test_p_spawn_strobe_flash_renders_exactly() {
        let params: HashMap<String, String> = [
            ("sector".to_string(), "SectorId".to_string()),
            ("fastOrSlow".to_string(), "i32".to_string()),
            ("inSync".to_string(), "i32".to_string()),
        ]
        .into_iter()
        .collect();
        let rendered = render_spawn_fn(
            &corpus_dir(),
            "p_lights.c",
            "P_SpawnStrobeFlash",
            "Strobe",
            &params,
            &field_types(&[
                ("sector", "SectorId"),
                ("count", "i32"),
                ("minlight", "i32"),
                ("maxlight", "i32"),
                ("darktime", "i32"),
                ("brighttime", "i32"),
            ]),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn P_SpawnStrobeFlash(sector: SectorId, fastOrSlow: i32, inSync: i32, world: &mut World, thinkers: &mut Arena<Thinker>) {
    let darktime = fastOrSlow;
    let brighttime = STROBEBRIGHT;
    let maxlight = world[sector].lightlevel;
    let mut minlight = P_FindMinSurroundingLight(sector, world[sector].lightlevel);
    if minlight == maxlight {
        minlight = 0;
    }
    world[sector].special = 0;
    let count = if inSync == 0 { (P_Random() & 7) + 1 } else { 1 };
    thinkers.insert(Thinker::Strobe(Strobe { sector, darktime, brighttime, maxlight, minlight, count }));
}";
        assert_eq!(rendered, expected);
    }

    /// The constructor back-reference idiom (`p_doors.c`'s door
    /// spawners): `sec->specialdata = door;` needs `door`'s real
    /// `Handle<Thinker>`, so the whole function renders in two phases --
    /// every field first, then the `insert` bound to `let handle = ...;`,
    /// then the back-reference (and the unrelated `sec->special = 0;`
    /// alongside it) rendered afterward. Also exercises `r#type` (a
    /// keyword-colliding field name) and `Option`-wrapping `specialdata`'s
    /// own value (`Some(handle)`, not a bare `handle` -- it maps to
    /// `Option<Handle<Thinker>>`, per `struct_fields.rs`'s own name-based
    /// special case for that field).
    fn vldoor_field_types() -> HashMap<String, String> {
        field_types(&[
            ("sector", "SectorId"),
            ("r#type", "i32"),
            ("topheight", "FixedT"),
            ("speed", "FixedT"),
            ("direction", "i32"),
            ("topwait", "i32"),
            ("topcountdown", "i32"),
        ])
    }

    /// `P_SpawnDoorCloseIn30` genuinely never sets `topheight`/`topwait`
    /// in the original C (left as whatever `Z_Malloc` happened to
    /// return) -- fine for C, where nothing else reads them for this
    /// particular door variant, but Rust's struct literal has no
    /// equivalent for "leave it uninitialized." Confirms `render_spawn_fn`
    /// catches this itself, loudly, rather than emitting an incomplete
    /// literal that would only fail later, confusingly, when the
    /// generated output is compiled.
    #[test]
    fn test_p_spawn_door_close_in_30_detects_missing_fields() {
        let params: HashMap<String, String> = [("sec".to_string(), "SectorId".to_string())]
            .into_iter()
            .collect();
        let err = render_spawn_fn(
            &corpus_dir(),
            "p_doors.c",
            "P_SpawnDoorCloseIn30",
            "VerticalDoor",
            &params,
            &vldoor_field_types(),
        )
        .expect_err("should detect the incomplete literal");
        assert!(err.contains("topheight"), "expected `topheight` in: {err}");
        assert!(err.contains("topwait"), "expected `topwait` in: {err}");
    }

    /// The constructor back-reference idiom (`p_doors.c`'s door
    /// spawners): `sec->specialdata = door;` needs `door`'s real
    /// `Handle<Thinker>`, so the whole function renders in two phases --
    /// every field first, then the `insert` bound to `let handle = ...;`,
    /// then the back-reference (and the unrelated `sec->special = 0;`
    /// alongside it) rendered afterward. Also exercises `r#type` (a
    /// keyword-colliding field name); `Option`-wrapping `specialdata`'s
    /// own value (`Some(handle)`, not a bare `handle` -- it maps to
    /// `Option<Handle<Thinker>>`); and `topheight` being refined by a
    /// *compound* assignment right after its own initial value
    /// (`door->topheight -= 4*FRACUNIT;`) -- which must render inline,
    /// *before* the `insert`, not deferred alongside the back-reference,
    /// or the inserted struct would carry the pre-refinement value.
    #[test]
    fn test_p_spawn_door_raise_in_5_mins_renders_exactly() {
        let params: HashMap<String, String> = [
            ("sec".to_string(), "SectorId".to_string()),
            ("secnum".to_string(), "i32".to_string()),
        ]
        .into_iter()
        .collect();
        let rendered = render_spawn_fn(
            &corpus_dir(),
            "p_doors.c",
            "P_SpawnDoorRaiseIn5Mins",
            "VerticalDoor",
            &params,
            &vldoor_field_types(),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn P_SpawnDoorRaiseIn5Mins(sec: SectorId, secnum: i32, world: &mut World, thinkers: &mut Arena<Thinker>) {
    let sector = sec;
    let direction = 2;
    let r#type = raiseIn5Mins;
    let speed = VDOORSPEED;
    let mut topheight = P_FindLowestCeilingSurrounding(sec);
    topheight -= 4 * FRACUNIT;
    let topwait = VDOORWAIT;
    let topcountdown = 5 * 60 * 35;
    let handle = thinkers.insert(Thinker::VerticalDoor(VerticalDoor { sector, direction, r#type, speed, topheight, topwait, topcountdown }));
    world[sec].specialdata = Some(handle);
    world[sec].special = 0;
}";
        assert_eq!(rendered, expected);
    }

    /// A third kind of function body -- no `self`-struct at all, just
    /// local variables and calls -- exercising several new pieces at
    /// once: `while ((secnum = P_FindSectorFromLineTag(line,secnum)) >=
    /// 0)` needs restructuring into `loop { ..; if !(..) { break; } .. }`
    /// (Rust's `while` can't re-run a hoisted assignment each pass);
    /// `&sectors[secnum]` needs the `SectorId`-wrapping special case
    /// (`sector_t*` already maps to a plain index, not a real pointer);
    /// `if (sec->specialdata) continue;` needs `.is_some()` truthiness,
    /// not `== 0` (`specialdata` is the one `Option`-typed field this
    /// renderer knows); and a plain local (`sector_t* sec;`) needs the
    /// same deferred-`let` handling `int` locals already had.
    #[test]
    fn test_ev_start_light_strobing_renders_exactly() {
        let params: HashMap<String, String> = [("line".to_string(), "LineId".to_string())]
            .into_iter()
            .collect();
        let locals: HashMap<String, String> = [("sec".to_string(), "SectorId".to_string())]
            .into_iter()
            .collect();
        let rendered = render_trigger_fn(
            &corpus_dir(),
            "p_lights.c",
            "EV_StartLightStrobing",
            &params,
            &locals,
        )
        .expect("should render cleanly");
        let expected = "\
pub fn EV_StartLightStrobing(line: LineId, world: &mut World, thinkers: &mut Arena<Thinker>) {
    let mut secnum;
    let mut sec;
    secnum = -1;
    loop {
        secnum = P_FindSectorFromLineTag(line, secnum);
        if !(secnum >= 0) {
            break;
        }
        sec = SectorId(secnum as u32);
        if world[sec].specialdata.is_some() {
            continue;
        }
        P_SpawnStrobeFlash(sec, SLOWDARK, 0);
    }
}";
        assert_eq!(rendered, expected);
    }
}
