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
    AssignOp, BinaryOp, BlockItem, Declaration, DirectDeclarator, Expr, ExternalDecl, ForInit,
    FunctionDef, IncDecOp, Initializer, ParamDeclarator, SizeofArg, Stmt, TypeSpecifier, UnaryOp,
};
use crate::parser::grammar::declarator_name;
use crate::parser::parse_full;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

/// Rust types this renderer knows are `World`-indexed cross-references,
/// not plain values -- see module docs.
const CROSS_REF_TYPES: &[&str] = &["SectorId", "VertexId", "SideId", "LineId", "SubsectorId"];

fn is_cross_ref(rust_type: &str) -> bool {
    CROSS_REF_TYPES.contains(&rust_type)
}

/// Like `rust_field_name`, but for a bare identifier appearing as a
/// *value* (a parameter/local reference, `type != lowerToFloor`), not a
/// struct field name: `true`/`false` are C's own `boolean.h` literal
/// tokens (`crush = false;`), matching the already-established
/// `boolean` -> `bool` type mapping -- valid, unescaped Rust boolean
/// literals as-is here, not identifiers that need `rust_field_name`'s
/// own escape-or-reject handling (which exists for *field* positions,
/// where `pub false: bool` is never valid in any form). Every other
/// keyword (`type`, `crate`, `self`, ...) still goes through
/// `rust_field_name` unchanged, since those really would be identifier-
/// position problems.
fn rust_ident_name(name: &str) -> Result<String, String> {
    if name == "true" || name == "false" {
        return Ok(name.to_string());
    }
    rust_field_name(name)
}

#[derive(Clone, Copy)]
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
    /// `ctor_var`'s own field-types map (mirrors `self_field_types`, just
    /// for the constructor-in-progress rather than an already-existing
    /// `self` value) -- lets `ctor_var->field` resolve its own
    /// cross-reference-ness when that field is itself used as the *base*
    /// of a further member access (`door->sector->soundorg`, once
    /// `sector` is a plain `SectorId` local). Empty whenever `ctor_var`
    /// is.
    ctor_field_types: &'a HashMap<String, String>,
    /// Set only by `render_trigger_fn`, for a trigger loop that
    /// constructs its thinker *inline* (`EV_DoCeiling`'s own `while`
    /// body does `Z_Malloc`/`P_AddThinker`/field-fill-in directly,
    /// unlike `EV_StartLightStrobing`'s call out to a separate
    /// `P_Spawn*` function) -- the constructor's own local variable
    /// name, its `Thinker` variant/struct name, its field-types map
    /// (exactly what `render_spawn_fn` needs, just supplied by the
    /// enclosing trigger instead of self-discovered from a top-level
    /// `Decl`, since the local is typically declared once, outside the
    /// loop that actually constructs it), and a field-defaults map (see
    /// `render_ctor_body`'s own doc comment). `render_compound_items`
    /// watches for this and switches into `render_ctor_body` partway
    /// through whichever block actually contains the `Z_Malloc` call.
    /// `None` everywhere else.
    embedded_ctor: Option<CtorSpec<'a>>,
    /// Set only inside `render_existing_thinker_mutation`'s own block
    /// (`EV_VerticalDoor`'s "reuse an already-active mover" branch) --
    /// unlike `self_param`/`ctor_var` (both a bare, already-real local
    /// binding, resolved with a plain `.field` access), a reference to
    /// `mutating_handle.var` needs a *fresh* `thinkers.get(..)`/
    /// `thinkers.get_mut(..)` call at each point of use, not one hoisted
    /// binding held across the whole block -- see its own doc comment
    /// for why a hoisted `&mut` binding doesn't work here. `None`
    /// everywhere else.
    mutating_handle: Option<MutatingHandle<'a>>,
    /// Names of `self_param`'s own sibling locals declared as a bare
    /// `int` at the function's top level (`render_fn`'s own
    /// `collect_plain_int_locals`) -- needed for a real gap `A_PosAttack`
    /// surfaced: C silently reinterprets an `angle_t` (`u32` under this
    /// project's own field-type mapping) as a plain `int` on assignment
    /// (`angle = actor->angle;`, `int angle;` vs. `angle_t angle;`), the
    /// same bit-reinterpreting idiom `EV_VerticalDoor`'s own `secnum =
    /// sec-sectors;` already needs a `.0 as i32` for -- confirmed a real
    /// compile error (`u32 += i32`), not a hypothetical, by actually
    /// compiling `A_PosAttack`'s first-draft output with `rustc` before
    /// this field existed. Assigning a `self_param` field whose
    /// registered type is `u32` into one of these names now renders an
    /// explicit `as i32`, matching C's own implicit conversion instead of
    /// leaving Rust to (wrongly, for this idiom) infer the local as
    /// `u32` from its first use. Empty for every context that isn't a
    /// plain tick/action function's own top-level locals (constructors,
    /// triggers, and every isolated-fragment test below never exercise
    /// this shape).
    plain_int_locals: &'a HashSet<String>,
    /// Set only while rendering the RHS of a write to one field of a
    /// `Handle<Thinker>`-typed local (the `P_SpawnMobj`-local write arm
    /// in `render_expr_stmt`) -- the name of that same local, if any
    /// (`"mo"` for `mo->momx = FixedMul(mo->info->speed, ...)`). A
    /// further `Member` read through *this exact* name inside that RHS
    /// (`mo->info`, a *different* field of the same handle) then resolves
    /// to `m.info` directly, reusing the write's own already-bound match
    /// variable, instead of a second independent `thinkers.get(mo)` call
    /// -- confirmed necessary (not hypothetical) while investigating
    /// `A_FatAttack1`: the write's own `if let Some(Thinker::Mobj(m)) =
    /// thinkers.get_mut(mo) { .. }` already holds `thinkers` mutably
    /// borrowed for the whole block, so a second, independent `thinkers.
    /// get(mo)` for the RHS read would be a real second borrow of the
    /// same value at the same time -- a genuine `rustc` rejection, not a
    /// style choice, the same class of borrow conflict `mutating_handle`
    /// was already designed around for a *different* (multi-statement)
    /// shape. `None` everywhere else, including inside `mutating_handle`'s
    /// own block (an unrelated mechanism for a different real corpus
    /// idiom) and every isolated-fragment test below that doesn't
    /// exercise this shape.
    same_handle_write: Option<&'a str>,
}

/// Everything needed to resolve a reference to an *existing* thinker's
/// own field, looked up fresh via `Handle` at each point of use rather
/// than through one hoisted binding -- see `FnBodyContext::
/// mutating_handle` and `render_existing_thinker_mutation`.
#[derive(Clone, Copy)]
struct MutatingHandle<'a> {
    /// The local name a `X->field` reference resolves through (`door`).
    var: &'a str,
    /// The `Thinker` variant/struct name to match out (`VerticalDoor`).
    rust_type: &'a str,
    /// Already-rendered text producing the `Handle<Thinker>` itself
    /// (`world[sec].specialdata.unwrap()`) -- computed once by
    /// `render_existing_thinker_mutation` and reused verbatim at every
    /// point of use, rather than re-rendering the original base
    /// expression each time.
    handle_expr: &'a str,
}

/// Everything needed to construct a value via `render_ctor_body` (see its
/// own doc comment): the local variable name being built, its `Thinker`
/// variant/struct name, its field-types map (for the completeness
/// check), and a field-defaults map (for a field this constructor
/// genuinely never sets on every path -- also its own doc comment).
/// Bundled into one struct, rather than a growing parameter list/tuple,
/// once `render_ctor_body` needed a fourth piece of ctor-specific data
/// (`field_defaults`) alongside `ctor_var` itself.
#[derive(Clone, Copy)]
pub struct CtorSpec<'a> {
    pub ctor_var: &'a str,
    pub ctor_rust_type: &'a str,
    pub ctor_field_types: &'a HashMap<String, String>,
    pub field_defaults: &'a HashMap<String, String>,
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

/// Whether `base` is `{self_param}.target` / `{self_param}.tracer` --
/// self-struct fields registered `Option<Handle<Thinker>>`-typed (only
/// `mobj_t` ever has either, per `struct_fields.rs`'s self-referential-
/// field mapping). Shared by `render_expr`'s target/tracer chain-through
/// arm and `body_has_target_deref`'s signature-extension scan below, so
/// both agree on exactly the same shape.
/// Whether `e` evaluates to a value of type `Option<Handle<Thinker>>`
/// known (corpus-checked) to always be `Mobj`-shaped when present:
/// either `{self_param}.target`/`.tracer` directly, or a plain local
/// alias assigned straight from one of those (`dest = actor->target;`,
/// `A_SkullAttack`'s own idiom, common corpus-wide -- tracked via
/// `aliases`, populated by `collect_target_tracer_aliases`). Takes its
/// dependencies as plain arguments rather than `&FnBodyContext` so it
/// can be reused both by `render_expr` (which has a `ctx`) and by the
/// `body_has_target_deref` family (which runs *before* `render_fn` has
/// built one, to decide whether the signature needs it at all).
/// Deliberately single-level: `aliases` itself is always built by
/// scanning for a direct `self.target`/`self.tracer` assignment, not by
/// resolving through another alias -- no corpus example so far aliases
/// an alias.
fn is_target_tracer_typed(
    e: &Expr,
    self_param: &str,
    self_field_types: &HashMap<String, String>,
    aliases: &HashMap<String, String>,
) -> bool {
    match e {
        Expr::Member {
            base: inner, field, ..
        } if matches!(inner.as_ref(), Expr::Ident(n) if n == self_param) => {
            (field == "target" || field == "tracer")
                && self_field_types.get(field.as_str()).map(String::as_str)
                    == Some("Option<Handle<Thinker>>")
        }
        Expr::Ident(name) => {
            aliases.get(name.as_str()).map(String::as_str) == Some("Option<Handle<Thinker>>")
        }
        _ => false,
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
            // `NULL` used as a plain value (not just the specialdata-
            // assignment shape `render_expr_stmt` already special-cases)
            // -- `S_StartSound(NULL, sfx_oof)`'s "no origin, play
            // globally" call, e.g. -- always means "no value" under this
            // project's own no-real-pointers memory model, so it's always
            // `None` wherever it appears as a value, not just on the RHS
            // of a `specialdata = ` assignment.
            if name == "NULL" {
                return Ok(("None".to_string(), false));
            }
            let is_crossref = ctx
                .extra_cross_ref_idents
                .get(name.as_str())
                .is_some_and(|t| is_cross_ref(t));
            Ok((rust_ident_name(name)?, is_crossref))
        }
        // `sides[i].sector` -- `side_t.sector` (unlike its other four
        // fields) is itself cross-reference-typed (`SectorId`), and the
        // general fallback below only knows how to look up a field's own
        // crossref-ness for `self_param`'s or `ctor_var`'s fields (both
        // backed by an explicit field-types map), not an arbitrary
        // `Expr::Index` result's fields -- there's no general struct-
        // field-type registry yet, so this is hand-matched narrowly, the
        // same way `&sectors[i]`/bare `sides[i]` are, rather than building
        // one ahead of more evidence it's needed.
        Expr::Member { base, field, .. }
            if field == "sector"
                && matches!(base.as_ref(), Expr::Index { base, .. } if matches!(base.as_ref(), Expr::Ident(n) if n == "sides")) =>
        {
            let (base_text, _) = render_expr(base, ctx)?;
            Ok((format!("world[{base_text}].sector"), true))
        }
        // `thing->player` -- `thing`'s own declared type is
        // `Handle<Thinker>` (a live thinker passed in, unlike
        // `SectorId`/etc. this isn't itself `World`-indexed, it needs a
        // real `Arena` lookup), and `Thinker` is a closed enum over ten
        // different thinker shapes, only one of which (`Mobj`) has a
        // `.player` field at all -- hand-matched narrowly, the same "no
        // general lookup ahead of evidence" reasoning as `sides[i].
        // sector`, rather than a general enum-variant-field mechanism no
        // second caller has needed yet.
        Expr::Member { base, field, .. }
            if field == "player"
                && matches!(base.as_ref(), Expr::Ident(n) if ctx.extra_cross_ref_idents.get(n.as_str()).map(String::as_str) == Some("Handle<Thinker>")) =>
        {
            let Expr::Ident(name) = base.as_ref() else {
                unreachable!("guarded above")
            };
            Ok((
                format!(
                    "match thinkers.get({name}) {{ Some(Thinker::Mobj(m)) => m.player, _ => None }}"
                ),
                false,
            ))
        }
        // `mo->info` read inside the RHS of a write to a *different*
        // field of that same `mo` (`mo->momx = FixedMul(mo->info->speed,
        // ..);`, `A_FatAttack1`'s own idiom) -- reuses the write's own
        // already-bound `m` (`ctx.same_handle_write`, set only while
        // rendering that one RHS) instead of a second, independent
        // `thinkers.get(mo)` call, which would be a genuine second borrow
        // of `thinkers` while the write's own `thinkers.get_mut(mo)` is
        // still live -- see `FnBodyContext::same_handle_write`'s own doc
        // comment for why this isn't just a style choice. Checked before
        // the general `Handle<Thinker>` arm just below so a same-handle
        // read never takes that arm's fresh-lookup path by mistake.
        Expr::Member { base, field, .. } if matches!(base.as_ref(), Expr::Ident(n) if ctx.same_handle_write == Some(n.as_str())) => {
            Ok((format!("m.{}", rust_field_name(field)?), false))
        }
        // `th->field` (any field, not just `player`) -- `th`'s own
        // declared type is `Handle<Thinker>`, the same "live thinker
        // needing a real `Arena` lookup" shape `thing->player` above
        // already covers, generalized to every field now that a second
        // real caller needs it: `A_Tracer`'s own `th = P_SpawnMobj(...);
        // th->momz = ...; th->tics -= ...;` (`collect_spawn_mobj_locals`
        // registers `th` here the same way `thing`'s own parameter type
        // is registered elsewhere). Unlike `thing->player`'s own `_ =>
        // None` fallback (correct there since `.player` is itself
        // `Option`-typed), this uses `_ => unreachable!()` -- corpus-
        // checked safe the same way `door->field`/target-tracer
        // dereferencing already are: every real `Handle<Thinker>`-typed
        // local reaching this arm is either a `mobj_t*` parameter or a
        // value fresh out of `P_SpawnMobj`, always the `Mobj` variant.
        Expr::Member { base, field, .. } if matches!(base.as_ref(), Expr::Ident(n) if ctx.extra_cross_ref_idents.get(n.as_str()).map(String::as_str) == Some("Handle<Thinker>")) =>
        {
            let Expr::Ident(name) = base.as_ref() else {
                unreachable!("guarded above")
            };
            Ok((
                format!(
                    "match thinkers.get({name}) {{ Some(Thinker::Mobj(m)) => m.{}, _ => unreachable!() }}",
                    rust_field_name(field)?
                ),
                false,
            ))
        }
        // `door->field` *read*, inside `render_existing_thinker_mutation`'s
        // own block (`ctx.mutating_handle` set) -- a fresh `thinkers.
        // get(..)` call at this exact point (not a hoisted `&` binding,
        // for the same borrow-conflict reason `render_expr_stmt`'s own
        // write-side special case explains). The `_ => unreachable!()`
        // arm is genuinely safe: this sector's `specialdata` was already
        // proven `Some` right before this block was entered, and by
        // construction only ever holds the variant this same trigger
        // itself builds.
        Expr::Member { base, field, .. } if matches!(ctx.mutating_handle, Some(mh) if matches!(base.as_ref(), Expr::Ident(n) if n == mh.var)) =>
        {
            let mh = ctx.mutating_handle.expect("guarded above");
            Ok((
                format!(
                    "match thinkers.get({}) {{ Some(Thinker::{}({})) => {}.{}, _ => unreachable!() }}",
                    mh.handle_expr,
                    mh.rust_type,
                    mh.var,
                    mh.var,
                    rust_field_name(field)?
                ),
                false,
            ))
        }
        // `p->field` -- `p`'s own declared type is `Option<PlayerId>`
        // (`EV_DoLockedDoor`'s own `player_t*` local, always immediately
        // null-checked right beside every real dereference in the
        // corpus). `.unwrap()` at the point of use, rather than
        // reshaping the whole function around a narrowed/shadowed
        // binding -- every real caller already guards each dereference
        // with its own `if (!p) return ...;` right next to it, so this
        // stays a close, simple translation of that same defensive style
        // rather than a fancier one nothing here needs yet.
        Expr::Member { base, field, .. } if matches!(base.as_ref(), Expr::Ident(n) if ctx.extra_cross_ref_idents.get(n.as_str()).map(String::as_str) == Some("Option<PlayerId>")) =>
        {
            let Expr::Ident(name) = base.as_ref() else {
                unreachable!("guarded above")
            };
            Ok((
                format!("world[{name}.unwrap()].{}", rust_field_name(field)?),
                false,
            ))
        }
        // `line->frontsector` -- unlike a self-struct field
        // (`self_field_types`) or a constructor-in-progress field
        // (`ctor_field_types`), a trigger function's own parameter (`line:
        // LineId`, tracked only in `extra_cross_ref_idents`) has no
        // generic field-type registry to say one of *its* fields is
        // itself cross-reference-typed -- hand-matched narrowly by name,
        // the same "no general struct-field-type registry yet" reasoning
        // as `sides[i].sector` above, so a further chain (`line->
        // frontsector->floorpic`, `EV_DoFloor`'s own
        // `raiseFloor24AndChange` case) resolves through `world[...]`
        // correctly instead of stopping one level short.
        Expr::Member { base, field, .. }
            if field == "frontsector"
                && matches!(base.as_ref(), Expr::Ident(n) if ctx.extra_cross_ref_idents.get(n.as_str()).map(String::as_str) == Some("LineId")) =>
        {
            let (base_text, _) = render_expr(base, ctx)?;
            Ok((format!("world[{base_text}].frontsector"), true))
        }
        // `actor->target->field` / `actor->tracer->field`, or the same
        // chain through a local alias (`dest->field` once `dest = actor->
        // target;`, `A_SkullAttack`'s own idiom) -- `target`/`tracer` are
        // self-struct fields themselves `Option<Handle<Thinker>>`-typed
        // (`struct_fields.rs`'s self-referential-field mapping), unlike
        // every cross-reference field handled above (which is a plain
        // index into `World`) -- dereferencing one needs a real `Arena`
        // lookup, not `world[...]`. Corpus-checked: only `mobj_t` ever
        // has `target`/`tracer`, so the looked-up thinker is always the
        // `Mobj` variant, making `_ => unreachable!()` genuinely safe
        // (matching `door->field`'s own `mutating_handle` arm's
        // reasoning above) -- not a defensive catch-all. `.unwrap()` at
        // the point of use, not a narrowed rebinding, the same `p->field`
        // (`Option<PlayerId>`) precedent: every real corpus dereference
        // site already guards this with its own adjacent `if (!actor->
        // target) return;`.
        Expr::Member { base, field, .. }
            if is_target_tracer_typed(
                base,
                ctx.self_param,
                ctx.self_field_types,
                ctx.extra_cross_ref_idents,
            ) =>
        {
            let (base_text, _) = render_expr(base, ctx)?;
            Ok((
                format!(
                    "match thinkers.get({base_text}.unwrap()) {{ Some(Thinker::Mobj(m)) => m.{}, _ => unreachable!() }}",
                    rust_field_name(field)?
                ),
                false,
            ))
        }
        Expr::Member { base, field, .. } => {
            if !ctx.ctor_var.is_empty()
                && matches!(base.as_ref(), Expr::Ident(n) if n == ctx.ctor_var)
            {
                let is_crossref = ctx
                    .ctor_field_types
                    .get(field.as_str())
                    .is_some_and(|t| is_cross_ref(t));
                return Ok((rust_field_name(field)?, is_crossref));
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
            Ok((
                format!("{base_text}.{}", rust_field_name(field)?),
                is_crossref,
            ))
        }
        // `sec-sectors` (`EV_VerticalDoor`'s own `secnum = sec-sectors;`)
        // -- real pointer-arithmetic C idiom for "the index of this
        // pointer within the array it came from," which is exactly what
        // a `SectorId` already *is* under this project's own memory-
        // model decision (a plain index, not a real pointer), so this
        // needs no arithmetic at all, just unwrapping the newtype's own
        // field -- confirmed this is what the expression actually
        // computes (not assumed), same rigor as every other pointer-
        // idiom special case in this module.
        Expr::Binary {
            op: BinaryOp::Sub,
            lhs,
            rhs,
        } if matches!(rhs.as_ref(), Expr::Ident(n) if n == "sectors") => {
            let (lhs_text, _) = render_expr(lhs, ctx)?;
            Ok((format!("{lhs_text}.0 as i32"), false))
        }
        Expr::Binary { op, lhs, rhs } => {
            let prec = binary_prec(*op);
            let lhs_text = render_binary_operand(lhs, *op, prec, false, ctx)?;
            let rhs_text = render_binary_operand(rhs, *op, prec, true, ctx)?;
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
        // `sides[i]` (bare, no `&` -- unlike `&sectors[i]` above, the
        // corpus reads this one as a plain value, immediately chaining a
        // further `.field` off it: `sides[line->sidenum[0]].sector`,
        // `EV_DoPlat`'s own idiom for "the sector on the other side of
        // this line"). Same reasoning as `&sectors[i]`: `side_t*` already
        // maps to `SideId` (a plain index), so this needs no real
        // indexing operation, just wrapping the index -- but since
        // nothing dereferences it via `&`, this returns `is_crossref:
        // true` directly (unlike the `&`-prefixed form, whose caller
        // already has its own dereference-free pointer semantics) so a
        // chained `.field` access resolves through `world[...]` the same
        // way any other cross-reference-typed value does.
        Expr::Index { base, index } if matches!(base.as_ref(), Expr::Ident(n) if n == "sides") => {
            let (index_text, _) = render_expr(index, ctx)?;
            Ok((format!("SideId({index_text} as u32)"), true))
        }
        // A plain fixed-size array *field* (`line->sidenum[0]`, `sidenum:
        // [i16; 2]` -- struct_fields.rs's own single-dimension array
        // support), not one of the special global cross-reference arrays
        // matched above -- ordinary Rust indexing syntax needs nothing
        // else, and the element itself isn't cross-reference-typed (no
        // corpus example needing that yet).
        Expr::Index { base, index } => {
            let (base_text, _) = render_expr(base, ctx)?;
            let (index_text, _) = render_expr(index, ctx)?;
            // A struct field used as an index (`textureheight[side->
            // bottomtexture]`, `EV_DoFloor`'s own `raiseToTexture` scan --
            // `bottomtexture` is a concrete `i16`, fixed by `Side`'s own
            // struct definition) needs an explicit cast: Rust arrays/`Vec`
            // only implement `Index<usize>`, and unlike a fresh, still-
            // type-inferred local (`sidenum[side ^ 1]`, where `side`'s own
            // type gets inferred as `usize` straight from this same
            // indexing use, needing no cast), a struct field's type is
            // already fixed elsewhere and can't retroactively change.
            // `finecosine`/`finesine` (`tables.h`) need the same cast even
            // for a *plain* index identifier (`finecosine[exact]`,
            // `A_Tracer`'s own idiom): unlike `sidenum[side^1]`'s fresh,
            // single-purpose local, `exact` (`angle_t`/`u32`) is *also*
            // used earlier in real `u32` arithmetic against `actor->angle`
            // (`exact - actor->angle > 0x80000000`), so Rust can't freely
            // infer it as `usize` here -- narrowly by-name, matching this
            // module's usual "hand-match the one real array identifier"
            // style (`sides`/`sectors`/`textureheight` above).
            let index_text = if matches!(index.as_ref(), Expr::Member { .. })
                || matches!(base.as_ref(), Expr::Ident(n) if n == "finecosine" || n == "finesine")
            {
                format!("{index_text} as usize")
            } else {
                index_text
            };
            Ok((format!("{base_text}[{index_text}]"), false))
        }
        Expr::Unary {
            op: UnaryOp::AddrOf,
            expr,
        } => {
            // Every other cross-reference-typed field (`SectorId` etc.)
            // has its own narrower special case above -- reaching here
            // means `expr` is a plain *value* field (e.g. `sector_t.
            // soundorg`, an embedded `degenmobj_t`, not a pointer/index at
            // all), so a real Rust `&` reference is the correct,
            // idiomatic, safe translation, not a pointer trick.
            let (inner_text, _) = render_expr(expr, ctx)?;
            let inner_text = parenthesize_if_needed(expr, &inner_text, u8::MAX, false);
            Ok((format!("&{inner_text}"), false))
        }
        // `!actor->target` used as one operand of a `&&`/`||` chain
        // (`A_CPosRefire`'s own `!actor->target || actor->target->health
        // <= 0 || ...`) -- unlike `render_bool_expr`'s own top-level
        // entry point (which already special-cases this), a `Binary`
        // logical chain's operands render through this generic
        // `render_expr` path (via `render_binary_operand`), not
        // `render_bool_expr`, so the same `Option`-awareness needs its
        // own arm here too rather than assuming every top-level condition
        // is a single bare `!x`. The plain `!` fallback just below stays
        // correct and unchanged for a genuinely `bool`-valued operand
        // (`!player->cards[idx]`, already real Rust `bool` since
        // `boolean` maps to it directly) -- only an `Option`-valued
        // operand needs `.is_none()` instead of `!`, which doesn't even
        // compile on an `Option`.
        Expr::Unary {
            op: UnaryOp::Not,
            expr,
        } if is_option_valued(expr, ctx) => {
            Ok((format!("{}.is_none()", render_expr(expr, ctx)?.0), false))
        }
        Expr::Unary { op, expr } => {
            let op_text = match op {
                UnaryOp::Minus => "-",
                UnaryOp::Plus => "+",
                UnaryOp::Not | UnaryOp::BitNot => "!",
                UnaryOp::AddrOf => unreachable!("handled by the dedicated arm above"),
                UnaryOp::Deref => {
                    return Err(
                        "render_expr: unary deref isn't supported yet -- translated code has no real pointers"
                            .to_string(),
                    );
                }
            };
            let (inner_text, _) = render_expr(expr, ctx)?;
            // Unary operators bind tighter than every binary operator this
            // renderer handles, so any binary child always needs parens.
            let inner_text = parenthesize_if_needed(expr, &inner_text, u8::MAX, false);
            Ok((format!("{op_text}{inner_text}"), false))
        }
        // A C cast, dropped entirely rather than rendered as `as
        // TargetType` -- every one seen so far is type-erasure noise
        // with no real value transformation behind it (matching the
        // `(actionf_p1)` cast around an action-function pointer, already
        // elided the same way by `ActionFn`'s own design): `(mobj_t *)
        // &door->sector->soundorg` reinterprets a `degenmobj_t`'s memory
        // layout as a `mobj_t*` purely so `S_StartSound`'s C signature
        // (`void* origin`) accepts it, not because the value is actually
        // becoming a `Mobj`. A cast reflecting a genuine numeric
        // conversion hasn't been seen yet; this would need revisiting if
        // one turns up.
        Expr::Cast { expr, .. } => render_expr(expr, ctx),
        // `player->refire++;` (`A_ReFire`) -- a bare increment/decrement
        // used as its own standalone statement, not as an `if`'s own
        // condition (`hoist_pre_inc_dec`'s own narrower shape, which
        // pattern-matches `PreIncDec` directly before ever reaching this
        // generic arm, so the two don't conflict) or a `for` loop's step
        // (`render_for_step`, which now just delegates here instead of
        // duplicating this same text). C doesn't distinguish pre/post
        // for a discarded result, so both render identically.
        Expr::PostIncDec { expr, op } | Expr::PreIncDec { expr, op } => {
            let (target_text, _) = render_expr(expr, ctx)?;
            let op_text = match op {
                IncDecOp::Inc => "+= 1",
                IncDecOp::Dec => "-= 1",
            };
            Ok((format!("{target_text} {op_text}"), false))
        }
        Expr::Call { callee, args } => {
            let (callee_text, _) = render_expr(callee, ctx)?;
            let mut rendered_args = Vec::with_capacity(args.len());
            for a in args {
                rendered_args.push(render_expr(a, ctx)?.0);
            }
            // `getSide`/`getSector` (`p_spec.c`) return `side_t*`/
            // `sector_t*` -> `SideId`/`SectorId` under the existing
            // memory-model decision, matched narrowly by name (the same
            // "not yet fully modeled callee" treatment `twoSided` already
            // gets above) -- needed so a `.field` chained *directly* off
            // the call result (`getSide(secnum,i,0)->sector`, `EV_DoFloor`'s
            // own `lowerAndChange` case) resolves through `world[...]`
            // correctly, not just once the result is first bound to an
            // already-known-typed local the way `side = getSide(..);`
            // already works.
            let is_crossref =
                matches!(callee.as_ref(), Expr::Ident(n) if n == "getSide" || n == "getSector");
            Ok((
                format!("{callee_text}({})", rendered_args.join(", ")),
                is_crossref,
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

/// Renders one operand of a `Expr::Binary { op: parent_op, .. }`,
/// applying `parenthesize_if_needed`'s ordinary precedence rule -- except
/// when `parent_op` is arithmetic (not itself a comparison/logical op)
/// and the operand is *itself* a comparison/logical expression
/// (`EV_DoFloor`'s own `(8*FRACUNIT)*(floortype == raiseFloorCrush)`,
/// C's bool-as-0-or-1 arithmetic idiom): a comparison already renders as
/// a real Rust `bool`, which -- unlike C's `int` -- can't be multiplied/
/// added/etc. directly, so it needs an explicit `as i32` cast. The
/// comparison itself still needs its own parens *before* that cast
/// (`as` binds tighter than every binary operator this renderer handles,
/// so `x == y as i32` would parse as `x == (y as i32)`, not what's
/// wanted here) -- hard-coded rather than derived from
/// `parenthesize_if_needed`'s precedence table, since a cast isn't a
/// binary operator with a precedence level of its own.
fn render_binary_operand(
    operand: &Expr,
    parent_op: BinaryOp,
    parent_prec: u8,
    is_right: bool,
    ctx: &FnBodyContext,
) -> Result<String, String> {
    // `(player->cmd.buttons & BT_ATTACK) && player->pendingweapon ==
    // wp_nochange && player->health` (`A_ReFire`) -- the reverse
    // direction of the bool-as-arithmetic cast just below: a `&&`/`||`
    // operand that's a bare field (`player->health`) or a non-comparison
    // `Binary` (`player->cmd.buttons & BT_ATTACK`) is still real C
    // truthiness -- any nonzero value is true -- the same reading
    // `render_bool_expr`'s own top-level entry point already gives a
    // *whole* condition, reused directly here rather than duplicated.
    // Deliberately scoped to just these two shapes, *not* every
    // non-comparison operand: a bare `Unary::Not` operand (`!world[p.
    // unwrap()].cards[idx]`, `EV_DoLockedDoor`/`EV_VerticalDoor`'s own
    // lock checks) already renders correctly as plain `!` via the
    // ordinary `Expr::Unary` arm below, since `cards` is a genuine Rust
    // `bool` array -- routing it through `render_bool_expr` too would
    // wrongly apply that function's generic *int*-truthiness `== 0`
    // fallback to an already-`bool` value (confirmed a real regression
    // against `test_ev_do_locked_door_renders_exactly`/`test_ev_
    // vertical_door_renders_exactly` when tried). No renderer here
    // tracks a field's real C type well enough to tell a genuinely-`int`
    // negation from a genuinely-`bool` one inside a chain, so this
    // leaves `Unary::Not` exactly as before rather than guessing.
    if matches!(parent_op, BinaryOp::LogAnd | BinaryOp::LogOr)
        && (matches!(operand, Expr::Member { .. })
            || matches!(operand, Expr::Binary { op, .. } if !is_comparison_or_logical(*op)))
    {
        return render_bool_expr(operand, ctx);
    }
    let (text, _) = render_expr(operand, ctx)?;
    if !is_comparison_or_logical(parent_op)
        && matches!(operand, Expr::Binary { op, .. } if is_comparison_or_logical(*op))
    {
        return Ok(format!("({text}) as i32"));
    }
    Ok(parenthesize_if_needed(
        operand,
        &text,
        parent_prec,
        is_right,
    ))
}

/// Whether `expr` renders to an `Option<_>`-typed Rust value -- a bare
/// local declared `Option<PlayerId>` (`p`), or a direct `thing->player`
/// dereference (matching `render_expr`'s own `Expr::Member` special case
/// for it). Narrowly by-name/by-shape, the same way `specialdata`'s own
/// `Option`-awareness is, not a general type-inference pass.
/// Whether `expr` is a call to a corpus function whose real declared C
/// return type is `boolean` (already Rust's native `bool`, per
/// `struct_fields.rs`), so it needs no `!= 0`/`== 0` truthiness cast at
/// all, bare or negated -- narrowly matched by name (this codebase's
/// usual "no callee-signature tracking, hand-match the real shape"
/// style), not a general return-type inference. Extend this list only
/// once a real corpus call site is found needing it, the same way
/// `twoSided` stayed its own separate `int`-flag-shaped arm rather than
/// being folded in here.
fn is_bool_returning_call(expr: &Expr) -> bool {
    matches!(expr, Expr::Call { callee, .. }
        if matches!(callee.as_ref(), Expr::Ident(n) if n == "P_CheckMeleeRange" || n == "P_CheckSight"))
}

fn is_option_valued(expr: &Expr, ctx: &FnBodyContext) -> bool {
    match expr {
        // Covers both a trigger function's own `Option<PlayerId>` local
        // (`p`) and a `Mobj`-shaped action function's own local alias of
        // `target`/`tracer` (`dest`, `A_SkullAttack`'s idiom) -- both
        // tracked the same way, via `ctx.extra_cross_ref_idents`.
        Expr::Ident(n) => matches!(
            ctx.extra_cross_ref_idents
                .get(n.as_str())
                .map(String::as_str),
            Some("Option<PlayerId>") | Some("Option<Handle<Thinker>>")
        ),
        Expr::Member { field, .. } if field == "player" => true,
        // `!actor->target` (`A_PosAttack` and friends) -- `target`'s
        // registered type (`Mobj.target: Option<Handle<Thinker>>`, per
        // `struct_fields.rs`'s own self-referential-field mapping) is the
        // general case `field == "player"` above only special-cased by
        // name for: any self-struct field whose `self_field_types` entry
        // is itself `Option<...>`-shaped gets the same `.is_none()`
        // treatment, not just that one hardcoded name.
        Expr::Member { base, field, .. }
            if matches!(base.as_ref(), Expr::Ident(n) if n == ctx.self_param)
                && ctx
                    .self_field_types
                    .get(field.as_str())
                    .is_some_and(|t| t.starts_with("Option<")) =>
        {
            true
        }
        _ => false,
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
        // `!p` (an `Option<PlayerId>`-typed local, `EV_DoLockedDoor`'s own
        // `player_t*`) or `!thing->player` (`EV_VerticalDoor`'s own
        // re-check further down, reading the same `Handle<Thinker>`-
        // dereferencing `Expr::Member` special case above) -- both
        // `Option`-valued, needing `.is_none()` rather than the `== 0`
        // every other (plain `int`) negated value gets, the same
        // Option-awareness `specialdata` already gets below, just for a
        // bare local/direct dereference instead of a struct field.
        Expr::Unary {
            op: UnaryOp::Not,
            expr,
        } if is_option_valued(expr, ctx) => Ok(format!("{}.is_none()", render_expr(expr, ctx)?.0)),
        // `! P_CheckSight (actor, actor->target)` (`P_CheckMeleeRange`) --
        // unlike a plain `int`-valued operand (the generic `== 0` arm just
        // below), a call to a real `boolean`-returning corpus function is
        // already a real Rust `bool`, so plain `!` is correct here, the
        // same "already `bool`, no cast" reasoning `is_bool_returning_call`
        // callers below already use for the *bare* (non-negated) case.
        Expr::Unary {
            op: UnaryOp::Not,
            expr,
        } if is_bool_returning_call(expr) => Ok(format!("!{}", render_expr(expr, ctx)?.0)),
        Expr::Unary {
            op: UnaryOp::Not,
            expr,
        } => Ok(format!("{} == 0", render_expr(expr, ctx)?.0)),
        // A bare value used for truthiness (not a comparison/negation) --
        // C's `if (x)` tests non-zero/non-null. `specialdata` is the one
        // corpus *field* known to be `Option`-typed (`struct_fields.rs`'s
        // own name-based special case, reused by `render_expr_stmt`'s
        // `Some(..)`-wrapping too), so a bare reference to it needs
        // `.is_some()`, not the `== 0` truthiness every other (plain
        // `int`) value gets.
        Expr::Member { field, .. } if field == "specialdata" => {
            Ok(format!("{}.is_some()", render_expr(cond, ctx)?.0))
        }
        // `if (actor->info->painsound)` (`A_Pain`) -- a bare struct-field
        // reference used for truthiness, same as `specialdata` above, but
        // this one is a genuinely plain `int` field (`mobjinfo_t.
        // painsound`), not `Option`-valued -- ordinary C truthiness,
        // `!= 0`, not `.is_some()`. Excludes anything `is_option_valued`
        // already claims (`player`, an `Option<PlayerId>`-typed extra
        // cross-ref ident) so a genuinely `Option`-typed field falls
        // through to its own correct handling instead of this one.
        Expr::Member { .. } if !is_option_valued(cond, ctx) => {
            Ok(format!("{} != 0", render_expr(cond, ctx)?.0))
        }
        // `if (actor->target->flags & MF_SHADOW)` (`A_FaceTarget`) -- a
        // bare non-comparison `Binary` (here bitwise AND against a flag
        // mask) used for C truthiness, the same "bare value, not a
        // comparison/negation" idiom as the bare `Member` arm just above,
        // just for a computed value instead of a direct field read.
        // `is_comparison_or_logical`'s own arm at the top of this match
        // already claims every `==`/`<`/`&&`/`||`/etc. shape, so this only
        // ever fires for a genuinely non-bool-valued `Binary` result
        // (`&`, `|`, `+`, ...) needing the ordinary `!= 0` cast.
        Expr::Binary { op, .. } if !is_comparison_or_logical(*op) => {
            Ok(format!("({}) != 0", render_expr(cond, ctx)?.0))
        }
        // `if (twoSided (secnum, i))` -- `EV_DoFloor`'s own adjacency scan,
        // the first bare (non-negated) *call result* used for truthiness
        // rather than a comparison/field. `twoSided` genuinely returns a
        // plain C `int` (a bitmasked flag, `flags & ML_TWOSIDED`, `p_spec.c`),
        // so this is ordinary `int` truthiness, not `Option`-valued like
        // `specialdata` above -- narrowly matched by name (this codebase's
        // usual style) rather than a general "any call is int-valued"
        // fallback, since a differently-shaped callee could just as easily
        // return something `Option`-valued the way `thing->player` already
        // does elsewhere.
        Expr::Call { callee, .. } if matches!(callee.as_ref(), Expr::Ident(n) if n == "twoSided") => {
            Ok(format!("{} != 0", render_expr(cond, ctx)?.0))
        }
        // `if (P_CheckMeleeRange (actor))` (`A_TroopAttack` and several
        // other melee-attack action functions) -- unlike `twoSided`,
        // `P_CheckMeleeRange`/`P_CheckSight`'s own real corpus
        // declarations (`p_enemy.c`/`p_local.h`) return `boolean`, not a
        // plain `int`, and `boolean` already maps to Rust's native `bool`
        // (`struct_fields.rs`'s own established decision) -- so a call to
        // either is already a real `bool` value, used directly with no
        // `!= 0` cast at all. Matched narrowly by name
        // (`is_bool_returning_call`), the same "hand-match the one real
        // corpus shape" style as `twoSided`, rather than inferring a
        // callee's C return type generically (nothing else here tracks
        // function signatures).
        Expr::Call { .. } if is_bool_returning_call(cond) => Ok(render_expr(cond, ctx)?.0),
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
            let (hoisted, target_text) = hoist_pre_inc_dec(expr, *op, ctx, depth)?;
            Ok((hoisted, format!("{target_text} != 0")))
        }
        // `if (!--door->topcountdown)` -- the same countdown-to-zero
        // idiom as the bare `--x` case above, just testing for zero
        // (C's `!` truthiness on the decremented result) rather than
        // nonzero. Extremely common in Doom's timer/countdown tick
        // functions (`T_VerticalDoor`'s own WAITING/INITIAL WAIT states).
        Expr::Unary {
            op: UnaryOp::Not,
            expr: not_expr,
        } if matches!(not_expr.as_ref(), Expr::PreIncDec { .. }) => {
            let Expr::PreIncDec { expr, op } = not_expr.as_ref() else {
                unreachable!("guarded above")
            };
            let (hoisted, target_text) = hoist_pre_inc_dec(expr, *op, ctx, depth)?;
            Ok((hoisted, format!("{target_text} == 0")))
        }
        _ => Ok((Vec::new(), render_bool_expr(cond, ctx)?)),
    }
}

/// Renders `--x`/`++x` as a statement to hoist immediately before an
/// `if`'s own test (both `render_condition` cases above need this, just
/// with a different final comparison), returning that statement plus the
/// plain target text for the caller to build its own comparison from.
fn hoist_pre_inc_dec(
    expr: &Expr,
    op: IncDecOp,
    ctx: &FnBodyContext,
    depth: usize,
) -> Result<(Vec<String>, String), String> {
    let (target_text, _) = render_expr(expr, ctx)?;
    let op_text = match op {
        IncDecOp::Inc => "+= 1",
        IncDecOp::Dec => "-= 1",
    };
    let hoisted = vec![format!("{}{target_text} {op_text};", indent(depth))];
    Ok((hoisted, target_text))
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
    // `int amount;` (a plain scalar), `unsigned an;` (`A_Fire`'s own
    // local, bit-shifted then used as a `finecosine`/`finesine` index --
    // same deferred-inference treatment as `int`, just a different bare
    // specifier with no declarator of its own to distinguish), and
    // `sector_t* sec;` (a single pointer to an already-known cross-
    // reference type, e.g. `EV_StartLightStrobing`'s own loop variable)
    // all render the same way: Rust infers the type from later use, so
    // no annotation is needed regardless of which C type it was. Anything
    // else (arrays, multiple declarators, an initializer) isn't supported
    // yet.
    if !matches!(
        d.specifiers.type_specifiers.as_slice(),
        [TypeSpecifier::Int] | [TypeSpecifier::Unsigned] | [TypeSpecifier::TypedefName(_)]
    ) {
        return Err(format!(
            "render_decl: only a bare `int` or single-pointer known-type declaration is supported so far, got {:?}",
            d.specifiers.type_specifiers
        ));
    }
    // `int secnum,rtn;` (`EV_DoDoor`'s own locals) declares more than one
    // plain scalar off one `int`, sharing the same (absent) type
    // annotation and initializer rules as the single-declarator case --
    // handled by rendering each declarator through the same checks below,
    // in source order, rather than requiring a caller to have already
    // split it into separate `int` decls the way `EV_DoCeiling` happens
    // to.
    let mut lines = Vec::new();
    for decl in &d.declarators {
        // `int minsize = MAXINT;` (`EV_DoFloor`'s own `raiseToTexture`
        // case) -- a plain expression initializer, rendered inline on the
        // same `let mut` this renderer already always uses for a
        // deferred-inference local (unconditionally `mut`, the same as
        // every uninitialized decl below, rather than analyzing whether
        // this particular one is ever reassigned).
        let init_text = match &decl.initializer {
            None => String::new(),
            Some(Initializer::Expr(e)) => format!(" = {}", render_expr(e, ctx)?.0),
            Some(Initializer::List(_)) => {
                return Err(
                    "render_decl: a brace-list initializer is not supported so far".to_string(),
                );
            }
        };
        if !matches!(decl.declarator.direct, DirectDeclarator::Ident(_)) {
            return Err(
                "render_decl: only a plain (non-array, non-function) declarator is supported so far"
                    .to_string(),
            );
        }
        let name = declarator_name(&decl.declarator)
            .ok_or_else(|| "render_decl: declarator has no plain name".to_string())?;
        // A trigger's own top-level `Foo* var;` for its embedded constructor
        // (`EV_DoCeiling`'s `ceiling_t* ceiling;`, declared once outside the
        // loop that actually builds it -- see `FnBodyContext::embedded_ctor`)
        // never becomes a real Rust binding at all: `render_ctor_body` gives
        // each of the constructed value's own *fields* their own `let`
        // instead, so `ceiling` itself is never assigned or read anywhere in
        // the translated output. Emitting `let mut ceiling;` for it would be
        // genuinely dead, uninferable code (Rust has nothing to infer its
        // type from), so it's dropped here rather than rendered.
        if let Some(spec) = ctx.embedded_ctor
            && name == spec.ctor_var
        {
            continue;
        }
        lines.push(format!("{}let mut {name}{init_text};", indent(depth)));
    }
    Ok(lines)
}

/// `if (X->specialdata) { CTOR_VAR = X->specialdata; ...uses of
/// CTOR_VAR->field...; }` -- `EV_VerticalDoor`'s own "reuse and mutate
/// an already-active mover instead of building a new one" branch, the
/// one shape in the corpus so far where a trigger looks up an *existing*
/// thinker via `Handle` rather than always constructing a new one.
/// Detected only when the block's very first statement assigns
/// `X->specialdata` straight into the embedded constructor's own
/// `ctor_var`, and `X` matches the `if`'s own condition -- narrow on
/// purpose, the same "hand-match the one real shape" style as
/// `sides[i].sector`, not a general "does this block start by
/// unwrapping a `specialdata`" mechanism. Returns the crossref base
/// expr (`X`) and every statement in the block *after* that first one.
fn existing_thinker_mutation_shape<'a>(
    cond: &'a Expr,
    then_branch: &'a Stmt,
    ctor_var: &str,
) -> Option<(&'a Expr, &'a [BlockItem])> {
    let Expr::Member {
        base: cond_base,
        field: cond_field,
        ..
    } = cond
    else {
        return None;
    };
    if cond_field != "specialdata" {
        return None;
    }
    let Expr::Ident(cond_name) = cond_base.as_ref() else {
        return None;
    };
    let Stmt::Compound(c) = then_branch else {
        return None;
    };
    let [first, rest @ ..] = c.items.as_slice() else {
        return None;
    };
    let BlockItem::Stmt(Stmt::Expr(Some(Expr::Assign {
        op: AssignOp::Assign,
        lhs,
        rhs,
    }))) = first
    else {
        return None;
    };
    if !matches!(lhs.as_ref(), Expr::Ident(n) if n == ctor_var) {
        return None;
    }
    let Expr::Member {
        base: rhs_base,
        field: rhs_field,
        ..
    } = rhs.as_ref()
    else {
        return None;
    };
    if rhs_field != "specialdata" || !matches!(rhs_base.as_ref(), Expr::Ident(n) if n == cond_name)
    {
        return None;
    }
    Some((cond_base, rest))
}

/// Renders the block `existing_thinker_mutation_shape` matched: the
/// `if`'s own condition (reusing the ordinary `specialdata.is_some()`
/// truthiness rendering), then every remaining statement rendered with
/// `FnBodyContext::mutating_handle` set so each `CTOR_VAR->field`
/// reference (read *or* write) gets its own fresh `thinkers.get(..)`/
/// `get_mut(..)` call at exactly that point.
///
/// **Deliberately NOT a single hoisted `let Thinker::Variant(door) =
/// thinkers.get_mut(..).unwrap() else { unreachable!() };` binding held
/// across the whole block**, even though that's what a tick function's
/// own `self`-typed receiver already does (`T_VerticalDoor`'s `door.
/// field`) -- tried first, and confirmed a real borrow-checker rejection
/// by actually compiling it with `rustc`: `EV_VerticalDoor`'s own block
/// reads `door->direction`, then (in one branch) calls `thing->player`
/// -- itself a *second*, unrelated `thinkers.get(..)` borrow -- before
/// writing `door->direction` again. A hoisted `&mut` binding stays alive
/// across that whole span (Rust can't shrink its lifetime past the later
/// write), so the intervening immutable borrow doesn't type-check. Fresh
/// per-access borrows, each scoped to just one statement, avoid this
/// outright and remain correct for any future control flow between
/// reads/writes, not just this one shape.
fn render_existing_thinker_mutation(
    base: &Expr,
    rest: &[BlockItem],
    spec: &CtorSpec,
    ctx: &FnBodyContext,
    depth: usize,
) -> Result<Vec<String>, String> {
    let (base_text, base_is_crossref) = render_expr(base, ctx)?;
    let base_text = if base_is_crossref {
        format!("world[{base_text}]")
    } else {
        base_text
    };
    let handle_expr = format!("{base_text}.specialdata.unwrap()");
    let mut lines = vec![format!(
        "{}if {base_text}.specialdata.is_some() {{",
        indent(depth)
    )];
    let inner_ctx = FnBodyContext {
        mutating_handle: Some(MutatingHandle {
            var: spec.ctor_var,
            rust_type: spec.ctor_rust_type,
            handle_expr: &handle_expr,
        }),
        ..*ctx
    };
    for item in rest {
        match item {
            BlockItem::Decl(d) => lines.extend(render_decl(d, &inner_ctx, depth + 1)?),
            BlockItem::Stmt(st) => lines.extend(render_stmt(st, &inner_ctx, depth + 1)?),
        }
    }
    lines.push(format!("{}}}", indent(depth)));
    Ok(lines)
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
            if else_branch.is_none()
                && let Some(spec) = ctx.embedded_ctor
                && let Some((base, rest)) =
                    existing_thinker_mutation_shape(cond, then_branch, spec.ctor_var)
            {
                return render_existing_thinker_mutation(base, rest, &spec, ctx, depth);
            }
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
        // A `case`'s own body wrapped in real braces (`case raiseToTexture:
        // { int minsize = MAXINT; side_t* side; ... } break;`, `EV_DoFloor`'s
        // own `p_floor.c`) parses as one bare `Stmt::Compound`, unlike
        // every other case in the same `switch` (flat, brace-less
        // siblings) -- `render_switch` hands each arm's own statements to
        // `render_stmt` directly, so this needed the same dispatch
        // `render_block` already has for a `Compound`, just reachable as
        // an ordinary statement rather than only an `if`/`while` body.
        // Rust's own `match` arm already provides equivalent block
        // scoping for `minsize`/`side`, so the case's extra braces need
        // no separate nesting of their own here.
        Stmt::Compound(c) => render_compound_items(&c.items, ctx, depth),
        Stmt::Switch { cond, body } => render_switch(cond, body, ctx, depth),
        Stmt::While { cond, body } => render_while(cond, body, ctx, depth),
        Stmt::For {
            init,
            cond,
            step,
            body,
        } => render_for(init, cond, step, body, ctx, depth),
        Stmt::Continue => Ok(vec![format!("{}continue;", indent(depth))]),
        // A real loop-exiting `break` (`EV_DoFloor`'s own `lowerAndChange`
        // case, several levels inside a `for` loop's own body) -- distinct
        // from the switch-case-delimiter `break` `render_switch` consumes
        // itself while splitting arms apart (that one is peeled off
        // before individual statements ever reach `render_stmt` at all).
        Stmt::Break => Ok(vec![format!("{}break;", indent(depth))]),
        _ => Err(format!("render_stmt: unsupported statement shape: {s:?}")),
    }
}

/// Peels off a chain of `case` labels sharing one body (`case blazeRaise:
/// case blazeClose: <shared body>`) -- C parses each `case X:` as a label
/// wrapping only the *next* statement, so multiple labels in a row
/// (nothing but another `case` between them) nest as `Case{X, stmt:
/// Case{Y, stmt: <real body>}}` rather than appearing as flat siblings.
/// Confirmed directly against the real parsed AST (`T_VerticalDoor`) --
/// not assumed -- before writing this. Returns every label's own
/// rendered text, in order, plus the first statement that isn't itself a
/// `case` label (which may be `Stmt::Break` itself, for a group of labels
/// whose shared body is empty -- `render_switch`'s own caller handles
/// that).
fn collect_case_labels<'a>(
    mut s: &'a Stmt,
    ctx: &FnBodyContext,
) -> Result<(Vec<String>, &'a Stmt), String> {
    let mut labels = Vec::new();
    while let Stmt::Case { expr, stmt } = s {
        labels.push(render_expr(expr, ctx)?.0);
        s = stmt;
    }
    Ok((labels, s))
}

/// Renders `switch (cond) { ... }` as a Rust `match` -- see module docs
/// for how C's per-statement `case`/`default` labels get re-grouped into
/// one block per arm, and why an implicit `_ => {}` is added when there's
/// no explicit `default:`.
/// One `case`/`default` label group, before fallthrough resolution:
/// its own labels (`None` for `default`), the statements written
/// directly under it, and whether it falls into whatever comes next
/// (no `break` before the next label, and something *does* follow).
struct RawArm<'a> {
    labels: Option<Vec<String>>,
    own_stmts: Vec<&'a Stmt>,
    falls_through: bool,
}

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

    // Pass 1: split into raw arms, in source order, recording (rather
    // than rejecting) whether each one falls through into the next --
    // real C fallthrough (`case silentCrushAndRaise: S_StartSound(..);
    // case fastCrushAndRaise: case crushAndRaise: ceiling->direction =
    // -1; break;`, confirmed against the real parsed AST, not assumed)
    // needs the *target*'s own statements folded into the source arm's
    // body, since Rust `match` arms never fall through -- done below, in
    // pass 2.
    let mut arms: Vec<RawArm> = Vec::new();
    let mut has_default = false;
    let mut i = 0;
    while i < stmts.len() {
        let (labels, first_stmt) = match stmts[i] {
            Stmt::Case { .. } => {
                let (labels, first_stmt) = collect_case_labels(stmts[i], ctx)?;
                (Some(labels), first_stmt)
            }
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
        // Several labels sharing one body (`case blazeRaise: case
        // blazeClose: ...;`) can peel all the way down to a bare `break;`
        // with nothing in between (`case blazeClose: case close: break;`
        // -- "these types intentionally do nothing") -- `collect_case_labels`
        // hands that back as `first_stmt` itself, not a separate sibling,
        // so it needs recognizing here rather than being pushed as a real
        // body statement.
        let mut saw_break = matches!(first_stmt, Stmt::Break);
        let mut own_stmts: Vec<&Stmt> = if saw_break {
            Vec::new()
        } else {
            vec![first_stmt]
        };
        while !saw_break && i < stmts.len() {
            match stmts[i] {
                Stmt::Case { .. } | Stmt::Default(_) => break,
                Stmt::Break => {
                    saw_break = true;
                    i += 1;
                    break;
                }
                other => {
                    own_stmts.push(other);
                    i += 1;
                }
            }
        }
        // `case 0: return;` (`A_Scream`) -- an arm can end in an
        // unconditional `return` with no `break` at all, reaching the
        // next label only because nothing else follows it in the source,
        // not because it falls through. Confirmed against the real
        // parsed AST before fixing rather than assumed: without this, the
        // naive "no `break` seen before the next label" rule would have
        // wrongly folded the *next* arm's own statements in as dead code
        // after the `return` -- harmless to runtime behavior (`return`
        // still exits first), but not the honest, clean-`match` output
        // this renderer is otherwise held to, and `cargo clippy` flags
        // the resulting unreachable code.
        let terminates = saw_break || matches!(own_stmts.last(), Some(Stmt::Return(_)));
        let falls_through = !terminates && i < stmts.len();
        arms.push(RawArm {
            labels,
            own_stmts,
            falls_through,
        });
    }

    // Pass 2: resolve each arm's real body, back to front, so a falling-
    // through arm can borrow its already-resolved successor's body --
    // handles multi-level fallthrough (an arm falling into another arm
    // that itself falls through) for free, since each arm only ever
    // looks one step ahead at its own already-finished neighbor.
    let mut resolved: Vec<Vec<&Stmt>> = vec![Vec::new(); arms.len()];
    for k in (0..arms.len()).rev() {
        let mut body_stmts = arms[k].own_stmts.clone();
        if arms[k].falls_through {
            body_stmts.extend(resolved[k + 1].iter().copied());
        }
        resolved[k] = body_stmts;
    }

    let mut lines = vec![format!("{}match {cond_text} {{", indent(depth))];
    for (k, arm) in arms.iter().enumerate() {
        let pattern = arm
            .labels
            .as_ref()
            .map(|ls| ls.join(" | "))
            .unwrap_or_else(|| "_".to_string());
        lines.push(format!("{}{pattern} => {{", indent(depth + 1)));
        for s in &resolved[k] {
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

/// Renders `for (init; cond; step) body` -- C's counted-loop idiom
/// (`EV_DoFloor`'s own two otherwise-identical `for (i = 0; i <
/// sec->linecount; i++)` adjacency scans). Only a plain-assignment init
/// (the loop counter is already declared earlier in the function -- the
/// only shape any real corpus `for` found so far uses, and the only one
/// C89 itself allows) and an `x++`/`x--` step are supported; both become
/// an ordinary statement, with the step appended *after* the body inside
/// a Rust `while` -- correct as long as the body itself never
/// `continue`s (rejected below), since C's own `for` still runs its step
/// on `continue`, unlike a bare Rust `while`/`loop`, which would jump
/// straight back to the condition and skip it.
fn render_for(
    init: &Option<ForInit>,
    cond: &Option<Expr>,
    step: &Option<Expr>,
    body: &Stmt,
    ctx: &FnBodyContext,
    depth: usize,
) -> Result<Vec<String>, String> {
    let Some(ForInit::Expr(init_expr)) = init else {
        return Err(format!("render_for: unsupported init shape: {init:?}"));
    };
    let Some(cond) = cond else {
        return Err("render_for: a missing condition is not supported yet".to_string());
    };
    let Some(step) = step else {
        return Err("render_for: a missing step is not supported yet".to_string());
    };
    let bare_continue = match body {
        Stmt::Compound(c) => body_has_bare_continue(&c.items),
        other => stmt_has_bare_continue(other),
    };
    if bare_continue {
        return Err(
            "render_for: `continue` inside a for-loop body is not supported yet (the translated `while` would skip the step, unlike C's `for`)"
                .to_string(),
        );
    }

    let init_text = render_expr_stmt(init_expr, ctx)?;
    let cond_text = render_bool_expr(cond, ctx)?;
    let step_text = render_for_step(step, ctx)?;

    let mut lines = vec![format!("{}{init_text};", indent(depth))];
    lines.push(format!("{}while {cond_text} {{", indent(depth)));
    lines.extend(render_block(body, ctx, depth + 1)?);
    lines.push(format!("{}{step_text};", indent(depth + 1)));
    lines.push(format!("{}}}", indent(depth)));
    Ok(lines)
}

/// `i++`/`i--`, or a compound-assign step (`A_BrainScream`'s own `for
/// (x = ...; x < ...; x += FRACUNIT*8)`, scanning 4096-unit-wide slices
/// of the map rather than counting by ones) -- delegates to
/// `render_expr_stmt`'s own already-general `Expr::Assign` handling
/// rather than duplicating it, since a `for` step is just an ordinary
/// statement rendered without its own trailing `;` (added by `render_for`
/// itself).
fn render_for_step(step: &Expr, ctx: &FnBodyContext) -> Result<String, String> {
    match step {
        Expr::PostIncDec { .. } | Expr::PreIncDec { .. } => Ok(render_expr(step, ctx)?.0),
        Expr::Assign { .. } => render_expr_stmt(step, ctx),
        other => Err(format!("render_for: unsupported step shape: {other:?}")),
    }
}

/// Detects a `continue` reaching a `for` loop's own body directly --
/// stops descending at a nested `while`/`for`, since that inner loop
/// consumes its own `continue` rather than letting it reach the outer
/// one. Mirrors `body_has_self_removal`'s own recursive-scan shape.
fn body_has_bare_continue(items: &[BlockItem]) -> bool {
    items.iter().any(|item| match item {
        BlockItem::Stmt(s) => stmt_has_bare_continue(s),
        BlockItem::Decl(_) => false,
    })
}

fn stmt_has_bare_continue(s: &Stmt) -> bool {
    match s {
        Stmt::Continue => true,
        Stmt::Compound(c) => body_has_bare_continue(&c.items),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            stmt_has_bare_continue(then_branch)
                || else_branch
                    .as_ref()
                    .is_some_and(|eb| stmt_has_bare_continue(eb))
        }
        Stmt::Switch { body, .. } => stmt_has_bare_continue(body),
        Stmt::Case { stmt, .. } => stmt_has_bare_continue(stmt),
        Stmt::Default(stmt) => stmt_has_bare_continue(stmt),
        Stmt::While { .. } | Stmt::For { .. } => false,
        _ => false,
    }
}

/// `P_RemoveThinker(&door->thinker);` -- a tick function removing
/// *itself* (`T_VerticalDoor`'s own "unlink and free" once a mover
/// finishes), the single most common self-removal shape in the corpus
/// (matching `Arena::remove`'s own doc comment on the same pattern). Only
/// this exact shape is recognized -- removing some *other* handle
/// (`P_RemoveThinker(&other->thinker)`) is a different, not-yet-attempted
/// case, since it would need to *name* that other handle somehow rather
/// than just reusing the receiver's own.
fn is_self_removal_call(e: &Expr, self_param: &str) -> bool {
    let Expr::Call { callee, args } = e else {
        return false;
    };
    if !matches!(callee.as_ref(), Expr::Ident(n) if n == "P_RemoveThinker") {
        return false;
    }
    let [arg] = args.as_slice() else {
        return false;
    };
    matches!(arg, Expr::Unary { op: UnaryOp::AddrOf, expr }
        if matches!(expr.as_ref(), Expr::Member { base, field, arrow: true }
            if field == "thinker" && matches!(base.as_ref(), Expr::Ident(n) if n == self_param)))
}

/// Whether `e` is built entirely from a known real `FixedT` source
/// (`FRACUNIT`, or a self-struct/`Handle<Thinker>`-local field
/// registered `"FixedT"` -- both share `ctx.self_field_types`, since a
/// `P_SpawnMobj` local is always the same `Mobj` shape `self_param` is)
/// threaded through plain arithmetic (`+`/`-`/`*`/`/`/unary `-`).
/// `false` for anything else -- a bare call (`P_Random()`), a plain
/// integer literal, or arithmetic built purely from those. Drives one
/// narrow gap `A_BrainExplode` surfaced: `th->momz = P_Random()*512;`
/// assigns a *plain* `i32` value (no `FixedT` source anywhere in it)
/// straight into a `FixedT` field -- valid, unremarkable C (`fixed_t`
/// is a bare `typedef int`, so this is just an `int` written into
/// another `int`), but Rust needs an explicit `FixedT(..)` wrap, the
/// same "C silently reinterprets the bits, Rust needs it spelled out"
/// idea as the already-established `angle_t`/plain-`int` cast pair.
/// Conservative on purpose: a `false` result only means "not provably
/// `FixedT`," not "definitely plain `int`," so this only ever adds a
/// wrap, never removes information a caller's already-typed value
/// needs.
fn expr_is_fixed_t_valued(e: &Expr, ctx: &FnBodyContext) -> bool {
    match e {
        Expr::Ident(n) => n == "FRACUNIT",
        Expr::Member { base, field, .. } => {
            let base_is_self_or_handle_local = matches!(base.as_ref(), Expr::Ident(n) if n == ctx.self_param)
                || matches!(base.as_ref(), Expr::Ident(n) if ctx.extra_cross_ref_idents.get(n.as_str()).map(String::as_str) == Some("Handle<Thinker>"));
            base_is_self_or_handle_local
                && ctx.self_field_types.get(field.as_str()).map(String::as_str) == Some("FixedT")
        }
        Expr::Binary {
            op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div,
            lhs,
            rhs,
        } => expr_is_fixed_t_valued(lhs, ctx) || expr_is_fixed_t_valued(rhs, ctx),
        Expr::Unary {
            op: UnaryOp::Minus,
            expr,
        } => expr_is_fixed_t_valued(expr, ctx),
        // `FixedMul(mo->info->speed, finecosine[an]);` (`A_FatAttack1`'s
        // own idiom, written straight into a spawned mobj's `momx`) --
        // `FixedMul`/`FixedDiv`/`FixedDiv2`'s own real declared C return
        // type is `fixed_t` (`m_fixed.h`, confirmed by direct read, not
        // assumed), matching `runtime/fixed.rs`'s already-implemented
        // `fixed_mul`/`fixed_div`/`fixed_div2` methods this module's
        // eventual translation of those functions would call into -- so
        // a bare call to one of them is already `FixedT`-valued and must
        // *not* additionally get wrapped in `FixedT(..)` (that would try
        // to build a `FixedT` from a `FixedT` argument, a real `rustc`
        // rejection this exact double-wrap was caught producing before
        // this arm existed). Scoped to these three names, this module's
        // usual "hand-match the one real callee" style, not a general
        // "assume every unmodeled call is `FixedT`" fallback (most
        // calls, like `P_Random()`, genuinely aren't).
        Expr::Call { callee, .. } => {
            matches!(callee.as_ref(), Expr::Ident(n) if n == "FixedMul" || n == "FixedDiv" || n == "FixedDiv2")
        }
        _ => false,
    }
}

/// `Expr::Assign` is rendered here (not in `render_expr`) since Doom's
/// tick functions only ever use it as a bare statement, never nested
/// inside a larger expression -- confirmed for `T_FireFlicker`, not yet
/// generalized.
fn render_expr_stmt(e: &Expr, ctx: &FnBodyContext) -> Result<String, String> {
    // `handle`/`arena` are the fixed names this renderer always uses for
    // a tick function's own `Handle<Thinker>`/`&mut Arena<Thinker>`
    // parameters, the same way `world`/`thinkers` are fixed names
    // elsewhere -- not yet threaded through `render_fn`'s own generated
    // signature (needs a complete real self-removing function to design
    // that against, not just this one statement shape in isolation; see
    // docs/03_TRANSPILER.md), so this only renders correctly once a
    // caller supplies them by hand.
    if is_self_removal_call(e, ctx.self_param) {
        return Ok("arena.remove(handle)".to_string());
    }
    if let Expr::Assign { op, lhs, rhs } = e {
        // `door->field = expr;`, inside `render_existing_thinker_mutation`'s
        // own block (`ctx.mutating_handle` set) -- a *write* through a
        // `Handle<Thinker>` needs its own fresh `thinkers.get_mut(..)`
        // call at exactly this point, wrapped in its own `if let`, rather
        // than reusing a single hoisted `&mut` binding for the whole
        // block: holding one across the block would keep `thinkers`
        // mutably borrowed even where a *different* statement in between
        // needs its own (immutable) borrow of `thinkers` (`EV_VerticalDoor`'s
        // own `if (!thing->player) return;`, sitting between the read of
        // `door->direction` and the write to it) -- confirmed as a real
        // borrow-checker rejection, not a hypothetical, by actually
        // compiling the hoisted-binding version with `rustc` first.
        // Intercepted here, before the general `render_expr(lhs, ..)`
        // call below, since that would otherwise render `lhs` as a
        // *read* (an un-assignable match expression).
        if *op == AssignOp::Assign
            && let Expr::Member {
                base,
                field: lhs_field,
                ..
            } = lhs.as_ref()
            && let Some(mh) = ctx.mutating_handle
            && matches!(base.as_ref(), Expr::Ident(n) if n == mh.var)
        {
            let (rhs_text, _) = render_expr(rhs, ctx)?;
            let field = rust_field_name(lhs_field)?;
            return Ok(format!(
                "if let Some(Thinker::{}({})) = thinkers.get_mut({}) {{ {}.{field} = {rhs_text}; }}",
                mh.rust_type, mh.var, mh.handle_expr, mh.var
            ));
        }
        // `th->field = expr;` / `th->field -= expr;` -- writing a field of
        // a `Handle<Thinker>`-typed local (`collect_spawn_mobj_locals`'s
        // own `th = P_SpawnMobj(...);`, `A_Tracer`'s idiom), the write
        // counterpart of the general read arm `render_expr`'s own
        // `Member` handling gained for the same case. A fresh `thinkers.
        // get_mut(..)` call at exactly this point, the same borrow-
        // scoping reasoning as `mutating_handle`'s own write arm just
        // above -- unlike that one, this also needs to support a
        // *compound* op (`th->tics -= P_Random()&3;`, `A_Tracer`'s own
        // countdown refinement, not just `EV_VerticalDoor`'s plain `=`),
        // so it renders whatever real `AssignOp` the source used rather
        // than hardcoding `=`. One more real wrinkle: `fog->target =
        // actor;` (`A_VileTarget`'s own idiom) stores the function's
        // *own* receiver as a value into the freshly-spawned mobj's
        // `target` field -- `self_param` alone renders as a `&mut Mobj`,
        // not the `Handle<Thinker>` this `Option<Handle<Thinker>>` field
        // actually needs, so this narrow case (scoped to the two fields
        // known `Option<Handle<Thinker>>`-typed on `Mobj`, mirroring
        // `is_target_tracer_typed`'s own pair) substitutes the fixed
        // `handle` name `render_fn_impl`'s own signature extension
        // supplies whenever a body needs its receiver's own handle as a
        // value (see `body_has_self_handle_value`).
        if let Expr::Member {
            base,
            field: lhs_field,
            ..
        } = lhs.as_ref()
            && matches!(base.as_ref(), Expr::Ident(n) if ctx.extra_cross_ref_idents.get(n.as_str()).map(String::as_str) == Some("Handle<Thinker>"))
        {
            let Expr::Ident(name) = base.as_ref() else {
                unreachable!("guarded above")
            };
            let field = rust_field_name(lhs_field)?;
            let is_option_handle_field = lhs_field == "target" || lhs_field == "tracer";
            let rhs_is_self = matches!(rhs.as_ref(), Expr::Ident(n) if n == ctx.self_param);
            // `th->momz = P_Random()*512;` (`A_BrainExplode`) -- a plain
            // `i32` expression (no `FixedT` source anywhere in it, see
            // `expr_is_fixed_t_valued`'s own doc comment) assigned into a
            // field this same `Mobj` shape's `self_field_types` registers
            // `FixedT`, needing an explicit wrap the same way `angle_t`'s
            // own bit-reinterpretation idiom already does elsewhere.
            // Scoped to a plain `=` only -- no real corpus example needs
            // this through a compound op yet.
            let is_fixed_t_field = ctx
                .self_field_types
                .get(lhs_field.as_str())
                .map(String::as_str)
                == Some("FixedT");
            // `mo->momx = FixedMul(mo->info->speed, ..);` (`A_FatAttack1`)
            // -- the RHS can itself read a *different* field of this same
            // `mo`, which must resolve to the write's own already-bound
            // `m` rather than a second, independent `thinkers.get(mo)`
            // call (a real borrow conflict with the `get_mut` below --
            // see `FnBodyContext::same_handle_write`'s own doc comment).
            let rhs_ctx = FnBodyContext {
                same_handle_write: Some(name.as_str()),
                ..*ctx
            };
            let rhs_text = if is_option_handle_field && rhs_is_self {
                "Some(handle)".to_string()
            } else if *op == AssignOp::Assign
                && is_fixed_t_field
                && !expr_is_fixed_t_valued(rhs, &rhs_ctx)
            {
                format!("FixedT({})", render_expr(rhs, &rhs_ctx)?.0)
            } else {
                render_expr(rhs, &rhs_ctx)?.0
            };
            return Ok(format!(
                "if let Some(Thinker::Mobj(m)) = thinkers.get_mut({name}) {{ m.{field} {} {rhs_text}; }}",
                render_assign_op(*op)
            ));
        }
        let (lhs_text, _) = render_expr(lhs, ctx)?;
        let (rhs_text, _) = render_expr(rhs, ctx)?;
        // `sector_t.specialdata`/`line_t.specialdata` map to
        // `Option<Handle<Thinker>>` (struct_fields.rs's own name-based
        // special case -- it's checked for truthiness/reset to `NULL`
        // corpus-wide, not dereferenced unconditionally), so any
        // assignment to it needs the same treatment every other
        // `Option`-typed field gets from its own corpus initializer, even
        // though this renderer has no general per-field-type awareness
        // beyond this one matching special case: `NULL` (a mover
        // resetting the field once it finishes, e.g. `T_VerticalDoor`'s
        // `door->sector->specialdata = NULL;`) becomes `None`; a
        // constructor's own back-reference (`sec->specialdata = door;`,
        // only reachable once `ctor_var_handle_name` is active -- see
        // module docs) becomes `Some(..)`.
        let lhs_is_specialdata =
            matches!(lhs.as_ref(), Expr::Member { field, .. } if field == "specialdata");
        let rhs_is_null = matches!(rhs.as_ref(), Expr::Ident(n) if n == "NULL");
        let rhs_is_ctor_var = !ctx.ctor_var_handle_name.is_empty()
            && matches!(rhs.as_ref(), Expr::Ident(n) if n == ctx.ctor_var);
        // `angle = actor->angle;` (`A_PosAttack`) -- `angle`'s true C
        // type is a plain `int` (`FnBodyContext::plain_int_locals`),
        // while `actor->angle`'s registered Rust type is `u32`
        // (`angle_t`, `struct_fields.rs`'s own mapping) -- C silently
        // reinterprets the bits on assignment, which Rust needs an
        // explicit `as i32` for, the same idea as `sec-sectors`'s own
        // `.0 as i32` elsewhere in this module. Scoped narrowly to a
        // direct `self_param` field read assigned straight into one of
        // these locals -- confirmed a real compile error otherwise (see
        // `FnBodyContext::plain_int_locals`'s own doc comment), not
        // guessed at.
        let lhs_is_plain_int_local =
            matches!(lhs.as_ref(), Expr::Ident(n) if ctx.plain_int_locals.contains(n.as_str()));
        let rhs_is_u32_self_field = matches!(rhs.as_ref(), Expr::Member { base, field, .. }
            if matches!(base.as_ref(), Expr::Ident(n) if n == ctx.self_param)
                && ctx.self_field_types.get(field.as_str()).map(String::as_str) == Some("u32"));
        // `actor->angle += (P_Random()-P_Random())<<21;` (`A_FaceTarget`)
        // -- the reverse direction of the `angle_t`-into-plain-`int` bug
        // just above: a *compound*-assign's RHS is ordinary `int`-typed
        // arithmetic (no operand here is itself a registered `u32`
        // field), but the LHS it's folded into is `angle_t`/`u32` --
        // confirmed a real `rustc` rejection (`cannot add-assign i32 to
        // u32`), not guessed at. Scoped to a *compound* op specifically
        // (`+=`/`-=`/...), not plain `=`: a plain assignment's RHS in
        // every real corpus case seen so far is already a call whose
        // real C return type is itself `angle_t`
        // (`R_PointToAngle2`) -- wrapping that in a redundant `as u32`
        // would be a no-op cast `clippy` flags, not a fix for a real
        // mismatch, so it's deliberately left alone unless a genuine
        // counterexample shows up.
        let lhs_is_u32_self_field = matches!(lhs.as_ref(), Expr::Member { base, field, .. }
            if matches!(base.as_ref(), Expr::Ident(n) if n == ctx.self_param)
                && ctx.self_field_types.get(field.as_str()).map(String::as_str) == Some("u32"));
        // `actor->tracer = fog;` (`A_VileTarget`'s own idiom) -- writing a
        // bare `Handle<Thinker>`-typed local (`fog`, fresh out of
        // `P_SpawnMobj`) straight into `self`'s own `target`/`tracer`
        // field needs the same `Some(..)` wrap `specialdata`'s own
        // constructor back-reference already gets above, generalized:
        // unlike `specialdata` (this renderer's only other `Option<
        // Handle<Thinker>>` field, keyed off `ctor_var_handle_name`
        // specifically), `target`/`tracer` are looked up the ordinary
        // way, via `self_field_types`, and the RHS is any local
        // registered `"Handle<Thinker>"` in `extra_cross_ref_idents`, not
        // just a constructor's own handle.
        let lhs_is_target_or_tracer_self_field = matches!(lhs.as_ref(), Expr::Member { base, field, .. }
            if matches!(base.as_ref(), Expr::Ident(n) if n == ctx.self_param)
                && (field == "target" || field == "tracer"));
        let rhs_is_handle_local = matches!(rhs.as_ref(), Expr::Ident(n) if ctx.extra_cross_ref_idents.get(n.as_str()).map(String::as_str) == Some("Handle<Thinker>"));
        let rhs_text = if lhs_is_specialdata && rhs_is_null {
            "None".to_string()
        } else if (lhs_is_specialdata && rhs_is_ctor_var)
            || (lhs_is_target_or_tracer_self_field && rhs_is_handle_local)
        {
            format!("Some({rhs_text})")
        } else if lhs_is_plain_int_local && rhs_is_u32_self_field {
            format!("{rhs_text} as i32")
        } else if lhs_is_u32_self_field && *op != AssignOp::Assign {
            format!("({rhs_text}) as u32")
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
    // A trigger with an inline constructor (`render_trigger_fn`'s own
    // `embedded_ctor`, e.g. `EV_DoCeiling`) watches every block it
    // renders for the `Z_Malloc` call that starts its known constructor
    // local's build-up -- whichever block actually contains it (a
    // `while` loop's own body, for the one real example so far) hands
    // everything from that point onward to `render_ctor_body` wholesale,
    // the same "process my whole remaining scope" behavior
    // `render_spawn_fn` already has for a full function.
    if let Some(spec) = ctx.embedded_ctor
        && let Some(idx) = items.iter().position(
            |item| matches!(item, BlockItem::Stmt(s) if is_malloc_assign(s, spec.ctor_var)),
        )
    {
        let mut out = Vec::new();
        for item in &items[..idx] {
            match item {
                BlockItem::Decl(d) => out.extend(render_decl(d, ctx, depth)?),
                BlockItem::Stmt(s) => out.extend(render_stmt(s, ctx, depth)?),
            }
        }
        out.extend(render_ctor_body(
            &items[idx..],
            &spec,
            ctx,
            depth,
            spec.ctor_var,
        )?);
        return Ok(out);
    }

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

/// Names declared as a bare, non-array, non-pointer `int` at a
/// function's own top level (`int angle;`, possibly several off one
/// specifier like `int secnum,rtn;`) -- see `FnBodyContext::
/// plain_int_locals`. Deliberately shallow (top-level items only, not
/// recursing into `if`/`switch`/`for` bodies): C89 declarations always
/// sit at the top of whichever block they belong to, and every real
/// corpus function needing this so far declares every local it needs
/// directly at the function's own top level, the same scope
/// `render_decl`'s own single-declarator-per-line precedent (`EV_DoDoor`'s
/// `int secnum,rtn;`) was measured against.
fn collect_plain_int_locals(items: &[BlockItem]) -> HashSet<String> {
    let mut names = HashSet::new();
    for item in items {
        let BlockItem::Decl(d) = item else { continue };
        if !matches!(
            d.specifiers.type_specifiers.as_slice(),
            [TypeSpecifier::Int]
        ) {
            continue;
        }
        for decl in &d.declarators {
            if let DirectDeclarator::Ident(_) = decl.declarator.direct
                && let Some(name) = declarator_name(&decl.declarator)
            {
                names.insert(name);
            }
        }
    }
    names
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
    render_fn_impl(
        corpus_dir,
        file,
        fn_name,
        self_rust_type,
        self_field_types,
        None,
    )
}

/// `render_fn`'s own `boolean`-returning twin -- `P_CheckMeleeRange`/
/// `P_CheckMissileRange` (`p_enemy.c`) are `boolean P_Check...(mobj_t*
/// actor)`, the same single-self-struct-parameter shape `render_fn`
/// already handles, just with a real return value instead of `void`.
/// A thin wrapper over the same `render_fn_impl` rather than a new
/// required parameter threaded through `render_fn`'s own 36 existing call
/// sites (every one of them a real `void A_*`/tick function, never
/// needing a return type) -- `boolean` already maps to Rust's native
/// `bool` (`struct_fields.rs`'s own decision), so `"bool"` is the only
/// return type this needs to support so far.
pub fn render_bool_fn(
    corpus_dir: &Path,
    file: &str,
    fn_name: &str,
    self_rust_type: &str,
    self_field_types: &HashMap<String, String>,
) -> Result<String, String> {
    render_fn_impl(
        corpus_dir,
        file,
        fn_name,
        self_rust_type,
        self_field_types,
        Some("bool"),
    )
}

fn render_fn_impl(
    corpus_dir: &Path,
    file: &str,
    fn_name: &str,
    self_rust_type: &str,
    self_field_types: &HashMap<String, String>,
    return_type: Option<&str>,
) -> Result<String, String> {
    let (_, unit) = parse_full(corpus_dir.join(file).to_str().unwrap())?;
    let f = find_function_def(&unit.items, fn_name)
        .ok_or_else(|| format!("{fn_name} not found in {file}"))?;
    let param_name = first_param_name(f)
        .ok_or_else(|| format!("{fn_name}: first parameter has no plain name"))?;
    let target_tracer_aliases =
        collect_target_tracer_aliases(&f.body.items, &param_name, self_field_types);
    let spawn_mobj_locals = collect_spawn_mobj_locals(&f.body.items);
    let mut extra_cross_ref_idents = target_tracer_aliases.clone();
    extra_cross_ref_idents.extend(
        spawn_mobj_locals
            .iter()
            .map(|(k, v)| (k.clone(), v.clone())),
    );
    let plain_int_locals = collect_plain_int_locals(&f.body.items);
    let ctx = FnBodyContext {
        self_param: &param_name,
        self_field_types,
        extra_cross_ref_idents: &extra_cross_ref_idents,
        ctor_var: "",
        ctor_var_handle_name: "",
        ctor_field_types: &HashMap::new(),
        embedded_ctor: None,
        mutating_handle: None,
        same_handle_write: None,
        plain_int_locals: &plain_int_locals,
    };
    let body_lines = render_compound_items(&f.body.items, &ctx, 1)?;
    // Only a tick function that actually removes itself somewhere in its
    // body (`is_self_removal_call`, possibly nested arbitrarily deep in
    // `switch`/`if` -- `T_VerticalDoor` buries several inside two levels
    // of `switch`) needs its own `Handle<Thinker>`/`&mut Arena<Thinker>`
    // -- confirmed against a first complete self-removing function
    // (`T_VerticalDoor`) rather than added speculatively to every tick
    // function's signature ahead of real evidence, matching the same
    // "measure, don't guess" call already made for `World`.
    // Same "measure, don't add speculatively" discipline as the self-
    // removal check just above: only a function that actually
    // dereferences *through* `target`/`tracer` (not just checks
    // truthiness or passes it opaquely) gets the extra `thinkers` read-
    // only lookup parameter.
    let needs_self_removal = body_has_self_removal(&f.body.items, &param_name);
    let needs_target_deref = body_has_target_deref(
        &f.body.items,
        &param_name,
        self_field_types,
        &target_tracer_aliases,
    );
    // A body that assigns a local from `P_SpawnMobj(...)` and then
    // writes one of its fields (`A_Tracer`'s own `th->momz = ...;`)
    // needs real *mutable* `Arena` access -- unlike `needs_target_deref`
    // (read-only), this reuses the same `thinkers` parameter name but
    // makes it `&mut Arena<Thinker>` instead, which still supports every
    // `needs_target_deref` read site too (`&mut Arena` can always
    // reborrow immutably for `.get(..)`), so the two compose for free
    // when a function (like `A_Tracer`) needs both at once: one mutable
    // `thinkers` parameter, not two conflicting ones. Every real
    // `P_SpawnMobj` local found is assumed to need mutation -- narrower
    // ("does it ever actually write a field") wasn't worth measuring
    // separately, since every real caller so far does.
    let needs_spawn_mut = !spawn_mobj_locals.is_empty();
    // `fog->target = actor;` (`A_VileTarget`'s own idiom) needs the
    // function's *own* receiver as a `Handle<Thinker>` *value* -- a
    // genuinely different need from self-removal's own `handle` (that
    // one also needs `arena.remove(handle)`; this one only ever reads
    // `handle` as a plain value), so it reuses the same fixed `handle`
    // parameter name without self-removal's own `arena` companion.
    let needs_self_handle_value =
        body_has_self_handle_value(&f.body.items, &param_name, &spawn_mobj_locals);
    if needs_self_removal && (needs_target_deref || needs_spawn_mut || needs_self_handle_value) {
        return Err(format!(
            "{fn_name}: needs both self-removal and target/tracer dereferencing or a spawned mobj's own handle -- not yet designed (see render_fn's own extra_params comment), fix by hand rather than guessing"
        ));
    }
    let handle_part = if needs_self_removal {
        ", handle: Handle<Thinker>, arena: &mut Arena<Thinker>"
    } else if needs_self_handle_value {
        ", handle: Handle<Thinker>"
    } else {
        ""
    };
    let thinkers_part = if needs_self_removal {
        ""
    } else if needs_spawn_mut {
        ", thinkers: &mut Arena<Thinker>"
    } else if needs_target_deref {
        ", thinkers: &Arena<Thinker>"
    } else {
        ""
    };
    let extra_params = format!("{thinkers_part}{handle_part}");
    let return_arrow = return_type.map(|t| format!(" -> {t}")).unwrap_or_default();
    Ok(format!(
        "pub fn {fn_name}({param_name}: &mut {self_rust_type}, world: &mut World{extra_params}){return_arrow} {{\n{}\n}}",
        body_lines.join("\n")
    ))
}

fn nth_param_name(f: &FunctionDef, n: usize) -> Option<String> {
    let DirectDeclarator::Function(_, params) = &f.declarator.direct else {
        return None;
    };
    let param = params.params.get(n)?;
    let ParamDeclarator::Named(d) = &param.declarator else {
        return None;
    };
    declarator_name(d)
}

/// Renders a `fn(player_t*, pspdef_t*)`-shaped action function (`state_t.
/// action`'s `acp2` variant, `action_fn.rs`'s `ActionFn::Weapon`) --
/// `A_Light0`/`A_Light1`/`A_Light2`, the first real examples. Unlike
/// `render_fn`'s single-struct tick-function shape, this has two real
/// parameters, neither a `Thinker`; only `player`'s fields are resolved
/// (`player_t` isn't struct-mapped in `struct_fields.rs` -- see module
/// docs -- so `player_field_types` is supplied directly by the caller,
/// the same as every other function-body test). No `world: &mut World`
/// parameter yet: none of the three real functions this was built against
/// touch a cross-reference field, so one isn't threaded through
/// speculatively -- add it if a future weapon action function needs one,
/// the same "measure, don't guess" reasoning `render_fn`'s own
/// self-removal parameters already follow.
pub fn render_weapon_fn(
    corpus_dir: &Path,
    file: &str,
    fn_name: &str,
    player_field_types: &HashMap<String, String>,
) -> Result<String, String> {
    let (_, unit) = parse_full(corpus_dir.join(file).to_str().unwrap())?;
    let f = find_function_def(&unit.items, fn_name)
        .ok_or_else(|| format!("{fn_name} not found in {file}"))?;
    let player_param = nth_param_name(f, 0)
        .ok_or_else(|| format!("{fn_name}: first parameter has no plain name"))?;
    let psp_param = nth_param_name(f, 1)
        .ok_or_else(|| format!("{fn_name}: second parameter has no plain name"))?;
    let no_extra_cross_refs = HashMap::new();
    let ctx = FnBodyContext {
        self_param: &player_param,
        self_field_types: player_field_types,
        extra_cross_ref_idents: &no_extra_cross_refs,
        ctor_var: "",
        ctor_var_handle_name: "",
        ctor_field_types: &HashMap::new(),
        embedded_ctor: None,
        mutating_handle: None,
        same_handle_write: None,
        plain_int_locals: &HashSet::new(),
    };
    let body_lines = render_compound_items(&f.body.items, &ctx, 1)?;
    Ok(format!(
        "pub fn {fn_name}({player_param}: &mut Player, {psp_param}: &mut PlayerSpriteState) {{\n{}\n}}",
        body_lines.join("\n")
    ))
}

fn body_has_self_removal(items: &[BlockItem], self_param: &str) -> bool {
    items.iter().any(|item| match item {
        BlockItem::Stmt(s) => stmt_has_self_removal(s, self_param),
        BlockItem::Decl(_) => false,
    })
}

fn stmt_has_self_removal(s: &Stmt, self_param: &str) -> bool {
    if let Stmt::Expr(Some(e)) = s
        && is_self_removal_call(e, self_param)
    {
        return true;
    }
    match s {
        Stmt::Compound(c) => body_has_self_removal(&c.items, self_param),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            stmt_has_self_removal(then_branch, self_param)
                || else_branch
                    .as_ref()
                    .is_some_and(|eb| stmt_has_self_removal(eb, self_param))
        }
        Stmt::Switch { body, .. } => stmt_has_self_removal(body, self_param),
        Stmt::Case { stmt, .. } => stmt_has_self_removal(stmt, self_param),
        Stmt::Default(stmt) => stmt_has_self_removal(stmt, self_param),
        Stmt::While { body, .. } => stmt_has_self_removal(body, self_param),
        _ => false,
    }
}

/// Whether `e` (anywhere in its subtree) dereferences *through*
/// `{self_param}.target`/`.tracer` to read one of the targeted mobj's own
/// fields (`render_expr`'s `is_target_tracer_typed` chain-through arm)
/// -- unlike `is_self_removal_call`'s single fixed top-level-statement
/// shape, this can appear nested arbitrarily deep in any expression
/// (a condition, a call argument, an assignment's RHS, ...), so it needs
/// a real expression-tree walk rather than one statement-shaped check.
/// Drives `render_fn`'s own signature extension: only a function that
/// actually needs a real thinker lookup gets the extra `thinkers: &Arena
/// <Thinker>` parameter, matching the same "measure, don't add
/// speculatively" discipline `body_has_self_removal` already set.
fn expr_has_target_deref(
    e: &Expr,
    self_param: &str,
    self_field_types: &HashMap<String, String>,
    aliases: &HashMap<String, String>,
) -> bool {
    if let Expr::Member { base, .. } = e
        && is_target_tracer_typed(base, self_param, self_field_types, aliases)
    {
        return true;
    }
    match e {
        Expr::Unary { expr, .. }
        | Expr::PreIncDec { expr, .. }
        | Expr::PostIncDec { expr, .. }
        | Expr::Cast { expr, .. }
        | Expr::Sizeof(SizeofArg::Expr(expr)) => {
            expr_has_target_deref(expr, self_param, self_field_types, aliases)
        }
        Expr::Binary { lhs, rhs, .. } | Expr::Comma(lhs, rhs) => {
            expr_has_target_deref(lhs, self_param, self_field_types, aliases)
                || expr_has_target_deref(rhs, self_param, self_field_types, aliases)
        }
        Expr::Assign { lhs, rhs, .. } => {
            expr_has_target_deref(lhs, self_param, self_field_types, aliases)
                || expr_has_target_deref(rhs, self_param, self_field_types, aliases)
        }
        Expr::Conditional {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_has_target_deref(cond, self_param, self_field_types, aliases)
                || expr_has_target_deref(then_expr, self_param, self_field_types, aliases)
                || expr_has_target_deref(else_expr, self_param, self_field_types, aliases)
        }
        Expr::Call { callee, args } => {
            expr_has_target_deref(callee, self_param, self_field_types, aliases)
                || args
                    .iter()
                    .any(|a| expr_has_target_deref(a, self_param, self_field_types, aliases))
        }
        Expr::Index { base, index } => {
            expr_has_target_deref(base, self_param, self_field_types, aliases)
                || expr_has_target_deref(index, self_param, self_field_types, aliases)
        }
        Expr::Member { base, .. } => {
            expr_has_target_deref(base, self_param, self_field_types, aliases)
        }
        _ => false,
    }
}

/// Locals directly aliased from `{self_param}.target`/`.tracer`
/// (`mobj_t* dest; ... dest = actor->target;`, `A_SkullAttack`'s own
/// idiom, common corpus-wide) -- registers each such local's name with
/// the same `"Option<Handle<Thinker>>"` type string `self_field_types`
/// uses for `target`/`tracer` themselves, into the same map shape a
/// trigger function's own `local_var_types` already produces
/// (`FnBodyContext::extra_cross_ref_idents`), so `is_target_tracer_typed`
/// resolves a self-field chain and a locally-aliased chain identically.
/// Deliberately single-level (see `is_target_tracer_typed`'s own doc
/// comment) -- scans for a *direct* `self.target`/`self.tracer`
/// assignment only, passing an empty map as its own `aliases` argument.
fn collect_target_tracer_aliases(
    items: &[BlockItem],
    self_param: &str,
    self_field_types: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    collect_target_tracer_aliases_in(items, self_param, self_field_types, &mut aliases);
    aliases
}

fn collect_target_tracer_aliases_in(
    items: &[BlockItem],
    self_param: &str,
    self_field_types: &HashMap<String, String>,
    aliases: &mut HashMap<String, String>,
) {
    let no_aliases = HashMap::new();
    for item in items {
        match item {
            BlockItem::Decl(d) => {
                for decl in &d.declarators {
                    if let Some(Initializer::Expr(e)) = &decl.initializer
                        && is_target_tracer_typed(e, self_param, self_field_types, &no_aliases)
                        && let Some(name) = declarator_name(&decl.declarator)
                    {
                        aliases.insert(name, "Option<Handle<Thinker>>".to_string());
                    }
                }
            }
            BlockItem::Stmt(s) => {
                collect_target_tracer_aliases_stmt(s, self_param, self_field_types, aliases)
            }
        }
    }
}

fn collect_target_tracer_aliases_stmt(
    s: &Stmt,
    self_param: &str,
    self_field_types: &HashMap<String, String>,
    aliases: &mut HashMap<String, String>,
) {
    let no_aliases = HashMap::new();
    if let Stmt::Expr(Some(Expr::Assign {
        op: AssignOp::Assign,
        lhs,
        rhs,
    })) = s
        && let Expr::Ident(name) = lhs.as_ref()
        && is_target_tracer_typed(rhs, self_param, self_field_types, &no_aliases)
    {
        aliases.insert(name.clone(), "Option<Handle<Thinker>>".to_string());
    }
    match s {
        Stmt::Compound(c) => {
            collect_target_tracer_aliases_in(&c.items, self_param, self_field_types, aliases)
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_target_tracer_aliases_stmt(then_branch, self_param, self_field_types, aliases);
            if let Some(eb) = else_branch {
                collect_target_tracer_aliases_stmt(eb, self_param, self_field_types, aliases);
            }
        }
        Stmt::Switch { body, .. } => {
            collect_target_tracer_aliases_stmt(body, self_param, self_field_types, aliases)
        }
        Stmt::Case { stmt, .. } => {
            collect_target_tracer_aliases_stmt(stmt, self_param, self_field_types, aliases)
        }
        Stmt::Default(stmt) => {
            collect_target_tracer_aliases_stmt(stmt, self_param, self_field_types, aliases)
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
            collect_target_tracer_aliases_stmt(body, self_param, self_field_types, aliases)
        }
        Stmt::For { body, .. } => {
            collect_target_tracer_aliases_stmt(body, self_param, self_field_types, aliases)
        }
        _ => {}
    }
}

/// Locals directly assigned from a fresh `P_SpawnMobj(...)`/
/// `P_SpawnMissile(...)` call (`th = P_SpawnMobj(...);`, `A_Tracer`'s/
/// `A_VileTarget`'s own idiom; `mo = P_SpawnMissile(...);`,
/// `A_FatAttack1`'s own idiom -- both real corpus functions declared
/// `mobj_t*`-returning, confirmed by direct read, so the identical
/// treatment applies to either name) -- registers each such local as
/// `"Handle<Thinker>"`-typed (bare, not `Option`-wrapped: no real corpus
/// call site ever null-checks either result, unlike `target`/`tracer`/
/// `specialdata`), into the same `extra_cross_ref_idents` map shape
/// `collect_target_tracer_aliases` already produces -- lets a later
/// `th->field` read/write resolve through a real `Arena` lookup
/// (`render_expr`'s and `render_expr_stmt`'s own generalized
/// `Handle<Thinker>`-base arms). Neither `P_SpawnMobj` nor
/// `P_SpawnMissile` is itself translated (both stay forward-reference
/// stubs, same as `S_StartSound`/`P_Random` elsewhere in this module) --
/// only each call site's own return value needs a real type here.
fn collect_spawn_mobj_locals(items: &[BlockItem]) -> HashMap<String, String> {
    let mut locals = HashMap::new();
    collect_spawn_mobj_locals_in(items, &mut locals);
    locals
}

fn collect_spawn_mobj_locals_in(items: &[BlockItem], locals: &mut HashMap<String, String>) {
    for item in items {
        if let BlockItem::Stmt(s) = item {
            collect_spawn_mobj_locals_stmt(s, locals);
        }
    }
}

fn collect_spawn_mobj_locals_stmt(s: &Stmt, locals: &mut HashMap<String, String>) {
    // `mo = P_SpawnMissile(actor, actor->target, MT_FATSHOT);`
    // (`A_FatAttack1`/`A_FatAttack2`/`A_FatAttack3`, `A_SkelMissile`) --
    // `P_SpawnMissile`'s own real declared return type (`p_mobj.c`) is
    // `mobj_t*`, the identical shape `P_SpawnMobj` already gets this
    // exact treatment for, confirmed by direct read rather than assumed.
    if let Stmt::Expr(Some(Expr::Assign {
        op: AssignOp::Assign,
        lhs,
        rhs,
    })) = s
        && let Expr::Ident(name) = lhs.as_ref()
        && matches!(rhs.as_ref(), Expr::Call { callee, .. } if matches!(callee.as_ref(), Expr::Ident(n) if n == "P_SpawnMobj" || n == "P_SpawnMissile"))
    {
        locals.insert(name.clone(), "Handle<Thinker>".to_string());
    }
    match s {
        Stmt::Compound(c) => collect_spawn_mobj_locals_in(&c.items, locals),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_spawn_mobj_locals_stmt(then_branch, locals);
            if let Some(eb) = else_branch {
                collect_spawn_mobj_locals_stmt(eb, locals);
            }
        }
        Stmt::Switch { body, .. } => collect_spawn_mobj_locals_stmt(body, locals),
        Stmt::Case { stmt, .. } => collect_spawn_mobj_locals_stmt(stmt, locals),
        Stmt::Default(stmt) => collect_spawn_mobj_locals_stmt(stmt, locals),
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
            collect_spawn_mobj_locals_stmt(body, locals)
        }
        Stmt::For { body, .. } => collect_spawn_mobj_locals_stmt(body, locals),
        _ => {}
    }
}

/// `X->target = actor;` / `X->tracer = actor;` (`A_VileTarget`'s own
/// `fog->target = actor;`) -- storing the function's *own* receiver
/// (`self_param`) as a value into a freshly-spawned `Handle<Thinker>`-
/// typed local's own `target`/`tracer` field. Unlike every other self-
/// struct reference in this module, this needs `self_param`'s own
/// identity as a real `Handle<Thinker>` *value*, not just `&mut self` --
/// a genuinely new need, on the same footing as `body_has_self_removal`
/// first needing a self-removing tick function's own handle, just for a
/// different reason (storing it elsewhere rather than removing it).
/// Drives `render_fn_impl`'s own signature extension (a `handle:
/// Handle<Thinker>` parameter, reusing the same fixed name self-removal
/// already established, without that case's own `arena: &mut
/// Arena<Thinker>` -- nothing here calls `Arena::remove`). Scoped to
/// `target`/`tracer` specifically, the only two `Mobj` fields known
/// `Option<Handle<Thinker>>`-typed so far (mirroring `is_target_tracer_
/// typed`'s own pair).
fn is_self_handle_value_assign(
    s: &Stmt,
    self_param: &str,
    spawn_locals: &HashMap<String, String>,
) -> bool {
    let Stmt::Expr(Some(Expr::Assign {
        op: AssignOp::Assign,
        lhs,
        rhs,
    })) = s
    else {
        return false;
    };
    let Expr::Member { base, field, .. } = lhs.as_ref() else {
        return false;
    };
    (field == "target" || field == "tracer")
        && matches!(base.as_ref(), Expr::Ident(n) if spawn_locals.get(n.as_str()).map(String::as_str) == Some("Handle<Thinker>"))
        && matches!(rhs.as_ref(), Expr::Ident(n) if n == self_param)
}

fn body_has_self_handle_value(
    items: &[BlockItem],
    self_param: &str,
    spawn_locals: &HashMap<String, String>,
) -> bool {
    items.iter().any(|item| match item {
        BlockItem::Stmt(s) => stmt_has_self_handle_value(s, self_param, spawn_locals),
        BlockItem::Decl(_) => false,
    })
}

fn stmt_has_self_handle_value(
    s: &Stmt,
    self_param: &str,
    spawn_locals: &HashMap<String, String>,
) -> bool {
    if is_self_handle_value_assign(s, self_param, spawn_locals) {
        return true;
    }
    match s {
        Stmt::Compound(c) => body_has_self_handle_value(&c.items, self_param, spawn_locals),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            stmt_has_self_handle_value(then_branch, self_param, spawn_locals)
                || else_branch
                    .as_ref()
                    .is_some_and(|eb| stmt_has_self_handle_value(eb, self_param, spawn_locals))
        }
        Stmt::Switch { body, .. } => stmt_has_self_handle_value(body, self_param, spawn_locals),
        Stmt::Case { stmt, .. } => stmt_has_self_handle_value(stmt, self_param, spawn_locals),
        Stmt::Default(stmt) => stmt_has_self_handle_value(stmt, self_param, spawn_locals),
        Stmt::While { body, .. } => stmt_has_self_handle_value(body, self_param, spawn_locals),
        _ => false,
    }
}

fn body_has_target_deref(
    items: &[BlockItem],
    self_param: &str,
    self_field_types: &HashMap<String, String>,
    aliases: &HashMap<String, String>,
) -> bool {
    items.iter().any(|item| match item {
        BlockItem::Decl(d) => d.declarators.iter().any(|decl| {
            matches!(&decl.initializer, Some(Initializer::Expr(e)) if expr_has_target_deref(e, self_param, self_field_types, aliases))
        }),
        BlockItem::Stmt(s) => stmt_has_target_deref(s, self_param, self_field_types, aliases),
    })
}

fn stmt_has_target_deref(
    s: &Stmt,
    self_param: &str,
    self_field_types: &HashMap<String, String>,
    aliases: &HashMap<String, String>,
) -> bool {
    match s {
        Stmt::Expr(Some(e)) => expr_has_target_deref(e, self_param, self_field_types, aliases),
        Stmt::Expr(None) => false,
        Stmt::Compound(c) => body_has_target_deref(&c.items, self_param, self_field_types, aliases),
        Stmt::If {
            cond,
            then_branch,
            else_branch,
        } => {
            expr_has_target_deref(cond, self_param, self_field_types, aliases)
                || stmt_has_target_deref(then_branch, self_param, self_field_types, aliases)
                || else_branch
                    .as_ref()
                    .is_some_and(|eb| stmt_has_target_deref(eb, self_param, self_field_types, aliases))
        }
        Stmt::Switch { cond, body } => {
            expr_has_target_deref(cond, self_param, self_field_types, aliases)
                || stmt_has_target_deref(body, self_param, self_field_types, aliases)
        }
        Stmt::Case { expr, stmt } => {
            expr_has_target_deref(expr, self_param, self_field_types, aliases)
                || stmt_has_target_deref(stmt, self_param, self_field_types, aliases)
        }
        Stmt::Default(stmt) => stmt_has_target_deref(stmt, self_param, self_field_types, aliases),
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            expr_has_target_deref(cond, self_param, self_field_types, aliases)
                || stmt_has_target_deref(body, self_param, self_field_types, aliases)
        }
        Stmt::For {
            init,
            cond,
            step,
            body,
        } => {
            init.as_ref().is_some_and(|i| match i {
                ForInit::Decl(d) => d.declarators.iter().any(|decl| {
                    matches!(&decl.initializer, Some(Initializer::Expr(e)) if expr_has_target_deref(e, self_param, self_field_types, aliases))
                }),
                ForInit::Expr(e) => expr_has_target_deref(e, self_param, self_field_types, aliases),
            }) || cond
                .as_ref()
                .is_some_and(|e| expr_has_target_deref(e, self_param, self_field_types, aliases))
                || step
                    .as_ref()
                    .is_some_and(|e| expr_has_target_deref(e, self_param, self_field_types, aliases))
                || stmt_has_target_deref(body, self_param, self_field_types, aliases)
        }
        Stmt::Return(Some(e)) => expr_has_target_deref(e, self_param, self_field_types, aliases),
        Stmt::Labeled { stmt, .. } => stmt_has_target_deref(stmt, self_param, self_field_types, aliases),
        _ => false,
    }
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

/// Collects, in first-seen order (deduplicated), every field name
/// assigned to `ctor_var->field` anywhere inside `s` -- used to recognize
/// a `switch` that's part of *constructing* the value, deciding some of
/// its fields per case (`EV_DoCeiling`'s own `switch(type) { case X:
/// ceiling->topheight = ...; ... }`, unlike every earlier constructor,
/// which only ever set fields via flat top-level statements). Once the
/// fields a switch touches are known, they can be pre-declared as `let
/// mut field;` right before it, and the switch itself then renders
/// completely unchanged -- `ctx`'s existing `ctor_var` resolution already
/// turns every `ceiling->field` reference inside its arms into a plain,
/// already-`let`-bound local either way, so no new statement-shape
/// handling is needed for the switch's own body.
fn collect_ctor_fields_in(s: &Stmt, ctor_var: &str, out: &mut Vec<String>) {
    if let Some(field) = ctor_field_assign_target(s, ctor_var) {
        if !out.iter().any(|f| f == field) {
            out.push(field.to_string());
        }
        return;
    }
    match s {
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_ctor_fields_in(then_branch, ctor_var, out);
            if let Some(eb) = else_branch {
                collect_ctor_fields_in(eb, ctor_var, out);
            }
        }
        Stmt::Switch { body, .. } => collect_ctor_fields_in(body, ctor_var, out),
        Stmt::Case { stmt, .. } => collect_ctor_fields_in(stmt, ctor_var, out),
        Stmt::Default(stmt) => collect_ctor_fields_in(stmt, ctor_var, out),
        Stmt::Compound(c) => {
            for item in &c.items {
                if let BlockItem::Stmt(s) = item {
                    collect_ctor_fields_in(s, ctor_var, out);
                }
            }
        }
        _ => {}
    }
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
        // `case`/`default` labels wrap only their own next statement (see
        // `render_switch`'s own module docs on this C parsing quirk) --
        // never recursed into here before this fix, since no constructor
        // attempted so far had a `switch` deciding any of its fields
        // (`EV_DoCeiling`'s own inline constructor is the first), so this
        // was a real, previously-unexercised gap in `mut`-detection.
        Stmt::Case { stmt, .. } => count_ctor_field_assigns_stmt(stmt, ctor_var, counts),
        Stmt::Default(stmt) => count_ctor_field_assigns_stmt(stmt, ctor_var, counts),
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
        rendered.push(format!("{}: {rust_type}", rust_field_name(&name)?));
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

    let base_ctx = FnBodyContext {
        self_param: "",
        self_field_types: &HashMap::new(),
        extra_cross_ref_idents: param_cross_ref_types,
        ctor_var: "",
        ctor_var_handle_name: "",
        ctor_field_types: &HashMap::new(),
        embedded_ctor: None,
        mutating_handle: None,
        same_handle_write: None,
        plain_int_locals: &HashSet::new(),
    };
    let no_field_defaults = HashMap::new();
    let spec = CtorSpec {
        ctor_var: &ctor_var,
        ctor_rust_type,
        ctor_field_types,
        field_defaults: &no_field_defaults,
    };
    let lines = render_ctor_body(&f.body.items, &spec, &base_ctx, 1, fn_name)?;
    Ok(format!(
        "pub fn {fn_name}({}, world: &mut World, thinkers: &mut Arena<Thinker>) {{\n{}\n}}",
        rendered_params.join(", "),
        lines.join("\n")
    ))
}

/// Renders a `Z_Malloc`+`P_AddThinker`+field-fill-in constructor
/// sequence: `render_spawn_fn`'s own core logic, generalized so it can
/// also run *embedded* inside a larger function (`render_trigger_fn`'s
/// `EV_DoCeiling`-style triggers that build their thinker inline,
/// mid-loop, rather than delegating to a separate `P_Spawn*` function --
/// see `render_compound_items`, which detects this and hands it exactly
/// the slice starting at the `Z_Malloc` call). `items` is consumed in
/// full, matching `render_spawn_fn`'s own "process my whole scope"
/// behavior -- there's no "natural end" heuristic, since in every real
/// example the constructor's own trailing statements (e.g. `EV_DoCeiling`'s
/// `P_AddActiveCeiling(ceiling);`) simply run to the end of whatever
/// block contains them. `base_ctx` supplies everything about the
/// *enclosing* function (its own parameters/locals) that field
/// expressions may need to resolve (`EV_DoCeiling`'s `ceiling->sector =
/// sec;` reads the trigger loop's own `sec` local) -- this function
/// layers its own `ctor_var` on top via `FnBodyContext: Copy`'s struct-
/// update syntax, rather than building a context from scratch.
///
/// `field_defaults` supplies an explicit rendered-expression string for
/// a field this constructor genuinely never sets anywhere (`EV_DoCeiling`
/// never touches `Ceiling.olddirection` at all -- only
/// `P_ActivateInStasisCeiling`/`EV_CeilingCrushStop`, two *other*
/// functions entirely, ever read or write it, and always write-before-
/// read, so the freshly-constructed garbage value is never actually
/// observed). This is never guessed at automatically -- only used when a
/// caller supplies an entry, after doing the same kind of corpus tracing
/// that got `P_SpawnDoorCloseIn30` correctly *rejected* for the opposite
/// reason (there, the unset field *was* reachable with no defined value).
/// A field with no entry here still fails the completeness check exactly
/// as before.
fn render_ctor_body(
    items: &[BlockItem],
    spec: &CtorSpec,
    base_ctx: &FnBodyContext,
    depth: usize,
    fn_name: &str,
) -> Result<Vec<String>, String> {
    let CtorSpec {
        ctor_var,
        ctor_rust_type,
        ctor_field_types,
        field_defaults,
    } = *spec;
    let ctx = FnBodyContext {
        ctor_var,
        ctor_var_handle_name: "",
        ctor_field_types,
        embedded_ctor: None,
        ..*base_ctx
    };

    let mut reassign_counts: HashMap<String, usize> = HashMap::new();
    count_ctor_field_assigns(items, ctor_var, &mut reassign_counts);

    // A back-reference (`sec->specialdata = door;`) needs the constructed
    // value's real `Handle` before it can be rendered at all, so its
    // presence anywhere in the body switches this whole scope to a
    // two-phase render: every constructor field first (regardless of
    // where its assignment fell in the original source, same reordering
    // argument as always), then the `Arena::insert` call bound to `let
    // handle = ...;`, then every "other" statement (queued into
    // `pending_other` below) rendered afterward with a bare `ctor_var`
    // resolving to `handle`. Without a back-reference, "other" statements
    // render immediately, interleaved in original source order exactly
    // as before -- this mode is unchanged from every earlier spawn
    // function this module already handles.
    let has_backreference = body_has_backreference(items, ctor_var);

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
    for item in items {
        let BlockItem::Stmt(s) = item else { continue };
        if is_malloc_assign(s, ctor_var)
            || is_add_thinker_call(s)
            || is_function_pointer_assign(s, ctor_var)
        {
            continue;
        }
        if let Some((field, then_rhs, else_rhs)) = if_else_ctor_field_assign(s, ctor_var) {
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
                "{}let {field} = if {cond_text} {{ {then_text} }} else {{ {else_text} }};",
                indent(depth)
            ));
            ctor_field_names.push(field);
            continue;
        }
        if let Some((field, rhs)) = ctor_field_assign(s, ctor_var) {
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
                lines.push(format!(
                    "{}let {mutability}{field} = {rhs_text};",
                    indent(depth)
                ));
            }
            ctor_field_names.push(field);
            continue;
        }
        if ctor_field_assign_target(s, ctor_var).is_some() {
            // A *compound* assignment refining an already-`let`-bound
            // field right after its own initial value
            // (`P_SpawnDoorRaiseIn5Mins`'s `door->topheight -=
            // 4*FRACUNIT;` -- `ctor_field_assign` above only matches
            // plain `=`, so reaching here means this is exactly that
            // case). Still part of *constructing* the value, so it's
            // rendered inline via the ordinary path right here, never
            // deferred to `pending_other` even when `has_backreference`
            // -- deferring it would insert the pre-refinement value.
            lines.extend(render_stmt(s, &ctx, depth)?);
            continue;
        }
        if let Stmt::Switch { .. } = s {
            let mut touched_fields = Vec::new();
            collect_ctor_fields_in(s, ctor_var, &mut touched_fields);
            if !touched_fields.is_empty() {
                // A `switch` deciding some of the constructed value's own
                // fields per case (`EV_DoCeiling`'s `switch(type) { case
                // X: ceiling->topheight = ...; ... }`), rather than a
                // flat statement -- pre-declare whichever of those fields
                // aren't already `let`-bound (always `mut`, since a
                // switch-decided field's value depends on which arm ran,
                // not a single unconditional computation), then render
                // the switch completely unchanged: `ctx`'s `ctor_var`
                // resolution already turns every `ceiling->field`
                // reference inside its arms into a plain reassignment of
                // that local, exactly like a compound-assignment
                // refinement already does outside a switch.
                //
                // **Every** touched field needs a `field_defaults` entry
                // here, seeding its own `let`, even one every *real* `case`
                // sets (confirmed empirically, not assumed, by actually
                // compiling `EV_DoCeiling`'s output with `rustc`): `type`
                // maps to a plain `i32`, not a real closed Rust `enum`, so
                // `render_switch`'s own synthetic `_ => {}` catch-all (for
                // a discriminant value outside the ones this `switch`
                // actually names -- unreachable in real Doom, since `type`
                // is always one of the known `ceiling_e` values, but Rust
                // can't see that from an `i32`) leaves every switch-only
                // field looking possibly-uninitialized to Rust's real
                // definite-assignment check, not just the ones with a
                // genuine per-arm gap like `bottomheight`.
                for field in &touched_fields {
                    let field = rust_field_name(field)?;
                    if ctor_field_names.contains(&field) {
                        continue;
                    }
                    let default_expr = field_defaults.get(&field).ok_or_else(|| {
                        format!(
                            "{fn_name}: `{field}` is only ever set inside a `switch`, and needs an explicit `field_defaults` entry to satisfy Rust's definite-assignment check (its own synthetic `_` catch-all arm never sets it)"
                        )
                    })?;
                    lines.push(format!(
                        "{}let mut {field} = {default_expr};",
                        indent(depth)
                    ));
                    ctor_field_names.push(field);
                }
                lines.extend(render_stmt(s, &ctx, depth)?);
                continue;
            }
        }
        // Once `has_backreference` is true, deferring is safe for *any*
        // statement, whatever shape -- `ctor_var_handle_name` resolves
        // `ctor_var` correctly wherever it appears once rendered with
        // `ctx_after`, not just in the specific back-reference-assignment
        // shape (`EV_DoCeiling`'s own `P_AddActiveCeiling(ceiling);`
        // passes `ctor_var` as a bare call argument, a different shape
        // again). Without a back-reference, there's no `handle` binding
        // to resolve to at all, so a bare `ctor_var` reference there is
        // still rejected loudly rather than silently mistranslated.
        if !has_backreference && stmt_uses_bare_ctor_ident(s, ctor_var) {
            return Err(format!(
                "{fn_name}: a statement referencing the constructed value in an unsupported way: {s:?}"
            ));
        }
        if has_backreference {
            pending_other.push(s);
        } else {
            lines.extend(render_stmt(s, &ctx, depth)?);
        }
    }

    // Sorted, not raw `HashMap::keys()` order: `EV_DoCeiling` only ever
    // had one field fall through to this default-filling loop
    // (`olddirection`), so `HashMap`'s per-process-randomized iteration
    // order was never actually observable in its output -- `EV_DoPlat`'s
    // `count`/`oldstatus` (two fields, neither touched by its own
    // `switch`) exposed this for real: an unsorted iteration would make
    // the generated code's own field order (and therefore this whole
    // renderer's output) nondeterministic across runs, not just
    // differently-ordered-but-equally-valid.
    let mut still_missing: Vec<&str> = Vec::new();
    let mut remaining_fields: Vec<&String> = ctor_field_types.keys().collect();
    remaining_fields.sort();
    for field in remaining_fields {
        if ctor_field_names.contains(field) {
            continue;
        }
        match field_defaults.get(field) {
            Some(default_expr) => {
                lines.push(format!("{}let {field} = {default_expr};", indent(depth)));
                ctor_field_names.push(field.clone());
            }
            None => still_missing.push(field.as_str()),
        }
    }
    if !still_missing.is_empty() {
        return Err(format!(
            "{fn_name}: never assigns {ctor_rust_type}'s field(s) {}, so the constructed literal would be incomplete",
            still_missing.join(", ")
        ));
    }

    let insert_expr = format!(
        "Thinker::{ctor_rust_type}({ctor_rust_type} {{ {} }})",
        ctor_field_names.join(", ")
    );
    if has_backreference {
        lines.push(format!(
            "{}let handle = thinkers.insert({insert_expr});",
            indent(depth)
        ));
        let ctx_after = FnBodyContext {
            ctor_var_handle_name: "handle",
            ..ctx
        };
        for s in pending_other {
            lines.extend(render_stmt(s, &ctx_after, depth)?);
        }
    } else {
        lines.push(format!("{}thinkers.insert({insert_expr});", indent(depth)));
    }
    Ok(lines)
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
/// `embedded_ctor` is `Some((ctor_var, ctor_rust_type, ctor_field_types))`
/// for a trigger that builds its own thinker *inline*, mid-loop
/// (`EV_DoCeiling`'s own `while` body does `Z_Malloc`/`P_AddThinker`/
/// field-fill-in directly, unlike `EV_StartLightStrobing`'s call out to a
/// separate `P_Spawn*` function) -- `render_compound_items` watches for
/// this via `FnBodyContext::embedded_ctor` and switches into
/// `render_ctor_body` partway through whichever block actually contains
/// the `Z_Malloc` call. `ctor_var` is supplied here rather than self-
/// discovered the way `render_spawn_fn` does, since the local is
/// typically declared once at the top of the function, not right before
/// the loop that actually constructs it. The last element is a field-
/// defaults map (see `render_ctor_body`'s own doc comment) for any field
/// the constructor genuinely never sets anywhere, once a caller has
/// traced that it's safe -- pass an empty map when every field really is
/// assigned somewhere.
///
/// `return_type` renders as the function's own `-> T` when `Some`
/// (`EV_DoCeiling`/`EV_DoFloor`-style triggers return `int`, whether a
/// sector was actually activated) -- `Stmt::Return` itself already
/// renders generically via `render_stmt`, so this is only about the
/// signature.
pub fn render_trigger_fn(
    corpus_dir: &Path,
    file: &str,
    fn_name: &str,
    param_types: &HashMap<String, String>,
    local_var_types: &HashMap<String, String>,
    embedded_ctor: Option<CtorSpec>,
    return_type: Option<&str>,
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
        ctor_field_types: &HashMap::new(),
        embedded_ctor,
        mutating_handle: None,
        same_handle_write: None,
        plain_int_locals: &HashSet::new(),
    };

    let body_lines = render_compound_items(&f.body.items, &ctx, 1)?;
    let return_arrow = return_type.map(|t| format!(" -> {t}")).unwrap_or_default();
    Ok(format!(
        "pub fn {fn_name}({}, world: &mut World, thinkers: &mut Arena<Thinker>){return_arrow} {{\n{}\n}}",
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

    /// The first function needing real `Arena` read access from inside a
    /// `Mobj`-shaped action function: `actor->target->x`/`.y`/`.flags`
    /// dereference *through* the `Option<Handle<Thinker>>`-typed `target`
    /// field, not just check its truthiness -- the architectural gap
    /// `docs/03_TRANSPILER.md` flagged as the next step after the
    /// target-attack batch. `render_fn` gains the extra `thinkers: &Arena
    /// <Thinker>` read-only parameter (`body_has_target_deref`) only
    /// because this function's body actually needs it -- the corpus
    /// fact that only `mobj_t` ever has `target`/`tracer` makes `_ =>
    /// unreachable!()` genuinely safe on the resulting lookup, not a
    /// defensive catch-all. Also exercises two smaller, real gaps this
    /// function's own body surfaced and got fixed alongside it: a bare
    /// non-comparison `Binary` (`actor->target->flags & MF_SHADOW`) used
    /// for C truthiness needed its own `render_bool_expr` arm (`!= 0`,
    /// the same idiom the bare-`Member` truthiness arm already had); and
    /// `actor->angle += (...)<<21;` needed an explicit `as u32` on the
    /// compound-assign RHS -- confirmed a real `rustc` rejection (`cannot
    /// add-assign i32 to u32`), the mirror image of the already-
    /// documented `angle_t`-into-plain-`int` bug, just in the opposite
    /// direction. Verified compiling for real (`rustc --edition 2021
    /// --crate-type lib`) against hand-written stand-in `World`/
    /// `Thinker`/`Arena`/`Handle`/`Mobj` shapes.
    #[test]
    fn test_a_face_target_renders_exactly() {
        let field_types = field_types(&[
            ("target", "Option<Handle<Thinker>>"),
            ("flags", "i32"),
            ("angle", "u32"),
        ]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_FaceTarget",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_FaceTarget(actor: &mut Mobj, world: &mut World, thinkers: &Arena<Thinker>) {\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             actor.flags &= !MF_AMBUSH;\n    \
             actor.angle = R_PointToAngle2(actor.x, actor.y, match thinkers.get(actor.target.unwrap()) { Some(Thinker::Mobj(m)) => m.x, _ => unreachable!() }, match thinkers.get(actor.target.unwrap()) { Some(Thinker::Mobj(m)) => m.y, _ => unreachable!() });\n    \
             if (match thinkers.get(actor.target.unwrap()) { Some(Thinker::Mobj(m)) => m.flags, _ => unreachable!() } & MF_SHADOW) != 0 {\n        \
             actor.angle += (P_Random() - P_Random() << 21) as u32;\n    \
             }\n\
             }"
        );
    }

    /// `A_CPosRefire`/`A_SpidRefire` -- identical shape, differing only
    /// in the `P_Random()` threshold, both reading `actor->target->
    /// health` inside a `||` chain alongside a bare `!actor->target` and
    /// a bare `!P_CheckSight(..)`. Surfaces a real gap distinct from
    /// `A_FaceTarget`'s own: `render_binary_operand`'s `&&`/`||` operands
    /// render through the *generic* `render_expr` path, not
    /// `render_bool_expr` -- so `!actor->target`'s own `Option`-aware
    /// `.is_none()` treatment (already correct at `render_bool_expr`'s
    /// top-level entry point) needed its own twin arm added directly to
    /// `render_expr`'s `Unary::Not` handling, or it would try to apply
    /// Rust's `!` operator to an `Option`, a real type error, not just a
    /// wrong-but-compiling translation. `!P_CheckSight(..)` needed no new
    /// code at all: `P_CheckSight`'s real corpus declaration
    /// (`p_local.h`) returns `boolean` (already real Rust `bool`), so
    /// plain `!` on its call result is already correct, unlike a plain-
    /// `int`-flag callee. Verified compiling for real.
    #[test]
    fn test_a_cpos_refire_renders_exactly() {
        let field_types = field_types(&[
            ("target", "Option<Handle<Thinker>>"),
            ("info", "&'static MobjInfo"),
            ("health", "i32"),
        ]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_CPosRefire",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_CPosRefire(actor: &mut Mobj, world: &mut World, thinkers: &Arena<Thinker>) {\n    \
             A_FaceTarget(actor);\n    \
             if P_Random() < 40 {\n        \
             return;\n    \
             }\n    \
             if actor.target.is_none() || match thinkers.get(actor.target.unwrap()) { Some(Thinker::Mobj(m)) => m.health, _ => unreachable!() } <= 0 || !P_CheckSight(actor, actor.target) {\n        \
             P_SetMobjState(actor, actor.info.seestate);\n    \
             }\n\
             }"
        );
    }

    #[test]
    fn test_a_spid_refire_renders_exactly() {
        let field_types = field_types(&[
            ("target", "Option<Handle<Thinker>>"),
            ("info", "&'static MobjInfo"),
            ("health", "i32"),
        ]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_SpidRefire",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_SpidRefire(actor: &mut Mobj, world: &mut World, thinkers: &Arena<Thinker>) {\n    \
             A_FaceTarget(actor);\n    \
             if P_Random() < 10 {\n        \
             return;\n    \
             }\n    \
             if actor.target.is_none() || match thinkers.get(actor.target.unwrap()) { Some(Thinker::Mobj(m)) => m.health, _ => unreachable!() } <= 0 || !P_CheckSight(actor, actor.target) {\n        \
             P_SetMobjState(actor, actor.info.seestate);\n    \
             }\n\
             }"
        );
    }

    /// `P_CheckMeleeRange` -- the first `boolean`-returning function this
    /// renderer produces (`render_bool_fn`, a thin `-> {return_type}`
    /// wrapper over the same `render_fn_impl` every `void A_*` action
    /// function already shares), and the first real corpus use of the
    /// `dest = actor->target;` local-alias chain-through combined with a
    /// *further* chain off the dereferenced result (`pl->info->radius`:
    /// `pl->info` resolves through the alias to a real `&'static
    /// MobjInfo`, then `.radius` chains off *that* through the ordinary
    /// generic `Expr::Member` fallback -- no new code needed, since the
    /// match-expression the alias arm produces is just an ordinary Rust
    /// value any further `.field` can chain off). Also surfaces a real
    /// bug, independent of anything built for `A_CPosRefire`'s own `&&`/
    /// `||`-chain fix: `render_bool_expr`'s own top-level `Unary::Not`
    /// handling (a *single* condition, not part of a logical chain) had
    /// no `bool`-returning-callee awareness at all, so `if (!
    /// P_CheckSight(..))` rendered as `P_CheckSight(..) == 0` -- syntactically
    /// valid but semantically backwards-compiling nonsense once
    /// `P_CheckSight` returns a real Rust `bool` (`== 0` doesn't even
    /// type-check against `bool`, so this would have been caught at the
    /// `rustc` smoke-compile step, but is fixed here before that point
    /// via the new `is_bool_returning_call` helper, shared with the
    /// already-existing bare (non-negated) `P_CheckMeleeRange` arm).
    /// Verified compiling for real.
    #[test]
    fn test_p_check_melee_range_renders_exactly() {
        let field_types = field_types(&[
            ("target", "Option<Handle<Thinker>>"),
            ("info", "&'static MobjInfo"),
        ]);
        let rendered = render_bool_fn(
            &corpus_dir(),
            "p_enemy.c",
            "P_CheckMeleeRange",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn P_CheckMeleeRange(actor: &mut Mobj, world: &mut World, thinkers: &Arena<Thinker>) -> bool {\n    \
             let mut pl;\n    \
             let mut dist;\n    \
             if actor.target.is_none() {\n        \
             return false;\n    \
             }\n    \
             pl = actor.target;\n    \
             dist = P_AproxDistance(match thinkers.get(pl.unwrap()) { Some(Thinker::Mobj(m)) => m.x, _ => unreachable!() } - actor.x, match thinkers.get(pl.unwrap()) { Some(Thinker::Mobj(m)) => m.y, _ => unreachable!() } - actor.y);\n    \
             if dist >= MELEERANGE - 20 * FRACUNIT + match thinkers.get(pl.unwrap()) { Some(Thinker::Mobj(m)) => m.info, _ => unreachable!() }.radius {\n        \
             return false;\n    \
             }\n    \
             if !P_CheckSight(actor, actor.target) {\n        \
             return false;\n    \
             }\n    \
             return true;\n\
             }"
        );
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
    /// return). Confirms `render_spawn_fn` catches this itself, loudly,
    /// rather than emitting an incomplete literal that would only fail
    /// later, confusingly, when the generated output is compiled.
    ///
    /// **This rejection is permanent, not a placeholder**: traced against
    /// the now-fully-translated `T_VerticalDoor`, a door spawned this way
    /// (`direction = 0`, `type = normal`) *can* reach code that reads
    /// `topheight` -- if crushed while closing, `type == normal` isn't
    /// `blazeClose`/`close` ("DO NOT GO BACK UP!", the corpus's own
    /// comment on that exclusion), so it reverses to `direction = 1` and
    /// the next tick reads `door->topheight` as `T_MovePlane`'s
    /// destination. `Z_Malloc` (`z_zone.c`) never zeroes memory (read in
    /// full, not assumed) -- so this is genuine reachable undefined
    /// behavior in the *original* C, not a translation gap: there is no
    /// well-defined C value to be faithful to, so no Rust default would
    /// be honest either. Refusing to guess is the correct answer here,
    /// not an incomplete one.
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
            None,
            None,
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

    /// `T_VerticalDoor` (`p_doors.c`) is far larger than anything
    /// translated so far -- nested `switch`es with cases sharing one
    /// body, self-removal via `P_RemoveThinker` (which would need
    /// `Thinker::tick`'s own signature revised to carry a `Handle` and
    /// `&mut Arena`, a consequential change not attempted yet), `NULL`
    /// assigned to `specialdata` (needs `None`, not yet handled) -- so
    /// this doesn't attempt the whole function. It does use
    /// `if (!--door->topcountdown)` twice (its WAITING/INITIAL-WAIT
    /// countdown states): the same countdown-to-zero idiom as
    /// `T_FireFlicker`'s bare `--flick->count`, just negated (testing
    /// for zero, not nonzero). This extracts that real condition
    /// sub-expression directly from the parsed corpus AST -- real,
    /// corpus-verified C, not a fabricated snippet -- to confirm
    /// `render_condition`'s new negated-`PreIncDec` case in isolation.
    #[test]
    fn test_negated_pre_dec_condition_against_real_t_vertical_door() {
        let path = corpus_dir().join("p_doors.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_doors.c should parse");
        let f = find_function_def(&unit.items, "T_VerticalDoor").expect("T_VerticalDoor not found");
        let Some(BlockItem::Stmt(Stmt::Switch { body, .. })) = f.body.items.get(1) else {
            panic!("expected T_VerticalDoor's second body item to be its outer switch");
        };
        let Stmt::Compound(c) = body.as_ref() else {
            panic!("expected the switch body to be a compound statement");
        };
        let Some(BlockItem::Stmt(Stmt::Case { stmt, .. })) = c.items.first() else {
            panic!("expected the switch body's first item to be a case label");
        };
        let Stmt::If { cond, .. } = stmt.as_ref() else {
            panic!("expected `case 0:`'s statement to be the `if (!--door->topcountdown)`");
        };

        let self_field_types = field_types(&[("topcountdown", "i32")]);
        let no_extra_cross_refs = HashMap::new();
        let ctx = FnBodyContext {
            self_param: "door",
            self_field_types: &self_field_types,
            extra_cross_ref_idents: &no_extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let (hoisted, cond_text) = render_condition(cond, &ctx, 2).expect("should render cleanly");
        assert_eq!(hoisted, vec!["        door.topcountdown -= 1;".to_string()]);
        assert_eq!(cond_text, "door.topcountdown == 0");
    }

    /// Recursively searches `s` (and anything nested inside it -- `if`/
    /// `else` branches, `switch` bodies, `case`/`default` labels,
    /// compound blocks) for the first statement matching `pred`, so a
    /// test can pull one real statement out of a large function's AST
    /// (e.g. `T_VerticalDoor`) without hand-indexing through its exact
    /// nesting shape.
    fn find_stmt<'a>(s: &'a Stmt, pred: &dyn Fn(&Stmt) -> bool) -> Option<&'a Stmt> {
        if pred(s) {
            return Some(s);
        }
        match s {
            Stmt::Compound(c) => find_stmt_in_body(&c.items, pred),
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => find_stmt(then_branch, pred)
                .or_else(|| else_branch.as_ref().and_then(|eb| find_stmt(eb, pred))),
            Stmt::Switch { body, .. } => find_stmt(body, pred),
            Stmt::Case { stmt, .. } => find_stmt(stmt, pred),
            Stmt::Default(stmt) => find_stmt(stmt, pred),
            Stmt::While { body, .. } => find_stmt(body, pred),
            Stmt::For { body, .. } => find_stmt(body, pred),
            _ => None,
        }
    }

    fn find_stmt_in_body<'a>(
        items: &'a [BlockItem],
        pred: &dyn Fn(&Stmt) -> bool,
    ) -> Option<&'a Stmt> {
        items.iter().find_map(|item| match item {
            BlockItem::Stmt(s) => find_stmt(s, pred),
            BlockItem::Decl(_) => None,
        })
    }

    /// `T_VerticalDoor`'s `door->sector->specialdata = NULL;` (run once
    /// the door finishes closing, clearing the sector's back-reference to
    /// it) needs `None`, not a bare `NULL` passthrough -- `specialdata`
    /// maps to `Option<Handle<Thinker>>`. Same real-AST-extraction
    /// approach as the negated-`--x` test above: pulls the one real
    /// statement out of `T_VerticalDoor` without attempting the whole
    /// function.
    #[test]
    fn test_null_to_none_against_real_t_vertical_door() {
        let path = corpus_dir().join("p_doors.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_doors.c should parse");
        let f = find_function_def(&unit.items, "T_VerticalDoor").expect("T_VerticalDoor not found");
        let is_specialdata_null_assign = |s: &Stmt| {
            matches!(s, Stmt::Expr(Some(Expr::Assign { lhs, rhs, .. }))
                if matches!(lhs.as_ref(), Expr::Member { field, .. } if field == "specialdata")
                && matches!(rhs.as_ref(), Expr::Ident(n) if n == "NULL"))
        };
        let stmt = f
            .body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => find_stmt(s, &is_specialdata_null_assign),
                BlockItem::Decl(_) => None,
            })
            .expect("expected a `specialdata = NULL;` statement somewhere in T_VerticalDoor");
        let Stmt::Expr(Some(e)) = stmt else {
            unreachable!("guarded by is_specialdata_null_assign")
        };

        let self_field_types = field_types(&[("sector", "SectorId")]);
        let no_extra_cross_refs = HashMap::new();
        let ctx = FnBodyContext {
            self_param: "door",
            self_field_types: &self_field_types,
            extra_cross_ref_idents: &no_extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let rendered = render_expr_stmt(e, &ctx).expect("should render cleanly");
        assert_eq!(rendered, "world[door.sector].specialdata = None");
    }

    /// `T_VerticalDoor`'s `P_RemoveThinker(&door->thinker);` -- a tick
    /// function removing itself, run right alongside the `specialdata =
    /// NULL;` reset tested above. Confirms `is_self_removal_call`
    /// recognizes the real shape and `render_expr_stmt` renders it as
    /// `arena.remove(handle)`, using the fixed `handle`/`arena` names
    /// this renderer reserves for a tick function's own removal context
    /// (not yet threaded through `render_fn`'s generated signature --
    /// see docs/03_TRANSPILER.md).
    #[test]
    fn test_self_removal_against_real_t_vertical_door() {
        let path = corpus_dir().join("p_doors.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_doors.c should parse");
        let f = find_function_def(&unit.items, "T_VerticalDoor").expect("T_VerticalDoor not found");
        let is_self_removal_stmt =
            |s: &Stmt| matches!(s, Stmt::Expr(Some(e)) if is_self_removal_call(e, "door"));
        let stmt = f
            .body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => find_stmt(s, &is_self_removal_stmt),
                BlockItem::Decl(_) => None,
            })
            .expect("expected a `P_RemoveThinker(&door->thinker);` statement in T_VerticalDoor");
        let Stmt::Expr(Some(e)) = stmt else {
            unreachable!("guarded by is_self_removal_stmt")
        };

        let no_self_fields = HashMap::new();
        let no_extra_cross_refs = HashMap::new();
        let ctx = FnBodyContext {
            self_param: "door",
            self_field_types: &no_self_fields,
            extra_cross_ref_idents: &no_extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let rendered = render_expr_stmt(e, &ctx).expect("should render cleanly");
        assert_eq!(rendered, "arena.remove(handle)");
    }

    /// `A_SkullAttack`'s `mobj_t* dest; ... dest = actor->target; ...
    /// dest->x - actor->x ...` -- the local-alias generalization of the
    /// `actor->target->field` chain-through arm (`is_target_tracer_typed`,
    /// `collect_target_tracer_aliases`), verified against the real parsed
    /// AST rather than a fabricated snippet. `A_SkullAttack` as a *whole*
    /// function isn't attempted here -- it has its own separate, unrelated
    /// gap (`dist`, declared plain `int`, is assigned `P_AproxDistance`'s
    /// fixed-point-`FixedT` result and later compared/reassigned as if it
    /// were still `int`; a different problem from anything this session's
    /// target/tracer work touches, not investigated here), matching this
    /// codebase's own "isolate the one clean new piece, leave the rest
    /// open" precedent (`T_VerticalDoor`'s own countdown/`NULL`/self-
    /// removal pieces before it rendered whole). `collect_target_tracer_
    /// aliases` runs against `A_SkullAttack`'s real, complete body (not a
    /// synthetic fragment) to confirm it actually discovers `dest` from
    /// real corpus source, then just the one real `dest->x - actor->x`
    /// sub-expression (pulled out of the real `P_AproxDistance(...)` call
    /// argument list) is rendered in isolation.
    #[test]
    fn test_target_tracer_local_alias_against_real_a_skull_attack() {
        let path = corpus_dir().join("p_enemy.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_enemy.c should parse");
        let f = find_function_def(&unit.items, "A_SkullAttack").expect("A_SkullAttack not found");
        let self_field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let aliases = collect_target_tracer_aliases(&f.body.items, "actor", &self_field_types);
        assert_eq!(
            aliases.get("dest").map(String::as_str),
            Some("Option<Handle<Thinker>>")
        );

        let is_aprox_distance_call = |s: &Stmt| {
            matches!(s, Stmt::Expr(Some(Expr::Assign { rhs, .. }))
                if matches!(rhs.as_ref(), Expr::Call { callee, .. } if matches!(callee.as_ref(), Expr::Ident(n) if n == "P_AproxDistance")))
        };
        let stmt = f
            .body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => find_stmt(s, &is_aprox_distance_call),
                BlockItem::Decl(_) => None,
            })
            .expect("expected a `dist = P_AproxDistance(..)` statement in A_SkullAttack");
        let Stmt::Expr(Some(Expr::Assign { rhs, .. })) = stmt else {
            unreachable!("guarded by is_aprox_distance_call")
        };
        let Expr::Call { args, .. } = rhs.as_ref() else {
            unreachable!("guarded by is_aprox_distance_call")
        };
        let first_arg = &args[0];

        let ctx = FnBodyContext {
            self_param: "actor",
            self_field_types: &self_field_types,
            extra_cross_ref_idents: &aliases,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let (rendered, _) = render_expr(first_arg, &ctx).expect("should render cleanly");
        assert_eq!(
            rendered,
            "match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.x, _ => unreachable!() } - actor.x"
        );
    }

    /// `T_VerticalDoor`'s innermost `switch(door->type) { case blazeClose:
    /// case close: break; default: ...; }` -- two `case` labels sharing
    /// one (here, empty) body. C parses `case blazeClose: case close:
    /// break;` as `Case{blazeClose, stmt: Case{close, stmt: Break}}`, not
    /// as flat siblings -- confirmed directly against the real parsed AST
    /// before writing `collect_case_labels`, which peels that chain back
    /// into one Rust match arm covering both patterns
    /// (`blazeClose | close => {}`). The whole function still can't
    /// render end-to-end yet (`S_StartSound`'s `(mobj_t *)&door->sector->
    /// soundorg` cast isn't handled -- a separate, unrelated gap), so
    /// this clones just the real shared-label `Case` chain and its real
    /// `switch` condition out of the parsed AST and re-wraps them in a
    /// synthetic `Switch`/`Compound` to test `render_switch`'s new
    /// shared-label handling in isolation, without needing the whole
    /// function to compile.
    #[test]
    fn test_shared_case_labels_against_real_t_vertical_door() {
        let path = corpus_dir().join("p_doors.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_doors.c should parse");
        let f = find_function_def(&unit.items, "T_VerticalDoor").expect("T_VerticalDoor not found");
        // `blazeClose` labels two different arms in `T_VerticalDoor`
        // (the `pastdest` switch's `case blazeRaise: case blazeClose:
        // specialdata = NULL; ...` and the `crushed` switch's own `case
        // blazeClose: case close: break;`) -- match the exact chain
        // shape (`blazeClose` -> `close` -> a bare `break;`, nothing
        // else) to find the second one specifically.
        let is_blaze_close_close_break_case = |s: &Stmt| {
            let Stmt::Case { expr, stmt } = s else {
                return false;
            };
            if !matches!(expr, Expr::Ident(n) if n == "blazeClose") {
                return false;
            }
            let Stmt::Case { expr, stmt } = stmt.as_ref() else {
                return false;
            };
            matches!(expr, Expr::Ident(n) if n == "close") && matches!(stmt.as_ref(), Stmt::Break)
        };
        let case_chain = f
            .body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => find_stmt(s, &is_blaze_close_close_break_case),
                BlockItem::Decl(_) => None,
            })
            .expect("expected a `case blazeClose:` statement somewhere in T_VerticalDoor")
            .clone();
        let switch_cond = Expr::Member {
            base: Box::new(Expr::Ident("door".to_string())),
            field: "type".to_string(),
            arrow: true,
        };
        let synthetic_switch = Stmt::Switch {
            cond: switch_cond,
            body: Box::new(Stmt::Compound(crate::parser::ast::CompoundStmt {
                items: vec![BlockItem::Stmt(case_chain)],
            })),
        };

        let no_self_fields = HashMap::new();
        let no_extra_cross_refs = HashMap::new();
        let ctx = FnBodyContext {
            self_param: "door",
            self_field_types: &no_self_fields,
            extra_cross_ref_idents: &no_extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let rendered = render_stmt(&synthetic_switch, &ctx, 1).expect("should render cleanly");
        assert_eq!(
            rendered,
            vec![
                "    match door.r#type {".to_string(),
                "        blazeClose | close => {".to_string(),
                "        }".to_string(),
                "        _ => {}".to_string(),
                "    }".to_string(),
            ]
        );
    }

    /// `T_VerticalDoor` end-to-end -- the function every other test in
    /// this module worked up to piece by piece (negated `--x`, `NULL` ->
    /// `None`, self-removal, shared `case` labels, the `S_StartSound`
    /// cast/`&`-reference pair), now proven complete rather than only in
    /// isolated fragments. Confirms `render_fn`'s own signature-extension
    /// logic (`body_has_self_removal`) correctly finds the self-removal
    /// calls buried two `switch` levels deep and adds `handle`/`arena` to
    /// the generated signature.
    #[test]
    fn test_t_vertical_door_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_doors.c",
            "T_VerticalDoor",
            "VerticalDoor",
            &vldoor_field_types(),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn T_VerticalDoor(door: &mut VerticalDoor, world: &mut World, handle: Handle<Thinker>, arena: &mut Arena<Thinker>) {
    let mut res;
    match door.direction {
        0 => {
            door.topcountdown -= 1;
            if door.topcountdown == 0 {
                match door.r#type {
                    blazeRaise => {
                        door.direction = -1;
                        S_StartSound(&world[door.sector].soundorg, sfx_bdcls);
                    }
                    normal => {
                        door.direction = -1;
                        S_StartSound(&world[door.sector].soundorg, sfx_dorcls);
                    }
                    close30ThenOpen => {
                        door.direction = 1;
                        S_StartSound(&world[door.sector].soundorg, sfx_doropn);
                    }
                    _ => {
                    }
                }
            }
        }
        2 => {
            door.topcountdown -= 1;
            if door.topcountdown == 0 {
                match door.r#type {
                    raiseIn5Mins => {
                        door.direction = 1;
                        door.r#type = normal;
                        S_StartSound(&world[door.sector].soundorg, sfx_doropn);
                    }
                    _ => {
                    }
                }
            }
        }
        -1 => {
            res = T_MovePlane(door.sector, door.speed, world[door.sector].floorheight, false, 1, door.direction);
            if res == pastdest {
                match door.r#type {
                    blazeRaise | blazeClose => {
                        world[door.sector].specialdata = None;
                        arena.remove(handle);
                        S_StartSound(&world[door.sector].soundorg, sfx_bdcls);
                    }
                    normal | close => {
                        world[door.sector].specialdata = None;
                        arena.remove(handle);
                    }
                    close30ThenOpen => {
                        door.direction = 0;
                        door.topcountdown = 35 * 30;
                    }
                    _ => {
                    }
                }
            } else {
                if res == crushed {
                    match door.r#type {
                        blazeClose | close => {
                        }
                        _ => {
                            door.direction = 1;
                            S_StartSound(&world[door.sector].soundorg, sfx_doropn);
                        }
                    }
                }
            }
        }
        1 => {
            res = T_MovePlane(door.sector, door.speed, door.topheight, false, 1, door.direction);
            if res == pastdest {
                match door.r#type {
                    blazeRaise | normal => {
                        door.direction = 0;
                        door.topcountdown = door.topwait;
                    }
                    close30ThenOpen | blazeOpen | open => {
                        world[door.sector].specialdata = None;
                        arena.remove(handle);
                    }
                    _ => {
                    }
                }
            }
        }
        _ => {}
    }
}";
        assert_eq!(rendered, expected);
    }

    /// `T_MoveFloor` (`p_floor.c`) -- structurally close to
    /// `T_VerticalDoor` (self-removal, `NULL` -> `None`, the sound-cast/
    /// reference pair, `r#type`), but with a genuine `switch` fallthrough
    /// Doom uses in two places: `case donutRaise: stmt1; stmt2; default:
    /// break;` -- `donutRaise` has *no* `break` of its own, falling into
    /// `default`'s. Since `default`'s own body is just `break;` (no
    /// statements), the fallthrough has no observable effect either way,
    /// so `render_switch` now recognizes falling into an empty `default:
    /// break;` specifically as safe and renders `donutRaise` as its own
    /// complete arm -- confirmed against the real parsed AST before
    /// relaxing the fallthrough rejection, and still narrow: falling into
    /// anything with real statements still errs loudly. Also exercises
    /// `!(leveltime&7)` (`Unary::Not` over a `Binary`, not `PreIncDec` --
    /// already handled generically by `render_bool_expr`, needing no new
    /// code) and confirms Rust's `&`-binds-tighter-than-`==` precedence
    /// matches C's exactly, so `leveltime & 7 == 0` needs no parens.
    #[test]
    fn test_t_move_floor_renders_exactly() {
        let field_types = field_types(&[
            ("sector", "SectorId"),
            ("r#type", "i32"),
            ("crush", "bool"),
            ("direction", "i32"),
            ("newspecial", "i32"),
            ("texture", "i16"),
            ("floordestheight", "FixedT"),
            ("speed", "FixedT"),
        ]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_floor.c",
            "T_MoveFloor",
            "FloorMove",
            &field_types,
        )
        .expect("should render cleanly");
        let expected = "\
pub fn T_MoveFloor(floor: &mut FloorMove, world: &mut World, handle: Handle<Thinker>, arena: &mut Arena<Thinker>) {
    let mut res;
    res = T_MovePlane(floor.sector, floor.speed, floor.floordestheight, floor.crush, 0, floor.direction);
    if leveltime & 7 == 0 {
        S_StartSound(&world[floor.sector].soundorg, sfx_stnmov);
    }
    if res == pastdest {
        world[floor.sector].specialdata = None;
        if floor.direction == 1 {
            match floor.r#type {
                donutRaise => {
                    world[floor.sector].special = floor.newspecial;
                    world[floor.sector].floorpic = floor.texture;
                }
                _ => {
                }
            }
        } else {
            if floor.direction == -1 {
                match floor.r#type {
                    lowerAndChange => {
                        world[floor.sector].special = floor.newspecial;
                        world[floor.sector].floorpic = floor.texture;
                    }
                    _ => {
                    }
                }
            }
        }
        arena.remove(handle);
        S_StartSound(&world[floor.sector].soundorg, sfx_pstop);
    }
}";
        assert_eq!(rendered, expected);
    }

    /// `T_MoveCeiling` (`p_ceilng.c`) -- real `switch` fallthrough with
    /// genuine content, not just an empty `default: break;`
    /// (`T_MoveFloor`'s narrow case). `case silentCrushAndRaise:
    /// S_StartSound(..); case crushAndRaise: ceiling->speed = CEILSPEED;
    /// case fastCrushAndRaise: ceiling->direction = 1; break;` is a real
    /// *three*-level fallthrough chain: `silentCrushAndRaise` needs all
    /// three statements, `crushAndRaise` needs the last two, and
    /// `fastCrushAndRaise` needs only its own. `render_switch`'s
    /// pass-2 back-to-front resolution (`resolved[k] = own_stmts[k] ++
    /// resolved[k+1]` when `k` falls through) builds exactly this,
    /// confirmed against the real parsed AST first -- each arm keeps its
    /// own identity and pattern (they can't share one, since their
    /// bodies genuinely differ), just with the right statements folded
    /// forward into each entry point. `Ceiling` has no self-removal here
    /// at all (`P_RemoveActiveCeiling`, a different, not-yet-translated
    /// function, not `T_VerticalDoor`'s literal `P_RemoveThinker(&self->
    /// thinker)` shape) -- confirms `render_fn`'s signature-extension
    /// logic doesn't false-positive on a similarly-named but structurally
    /// different removal call.
    #[test]
    fn test_t_move_ceiling_renders_exactly() {
        let field_types = field_types(&[
            ("sector", "SectorId"),
            ("r#type", "i32"),
            ("bottomheight", "FixedT"),
            ("topheight", "FixedT"),
            ("speed", "FixedT"),
            ("crush", "bool"),
            ("direction", "i32"),
            ("tag", "i32"),
            ("olddirection", "i32"),
        ]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_ceilng.c",
            "T_MoveCeiling",
            "Ceiling",
            &field_types,
        )
        .expect("should render cleanly");
        let expected = "\
pub fn T_MoveCeiling(ceiling: &mut Ceiling, world: &mut World) {
    let mut res;
    match ceiling.direction {
        0 => {
        }
        1 => {
            res = T_MovePlane(ceiling.sector, ceiling.speed, ceiling.topheight, false, 1, ceiling.direction);
            if leveltime & 7 == 0 {
                match ceiling.r#type {
                    silentCrushAndRaise => {
                    }
                    _ => {
                        S_StartSound(&world[ceiling.sector].soundorg, sfx_stnmov);
                    }
                }
            }
            if res == pastdest {
                match ceiling.r#type {
                    raiseToHighest => {
                        P_RemoveActiveCeiling(ceiling);
                    }
                    silentCrushAndRaise => {
                        S_StartSound(&world[ceiling.sector].soundorg, sfx_pstop);
                        ceiling.direction = -1;
                    }
                    fastCrushAndRaise | crushAndRaise => {
                        ceiling.direction = -1;
                    }
                    _ => {
                    }
                }
            }
        }
        -1 => {
            res = T_MovePlane(ceiling.sector, ceiling.speed, ceiling.bottomheight, ceiling.crush, 1, ceiling.direction);
            if leveltime & 7 == 0 {
                match ceiling.r#type {
                    silentCrushAndRaise => {
                    }
                    _ => {
                        S_StartSound(&world[ceiling.sector].soundorg, sfx_stnmov);
                    }
                }
            }
            if res == pastdest {
                match ceiling.r#type {
                    silentCrushAndRaise => {
                        S_StartSound(&world[ceiling.sector].soundorg, sfx_pstop);
                        ceiling.speed = CEILSPEED;
                        ceiling.direction = 1;
                    }
                    crushAndRaise => {
                        ceiling.speed = CEILSPEED;
                        ceiling.direction = 1;
                    }
                    fastCrushAndRaise => {
                        ceiling.direction = 1;
                    }
                    lowerAndCrush | lowerToFloor => {
                        P_RemoveActiveCeiling(ceiling);
                    }
                    _ => {
                    }
                }
            } else {
                if res == crushed {
                    match ceiling.r#type {
                        silentCrushAndRaise | crushAndRaise | lowerAndCrush => {
                            ceiling.speed = CEILSPEED / 8;
                        }
                        _ => {
                        }
                    }
                }
            }
        }
        _ => {}
    }
}";
        assert_eq!(rendered, expected);
    }

    /// `EV_DoCeiling` (`p_ceilng.c`) -- the fourth function shape closed
    /// out end-to-end: a trigger that constructs its thinker *inline*,
    /// mid-loop (`render_trigger_fn`'s own `embedded_ctor`), with fields
    /// decided by a `switch(type)` that's genuinely part of construction,
    /// not a flat statement. Exercises everything built for this shape at
    /// once: `render_ctor_body` embedded inside `render_compound_items`;
    /// `collect_ctor_fields_in` recognizing the switch's own fields
    /// (`crush` already `let`-bound from a flat statement before it, just
    /// reassigned inside two arms; `topheight`/`bottomheight`/`direction`/
    /// `speed` bound for the first time by the switch itself); and
    /// `field_defaults`, required for *every* switch-touched field here
    /// (not just the ones with a genuine per-arm gap like
    /// `bottomheight`'s own `raiseToHighest` case) -- confirmed by
    /// actually compiling a reduced repro with `rustc` that `type: i32`
    /// (not a real closed Rust `enum`) means `render_switch`'s own
    /// synthetic `_ => {}` catch-all leaves *every* field it touches
    /// looking possibly-uninitialized to Rust's real definite-assignment
    /// check, even ones every genuine `case` sets. `olddirection` needs a
    /// default for a different reason: `EV_DoCeiling` never touches it at
    /// all -- only `P_ActivateInStasisCeiling`/`EV_CeilingCrushStop`, two
    /// unrelated functions, ever read or write it, always write-before-
    /// read, confirmed by tracing every real reference in the corpus.
    /// Two more real bugs caught working through this: the parameter
    /// `type` (a Rust keyword) was escaped nowhere at all, not in its own
    /// declaration nor in any bare reference to it (fixed generally, for
    /// every bare identifier, via a new `rust_ident_name` -- distinct
    /// from `rust_field_name` since `true`/`false` are real Rust boolean
    /// literals when they appear as *values*, matching the already-
    /// established `boolean` -> `bool` mapping, not identifiers needing
    /// escaping or rejecting the way a field literally named `false`
    /// would); and the constructor's own top-level `ceiling_t* ceiling;`
    /// declaration became dead, uninferable code once every one of its
    /// fields got its own separate `let` instead, so `render_decl` now
    /// drops a local matching `embedded_ctor`'s own variable entirely.
    /// **Verified compiling for real**, including the switch/field-
    /// defaults/keyword fixes together -- zero errors, zero warnings.
    #[test]
    fn test_ev_do_ceiling_renders_exactly() {
        let params: HashMap<String, String> = [
            ("line".to_string(), "LineId".to_string()),
            ("type".to_string(), "i32".to_string()),
        ]
        .into_iter()
        .collect();
        let locals: HashMap<String, String> = [("sec".to_string(), "SectorId".to_string())]
            .into_iter()
            .collect();
        let ctor_field_types = field_types(&[
            ("sector", "SectorId"),
            ("r#type", "i32"),
            ("bottomheight", "FixedT"),
            ("topheight", "FixedT"),
            ("speed", "FixedT"),
            ("crush", "bool"),
            ("direction", "i32"),
            ("tag", "i32"),
            ("olddirection", "i32"),
        ]);
        let field_defaults = field_types(&[
            ("olddirection", "0"),
            ("topheight", "world[sec].ceilingheight"),
            ("bottomheight", "world[sec].floorheight"),
            ("direction", "0"),
            ("speed", "CEILSPEED"),
        ]);
        let rendered = render_trigger_fn(
            &corpus_dir(),
            "p_ceilng.c",
            "EV_DoCeiling",
            &params,
            &locals,
            Some(CtorSpec {
                ctor_var: "ceiling",
                ctor_rust_type: "Ceiling",
                ctor_field_types: &ctor_field_types,
                field_defaults: &field_defaults,
            }),
            Some("i32"),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn EV_DoCeiling(line: LineId, r#type: i32, world: &mut World, thinkers: &mut Arena<Thinker>) -> i32 {
    let mut secnum;
    let mut rtn;
    let mut sec;
    secnum = -1;
    rtn = 0;
    match r#type {
        fastCrushAndRaise | silentCrushAndRaise | crushAndRaise => {
            P_ActivateInStasisCeiling(line);
        }
        _ => {
        }
    }
    loop {
        secnum = P_FindSectorFromLineTag(line, secnum);
        if !(secnum >= 0) {
            break;
        }
        sec = SectorId(secnum as u32);
        if world[sec].specialdata.is_some() {
            continue;
        }
        rtn = 1;
        let sector = sec;
        let mut crush = false;
        let mut topheight = world[sec].ceilingheight;
        let mut bottomheight = world[sec].floorheight;
        let mut direction = 0;
        let mut speed = CEILSPEED;
        match r#type {
            fastCrushAndRaise => {
                crush = true;
                topheight = world[sec].ceilingheight;
                bottomheight = world[sec].floorheight + 8 * FRACUNIT;
                direction = -1;
                speed = CEILSPEED * 2;
            }
            silentCrushAndRaise | crushAndRaise => {
                crush = true;
                topheight = world[sec].ceilingheight;
                bottomheight = world[sec].floorheight;
                if r#type != lowerToFloor {
                    bottomheight += 8 * FRACUNIT;
                }
                direction = -1;
                speed = CEILSPEED;
            }
            lowerAndCrush | lowerToFloor => {
                bottomheight = world[sec].floorheight;
                if r#type != lowerToFloor {
                    bottomheight += 8 * FRACUNIT;
                }
                direction = -1;
                speed = CEILSPEED;
            }
            raiseToHighest => {
                topheight = P_FindHighestCeilingSurrounding(sec);
                direction = 1;
                speed = CEILSPEED;
            }
            _ => {}
        }
        let tag = world[sec].tag;
        let olddirection = 0;
        let handle = thinkers.insert(Thinker::Ceiling(Ceiling { sector, crush, topheight, bottomheight, direction, speed, tag, r#type, olddirection }));
        world[sec].specialdata = Some(handle);
        P_AddActiveCeiling(handle);
    }
    return rtn;
}";
        assert_eq!(rendered, expected);
    }

    /// `EV_DoDoor` (`p_doors.c`) -- the same trigger-with-inline-
    /// constructor shape as `EV_DoCeiling`, reusing the whole
    /// `collect_ctor_fields_in`/`field_defaults` machinery unchanged, but
    /// surfacing two more real gaps neither prior function happened to
    /// exercise: (1) `int secnum,rtn;` declares two locals off one `int`
    /// specifier -- `render_decl` previously only accepted a single
    /// declarator (every earlier function's locals each got their own
    /// line), fixed by looping over `d.declarators` instead of
    /// destructuring a one-element slice. (2) `door->sector->soundorg`
    /// reads a cross-reference-typed constructor field (`sector:
    /// SectorId`, set from `door->sector = sec;` earlier) back out and
    /// dereferences *through* it -- the `Expr::Member` branch handling a
    /// bare `ctor_var->field` read hard-coded `is_crossref: false`
    /// unconditionally (it only ever needed to name the field's own
    /// local before), so this rendered as the ill-typed `sector.soundorg`
    /// instead of `world[sector].soundorg`. Fixed by adding
    /// `FnBodyContext::ctor_field_types` (mirroring `self_field_types`)
    /// and checking it the same way the general `Member` fallback already
    /// checks `self_field_types`. Both bugs were real and previously
    /// unexercised, not design gaps anticipated ahead of time. `speed` is
    /// set once before the switch and reassigned inside two arms (making
    /// it `mut` from the ordinary reassign-count path, not the new
    /// switch-field-synthesis path, since it's already `let`-bound by the
    /// time the switch is reached); `topheight`/`direction` are switch-
    /// only and need `field_defaults`; `topcountdown` is never touched at
    /// all in `EV_DoDoor` (only read when `door.direction` is `0` or `2`,
    /// values `EV_DoDoor` never produces -- confirmed by tracing every
    /// read in `T_VerticalDoor`), so it falls through to the general
    /// completeness-check default exactly like `EV_DoCeiling`'s
    /// `olddirection`. `door->type = type;` is a plain passthrough
    /// (already-handled dedup: no `let` line, since the field name and
    /// the already-in-scope parameter's rendered name are identical).
    /// Verified compiling the complete function with `rustc` directly
    /// (hand-written `World`/`Sector`/`VerticalDoor`/`Arena`/`Handle`/
    /// `FixedT` stand-ins), zero errors.
    #[test]
    fn test_ev_do_door_renders_exactly() {
        let params: HashMap<String, String> = [
            ("line".to_string(), "LineId".to_string()),
            ("type".to_string(), "i32".to_string()),
        ]
        .into_iter()
        .collect();
        let locals: HashMap<String, String> = [("sec".to_string(), "SectorId".to_string())]
            .into_iter()
            .collect();
        let ctor_field_types = vldoor_field_types();
        let field_defaults = field_types(&[
            ("topheight", "world[sec].ceilingheight"),
            ("direction", "0"),
            ("topcountdown", "0"),
        ]);
        let rendered = render_trigger_fn(
            &corpus_dir(),
            "p_doors.c",
            "EV_DoDoor",
            &params,
            &locals,
            Some(CtorSpec {
                ctor_var: "door",
                ctor_rust_type: "VerticalDoor",
                ctor_field_types: &ctor_field_types,
                field_defaults: &field_defaults,
            }),
            Some("i32"),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn EV_DoDoor(line: LineId, r#type: i32, world: &mut World, thinkers: &mut Arena<Thinker>) -> i32 {
    let mut secnum;
    let mut rtn;
    let mut sec;
    secnum = -1;
    rtn = 0;
    loop {
        secnum = P_FindSectorFromLineTag(line, secnum);
        if !(secnum >= 0) {
            break;
        }
        sec = SectorId(secnum as u32);
        if world[sec].specialdata.is_some() {
            continue;
        }
        rtn = 1;
        let sector = sec;
        let topwait = VDOORWAIT;
        let mut speed = VDOORSPEED;
        let mut topheight = world[sec].ceilingheight;
        let mut direction = 0;
        match r#type {
            blazeClose => {
                topheight = P_FindLowestCeilingSurrounding(sec);
                topheight -= 4 * FRACUNIT;
                direction = -1;
                speed = VDOORSPEED * 4;
                S_StartSound(&world[sector].soundorg, sfx_bdcls);
            }
            close => {
                topheight = P_FindLowestCeilingSurrounding(sec);
                topheight -= 4 * FRACUNIT;
                direction = -1;
                S_StartSound(&world[sector].soundorg, sfx_dorcls);
            }
            close30ThenOpen => {
                topheight = world[sec].ceilingheight;
                direction = -1;
                S_StartSound(&world[sector].soundorg, sfx_dorcls);
            }
            blazeRaise | blazeOpen => {
                direction = 1;
                topheight = P_FindLowestCeilingSurrounding(sec);
                topheight -= 4 * FRACUNIT;
                speed = VDOORSPEED * 4;
                if topheight != world[sec].ceilingheight {
                    S_StartSound(&world[sector].soundorg, sfx_bdopn);
                }
            }
            normal | open => {
                direction = 1;
                topheight = P_FindLowestCeilingSurrounding(sec);
                topheight -= 4 * FRACUNIT;
                if topheight != world[sec].ceilingheight {
                    S_StartSound(&world[sector].soundorg, sfx_doropn);
                }
            }
            _ => {
            }
        }
        let topcountdown = 0;
        let handle = thinkers.insert(Thinker::VerticalDoor(VerticalDoor { sector, r#type, topwait, speed, topheight, direction, topcountdown }));
        world[sec].specialdata = Some(handle);
    }
    return rtn;
}";
        assert_eq!(rendered, expected);
    }

    /// `EV_DoPlat` (`p_plats.c`) -- a third trigger-with-inline-
    /// constructor, again reusing `EV_DoCeiling`'s whole machinery
    /// unchanged, but needing two genuinely new expression-rendering
    /// features and exposing a real nondeterminism bug. (1)
    /// `sides[line->sidenum[0]].sector->floorpic` chains a bare (non-`&`)
    /// global-array index (`sides[i]`, mirroring `&sectors[i]`'s existing
    /// special case but without the address-of, so it returns
    /// `is_crossref: true` directly rather than relying on a caller's own
    /// dereference) into a cross-reference-typed field (`side_t.sector`,
    /// hand-matched narrowly since there's no general struct-field-type
    /// registry yet) into a further plain field read -- renders as double
    /// `world[...]` indirection (`world[world[SideId(..)].sector].
    /// floorpic`), which is exactly correct: `sides[i]` is a `SideId`
    /// *value*, and `.sector` reads a *further* index out of it, so two
    /// separate lookups are genuinely needed. `line->sidenum[0]` (the
    /// index expression itself) needed a new, fully generic
    /// `Expr::Index` fallback arm for a plain fixed-size array field
    /// (`sidenum: [i16; 2]`) -- nothing before this needed indexing
    /// *into* an ordinary field, only the two special global arrays.
    /// `World` gained a second real field (`sides: Vec<Side>` +
    /// `Index`/`IndexMut<SideId>`, mirroring `sectors` exactly) to give
    /// the new `SideId` lookups somewhere to resolve to -- the same
    /// "grows when a real body needs it" incremental pattern `sectors`
    /// itself followed. (2) `plat->sector->specialdata = plat;` writes
    /// *through* a constructor field the same way `EV_DoDoor`'s
    /// `door->sector->soundorg` read through one -- already fixed for
    /// free by that same `ctor_field_types` fix, no new code needed here.
    /// **A real nondeterminism bug found while building this**: `count`
    /// and `oldstatus` are both untouched by `EV_DoPlat`'s own `switch`
    /// (needing `field_defaults` from the general completeness-check
    /// fallback, not the switch-synthesis path), and that fallback
    /// iterated `ctor_field_types.keys()` -- a `HashMap`, whose iteration
    /// order is randomized per-process. `EV_DoCeiling` never exposed this
    /// (`olddirection` was its only such field, so order was never
    /// observable); with two fields here, an unsorted iteration would
    /// make the renderer's own output nondeterministic across runs, not
    /// just differently-but-validly ordered. Fixed by sorting the
    /// remaining field names before that loop. `speed`/`high`/`wait`/
    /// `status`/`low` are all switch-only, needing `field_defaults`
    /// unconditionally per the same empirically-verified rule as
    /// `EV_DoCeiling`; `count`/`oldstatus` are safe to default (confirmed
    /// by tracing every read in `T_PlatRaise`/`P_ActivateInStasis`/
    /// `EV_StopPlat`: both are always written before ever being read, the
    /// same write-before-read safety as `EV_DoCeiling`'s `olddirection`).
    /// Verified compiling the complete function with `rustc` directly
    /// (hand-written `World`/`Sector`/`Side`/`Line`/`Plat`/`Arena`/
    /// `Handle`/`FixedT` stand-ins), zero errors.
    #[test]
    fn test_ev_do_plat_renders_exactly() {
        let params: HashMap<String, String> = [
            ("line".to_string(), "LineId".to_string()),
            ("type".to_string(), "i32".to_string()),
            ("amount".to_string(), "i32".to_string()),
        ]
        .into_iter()
        .collect();
        let locals: HashMap<String, String> = [("sec".to_string(), "SectorId".to_string())]
            .into_iter()
            .collect();
        let ctor_field_types = field_types(&[
            ("sector", "SectorId"),
            ("speed", "FixedT"),
            ("low", "FixedT"),
            ("high", "FixedT"),
            ("wait", "i32"),
            ("count", "i32"),
            ("status", "i32"),
            ("oldstatus", "i32"),
            ("crush", "bool"),
            ("tag", "i32"),
            ("r#type", "i32"),
        ]);
        let field_defaults = field_types(&[
            ("speed", "PLATSPEED"),
            ("low", "world[sec].floorheight"),
            ("high", "world[sec].ceilingheight"),
            ("wait", "0"),
            ("status", "0"),
            ("count", "0"),
            ("oldstatus", "0"),
        ]);
        let rendered = render_trigger_fn(
            &corpus_dir(),
            "p_plats.c",
            "EV_DoPlat",
            &params,
            &locals,
            Some(CtorSpec {
                ctor_var: "plat",
                ctor_rust_type: "Plat",
                ctor_field_types: &ctor_field_types,
                field_defaults: &field_defaults,
            }),
            Some("i32"),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn EV_DoPlat(line: LineId, r#type: i32, amount: i32, world: &mut World, thinkers: &mut Arena<Thinker>) -> i32 {
    let mut secnum;
    let mut rtn;
    let mut sec;
    secnum = -1;
    rtn = 0;
    match r#type {
        perpetualRaise => {
            P_ActivateInStasis(world[line].tag);
        }
        _ => {
        }
    }
    loop {
        secnum = P_FindSectorFromLineTag(line, secnum);
        if !(secnum >= 0) {
            break;
        }
        sec = SectorId(secnum as u32);
        if world[sec].specialdata.is_some() {
            continue;
        }
        rtn = 1;
        let sector = sec;
        let crush = false;
        let tag = world[line].tag;
        let mut speed = PLATSPEED;
        let mut high = world[sec].ceilingheight;
        let mut wait = 0;
        let mut status = 0;
        let mut low = world[sec].floorheight;
        match r#type {
            raiseToNearestAndChange => {
                speed = PLATSPEED / 2;
                world[sec].floorpic = world[world[SideId(world[line].sidenum[0] as u32)].sector].floorpic;
                high = P_FindNextHighestFloor(sec, world[sec].floorheight);
                wait = 0;
                status = up;
                world[sec].special = 0;
                S_StartSound(&world[sec].soundorg, sfx_stnmov);
            }
            raiseAndChange => {
                speed = PLATSPEED / 2;
                world[sec].floorpic = world[world[SideId(world[line].sidenum[0] as u32)].sector].floorpic;
                high = world[sec].floorheight + amount * FRACUNIT;
                wait = 0;
                status = up;
                S_StartSound(&world[sec].soundorg, sfx_stnmov);
            }
            downWaitUpStay => {
                speed = PLATSPEED * 4;
                low = P_FindLowestFloorSurrounding(sec);
                if low > world[sec].floorheight {
                    low = world[sec].floorheight;
                }
                high = world[sec].floorheight;
                wait = 35 * PLATWAIT;
                status = down;
                S_StartSound(&world[sec].soundorg, sfx_pstart);
            }
            blazeDWUS => {
                speed = PLATSPEED * 8;
                low = P_FindLowestFloorSurrounding(sec);
                if low > world[sec].floorheight {
                    low = world[sec].floorheight;
                }
                high = world[sec].floorheight;
                wait = 35 * PLATWAIT;
                status = down;
                S_StartSound(&world[sec].soundorg, sfx_pstart);
            }
            perpetualRaise => {
                speed = PLATSPEED;
                low = P_FindLowestFloorSurrounding(sec);
                if low > world[sec].floorheight {
                    low = world[sec].floorheight;
                }
                high = P_FindHighestFloorSurrounding(sec);
                if high < world[sec].floorheight {
                    high = world[sec].floorheight;
                }
                wait = 35 * PLATWAIT;
                status = P_Random() & 1;
                S_StartSound(&world[sec].soundorg, sfx_pstart);
            }
            _ => {}
        }
        let count = 0;
        let oldstatus = 0;
        let handle = thinkers.insert(Thinker::Plat(Plat { r#type, sector, crush, tag, speed, high, wait, status, low, count, oldstatus }));
        world[sector].specialdata = Some(handle);
        P_AddActivePlat(handle);
    }
    return rtn;
}";
        assert_eq!(rendered, expected);
    }

    /// `EV_DoLockedDoor` (`p_doors.c`) -- a plain trigger (no embedded
    /// constructor at all, it only early-returns or delegates to the
    /// already-translated `EV_DoDoor`), but the first function needing
    /// three genuinely new capabilities, all narrowly scoped to exactly
    /// what this function's own body needs. (1) `thing->player`: `thing`'s
    /// own declared type is `Handle<Thinker>` (a live thinker passed in,
    /// surveyed as the harder blocker for both this function and
    /// `EV_VerticalDoor`), needing a real `Arena` lookup plus picking out
    /// the one variant (`Mobj`) with a `.player` field at all, out of
    /// `Thinker`'s ten. Hand-matched narrowly (only `Mobj` reached this
    /// way so far), not a general enum-variant-field lookup mechanism.
    /// (2) `p->cards[..]`/`p->message`: `p`'s own declared type is
    /// `Option<PlayerId>` (a `player_t*` local) -- rather than
    /// reshaping the function around a narrowed/shadowed binding once
    /// null-checked (real `let-else` unwrapping), every dereference is
    /// `.unwrap()`-ed at its own point of use, since the corpus itself
    /// already guards each one with its own adjacent `if (!p) return
    /// ...;` (redundant after the first, since `p` can't become "null"
    /// again -- rendered as-is anyway, a close translation of the
    /// original's own defensive redundancy rather than a new "prove and
    /// elide dead code" feature this one function doesn't need). (3)
    /// `!p` (bare `Option`-typed local truthiness) needed
    /// `render_bool_expr`'s own `Unary::Not` handling to become aware of
    /// `Option<PlayerId>`, alongside the pre-existing `specialdata`-field
    /// special case -- `.is_none()`, not the `== 0` every other negated
    /// (plain `int`) value gets. `S_StartSound(NULL, sfx_oof)` (no sound
    /// origin) surfaced a real, more general gap: `NULL` was only ever
    /// converted to `None` in the one `specialdata = NULL;` assignment
    /// shape `render_expr_stmt` already special-cased, not as a plain
    /// value anywhere else -- generalized so `Expr::Ident("NULL")`
    /// always renders `None`, matching this project's own no-real-
    /// pointers memory model regardless of where it appears. `World`
    /// gained a third real field, `players: [Player; MAXPLAYERS]` --
    /// deliberately a fixed array, not `Vec`, matching `runtime/
    /// player.rs`'s own already-documented design (`player_t
    /// players[MAXPLAYERS]` is genuinely fixed-size, unlike `sectors`/
    /// `sides`). `!player->cards[idx]` (used inside `&&`, not as a
    /// whole `if`'s own top-level condition) needed no new code at all
    /// -- it flows through `render_expr`'s already-generic `Unary::Not`
    /// arm (plain `!`), not `render_bool_expr`'s specialized top-level
    /// handling, and a real `bool` field negates correctly with Rust's
    /// own `!` already. Verified compiling the complete function with
    /// `rustc` directly (hand-written `World`/`Player`/`Mobj`/`Thinker`/
    /// `Arena`/`Handle` stand-ins), zero errors.
    #[test]
    fn test_ev_do_locked_door_renders_exactly() {
        let params: HashMap<String, String> = [
            ("line".to_string(), "LineId".to_string()),
            ("type".to_string(), "i32".to_string()),
            ("thing".to_string(), "Handle<Thinker>".to_string()),
        ]
        .into_iter()
        .collect();
        let locals: HashMap<String, String> = [("p".to_string(), "Option<PlayerId>".to_string())]
            .into_iter()
            .collect();
        let rendered = render_trigger_fn(
            &corpus_dir(),
            "p_doors.c",
            "EV_DoLockedDoor",
            &params,
            &locals,
            None,
            Some("i32"),
        )
        .expect("should render cleanly");
        let expected = "\
pub fn EV_DoLockedDoor(line: LineId, r#type: i32, thing: Handle<Thinker>, world: &mut World, thinkers: &mut Arena<Thinker>) -> i32 {
    let mut p;
    p = match thinkers.get(thing) { Some(Thinker::Mobj(m)) => m.player, _ => None };
    if p.is_none() {
        return 0;
    }
    match world[line].special {
        99 | 133 => {
            if p.is_none() {
                return 0;
            }
            if !world[p.unwrap()].cards[it_bluecard] && !world[p.unwrap()].cards[it_blueskull] {
                world[p.unwrap()].message = PD_BLUEO;
                S_StartSound(None, sfx_oof);
                return 0;
            }
        }
        134 | 135 => {
            if p.is_none() {
                return 0;
            }
            if !world[p.unwrap()].cards[it_redcard] && !world[p.unwrap()].cards[it_redskull] {
                world[p.unwrap()].message = PD_REDO;
                S_StartSound(None, sfx_oof);
                return 0;
            }
        }
        136 | 137 => {
            if p.is_none() {
                return 0;
            }
            if !world[p.unwrap()].cards[it_yellowcard] && !world[p.unwrap()].cards[it_yellowskull] {
                world[p.unwrap()].message = PD_YELLOWO;
                S_StartSound(None, sfx_oof);
                return 0;
            }
        }
        _ => {}
    }
    return EV_DoDoor(line, r#type);
}";
        assert_eq!(rendered, expected);
    }

    /// `EV_VerticalDoor` (`p_doors.c`) -- the function that motivated
    /// surveying this whole area: reuses `EV_DoLockedDoor`'s
    /// `thing->player`/`Option<PlayerId>` infrastructure verbatim for its
    /// own three-lock-check preamble (confirmed identical shape, just
    /// different local/message names), reuses `EV_DoPlat`'s `sides[i].
    /// sector` chain for `sec = sides[line->sidenum[side^1]].sector;`,
    /// and adds three more pieces: (1) `secnum = sec-sectors;` -- real
    /// pointer arithmetic (computed but genuinely never read again,
    /// confirmed by grep, though translated honestly rather than
    /// dropped): since a `SectorId` already *is* "the index of this
    /// pointer," this needs no arithmetic at all, just `sec.0 as i32`.
    /// (2) The function's own core new capability -- `if (sec->
    /// specialdata) { door = sec->specialdata; ...door->field...; }`,
    /// reusing an *existing* thinker instead of always constructing a
    /// new one. `existing_thinker_mutation_shape` detects this exact
    /// assignment-from-specialdata shape (narrowly, the same "hand-match
    /// the one real corpus shape" style as everywhere else in this
    /// module); `render_existing_thinker_mutation` renders every
    /// remaining statement in the block with `FnBodyContext::
    /// mutating_handle` set, so each `door->field` reference (read or
    /// write) gets its own *fresh* `thinkers.get`/`get_mut` call.
    /// **Deliberately not a single hoisted `let Thinker::VerticalDoor
    /// (door) = thinkers.get_mut(..).unwrap() else { unreachable!() };`
    /// binding** (the obvious first design, matching how a tick
    /// function's own `self` receiver already works) -- tried first, and
    /// rejected by `rustc` itself, not by inspection: this exact block
    /// reads `door->direction`, then (in the `else` branch) calls
    /// `thing->player` -- a *second*, unrelated `thinkers.get(..)` call
    /// -- before writing `door->direction` again, and a hoisted `&mut`
    /// binding must stay borrowed across that whole span, conflicting
    /// with the intervening immutable borrow. Fresh per-access borrows
    /// sidestep this outright. (3) `S_StartSound` is called both with a
    /// real `&SoundOrg` (the "for proper sound" switch) and with `NULL`
    /// -> `None` (the lock-check preamble) *within this one function* --
    /// a real, newly-surfaced gap this renderer doesn't resolve (no
    /// cross-function awareness of `S_StartSound`'s own real parameter
    /// type to know whether the real-reference call sites need `Some(..)`
    /// wrapping too), verified compiling only by giving the scratch
    /// stub's own `S_StartSound` a deliberately generic signature -- left
    /// as a documented, out-of-scope limitation (see docs/03_TRANSPILER.md),
    /// not fixed here. `r#type`/`topcountdown` need `field_defaults`
    /// (switch-only and never-touched respectively, same reasoning as
    /// every earlier trigger-with-inline-constructor). Verified compiling
    /// the complete function with `rustc` directly (hand-written `World`/
    /// `Sector`/`Side`/`Line`/`Player`/`Mobj`/`VerticalDoor`/`Thinker`/
    /// `Arena`/`Handle` stand-ins), zero errors.
    #[test]
    fn test_ev_vertical_door_renders_exactly() {
        let params: HashMap<String, String> = [
            ("line".to_string(), "LineId".to_string()),
            ("thing".to_string(), "Handle<Thinker>".to_string()),
        ]
        .into_iter()
        .collect();
        let locals: HashMap<String, String> = [
            ("player".to_string(), "Option<PlayerId>".to_string()),
            ("sec".to_string(), "SectorId".to_string()),
        ]
        .into_iter()
        .collect();
        let ctor_field_types = vldoor_field_types();
        let field_defaults = field_types(&[("r#type", "normal"), ("topcountdown", "0")]);
        let rendered = render_trigger_fn(
            &corpus_dir(),
            "p_doors.c",
            "EV_VerticalDoor",
            &params,
            &locals,
            Some(CtorSpec {
                ctor_var: "door",
                ctor_rust_type: "VerticalDoor",
                ctor_field_types: &ctor_field_types,
                field_defaults: &field_defaults,
            }),
            None,
        )
        .expect("should render cleanly");
        let expected = "\
pub fn EV_VerticalDoor(line: LineId, thing: Handle<Thinker>, world: &mut World, thinkers: &mut Arena<Thinker>) {
    let mut player;
    let mut secnum;
    let mut sec;
    let mut side;
    side = 0;
    player = match thinkers.get(thing) { Some(Thinker::Mobj(m)) => m.player, _ => None };
    match world[line].special {
        26 | 32 => {
            if player.is_none() {
                return;
            }
            if !world[player.unwrap()].cards[it_bluecard] && !world[player.unwrap()].cards[it_blueskull] {
                world[player.unwrap()].message = PD_BLUEK;
                S_StartSound(None, sfx_oof);
                return;
            }
        }
        27 | 34 => {
            if player.is_none() {
                return;
            }
            if !world[player.unwrap()].cards[it_yellowcard] && !world[player.unwrap()].cards[it_yellowskull] {
                world[player.unwrap()].message = PD_YELLOWK;
                S_StartSound(None, sfx_oof);
                return;
            }
        }
        28 | 33 => {
            if player.is_none() {
                return;
            }
            if !world[player.unwrap()].cards[it_redcard] && !world[player.unwrap()].cards[it_redskull] {
                world[player.unwrap()].message = PD_REDK;
                S_StartSound(None, sfx_oof);
                return;
            }
        }
        _ => {}
    }
    sec = world[SideId(world[line].sidenum[side ^ 1] as u32)].sector;
    secnum = sec.0 as i32;
    if world[sec].specialdata.is_some() {
        match world[line].special {
            1 | 26 | 27 | 28 | 117 => {
                if match thinkers.get(world[sec].specialdata.unwrap()) { Some(Thinker::VerticalDoor(door)) => door.direction, _ => unreachable!() } == -1 {
                    if let Some(Thinker::VerticalDoor(door)) = thinkers.get_mut(world[sec].specialdata.unwrap()) { door.direction = 1; };
                } else {
                    if match thinkers.get(thing) { Some(Thinker::Mobj(m)) => m.player, _ => None }.is_none() {
                        return;
                    }
                    if let Some(Thinker::VerticalDoor(door)) = thinkers.get_mut(world[sec].specialdata.unwrap()) { door.direction = -1; };
                }
                return;
            }
            _ => {}
        }
    }
    match world[line].special {
        117 | 118 => {
            S_StartSound(&world[sec].soundorg, sfx_bdopn);
        }
        1 | 31 => {
            S_StartSound(&world[sec].soundorg, sfx_doropn);
        }
        _ => {
            S_StartSound(&world[sec].soundorg, sfx_doropn);
        }
    }
    let sector = sec;
    let direction = 1;
    let mut speed = VDOORSPEED;
    let topwait = VDOORWAIT;
    let mut r#type = normal;
    match world[line].special {
        1 | 26 | 27 | 28 => {
            r#type = normal;
        }
        31 | 32 | 33 | 34 => {
            r#type = open;
            world[line].special = 0;
        }
        117 => {
            r#type = blazeRaise;
            speed = VDOORSPEED * 4;
        }
        118 => {
            r#type = blazeOpen;
            world[line].special = 0;
            speed = VDOORSPEED * 4;
        }
        _ => {}
    }
    let mut topheight = P_FindLowestCeilingSurrounding(sec);
    topheight -= 4 * FRACUNIT;
    let topcountdown = 0;
    let handle = thinkers.insert(Thinker::VerticalDoor(VerticalDoor { sector, direction, speed, topwait, r#type, topheight, topcountdown }));
    world[sec].specialdata = Some(handle);
}";
        assert_eq!(rendered, expected);
    }

    /// `EV_DoFloor`'s own `floor->floordestheight -= (8*FRACUNIT)*
    /// (floortype == raiseFloorCrush);` -- C's bool-as-0-or-1 arithmetic
    /// idiom (`raiseFloorCrush`'s own extra 8-unit lowering, applied only
    /// when `floortype` really is `raiseFloorCrush`, folded into one
    /// expression rather than an `if`). A comparison already renders as a
    /// real Rust `bool`, which can't multiply/add/etc. directly the way
    /// C's `int`-valued comparisons can -- confirmed as a real, distinct
    /// gap (not yet exercised by any prior function) directly against
    /// this real parsed statement before designing `render_binary_
    /// operand`'s fix, the same real-AST-extraction approach as the
    /// negated-`--x`/`NULL`-to-`None` tests above: pulls just this one
    /// statement out of the real `EV_DoFloor` AST without attempting the
    /// rest of that much harder function (deliberately deferred -- see
    /// docs/03_TRANSPILER.md).
    #[test]
    fn test_bool_as_arithmetic_against_real_ev_do_floor() {
        let path = corpus_dir().join("p_floor.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_floor.c should parse");
        let f = find_function_def(&unit.items, "EV_DoFloor").expect("EV_DoFloor not found");
        let is_floordestheight_sub_assign = |s: &Stmt| {
            matches!(s, Stmt::Expr(Some(Expr::Assign { op: AssignOp::SubAssign, lhs, .. }))
                if matches!(lhs.as_ref(), Expr::Member { field, .. } if field == "floordestheight"))
        };
        let stmt = f
            .body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => find_stmt(s, &is_floordestheight_sub_assign),
                BlockItem::Decl(_) => None,
            })
            .expect("expected a `floordestheight -= ...;` statement somewhere in EV_DoFloor");
        let Stmt::Expr(Some(e)) = stmt else {
            unreachable!("guarded by is_floordestheight_sub_assign")
        };

        let self_field_types = field_types(&[("floordestheight", "FixedT")]);
        let no_extra_cross_refs = HashMap::new();
        let ctx = FnBodyContext {
            self_param: "floor",
            self_field_types: &self_field_types,
            extra_cross_ref_idents: &no_extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let rendered = render_expr_stmt(e, &ctx).expect("should render cleanly");
        assert_eq!(
            rendered,
            "floor.floordestheight -= 8 * FRACUNIT * (floortype == raiseFloorCrush) as i32"
        );
    }

    /// `EV_DoFloor`'s second piece: the `for (i = 0; i < sec->linecount;
    /// i++)` header both `raiseToTexture` and `lowerAndChange` use to scan
    /// a sector's own lines (a genuinely new statement shape -- no
    /// function translated so far has needed a real `for` loop). Pulls
    /// the real `Stmt::For` straight out of `EV_DoFloor`'s own parsed AST
    /// (whichever of the two identical-header loops `find_stmt` reaches
    /// first), then re-wraps its real `init`/`cond`/`step` around a
    /// synthetic empty body -- the same "clone the real subtree, swap in
    /// a synthetic body/wrapper" approach already used for the shared-
    /// case-labels test against `T_VerticalDoor`, since the loop bodies
    /// themselves depend on pieces (`twoSided`/`getSide`/`textureheight[]`)
    /// not modeled yet and tracked separately in docs/03_TRANSPILER.md.
    #[test]
    fn test_for_loop_header_against_real_ev_do_floor() {
        let path = corpus_dir().join("p_floor.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_floor.c should parse");
        let f = find_function_def(&unit.items, "EV_DoFloor").expect("EV_DoFloor not found");
        let is_for_loop = |s: &Stmt| matches!(s, Stmt::For { .. });
        let stmt = f
            .body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => find_stmt(s, &is_for_loop),
                BlockItem::Decl(_) => None,
            })
            .expect("expected a `for` loop somewhere in EV_DoFloor");
        let Stmt::For {
            init, cond, step, ..
        } = stmt
        else {
            unreachable!("guarded by is_for_loop")
        };
        let synthetic = Stmt::For {
            init: init.clone(),
            cond: cond.clone(),
            step: step.clone(),
            body: Box::new(Stmt::Compound(crate::parser::ast::CompoundStmt {
                items: Vec::new(),
            })),
        };

        let extra_cross_refs = field_types(&[("sec", "SectorId")]);
        let ctx = FnBodyContext {
            self_param: "",
            self_field_types: &HashMap::new(),
            extra_cross_ref_idents: &extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let rendered = render_stmt(&synthetic, &ctx, 0).expect("should render cleanly");
        assert_eq!(
            rendered,
            vec![
                "i = 0;".to_string(),
                "while i < world[sec].linecount {".to_string(),
                "    i += 1;".to_string(),
                "}".to_string(),
            ]
        );
    }

    /// `EV_DoFloor`'s third piece: `raiseToTexture`'s adjacency scan,
    /// `if (twoSided (secnum, i)) { side = getSide(secnum,i,0); ... }` --
    /// `p_spec.c`'s `twoSided`/`getSide`/`getSector` helpers, real corpus
    /// functions (not macros) over the same sector/line/side adjacency
    /// used already-translated corpus-wide (`sides[i].sector`), not yet
    /// modeled at all before this. `twoSided` genuinely returns a plain
    /// `int`, used here as a bare (non-negated) call-result condition for
    /// the first time -- needs `render_bool_expr`'s new `twoSided` arm.
    /// `getSide` returns `side_t*` -> `SideId` under the existing memory-
    /// model decision, exactly like `sec: sector_t*` already does, so
    /// `side = getSide(...)` and a later `side->field` need no new code at
    /// all once the caller supplies `side`'s own declared type via
    /// `extra_cross_ref_idents` -- the same generic mechanism `sec`
    /// already exercises. Clones the real `cond` plus the real first
    /// statement of the real `then_branch` (`side = getSide(secnum,i,0);`)
    /// out of `EV_DoFloor`'s own parsed AST, re-wrapped around a synthetic
    /// one-statement body -- the rest of that `if`'s body still depends on
    /// `textureheight[...]` (not modeled yet, tracked separately in
    /// docs/03_TRANSPILER.md), so it's deliberately left out here.
    #[test]
    fn test_two_sided_adjacency_scan_against_real_ev_do_floor() {
        let path = corpus_dir().join("p_floor.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_floor.c should parse");
        let f = find_function_def(&unit.items, "EV_DoFloor").expect("EV_DoFloor not found");
        let is_two_sided_if = |s: &Stmt| {
            matches!(s, Stmt::If { cond: Expr::Call { callee, .. }, .. }
                if matches!(callee.as_ref(), Expr::Ident(n) if n == "twoSided"))
        };
        let stmt = f
            .body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => find_stmt(s, &is_two_sided_if),
                BlockItem::Decl(_) => None,
            })
            .expect("expected an `if (twoSided(..))` statement somewhere in EV_DoFloor");
        let Stmt::If {
            cond, then_branch, ..
        } = stmt
        else {
            unreachable!("guarded by is_two_sided_if")
        };
        let Stmt::Compound(then_body) = then_branch.as_ref() else {
            panic!("expected the `if (twoSided(..))` body to be a compound statement");
        };
        let first_stmt = then_body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => Some(s),
                BlockItem::Decl(_) => None,
            })
            .expect("expected at least one statement in the `if (twoSided(..))` body");

        let synthetic = Stmt::If {
            cond: cond.clone(),
            then_branch: Box::new(Stmt::Compound(crate::parser::ast::CompoundStmt {
                items: vec![BlockItem::Stmt(first_stmt.clone())],
            })),
            else_branch: None,
        };

        let extra_cross_refs = field_types(&[("side", "SideId")]);
        let ctx = FnBodyContext {
            self_param: "",
            self_field_types: &HashMap::new(),
            extra_cross_ref_idents: &extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let rendered = render_stmt(&synthetic, &ctx, 0).expect("should render cleanly");
        assert_eq!(
            rendered,
            vec![
                "if twoSided(secnum, i) != 0 {".to_string(),
                "    side = getSide(secnum, i, 0);".to_string(),
                "}".to_string(),
            ]
        );
    }

    /// `EV_DoFloor`'s fourth piece: `textureheight[...]`, a genuinely new
    /// kind of table -- unlike `mobjinfo[]`/`states[]` (compile-time
    /// literal data straight from the corpus source, rendered as a
    /// `pub static` array), `textureheight` (`r_data.c`) is allocated at
    /// runtime (`Z_Malloc`) and filled from WAD data, so it has no
    /// literal corpus initializer to embed -- for this function body's
    /// own purposes it's just an opaque global identifier, indexed like
    /// any other array (the *setup* code that allocates/fills it,
    /// `R_InitTextures`, is a separate not-yet-transpiled function, the
    /// same already-accepted kind of forward-reference gap as every
    /// other not-yet-wired cross-function call). The one real new piece:
    /// `textureheight[side->bottomtexture]` indexes by a *struct field*
    /// (`bottomtexture: i16`, a concrete type fixed by `Side`'s own
    /// definition) rather than a literal or a fresh, still-inferred local
    /// (`sidenum[side ^ 1]`, already rendering fine with no cast) --
    /// Rust's `Index<usize>` needs an explicit `as usize` here that the
    /// generic `Expr::Index` fallback never had to add before. Clones the
    /// real nested `if (side->bottomtexture >= 0) if (textureheight[..]
    /// < minsize) minsize = textureheight[..];` straight out of
    /// `EV_DoFloor`'s own parsed AST (whichever of the two identical
    /// `getSide(..,0)`/`getSide(..,1)` occurrences `find_stmt` reaches
    /// first) -- no synthetic wrapper needed this time, the whole
    /// fragment renders as-is.
    #[test]
    fn test_textureheight_index_against_real_ev_do_floor() {
        let path = corpus_dir().join("p_floor.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_floor.c should parse");
        let f = find_function_def(&unit.items, "EV_DoFloor").expect("EV_DoFloor not found");
        let is_bottomtexture_check = |s: &Stmt| {
            matches!(s, Stmt::If { cond: Expr::Binary { op: BinaryOp::Ge, lhs, .. }, .. }
                if matches!(lhs.as_ref(), Expr::Member { field, .. } if field == "bottomtexture"))
        };
        let stmt = f
            .body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => find_stmt(s, &is_bottomtexture_check),
                BlockItem::Decl(_) => None,
            })
            .expect(
                "expected an `if (side->bottomtexture >= 0)` statement somewhere in EV_DoFloor",
            );

        let extra_cross_refs = field_types(&[("side", "SideId")]);
        let ctx = FnBodyContext {
            self_param: "",
            self_field_types: &HashMap::new(),
            extra_cross_ref_idents: &extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let rendered = render_stmt(stmt, &ctx, 0).expect("should render cleanly");
        assert_eq!(
            rendered,
            vec![
                "if world[side].bottomtexture >= 0 {".to_string(),
                "    if textureheight[world[side].bottomtexture as usize] < minsize {".to_string(),
                "        minsize = textureheight[world[side].bottomtexture as usize];".to_string(),
                "    }".to_string(),
                "}".to_string(),
            ]
        );
    }

    /// `EV_DoFloor`'s fifth piece: `raiseFloor24AndChange`'s own
    /// `sec->floorpic = line->frontsector->floorpic; sec->special =
    /// line->frontsector->special;` -- crossref chained through crossref
    /// via a *different* base shape than the already-built `sides[i].
    /// sector`: `line`'s own field `frontsector` (`LineId` -> `SectorId`)
    /// is a trigger function's own *parameter*, tracked only in
    /// `extra_cross_ref_idents` (no generic field-type registry exists
    /// for a parameter's own fields the way `self_field_types`/
    /// `ctor_field_types` cover `self`/a constructor-in-progress), so
    /// `line->frontsector` needed its own narrow by-name special case to
    /// let the *further* `.floorpic`/`.special` chain resolve through
    /// `world[...]` correctly. Clones both real assignment statements
    /// straight out of `EV_DoFloor`'s own parsed AST.
    #[test]
    fn test_frontsector_crossref_chain_against_real_ev_do_floor() {
        let path = corpus_dir().join("p_floor.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_floor.c should parse");
        let f = find_function_def(&unit.items, "EV_DoFloor").expect("EV_DoFloor not found");
        let is_frontsector_field_assign = |field_name: &'static str| {
            move |s: &Stmt| {
                matches!(s, Stmt::Expr(Some(Expr::Assign { lhs, rhs, .. }))
                    if matches!(lhs.as_ref(), Expr::Member { field, .. } if field == field_name)
                    && matches!(rhs.as_ref(), Expr::Member { base, field, .. }
                        if field == field_name
                        && matches!(base.as_ref(), Expr::Member { field, .. } if field == "frontsector")))
            }
        };
        let find_one = |field_name: &'static str| -> &Stmt {
            let pred = is_frontsector_field_assign(field_name);
            f.body
                .items
                .iter()
                .find_map(|item| match item {
                    BlockItem::Stmt(s) => find_stmt(s, &pred),
                    BlockItem::Decl(_) => None,
                })
                .unwrap_or_else(|| {
                    panic!("expected a `{field_name} = line->frontsector->{field_name};` statement somewhere in EV_DoFloor")
                })
        };
        let floorpic_stmt = find_one("floorpic").clone();
        let special_stmt = find_one("special").clone();

        let extra_cross_refs = field_types(&[("sec", "SectorId"), ("line", "LineId")]);
        let ctx = FnBodyContext {
            self_param: "",
            self_field_types: &HashMap::new(),
            extra_cross_ref_idents: &extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let mut rendered = render_stmt(&floorpic_stmt, &ctx, 0).expect("should render cleanly");
        rendered.extend(render_stmt(&special_stmt, &ctx, 0).expect("should render cleanly"));
        assert_eq!(
            rendered,
            vec![
                "world[sec].floorpic = world[world[line].frontsector].floorpic;".to_string(),
                "world[sec].special = world[world[line].frontsector].special;".to_string(),
            ]
        );
    }

    /// `EV_DoFloor`'s sixth and final piece: `lowerAndChange`'s own
    /// `if (getSide(secnum,i,0)->sector-sectors == secnum) { sec =
    /// getSector(secnum,i,1); ... } else { sec = getSector(secnum,i,0);
    /// ... }` -- pointer arithmetic (the already-built `X-sectors` idiom,
    /// `EV_VerticalDoor`'s own `secnum = sec-sectors;`) nested inside a
    /// comparison for the first time, chained directly off a call result
    /// rather than a local (`getSide(secnum,i,0)->sector`, needing the
    /// `Expr::Call` arm's new `getSide`/`getSector` -> cross-ref-typed
    /// special case, mirroring `twoSided`'s own by-name treatment), plus
    /// `sec` genuinely reassigned mid-construction (`sec = getSector(..);`,
    /// a plain local rebind -- harmless since `floor->sector = sec;`
    /// earlier in the case already copied the *old* value by-value, a
    /// `SectorId` being `Copy`, exactly like C's own pointer-copy
    /// semantics). Also surfaced a real, separate gap: a loop-exiting
    /// `break` reached as an ordinary statement (not `render_switch`'s own
    /// case-delimiter, which is peeled off before individual statements
    /// ever reach `render_stmt`) had no generic arm at all -- `Stmt::Break`
    /// now renders the same way `Stmt::Continue` already did. Clones the
    /// real `if`/`else` straight out of `EV_DoFloor`'s own parsed AST.
    /// **Combined with the five pieces above, `EV_DoFloor`'s entire body
    /// is now covered piece-by-piece** (not yet assembled/compiled as one
    /// whole function -- see docs/03_TRANSPILER.md).
    #[test]
    fn test_lower_and_change_sec_reassignment_against_real_ev_do_floor() {
        let path = corpus_dir().join("p_floor.c");
        let (_, unit) = parse_full(path.to_str().unwrap()).expect("p_floor.c should parse");
        let f = find_function_def(&unit.items, "EV_DoFloor").expect("EV_DoFloor not found");
        let is_side_sector_check = |s: &Stmt| {
            matches!(s, Stmt::If { cond: Expr::Binary { op: BinaryOp::Eq, lhs, rhs }, else_branch: Some(_), .. }
                if matches!(rhs.as_ref(), Expr::Ident(n) if n == "secnum")
                && matches!(lhs.as_ref(), Expr::Binary { op: BinaryOp::Sub, rhs: sub_rhs, .. }
                    if matches!(sub_rhs.as_ref(), Expr::Ident(n) if n == "sectors")))
        };
        let stmt = f
            .body
            .items
            .iter()
            .find_map(|item| match item {
                BlockItem::Stmt(s) => find_stmt(s, &is_side_sector_check),
                BlockItem::Decl(_) => None,
            })
            .expect(
                "expected an `if (getSide(..)->sector-sectors == secnum)` statement somewhere in EV_DoFloor",
            );

        let self_field_types = field_types(&[
            ("floordestheight", "FixedT"),
            ("texture", "i16"),
            ("newspecial", "i32"),
        ]);
        let extra_cross_refs = field_types(&[("sec", "SectorId")]);
        let ctx = FnBodyContext {
            self_param: "floor",
            self_field_types: &self_field_types,
            extra_cross_ref_idents: &extra_cross_refs,
            ctor_var: "",
            ctor_var_handle_name: "",
            ctor_field_types: &HashMap::new(),
            embedded_ctor: None,
            mutating_handle: None,
            same_handle_write: None,
            plain_int_locals: &HashSet::new(),
        };
        let rendered = render_stmt(stmt, &ctx, 0).expect("should render cleanly");
        assert_eq!(
            rendered,
            vec![
                "if world[getSide(secnum, i, 0)].sector.0 as i32 == secnum {".to_string(),
                "    sec = getSector(secnum, i, 1);".to_string(),
                "    if world[sec].floorheight == floor.floordestheight {".to_string(),
                "        floor.texture = world[sec].floorpic;".to_string(),
                "        floor.newspecial = world[sec].special;".to_string(),
                "        break;".to_string(),
                "    }".to_string(),
                "} else {".to_string(),
                "    sec = getSector(secnum, i, 0);".to_string(),
                "    if world[sec].floorheight == floor.floordestheight {".to_string(),
                "        floor.texture = world[sec].floorpic;".to_string(),
                "        floor.newspecial = world[sec].special;".to_string(),
                "        break;".to_string(),
                "    }".to_string(),
                "}".to_string(),
            ]
        );
    }

    /// Attempting `EV_DoFloor`'s full end-to-end assembly (via
    /// `render_trigger_fn`, the same integration point `EV_DoCeiling`/
    /// `EV_DoDoor`/`EV_DoPlat` already went through) surfaces a genuine
    /// latent bug in the original C, the same class already documented
    /// for `P_SpawnDoorCloseIn30`: `floor->newspecial` is *only* ever set
    /// deep inside the `lowerAndChange` case's own `for` loop, inside an
    /// `if (sec->floorheight == floor->floordestheight)` that isn't
    /// guaranteed to match any adjacent sector -- unlike `floor->texture`
    /// (also refined there, but *first* given an unconditional value
    /// right before the loop, `floor->texture = sec->floorpic;`).
    /// `T_MoveFloor` (already translated, `p_floor.c`) reads
    /// `floor->newspecial` unconditionally once a `lowerAndChange` floor
    /// reaches `pastdest`, regardless of whether that `for` loop ever
    /// found a match -- so a `lowerAndChange` floor whose sector has no
    /// two-sided neighbor at exactly its own destination height reads
    /// genuinely uninitialized `Z_Malloc` memory in the real original
    /// game, real reachable UB (confirmed by tracing the real call sites
    /// of both functions, not assumed). Since there's no well-defined C
    /// value to be faithful *to* here, `render_ctor_body`'s rejection is
    /// the correct, honest answer, not a gap to route around with a
    /// fabricated default -- unlike this test's *other* four switch-only
    /// fields (`direction`/`sector`/`speed`/`floordestheight`), which
    /// really are set by every one of `floor_e`'s 12 real call-site
    /// values across the whole corpus (grepped directly, not assumed);
    /// only the 13th variant, `donutRaise`, would reach the switch's own
    /// empty `default:` arm unset, and `EV_DoFloor` is never actually
    /// called with it anywhere in the corpus -- provably dead code, the
    /// same "never observed" reasoning already used for `EV_DoCeiling`'s
    /// own `Ceiling.olddirection`, so those four get real placeholder
    /// defaults instead.
    #[test]
    fn test_ev_do_floor_detects_missing_newspecial_default() {
        let params = field_types(&[("line", "LineId"), ("floortype", "i32")]);
        let locals = field_types(&[("sec", "SectorId")]);
        let ctor_field_types = field_types(&[
            ("r#type", "i32"),
            ("crush", "bool"),
            ("sector", "SectorId"),
            ("direction", "i32"),
            ("newspecial", "i32"),
            ("texture", "i16"),
            ("floordestheight", "FixedT"),
            ("speed", "FixedT"),
        ]);
        let field_defaults = field_types(&[
            ("direction", "0"),
            ("sector", "sec"),
            ("speed", "FLOORSPEED"),
            ("floordestheight", "world[sec].floorheight"),
            ("texture", "world[sec].floorpic"),
        ]);
        let err = render_trigger_fn(
            &corpus_dir(),
            "p_floor.c",
            "EV_DoFloor",
            &params,
            &locals,
            Some(CtorSpec {
                ctor_var: "floor",
                ctor_rust_type: "FloorMove",
                ctor_field_types: &ctor_field_types,
                field_defaults: &field_defaults,
            }),
            Some("i32"),
        )
        .expect_err("newspecial should be detected as missing a safe default");
        assert!(
            err.contains("newspecial"),
            "expected `newspecial` in: {err}"
        );
    }

    /// First `ActionFn::Mobj`-shaped function (`state_t.action`'s `acp1`
    /// variant, see `action_fn.rs`) translated end-to-end: `fn(mobj_t*)`
    /// is exactly `render_fn`'s own existing `self_param: &mut T` shape,
    /// needing no new renderer capability at all -- `actor->flags &=
    /// ~MF_SOLID;` is a plain self-field compound assignment
    /// (`render_assign_op`/`UnaryOp::BitNot` are already fully generic),
    /// and `MF_SOLID` itself is an already-mapped plain `i32` corpus
    /// constant (`mobjinfo_data.rs`), not a new identifier kind.
    #[test]
    fn test_a_fall_renders_exactly() {
        let field_types = field_types(&[("flags", "i32")]);
        let rendered = render_fn(&corpus_dir(), "p_enemy.c", "A_Fall", "Mobj", &field_types)
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Fall(actor: &mut Mobj, world: &mut World) {\n    actor.flags &= !MF_SOLID;\n}"
        );
    }

    /// Confirms a bare `self_param` (`actor`, not `actor->field`) passed
    /// directly as a call argument -- `S_StartSound`'s real first
    /// parameter is `void* origin`, most commonly a `mobj_t*` itself, not
    /// a field access off one -- already renders correctly through the
    /// fully generic `Expr::Ident`/`Expr::Call` paths, with no special
    /// case needed (unlike every prior `S_StartSound` call this renderer
    /// has seen, all of which passed `&world[..].soundorg` or `NULL`).
    #[test]
    fn test_a_xscream_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_XScream",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_XScream(actor: &mut Mobj, world: &mut World) {\n    S_StartSound(actor, sfx_slop);\n}"
        );
    }

    /// `actor->info->painsound` -- a two-level member chain through
    /// `info: &'static MobjInfo` (not a `World`-indexed cross-reference,
    /// so no `world[..]` wrapping applies) -- confirms `Expr::Member`'s
    /// fully generic fallback arm already resolves a chain through a
    /// plain-reference-typed self field with no new code, the same way
    /// it already resolves a chain through a cross-reference-typed one.
    #[test]
    fn test_a_pain_renders_exactly() {
        let field_types = field_types(&[("info", "&'static MobjInfo")]);
        let rendered = render_fn(&corpus_dir(), "p_enemy.c", "A_Pain", "Mobj", &field_types)
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Pain(actor: &mut Mobj, world: &mut World) {\n    if actor.info.painsound != 0 {\n        S_StartSound(actor, actor.info.painsound);\n    }\n}"
        );
    }

    /// `A_Hoof`/`A_Metal`/`A_BabyMetal` (`p_enemy.c`) -- identical two-
    /// statement shape (a sound, then a monster-generic `A_Chase(mo);`
    /// tail call), needing no new renderer capability: `A_Chase` isn't
    /// translated yet, but a bare forward-referencing call by name
    /// already renders correctly through the fully generic `Expr::Call`
    /// path (the same accepted "cross-function signature wiring...
    /// unresolved" gap already documented for other calls). Confirms
    /// `self_param`'s own name (`mo`, not `actor`) is picked up correctly
    /// too -- `first_param_name` was already name-agnostic, just not
    /// exercised by a real second name until now.
    #[test]
    fn test_a_hoof_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_Hoof",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Hoof(mo: &mut Mobj, world: &mut World) {\n    S_StartSound(mo, sfx_hoof);\n    A_Chase(mo);\n}"
        );
    }

    #[test]
    fn test_a_metal_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_Metal",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Metal(mo: &mut Mobj, world: &mut World) {\n    S_StartSound(mo, sfx_metal);\n    A_Chase(mo);\n}"
        );
    }

    /// First `ActionFn::Weapon`-shaped (`fn(player_t*, pspdef_t*)`)
    /// action functions translated, via the new `render_weapon_fn`.
    /// `player_t` isn't struct-mapped (see `render_weapon_fn`'s own
    /// docs), so `extralight`'s type is supplied directly, the same
    /// pattern already established for `EV_DoLockedDoor`.
    #[test]
    fn test_a_light0_renders_exactly() {
        let field_types = field_types(&[("extralight", "i32")]);
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_Light0", &field_types)
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Light0(player: &mut Player, psp: &mut PlayerSpriteState) {\n    player.extralight = 0;\n}"
        );
    }

    #[test]
    fn test_a_light1_renders_exactly() {
        let field_types = field_types(&[("extralight", "i32")]);
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_Light1", &field_types)
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Light1(player: &mut Player, psp: &mut PlayerSpriteState) {\n    player.extralight = 1;\n}"
        );
    }

    #[test]
    fn test_a_light2_renders_exactly() {
        let field_types = field_types(&[("extralight", "i32")]);
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_Light2", &field_types)
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Light2(player: &mut Player, psp: &mut PlayerSpriteState) {\n    player.extralight = 2;\n}"
        );
    }

    /// `A_OpenShotgun2`/`A_LoadShotgun2`/`A_CloseShotgun2` (`p_enemy.c`) --
    /// multi-line declarator style (`void\nA_OpenShotgun2\n( player_t*\t
    /// player,\n  pspdef_t*\tpsp )`), confirming `render_weapon_fn` isn't
    /// sensitive to that formatting. `player->mo` needs no entry at all in
    /// `player_field_types`: the generic `Expr::Member` fallback only
    /// treats a field as a cross-reference when its *registered* type is
    /// one of `CROSS_REF_TYPES`, so an unregistered field already falls
    /// through to plain, correct `player.mo` access. `A_CloseShotgun2`
    /// also calls not-yet-translated `A_ReFire(player,psp)` -- the same
    /// accepted forward-reference-call gap as `A_Hoof`'s own `A_Chase`.
    #[test]
    fn test_a_open_shotgun2_renders_exactly() {
        let rendered = render_weapon_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_OpenShotgun2",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_OpenShotgun2(player: &mut Player, psp: &mut PlayerSpriteState) {\n    S_StartSound(player.mo, sfx_dbopn);\n}"
        );
    }

    #[test]
    fn test_a_load_shotgun2_renders_exactly() {
        let rendered = render_weapon_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_LoadShotgun2",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_LoadShotgun2(player: &mut Player, psp: &mut PlayerSpriteState) {\n    S_StartSound(player.mo, sfx_dbload);\n}"
        );
    }

    #[test]
    fn test_a_close_shotgun2_renders_exactly() {
        let rendered = render_weapon_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_CloseShotgun2",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_CloseShotgun2(player: &mut Player, psp: &mut PlayerSpriteState) {\n    S_StartSound(player.mo, sfx_dbcls);\n    A_ReFire(player, psp);\n}"
        );
    }

    #[test]
    fn test_a_baby_metal_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_BabyMetal",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_BabyMetal(mo: &mut Mobj, world: &mut World) {\n    S_StartSound(mo, sfx_bspwlk);\n    A_Chase(mo);\n}"
        );
    }

    /// `A_PosAttack`/`A_SPosAttack`/`A_CPosAttack` (`p_enemy.c`) -- the
    /// first functions to check `!actor->target` (`Mobj.target: Option
    /// <Handle<Thinker>>`, per `struct_fields.rs`'s own self-referential-
    /// field mapping), generalizing `is_option_valued`'s `Expr::Member`
    /// handling beyond the one hardcoded `player`-named field: any self-
    /// struct field whose registered `self_field_types` entry is itself
    /// `Option<...>`-shaped now gets the same `.is_none()` treatment.
    /// Neither function dereferences *through* `target` any further than
    /// this truthiness check (that needs real `Arena` read access from
    /// inside a `Mobj`-shaped action function -- not yet built, see
    /// `A_FaceTarget`'s own deferred investigation in the module docs),
    /// so both stay within the fully generic call/arithmetic paths
    /// otherwise: a forward-referencing call to not-yet-translated
    /// `A_FaceTarget`, and C's familiar `(P_Random()-P_Random())<<20`/
    /// `((P_Random()%5)+1)*3` damage-roll idiom, exercised here for the
    /// first time with `%` inside an already-parenthesized sub-
    /// expression rather than at the top level. **A real bug caught by
    /// actually compiling this function's first-draft output with
    /// `rustc`, not just unit-testing its text**: `angle = actor->angle;`
    /// assigns `angle_t` (`u32`, `struct_fields.rs`'s own mapping) into a
    /// plain C `int` local -- Rust's own deferred-`let` inference wrongly
    /// picked up `u32` from this first use, then broke on the very next
    /// line's `i32`-valued `P_Random()` arithmetic (`u32 += i32`, a real
    /// `rustc` error, not hypothetical). Fixed generally, not just here:
    /// `FnBodyContext::plain_int_locals` (`render_fn`'s own
    /// `collect_plain_int_locals`) tracks which locals are genuinely
    /// declared `int`, and assigning a `u32`-registered `self_param`
    /// field straight into one now renders an explicit `as i32`,
    /// matching C's own implicit bit-reinterpreting conversion -- the
    /// same idea as `EV_VerticalDoor`'s own `secnum = sec-sectors;`
    /// needing `.0 as i32`.
    #[test]
    fn test_a_pos_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>"), ("angle", "u32")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_PosAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_PosAttack(actor: &mut Mobj, world: &mut World) {\n    \
             let mut angle;\n    \
             let mut damage;\n    \
             let mut slope;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             angle = actor.angle as i32;\n    \
             slope = P_AimLineAttack(actor, angle, MISSILERANGE);\n    \
             S_StartSound(actor, sfx_pistol);\n    \
             angle += P_Random() - P_Random() << 20;\n    \
             damage = (P_Random() % 5 + 1) * 3;\n    \
             P_LineAttack(actor, angle, MISSILERANGE, slope, damage);\n\
             }"
        );
    }

    /// `A_SPosAttack` -- the same `!actor->target` check plus a real
    /// `for` loop (`render_for`'s existing plain-assignment-init shape,
    /// `EV_DoFloor`'s own precedent) firing three shots, plus the same
    /// `bangle = actor->angle;` `as i32` reinterpretation as `A_PosAttack`.
    #[test]
    fn test_a_spos_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>"), ("angle", "u32")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_SPosAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_SPosAttack(actor: &mut Mobj, world: &mut World) {\n    \
             let mut i;\n    \
             let mut angle;\n    \
             let mut bangle;\n    \
             let mut damage;\n    \
             let mut slope;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             S_StartSound(actor, sfx_shotgn);\n    \
             A_FaceTarget(actor);\n    \
             bangle = actor.angle as i32;\n    \
             slope = P_AimLineAttack(actor, bangle, MISSILERANGE);\n    \
             i = 0;\n    \
             while i < 3 {\n        \
             angle = bangle + (P_Random() - P_Random() << 20);\n        \
             damage = (P_Random() % 5 + 1) * 3;\n        \
             P_LineAttack(actor, angle, MISSILERANGE, slope, damage);\n        \
             i += 1;\n    \
             }\n\
             }"
        );
    }

    /// `A_CPosAttack` -- `A_SPosAttack`'s own single-shot sibling (no
    /// loop, otherwise byte-for-byte the same damage-roll shape).
    #[test]
    fn test_a_cpos_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>"), ("angle", "u32")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_CPosAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_CPosAttack(actor: &mut Mobj, world: &mut World) {\n    \
             let mut angle;\n    \
             let mut bangle;\n    \
             let mut damage;\n    \
             let mut slope;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             S_StartSound(actor, sfx_shotgn);\n    \
             A_FaceTarget(actor);\n    \
             bangle = actor.angle as i32;\n    \
             slope = P_AimLineAttack(actor, bangle, MISSILERANGE);\n    \
             angle = bangle + (P_Random() - P_Random() << 20);\n    \
             damage = (P_Random() % 5 + 1) * 3;\n    \
             P_LineAttack(actor, angle, MISSILERANGE, slope, damage);\n\
             }"
        );
    }

    /// `A_PainAttack` -- reuses the same `!actor->target` check, then
    /// two forward-referencing calls (`A_FaceTarget`, not-yet-translated
    /// `A_PainShootSkull`), no new capability needed.
    #[test]
    fn test_a_pain_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_PainAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_PainAttack(actor: &mut Mobj, world: &mut World) {\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             A_PainShootSkull(actor, actor.angle);\n\
             }"
        );
    }

    /// `A_PainDie` -- `actor->angle+ANG90`/`+ANG180`/`+ANG270`: `ANG90`
    /// and friends (`tables.h`) are plain `#define`d hex-literal macros,
    /// not enum constants, so they render as bare pass-through
    /// identifiers the same way `MISSILERANGE` already does -- this
    /// renderer never evaluates a macro, just emits whatever identifier
    /// text the AST already has, trusting some later stage to have a
    /// real Rust `const` with that same name (the same accepted
    /// forward-reference gap as every other not-yet-wired global).
    #[test]
    fn test_a_pain_die_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_PainDie",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_PainDie(actor: &mut Mobj, world: &mut World) {\n    \
             A_Fall(actor);\n    \
             A_PainShootSkull(actor, actor.angle + ANG90);\n    \
             A_PainShootSkull(actor, actor.angle + ANG180);\n    \
             A_PainShootSkull(actor, actor.angle + ANG270);\n\
             }"
        );
    }

    /// `A_Scream` -- a `switch` whose very first arm (`case 0: return;`)
    /// exercises the `render_switch` fallthrough fix from the previous
    /// commit against real corpus code (not just a hand-built repro),
    /// alongside two ordinary shared-case-label groups and a `default`.
    /// `actor->type==MT_SPIDER || actor->type == MT_CYBORG` is the first
    /// real corpus `==` comparison nested inside `||` this renderer has
    /// hit -- ordinary precedence-aware `Binary` rendering, no new code.
    #[test]
    fn test_a_scream_renders_exactly() {
        let field_types = field_types(&[("info", "&'static MobjInfo")]);
        let rendered = render_fn(&corpus_dir(), "p_enemy.c", "A_Scream", "Mobj", &field_types)
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Scream(actor: &mut Mobj, world: &mut World) {\n    \
             let mut sound;\n    \
             match actor.info.deathsound {\n        \
             0 => {\n            \
             return;\n        \
             }\n        \
             sfx_podth1 | sfx_podth2 | sfx_podth3 => {\n            \
             sound = sfx_podth1 + P_Random() % 3;\n        \
             }\n        \
             sfx_bgdth1 | sfx_bgdth2 => {\n            \
             sound = sfx_bgdth1 + P_Random() % 2;\n        \
             }\n        \
             _ => {\n            \
             sound = actor.info.deathsound;\n        \
             }\n    \
             }\n    \
             if actor.r#type == MT_SPIDER || actor.r#type == MT_CYBORG {\n        \
             S_StartSound(None, sound);\n    \
             } else {\n        \
             S_StartSound(actor, sound);\n    \
             }\n\
             }"
        );
    }

    /// `A_BspiAttack`/`A_CyberAttack` (`p_enemy.c`) -- the first
    /// functions to pass `actor->target` itself (not a field reached
    /// *through* it) as a bare call argument to a not-yet-translated
    /// function (`P_SpawnMissile`) -- needs no new capability at all,
    /// since the already-generic `Expr::Member`/`Expr::Call` paths
    /// already resolve a bare `Option<Handle<Thinker>>`-valued field
    /// read correctly; only *dereferencing through* it needs the not-
    /// yet-built `Arena` read access this module still defers (see
    /// `A_FaceTarget`).
    #[test]
    fn test_a_bspi_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_BspiAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_BspiAttack(actor: &mut Mobj, world: &mut World) {\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             P_SpawnMissile(actor, actor.target, MT_ARACHPLAZ);\n\
             }"
        );
    }

    #[test]
    fn test_a_cyber_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_CyberAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_CyberAttack(actor: &mut Mobj, world: &mut World) {\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             P_SpawnMissile(actor, actor.target, MT_ROCKET);\n\
             }"
        );
    }

    /// `A_TroopAttack`/`A_HeadAttack` -- a melee-or-missile branch via
    /// `if (P_CheckMeleeRange (actor))`, the first bare-`boolean`-call
    /// condition this renderer has hit: `P_CheckMeleeRange`'s own real
    /// declared return type is `boolean` (not `int`, unlike `twoSided`),
    /// which already maps to Rust's native `bool` -- used directly with
    /// no `!= 0` cast, a new narrow `render_bool_expr` arm.
    #[test]
    fn test_a_troop_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_TroopAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_TroopAttack(actor: &mut Mobj, world: &mut World) {\n    \
             let mut damage;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             if P_CheckMeleeRange(actor) {\n        \
             S_StartSound(actor, sfx_claw);\n        \
             damage = (P_Random() % 8 + 1) * 3;\n        \
             P_DamageMobj(actor.target, actor, actor, damage);\n        \
             return;\n    \
             }\n    \
             P_SpawnMissile(actor, actor.target, MT_TROOPSHOT);\n\
             }"
        );
    }

    /// `A_SargAttack` -- the melee branch has no `return`/missile
    /// fallback at all (a pure gate: does nothing if out of range).
    #[test]
    fn test_a_sarg_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_SargAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_SargAttack(actor: &mut Mobj, world: &mut World) {\n    \
             let mut damage;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             if P_CheckMeleeRange(actor) {\n        \
             damage = (P_Random() % 10 + 1) * 4;\n        \
             P_DamageMobj(actor.target, actor, actor, damage);\n    \
             }\n\
             }"
        );
    }

    #[test]
    fn test_a_head_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_HeadAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_HeadAttack(actor: &mut Mobj, world: &mut World) {\n    \
             let mut damage;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             if P_CheckMeleeRange(actor) {\n        \
             damage = (P_Random() % 6 + 1) * 10;\n        \
             P_DamageMobj(actor.target, actor, actor, damage);\n        \
             return;\n    \
             }\n    \
             P_SpawnMissile(actor, actor.target, MT_HEADSHOT);\n\
             }"
        );
    }

    /// `A_BruisAttack` -- the one melee/missile attack in this group
    /// with no `A_FaceTarget` call at all anywhere in its body, confirmed
    /// directly against the real source rather than assumed missing --
    /// translated exactly as the original has it, quirk and all.
    #[test]
    fn test_a_bruis_attack_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_BruisAttack",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_BruisAttack(actor: &mut Mobj, world: &mut World) {\n    \
             let mut damage;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             if P_CheckMeleeRange(actor) {\n        \
             S_StartSound(actor, sfx_claw);\n        \
             damage = (P_Random() % 8 + 1) * 10;\n        \
             P_DamageMobj(actor.target, actor, actor, damage);\n        \
             return;\n    \
             }\n    \
             P_SpawnMissile(actor, actor.target, MT_BRUISERSHOT);\n\
             }"
        );
    }

    /// `A_VileStart` -- a single-statement sound call, same shape as
    /// `A_XScream`.
    #[test]
    fn test_a_vile_start_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_VileStart",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_VileStart(actor: &mut Mobj, world: &mut World) {\n    S_StartSound(actor, sfx_vilatk);\n}"
        );
    }

    /// `A_StartFire`/`A_FireCrackle` -- sound-then-forward-reference-call,
    /// the same two-statement shape as `A_Hoof`/`A_Metal`, just calling
    /// not-yet-translated `A_Fire` instead of `A_Chase`.
    #[test]
    fn test_a_start_fire_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_StartFire",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_StartFire(actor: &mut Mobj, world: &mut World) {\n    S_StartSound(actor, sfx_flamst);\n    A_Fire(actor);\n}"
        );
    }

    #[test]
    fn test_a_fire_crackle_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_FireCrackle",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_FireCrackle(actor: &mut Mobj, world: &mut World) {\n    S_StartSound(actor, sfx_flame);\n    A_Fire(actor);\n}"
        );
    }

    /// `A_BrainPain` -- a single `S_StartSound(NULL, ..)` call, `A_Look`'s
    /// own "full volume" idiom, confirming `NULL` -> `None` still needs
    /// no special casing when it's the *only* statement in the function.
    #[test]
    fn test_a_brain_pain_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_BrainPain",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_BrainPain(mo: &mut Mobj, world: &mut World) {\n    S_StartSound(None, sfx_bospn);\n}"
        );
    }

    /// `A_BrainDie` -- a single bare forward-referencing call with no
    /// arguments at all, the simplest shape yet.
    #[test]
    fn test_a_brain_die_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_BrainDie",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_BrainDie(mo: &mut Mobj, world: &mut World) {\n    G_ExitLevel();\n}"
        );
    }

    /// `A_SpawnSound` -- `A_Hoof`'s own sound-then-tail-call shape, just
    /// calling `A_SpawnFly` instead of `A_Chase`.
    #[test]
    fn test_a_spawn_sound_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_SpawnSound",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_SpawnSound(mo: &mut Mobj, world: &mut World) {\n    S_StartSound(mo, sfx_boscub);\n    A_SpawnFly(mo);\n}"
        );
    }

    /// `A_PlayerScream` -- the first function with a scalar local
    /// declared *with* an initializer (`int sound = sfx_pldeth;`,
    /// `render_decl`'s existing inline-initializer support, `EV_DoFloor`'s
    /// own `minsize` precedent) outside any constructor context, plus a
    /// `&&` of an unregistered global (`gamemode`, rendered as a bare
    /// pass-through identifier) and a self-field comparison against a
    /// negative literal.
    #[test]
    fn test_a_player_scream_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_PlayerScream",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_PlayerScream(mo: &mut Mobj, world: &mut World) {\n    \
             let mut sound = sfx_pldeth;\n    \
             if gamemode == commercial && mo.health < -50 {\n        \
             sound = sfx_pdiehi;\n    \
             }\n    \
             S_StartSound(mo, sound);\n\
             }"
        );
    }

    /// `A_Tracer` -- the first function needing real `Arena` *write*
    /// access from inside a `Mobj`-shaped action function: `th =
    /// P_SpawnMobj(...); th->momz = ...; th->tics -= ...;`. `th` is
    /// registered `"Handle<Thinker>"`-typed by the new
    /// `collect_spawn_mobj_locals` scan (mirroring `collect_target_
    /// tracer_aliases`'s own shape, just triggered by a `P_SpawnMobj`
    /// call instead of a `target`/`tracer` read); every subsequent
    /// `th->field` read/write resolves through the new generalized
    /// `Handle<Thinker>`-base arms in `render_expr`/`render_expr_stmt`
    /// (a fresh `thinkers.get`/`get_mut` at each point of use, the same
    /// per-access-borrow discipline `render_existing_thinker_mutation`
    /// already established for `EV_VerticalDoor`'s own reused-thinker
    /// case -- a single hoisted binding was already proven wrong by
    /// `rustc` there, so it's never attempted here either). Also the
    /// first function needing a *compound* assign through a `Handle<
    /// Thinker>`-typed base (`th->tics -= P_Random()&3;`), which the
    /// sibling `mutating_handle` write arm never needed (`EV_VerticalDoor`'s
    /// own reused-thinker writes are all plain `=`) -- this one renders
    /// whatever real `AssignOp` the source used instead. `dest =
    /// actor->tracer;` reuses the already-existing target/tracer local-
    /// alias mechanism unchanged. New `FixedT` arithmetic surfaced by
    /// this function's own tail (`40*FRACUNIT`, `dist / actor->info->
    /// speed`, `FRACUNIT/8`, `actor->momz -= FRACUNIT/8`): `fixed_t` is
    /// a bare `typedef int` in the original, so none of these ever go
    /// through the rescaling `FixedMul`/`FixedDiv` -- `runtime/fixed.rs`
    /// gains `Mul<FixedT> for i32`/`Mul<i32> for FixedT`/`Div<i32> for
    /// FixedT` (raw representation arithmetic, the same idea `Add`/`Sub`
    /// already model for `+`/`-`) plus `AddAssign`/`SubAssign`. `dist`
    /// itself is treated as a plain scalar throughout (declared `fixed_t`
    /// in the original, but only ever divided by/compared against plain
    /// `int`s after `P_AproxDistance` computes it here) -- the
    /// verification harness's own `P_AproxDistance` stub returns `i32`
    /// to match, the same "stub signature matches this real call site's
    /// own usage, not necessarily the callee's eventual one" precedent
    /// already documented for `S_StartSound` (`EV_VerticalDoor`).
    /// `finecosine[exact]`/`finesine[exact]` need an explicit `as usize`
    /// even though `exact` is a plain index identifier (unlike
    /// `sidenum[side^1]`'s fresh, single-purpose local): `exact` is
    /// *also* used earlier in real `u32` arithmetic against `actor->
    /// angle`, so Rust can't freely infer it `usize` -- narrowly matched
    /// by the array's own name (`finecosine`/`finesine`), the same
    /// "hand-match the one real array identifier" style as `sides`/
    /// `sectors`/`textureheight`. Verified compiling for real (`rustc
    /// --edition 2021 --crate-type lib`) against hand-written stand-in
    /// `World`/`Thinker`/`Mobj`/`MobjInfo`/`Arena`/`Handle`/`FixedT`
    /// shapes and stub `P_SpawnPuff`/`P_SpawnMobj`/`R_PointToAngle2`/
    /// `P_AproxDistance`/`FixedMul`/`P_Random` functions -- zero errors.
    #[test]
    fn test_a_tracer_renders_exactly() {
        let field_types = field_types(&[
            ("x", "FixedT"),
            ("y", "FixedT"),
            ("z", "FixedT"),
            ("momx", "FixedT"),
            ("momy", "FixedT"),
            ("momz", "FixedT"),
            ("tics", "i32"),
            ("angle", "u32"),
            ("tracer", "Option<Handle<Thinker>>"),
            ("info", "&'static MobjInfo"),
        ]);
        let rendered = render_fn(&corpus_dir(), "p_enemy.c", "A_Tracer", "Mobj", &field_types)
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Tracer(actor: &mut Mobj, world: &mut World, thinkers: &mut Arena<Thinker>) {\n    \
             let mut exact;\n    \
             let mut dist;\n    \
             let mut slope;\n    \
             let mut dest;\n    \
             let mut th;\n    \
             if (gametic & 3) != 0 {\n        \
             return;\n    \
             }\n    \
             P_SpawnPuff(actor.x, actor.y, actor.z);\n    \
             th = P_SpawnMobj(actor.x - actor.momx, actor.y - actor.momy, actor.z, MT_SMOKE);\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(th) { m.momz = FRACUNIT; };\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(th) { m.tics -= P_Random() & 3; };\n    \
             if match thinkers.get(th) { Some(Thinker::Mobj(m)) => m.tics, _ => unreachable!() } < 1 {\n        \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(th) { m.tics = 1; };\n    \
             }\n    \
             dest = actor.tracer;\n    \
             if dest.is_none() || match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.health, _ => unreachable!() } <= 0 {\n        \
             return;\n    \
             }\n    \
             exact = R_PointToAngle2(actor.x, actor.y, match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.x, _ => unreachable!() }, match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.y, _ => unreachable!() });\n    \
             if exact != actor.angle {\n        \
             if exact - actor.angle > 0x80000000 {\n            \
             actor.angle -= (TRACEANGLE) as u32;\n            \
             if exact - actor.angle < 0x80000000 {\n                \
             actor.angle = exact;\n            \
             }\n        \
             } else {\n            \
             actor.angle += (TRACEANGLE) as u32;\n            \
             if exact - actor.angle > 0x80000000 {\n                \
             actor.angle = exact;\n            \
             }\n        \
             }\n    \
             }\n    \
             exact = actor.angle >> ANGLETOFINESHIFT;\n    \
             actor.momx = FixedMul(actor.info.speed, finecosine[exact as usize]);\n    \
             actor.momy = FixedMul(actor.info.speed, finesine[exact as usize]);\n    \
             dist = P_AproxDistance(match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.x, _ => unreachable!() } - actor.x, match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.y, _ => unreachable!() } - actor.y);\n    \
             dist = dist / actor.info.speed;\n    \
             if dist < 1 {\n        \
             dist = 1;\n    \
             }\n    \
             slope = (match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.z, _ => unreachable!() } + 40 * FRACUNIT - actor.z) / dist;\n    \
             if slope < actor.momz {\n        \
             actor.momz -= FRACUNIT / 8;\n    \
             } else {\n        \
             actor.momz += FRACUNIT / 8;\n    \
             }\n\
             }"
        );
    }

    /// `A_VileTarget` -- the second function needing the new `P_SpawnMobj`
    /// write-access mechanism, and the first needing the function's own
    /// *receiver* as a `Handle<Thinker>` *value* (`fog->target = actor;`,
    /// storing `actor` itself into the freshly-spawned `fog`'s own
    /// `target` field) -- a genuinely new need, on the same footing as
    /// `body_has_self_removal` first needing a self-removing tick
    /// function's own handle, just to store it elsewhere rather than
    /// remove it (`body_has_self_handle_value`, a new `render_fn_impl`
    /// signature-extension case adding a bare `handle: Handle<Thinker>`
    /// parameter, reusing self-removal's own fixed name without its
    /// `arena: &mut Arena<Thinker>` companion, since nothing here calls
    /// `Arena::remove`). `actor->tracer = fog;` (self's own `tracer`
    /// field, a bare `Handle<Thinker>` RHS) needed the mirror-image
    /// generalization: `render_expr_stmt`'s existing `specialdata`-only
    /// `Some(..)`-wrap now also covers `target`/`tracer` self-writes
    /// whenever the RHS is a registered `Handle<Thinker>` local. `fog->
    /// tracer = actor->target;` needs no wrap at all -- `actor->target`
    /// (a bare, non-chained self-field read) is already `Option<Handle<
    /// Thinker>>`-shaped, confirming the new write arm doesn't double-
    /// wrap an already-`Option` RHS. A genuine latent bug in the
    /// original C, preserved faithfully rather than silently corrected:
    /// `fog = P_SpawnMobj (actor->target->x, actor->target->x,
    /// actor->target->z, MT_FIRE);` passes `actor->target->x` twice --
    /// the second argument should almost certainly be `actor->target->y`,
    /// but the real corpus source (`p_enemy.c`) really does say `x`
    /// twice; translated as-written, not "fixed," matching this project's
    /// own standing practice for genuine original-game defects. Verified
    /// compiling for real alongside `A_Tracer`, same stand-in shapes.
    #[test]
    fn test_a_vile_target_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_VileTarget",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_VileTarget(actor: &mut Mobj, world: &mut World, thinkers: &mut Arena<Thinker>, handle: Handle<Thinker>) {\n    \
             let mut fog;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             fog = P_SpawnMobj(match thinkers.get(actor.target.unwrap()) { Some(Thinker::Mobj(m)) => m.x, _ => unreachable!() }, match thinkers.get(actor.target.unwrap()) { Some(Thinker::Mobj(m)) => m.x, _ => unreachable!() }, match thinkers.get(actor.target.unwrap()) { Some(Thinker::Mobj(m)) => m.z, _ => unreachable!() }, MT_FIRE);\n    \
             actor.tracer = Some(fog);\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(fog) { m.target = Some(handle); };\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(fog) { m.tracer = actor.target; };\n    \
             A_Fire(fog);\n\
             }"
        );
    }

    /// `A_BrainExplode` -- a second, simpler `P_SpawnMobj`-write-access
    /// caller than `A_Tracer`'s own (no target/tracer dereferencing, no
    /// self-handle-value need, just a straight-line spawn-then-mutate),
    /// but the one that actually surfaced the mixed `FixedT`/`i32`
    /// arithmetic gap this session's own `runtime/fixed.rs` additions
    /// exist for: `x = mo->x + (P_Random()-P_Random())*2048;` and `z =
    /// 128 + P_Random()*2*FRACUNIT;` need `Add<i32>`/`Add<FixedT>` in
    /// both operand orders (the real corpus uses both), and `th->momz =
    /// P_Random()*512;` -- a *plain* `i32` expression with no `FixedT`
    /// source in it anywhere -- needs the new `expr_is_fixed_t_valued`
    /// check to wrap it `FixedT(..)` before assigning into a `FixedT`
    /// field, the same "C silently reinterprets the bits" idiom the
    /// `angle_t`/plain-`int` pair already established, just the mirror
    /// direction (plain `int` *into* a narrower Rust type, not out of
    /// one). Verified compiling for real.
    #[test]
    fn test_a_brain_explode_renders_exactly() {
        let field_types = field_types(&[("x", "FixedT"), ("momz", "FixedT"), ("tics", "i32")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_BrainExplode",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_BrainExplode(mo: &mut Mobj, world: &mut World, thinkers: &mut Arena<Thinker>) {\n    \
             let mut x;\n    \
             let mut y;\n    \
             let mut z;\n    \
             let mut th;\n    \
             x = mo.x + (P_Random() - P_Random()) * 2048;\n    \
             y = mo.y;\n    \
             z = 128 + P_Random() * 2 * FRACUNIT;\n    \
             th = P_SpawnMobj(x, y, z, MT_ROCKET);\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(th) { m.momz = FixedT(P_Random() * 512); };\n    \
             P_SetMobjState(th, S_BRAINEXPLODE1);\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(th) { m.tics -= P_Random() & 7; };\n    \
             if match thinkers.get(th) { Some(Thinker::Mobj(m)) => m.tics, _ => unreachable!() } < 1 {\n        \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(th) { m.tics = 1; };\n    \
             }\n\
             }"
        );
    }

    /// `A_Explode` -- a single opaque forward-referencing call passing
    /// `thingy->target` bare (no dereference), the same already-
    /// established shape every other `P_RadiusAttack`/`P_DamageMobj`-
    /// style call uses.
    #[test]
    fn test_a_explode_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_Explode",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Explode(thingy: &mut Mobj, world: &mut World) {\n    \
             P_RadiusAttack(thingy, thingy.target, 128);\n\
             }"
        );
    }

    #[test]
    fn test_a_skel_whoosh_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_SkelWhoosh",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_SkelWhoosh(actor: &mut Mobj, world: &mut World) {\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             S_StartSound(actor, sfx_skeswg);\n\
             }"
        );
    }

    #[test]
    fn test_a_skel_fist_renders_exactly() {
        let field_types = field_types(&[("target", "Option<Handle<Thinker>>")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_SkelFist",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_SkelFist(actor: &mut Mobj, world: &mut World) {\n    \
             let mut damage;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             if P_CheckMeleeRange(actor) {\n        \
             damage = (P_Random() % 10 + 1) * 6;\n        \
             S_StartSound(actor, sfx_skepch);\n        \
             P_DamageMobj(actor.target, actor, actor, damage);\n    \
             }\n\
             }"
        );
    }

    #[test]
    fn test_a_fat_raise_renders_exactly() {
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_FatRaise",
            "Mobj",
            &HashMap::new(),
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_FatRaise(actor: &mut Mobj, world: &mut World) {\n    \
             A_FaceTarget(actor);\n    \
             S_StartSound(actor, sfx_manatk);\n\
             }"
        );
    }

    /// `A_BFGsound` -- identical shape to the already-covered
    /// `A_OpenShotgun2`/`A_LoadShotgun2` (a single `S_StartSound(player->
    /// mo, sfx)` call), confirming `render_weapon_fn` generalizes with no
    /// changes needed.
    #[test]
    fn test_a_bfgsound_renders_exactly() {
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_BFGsound", &HashMap::new())
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_BFGsound(player: &mut Player, psp: &mut PlayerSpriteState) {\n    \
             S_StartSound(player.mo, sfx_bfg);\n\
             }"
        );
    }

    /// `A_CheckReload` -- a single opaque call, `P_CheckAmmo(player)`; the
    /// real corpus body also has an `#if 0`'d-out `P_SetPsprite` call
    /// right after it, confirming the parser's own preprocessing already
    /// strips dead `#if 0` blocks before this renderer ever sees them
    /// (the same "confirmed by direct read, not assumed" `#if 0` handling
    /// already established for `slidedoor_t`/`FixedDiv2`'s own dead
    /// alternate path).
    #[test]
    fn test_a_check_reload_renders_exactly() {
        let rendered =
            render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_CheckReload", &HashMap::new())
                .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_CheckReload(player: &mut Player, psp: &mut PlayerSpriteState) {\n    \
             P_CheckAmmo(player);\n\
             }"
        );
    }

    /// `A_ReFire` -- the function that motivated `render_binary_operand`'s
    /// own new arithmetic-as-bool generalization: `(player->cmd.buttons &
    /// BT_ATTACK) && player->pendingweapon == wp_nochange &&
    /// player->health` chains a bitwise-flag test and a bare `int` field
    /// truthiness check together with an already-`bool` comparison, none
    /// of which previously rendered as valid Rust `bool` text inside a
    /// `&&` chain. Also the first standalone (non-condition, non-for-step)
    /// `player->refire++;` statement, exercising `render_expr`'s new
    /// generic `PostIncDec`/`PreIncDec` arm.
    #[test]
    fn test_a_re_fire_renders_exactly() {
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_ReFire", &HashMap::new())
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_ReFire(player: &mut Player, psp: &mut PlayerSpriteState) {\n    \
             if (player.cmd.buttons & BT_ATTACK) != 0 && player.pendingweapon == wp_nochange && player.health != 0 {\n        \
             player.refire += 1;\n        \
             P_FireWeapon(player);\n    \
             } else {\n        \
             player.refire = 0;\n        \
             P_CheckAmmo(player);\n    \
             }\n\
             }"
        );
    }

    /// `A_Lower`/`A_Raise`/`A_GunFlash`/`A_FireMissile`/`A_FireBFG`/
    /// `A_FirePlasma` (`p_pspr.c`) -- six more `render_weapon_fn` bodies,
    /// all rendering correctly through already-existing machinery with no
    /// new capability: `psp->sy`/`weaponinfo[player->readyweapon].field`
    /// both compose for free out of pieces this module already has --
    /// `psp` (an ordinary, unregistered second parameter) falls through
    /// the generic `Expr::Member` case exactly like `player->mo` already
    /// does; `weaponinfo[..]` is a plain, unregistered global array (no
    /// `World`/cross-ref wrapping needed, unlike `sectors`/`sides`), so
    /// indexing it and then reading a further field off the result is
    /// just two ordinary, already-generic `Expr::Index`/`Expr::Member`
    /// steps stacked -- the existing `as usize` cast (triggered whenever
    /// an index expression is itself an `Expr::Member`) already fires
    /// correctly for `player->readyweapon` used as an index into it.
    /// `player->ammo[weaponinfo[player->readyweapon].ammo]--;` is the
    /// first real corpus use of the generic standalone `PostIncDec`
    /// statement arm against a doubly-nested index chain rather than a
    /// bare identifier. **Deliberately not attempted alongside these**:
    /// `A_FirePistol`/`A_FireShotgun`/`A_FireCGun` all pass `!player->
    /// refire` (negating a plain, non-`bool` `int` field) directly as a
    /// bare function-call *argument* -- a third context beyond the two
    /// this renderer already has truthiness-aware negation for (a
    /// top-level `if` condition, via `render_bool_expr`; a `&&`/`||`
    /// chain operand, via `render_binary_operand`) -- so today it would
    /// silently render as Rust's bitwise `!` on an `i32` (wrong) rather
    /// than `== 0`, the same trap already documented and deliberately
    /// left alone elsewhere in this module. Left for a future increment
    /// once a real need justifies generalizing negation to call-argument
    /// position, rather than guessed at here.
    #[test]
    fn test_a_lower_renders_exactly() {
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_Lower", &HashMap::new())
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Lower(player: &mut Player, psp: &mut PlayerSpriteState) {\n    \
             psp.sy += LOWERSPEED;\n    \
             if psp.sy < WEAPONBOTTOM {\n        \
             return;\n    \
             }\n    \
             if player.playerstate == PST_DEAD {\n        \
             psp.sy = WEAPONBOTTOM;\n        \
             return;\n    \
             }\n    \
             if player.health == 0 {\n        \
             P_SetPsprite(player, ps_weapon, S_NULL);\n        \
             return;\n    \
             }\n    \
             player.readyweapon = player.pendingweapon;\n    \
             P_BringUpWeapon(player);\n\
             }"
        );
    }

    #[test]
    fn test_a_raise_renders_exactly() {
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_Raise", &HashMap::new())
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Raise(player: &mut Player, psp: &mut PlayerSpriteState) {\n    \
             let mut newstate;\n    \
             psp.sy -= RAISESPEED;\n    \
             if psp.sy > WEAPONTOP {\n        \
             return;\n    \
             }\n    \
             psp.sy = WEAPONTOP;\n    \
             newstate = weaponinfo[player.readyweapon as usize].readystate;\n    \
             P_SetPsprite(player, ps_weapon, newstate);\n\
             }"
        );
    }

    #[test]
    fn test_a_gun_flash_renders_exactly() {
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_GunFlash", &HashMap::new())
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_GunFlash(player: &mut Player, psp: &mut PlayerSpriteState) {\n    \
             P_SetMobjState(player.mo, S_PLAY_ATK2);\n    \
             P_SetPsprite(player, ps_flash, weaponinfo[player.readyweapon as usize].flashstate);\n\
             }"
        );
    }

    #[test]
    fn test_a_fire_missile_renders_exactly() {
        let rendered =
            render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_FireMissile", &HashMap::new())
                .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_FireMissile(player: &mut Player, psp: &mut PlayerSpriteState) {\n    \
             player.ammo[weaponinfo[player.readyweapon as usize].ammo as usize] -= 1;\n    \
             P_SpawnPlayerMissile(player.mo, MT_ROCKET);\n\
             }"
        );
    }

    #[test]
    fn test_a_fire_bfg_renders_exactly() {
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_FireBFG", &HashMap::new())
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_FireBFG(player: &mut Player, psp: &mut PlayerSpriteState) {\n    \
             player.ammo[weaponinfo[player.readyweapon as usize].ammo as usize] -= BFGCELLS;\n    \
             P_SpawnPlayerMissile(player.mo, MT_BFG);\n\
             }"
        );
    }

    #[test]
    fn test_a_fire_plasma_renders_exactly() {
        let rendered = render_weapon_fn(&corpus_dir(), "p_pspr.c", "A_FirePlasma", &HashMap::new())
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_FirePlasma(player: &mut Player, psp: &mut PlayerSpriteState) {\n    \
             player.ammo[weaponinfo[player.readyweapon as usize].ammo as usize] -= 1;\n    \
             P_SetPsprite(player, ps_flash, weaponinfo[player.readyweapon as usize].flashstate + (P_Random() & 1));\n    \
             P_SpawnPlayerMissile(player.mo, MT_PLASMA);\n\
             }"
        );
    }

    /// `A_BrainScream` -- `A_BrainExplode`'s own already-translated
    /// idiom (spawn a rocket-shaped mobj, give it a random downward tics
    /// countdown clamped to at least 1) repeated across a horizontal
    /// slice of the map, motivating `render_for_step`'s own generalization
    /// to a compound-assign step (`x += FRACUNIT*8`, not `x++`/`x--`) --
    /// the real corpus counterpart `docs/03_TRANSPILER.md`'s own "not yet
    /// done" list already flagged this gap against. `y`/`z` are freshly
    /// recomputed every iteration (not hoisted out of the loop), and `x`
    /// itself is the loop's own already-declared-at-top counter, the only
    /// shape `render_for`'s init handling supports.
    #[test]
    fn test_a_brain_scream_renders_exactly() {
        let field_types = field_types(&[("x", "FixedT"), ("momz", "FixedT"), ("tics", "i32")]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_BrainScream",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_BrainScream(mo: &mut Mobj, world: &mut World, thinkers: &mut Arena<Thinker>) {\n    \
             let mut x;\n    \
             let mut y;\n    \
             let mut z;\n    \
             let mut th;\n    \
             x = mo.x - 196 * FRACUNIT;\n    \
             while x < mo.x + 320 * FRACUNIT {\n        \
             y = mo.y - 320 * FRACUNIT;\n        \
             z = 128 + P_Random() * 2 * FRACUNIT;\n        \
             th = P_SpawnMobj(x, y, z, MT_ROCKET);\n        \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(th) { m.momz = FixedT(P_Random() * 512); };\n        \
             P_SetMobjState(th, S_BRAINEXPLODE1);\n        \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(th) { m.tics -= P_Random() & 7; };\n        \
             if match thinkers.get(th) { Some(Thinker::Mobj(m)) => m.tics, _ => unreachable!() } < 1 {\n            \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(th) { m.tics = 1; };\n        \
             }\n        \
             x += FRACUNIT * 8;\n    \
             }\n    \
             S_StartSound(None, sfx_bosdth);\n\
             }"
        );
    }

    /// `A_Fire` -- `A_StartFire`/`A_FireCrackle`'s own shared tail call,
    /// itself never directly translated before. `dest = actor->tracer;`
    /// is the already-established target/tracer local-alias idiom
    /// (`collect_target_tracer_aliases`), and `dest->angle`/`.x`/`.y`/
    /// `.z` confirm that alias-dereference generalizes to *any* field,
    /// not just the `x`/`y`/`z` triple `A_SkullAttack` already exercised.
    /// `unsigned an;` motivates `render_decl`'s own small generalization
    /// (a bare `unsigned` specifier, not just `int`); `finecosine[an]`/
    /// `finesine[an]` reuse the already-established by-name `as usize`
    /// cast unchanged.
    #[test]
    fn test_a_fire_renders_exactly() {
        let field_types = field_types(&[
            ("target", "Option<Handle<Thinker>>"),
            ("tracer", "Option<Handle<Thinker>>"),
            ("x", "FixedT"),
            ("y", "FixedT"),
            ("z", "FixedT"),
            ("angle", "u32"),
        ]);
        let rendered = render_fn(&corpus_dir(), "p_enemy.c", "A_Fire", "Mobj", &field_types)
            .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_Fire(actor: &mut Mobj, world: &mut World, thinkers: &Arena<Thinker>) {\n    \
             let mut dest;\n    \
             let mut an;\n    \
             dest = actor.tracer;\n    \
             if dest.is_none() {\n        \
             return;\n    \
             }\n    \
             if !P_CheckSight(actor.target, dest) {\n        \
             return;\n    \
             }\n    \
             an = match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.angle, _ => unreachable!() } >> ANGLETOFINESHIFT;\n    \
             P_UnsetThingPosition(actor);\n    \
             actor.x = match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.x, _ => unreachable!() } + FixedMul(24 * FRACUNIT, finecosine[an as usize]);\n    \
             actor.y = match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.y, _ => unreachable!() } + FixedMul(24 * FRACUNIT, finesine[an as usize]);\n    \
             actor.z = match thinkers.get(dest.unwrap()) { Some(Thinker::Mobj(m)) => m.z, _ => unreachable!() };\n    \
             P_SetThingPosition(actor);\n\
             }"
        );
    }

    /// `A_FatAttack1` -- the function that motivated `collect_spawn_mobj_
    /// locals`'s own generalization to `P_SpawnMissile` (not just
    /// `P_SpawnMobj`, confirmed identical `mobj_t*`-returning shape by
    /// reading `p_mobj.c` directly) and `FnBodyContext::same_handle_
    /// write`'s new same-handle-RHS-read mechanism: `mo->momx = FixedMul
    /// (mo->info->speed, finecosine[an]);` writes one field of the
    /// freshly-spawned `mo` while its own RHS reads a *different* field
    /// (`mo->info`) of that same handle -- the read resolves to the
    /// write's own already-bound `m.info`, not a second, conflicting
    /// `thinkers.get(mo)` borrow. Also confirms `expr_is_fixed_t_valued`'s
    /// new `FixedMul`/`FixedDiv`/`FixedDiv2` recognition: without it,
    /// `momx`'s own `FixedT`-field write would have wrongly double-wrapped
    /// an already-`FixedT`-valued `FixedMul(..)` result in another
    /// `FixedT(..)`, a real `rustc` rejection caught while designing this
    /// (not shipped). The leading bare `P_SpawnMissile(actor, actor->
    /// target, MT_FATSHOT);` (no assignment, deliberately discarding its
    /// own return value -- confirmed a real corpus idiom, not a typo, by
    /// reading the surrounding lines) renders through the already-generic
    /// bare-call-statement path with no special casing at all.
    #[test]
    fn test_a_fat_attack1_renders_exactly() {
        let field_types = field_types(&[
            ("angle", "u32"),
            ("target", "Option<Handle<Thinker>>"),
            ("momx", "FixedT"),
            ("momy", "FixedT"),
        ]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_FatAttack1",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_FatAttack1(actor: &mut Mobj, world: &mut World, thinkers: &mut Arena<Thinker>) {\n    \
             let mut mo;\n    \
             let mut an;\n    \
             A_FaceTarget(actor);\n    \
             actor.angle += (FATSPREAD) as u32;\n    \
             P_SpawnMissile(actor, actor.target, MT_FATSHOT);\n    \
             mo = P_SpawnMissile(actor, actor.target, MT_FATSHOT);\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.angle += FATSPREAD; };\n    \
             an = match thinkers.get(mo) { Some(Thinker::Mobj(m)) => m.angle, _ => unreachable!() } >> ANGLETOFINESHIFT;\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.momx = FixedMul(m.info.speed, finecosine[an as usize]); };\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.momy = FixedMul(m.info.speed, finesine[an as usize]); };\n\
             }"
        );
    }

    /// `A_FatAttack2` -- identical shape to `A_FatAttack1`, differing only
    /// in `-=`/`*2` (`mo->angle -= FATSPREAD*2;`, a compound-assign RHS
    /// that's itself a `Binary` rather than a bare macro identifier),
    /// confirming the same-handle-write machinery generalizes with no
    /// further changes.
    #[test]
    fn test_a_fat_attack2_renders_exactly() {
        let field_types = field_types(&[
            ("angle", "u32"),
            ("target", "Option<Handle<Thinker>>"),
            ("momx", "FixedT"),
            ("momy", "FixedT"),
        ]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_FatAttack2",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_FatAttack2(actor: &mut Mobj, world: &mut World, thinkers: &mut Arena<Thinker>) {\n    \
             let mut mo;\n    \
             let mut an;\n    \
             A_FaceTarget(actor);\n    \
             actor.angle -= (FATSPREAD) as u32;\n    \
             P_SpawnMissile(actor, actor.target, MT_FATSHOT);\n    \
             mo = P_SpawnMissile(actor, actor.target, MT_FATSHOT);\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.angle -= FATSPREAD * 2; };\n    \
             an = match thinkers.get(mo) { Some(Thinker::Mobj(m)) => m.angle, _ => unreachable!() } >> ANGLETOFINESHIFT;\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.momx = FixedMul(m.info.speed, finecosine[an as usize]); };\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.momy = FixedMul(m.info.speed, finesine[an as usize]); };\n\
             }"
        );
    }

    /// `A_FatAttack3` -- two spawns in sequence, `mo` reused (rebound,
    /// not redeclared) for each -- confirms `render_decl`'s single
    /// top-level `let mut mo;` correctly serves *both* assignments, and
    /// that reusing the same handle name across two independent spawns
    /// doesn't confuse `same_handle_write`'s own by-name matching (each
    /// write's own RHS only ever sees the *current* value of `mo`, the
    /// same as the real C reassigning the same pointer variable twice).
    #[test]
    fn test_a_fat_attack3_renders_exactly() {
        let field_types = field_types(&[
            ("angle", "u32"),
            ("target", "Option<Handle<Thinker>>"),
            ("momx", "FixedT"),
            ("momy", "FixedT"),
        ]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_FatAttack3",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_FatAttack3(actor: &mut Mobj, world: &mut World, thinkers: &mut Arena<Thinker>) {\n    \
             let mut mo;\n    \
             let mut an;\n    \
             A_FaceTarget(actor);\n    \
             mo = P_SpawnMissile(actor, actor.target, MT_FATSHOT);\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.angle -= FATSPREAD / 2; };\n    \
             an = match thinkers.get(mo) { Some(Thinker::Mobj(m)) => m.angle, _ => unreachable!() } >> ANGLETOFINESHIFT;\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.momx = FixedMul(m.info.speed, finecosine[an as usize]); };\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.momy = FixedMul(m.info.speed, finesine[an as usize]); };\n    \
             mo = P_SpawnMissile(actor, actor.target, MT_FATSHOT);\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.angle += FATSPREAD / 2; };\n    \
             an = match thinkers.get(mo) { Some(Thinker::Mobj(m)) => m.angle, _ => unreachable!() } >> ANGLETOFINESHIFT;\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.momx = FixedMul(m.info.speed, finecosine[an as usize]); };\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.momy = FixedMul(m.info.speed, finesine[an as usize]); };\n\
             }"
        );
    }

    /// `A_SkelMissile` -- `mo->x += mo->momx;`/`mo->y += mo->momy;` are
    /// the same-handle-write shape through a *compound* assign (not
    /// `A_FatAttack1`'s own plain `=`), confirming `is_fixed_t_field`'s
    /// FixedT-wrap check (scoped to a plain `=` only, per its own doc
    /// comment) correctly stays out of the way for a compound op, letting
    /// the same-handle read resolve through unwrapped. `mo->tracer =
    /// actor->target;` writes an `Option<Handle<Thinker>>` local's own
    /// field straight from a *plain* (not further-dereferenced)
    /// `actor->target` read -- confirms this doesn't need the `Some(..)`
    /// wrap `fog->target = actor;` (`A_VileTarget`'s own idiom, a bare
    /// receiver stored as a value) already gets, since the RHS here is
    /// already `Option`-typed.
    #[test]
    fn test_a_skel_missile_renders_exactly() {
        let field_types = field_types(&[
            ("target", "Option<Handle<Thinker>>"),
            ("tracer", "Option<Handle<Thinker>>"),
            ("x", "FixedT"),
            ("y", "FixedT"),
            ("z", "FixedT"),
            ("momx", "FixedT"),
            ("momy", "FixedT"),
        ]);
        let rendered = render_fn(
            &corpus_dir(),
            "p_enemy.c",
            "A_SkelMissile",
            "Mobj",
            &field_types,
        )
        .expect("should render cleanly");
        assert_eq!(
            rendered,
            "pub fn A_SkelMissile(actor: &mut Mobj, world: &mut World, thinkers: &mut Arena<Thinker>) {\n    \
             let mut mo;\n    \
             if actor.target.is_none() {\n        \
             return;\n    \
             }\n    \
             A_FaceTarget(actor);\n    \
             actor.z += 16 * FRACUNIT;\n    \
             mo = P_SpawnMissile(actor, actor.target, MT_TRACER);\n    \
             actor.z -= 16 * FRACUNIT;\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.x += m.momx; };\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.y += m.momy; };\n    \
             if let Some(Thinker::Mobj(m)) = thinkers.get_mut(mo) { m.tracer = actor.target; };\n\
             }"
        );
    }
}
