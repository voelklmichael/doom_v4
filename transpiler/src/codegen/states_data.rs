//! Phase 3: `states[]` Data Table
//!
//! The other half of the table pair `mobjinfo_data.rs` started: `info.c`'s
//! `state_t states[NUMSTATES] = { {...}, ... };` (967 positional entries,
//! 7 values each), rendered as `pub static STATES: [State; 967] = [...];`.
//! Reuses that module's `rough_scan`/`render_value_expr`/
//! `build_constant_index` directly for `sprite`/`frame`/`tics`/
//! `nextstate`/`misc1`/`misc2` -- only `action` needs its own renderer
//! here, since `actionf_t` is a *union*, so its own C initializer is a
//! nested single-element compound literal (`{NULL}`/`{A_Function}`),
//! confirmed directly against the parser's real output rather than
//! assumed: `NULL` stays `Expr::Ident("NULL")` (no macro expansion), never
//! folded to a literal.
//!
//! **Round 19 -- `ActionFn` became a name-tagged closed enum, not a
//! 2-variant `fn`-pointer wrapper**: `docs/03_TRANSPILER.md`'s Round 18
//! entry found the old `Mobj(fn(&mut Mobj))`/`Weapon(fn(&mut Player, &mut
//! PlayerSpriteState))` design provably wrong -- a real Rust `fn` pointer
//! type is monomorphic, but the corpus's real `A_*` bodies need at least
//! 8 distinct concrete parameter lists (`action_fn.rs`'s own doc comment
//! has the full breakdown), so no single `fn(&mut Mobj)` type could ever
//! hold `A_Chase` (needs `world`/`thinkers` too). `render_action_field`
//! now just tags each `states[]` entry's `action` with its own bare enum
//! variant (`Some(ActionFn::AChase)`, no payload) -- the *dispatcher*
//! (`P_SetMobjState`, `function_body.rs`) does the real work, a
//! hand-written `match` calling each real function by name with
//! whatever args *that* function's own already-shipped signature needs,
//! drawn from the dispatcher's own scope. This also means a tagged
//! variant needs no cross-module `use` import at the `states[]`-table
//! level any more (the old `Mobj(A_Chase)` literal needed `A_Chase`'s
//! real value in scope right there; a bare `ActionFn::AChase` tag
//! doesn't) -- imports for actions specifically are gone from this
//! module's own `StatesTable::imports` output; only the dispatcher's own
//! module needs those.
//!
//! **Resolving which name needs a variant at all**: `info.c`'s own
//! forward declarations of the 132 action functions (`void A_Foo();`) use
//! C89's "unspecified arguments" `()` shape, which can't even confirm a
//! name is real -- so `build_action_function_index` reads the *real*
//! definitions in `p_enemy.c`/`p_pspr.c` instead, via `parse_full` (both
//! parse cleanly standalone, same as `info.c` did for `mobjinfo_data.rs`).
//! Not every one of those 132 real definitions is actually a variant
//! `ActionFn` needs, though: `A_PainShootSkull` is a plain *helper*,
//! called directly from `A_PainAttack`/`A_PainDie`'s own bodies with an
//! extra `angle` argument no `state_t.action` slot could ever supply --
//! it never appears as a `states[]` entry's own `{A_PainShootSkull}`
//! value, confirmed by direct grep of `info.c`, not assumed from its
//! `p_enemy.c` file location. So `render_states_table` derives the
//! exhaustive variant set from `states[]` itself (74 distinct names
//! actually referenced, out of 132 real definitions), not from the
//! action-function index directly -- `StatesTable::action_names`, which
//! `action_fn.rs`'s own `render_action_fn_enum` reuses rather than
//! re-scanning, so there's exactly one true computation of "which names
//! need a variant."

use crate::codegen::mobjinfo_data::{build_constant_index, render_value_expr, rough_scan};
use crate::codegen::struct_fields::{
    collect_enum_typedef_names, find_typedef_struct, map_struct_fields,
};
use crate::parser::ast::{Declaration, DirectDeclarator, Expr, ExternalDecl, Initializer};
use crate::parser::grammar::declarator_name;
use crate::parser::parse_full;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// Same enum sources `mobjinfo_data.rs` uses -- `sprite`/`nextstate` are
/// `spritenum_t`/`statenum_t` (`info.h`), both already covered there.
const CONSTANT_SOURCES: &[(&str, &str)] = &[
    ("info.h", "info"),
    ("sounds.h", "sounds"),
    ("p_mobj.h", "p_mobj"),
];

/// Where the 132 real `A_*` action function *definitions* live -- not
/// `info.h`'s own forward declarations, which can't tell arity apart (see
/// module docs).
const ACTION_FUNCTION_SOURCES: &[(&str, &str)] =
    &[("p_enemy.c", "p_enemy"), ("p_pspr.c", "p_pspr")];

/// `name -> (home module, parameter count)` for every `A_*` function
/// actually *defined* in `ACTION_FUNCTION_SOURCES`.
fn build_action_function_index(corpus_dir: &Path) -> HashMap<String, (String, usize)> {
    let mut out = HashMap::new();
    for (file, module) in ACTION_FUNCTION_SOURCES {
        let Ok((_, unit)) = parse_full(corpus_dir.join(file).to_str().unwrap()) else {
            continue;
        };
        for item in &unit.items {
            if let ExternalDecl::FunctionDef(f) = item
                && let DirectDeclarator::Function(base, params) = &f.declarator.direct
                && let DirectDeclarator::Ident(name) = base.as_ref()
                && name.starts_with("A_")
            {
                out.entry(name.clone())
                    .or_insert((module.to_string(), params.params.len()));
            }
        }
    }
    out
}

/// `A_Chase` -> `AChase` -- `ActionFn`'s own variant-naming rule (this
/// module's doc comment, and `action_fn.rs`'s): every real action
/// function name starts with the single `A_` prefix, so stripping just
/// that one underscore (not every underscore -- none of these 74 names
/// have a second one) gives a valid, collision-free Rust identifier
/// directly, no further casing needed.
pub fn action_variant_name(c_name: &str) -> String {
    c_name.replacen('_', "", 1)
}

/// Renders `state_t.action`'s own nested union-initializer value
/// (`{NULL}`/`{A_Function}`) as an `Option<ActionFn>` tag literal (no
/// cross-module import needed for it any more -- see module docs),
/// recording the real name into `referenced` whenever it resolves to one.
fn render_action_field(
    init: &Initializer,
    actions: &HashMap<String, (String, usize)>,
    referenced: &mut BTreeSet<String>,
) -> Option<String> {
    let Initializer::List(items) = init else {
        return None;
    };
    let [Initializer::Expr(e)] = items.as_slice() else {
        return None;
    };
    match e {
        Expr::Ident(name) if name == "NULL" => Some("None".to_string()),
        Expr::IntLiteral(text) if text == "0" => Some("None".to_string()),
        Expr::Ident(name) => {
            actions.get(name)?;
            referenced.insert(name.clone());
            Some(format!("Some(ActionFn::{})", action_variant_name(name)))
        }
        _ => None,
    }
}

/// `states[]`'s rendered Rust text, the cross-module `use` imports it
/// needs (no longer includes action-function names -- see module docs),
/// and the exhaustive set of real `A_*` names actually referenced from
/// some entry's `action` field -- the one true source `action_fn.rs`'s
/// own `render_action_fn_enum` reuses for its variant list, rather than
/// re-deriving it.
pub struct StatesTable {
    pub rendered: String,
    pub imports: BTreeMap<String, BTreeSet<String>>,
    pub action_names: BTreeSet<String>,
}

/// `state_index` -- hand-rendered literal text, not corpus-mapped, the
/// same category as `world.rs`'s own `render_world_struct` (new,
/// invented Rust-only infrastructure with no direct corpus counterpart).
/// Needed the first time a real function body computes real C pointer
/// arithmetic between two `state_t*` values (`A_FireCGun`'s own
/// `weaponinfo[player->readyweapon].flashstate + psp->state -
/// &states[S_CHAIN1]`, `p_pspr.c`): unlike `sector_t*`/`side_t*`/etc.
/// (plain index newtypes under this project's own memory model, so
/// "index of this pointer" is a free `.0` field read, see `EV_
/// VerticalDoor`'s own `sec-sectors`), `state_t*` maps to a real
/// `&'static State` reference into the `STATES` table (`struct_
/// fields.rs`'s own established decision -- `mobjinfo_t`/`state_t*`
/// point at a static, program-lifetime, read-only table, so a real
/// reference is simpler than another index newtype), which has no
/// index of its own to read back out. Computed via plain pointer-to-
/// `usize` address arithmetic -- safe, well-defined Rust for two
/// pointers known to be elements of the *same* array (no dereferencing
/// of a possibly-invalid pointer happens at all, only integer
/// arithmetic on addresses), matching the byte-distance-divided-by-
/// element-size semantics C's own pointer subtraction has, without
/// needing an `unsafe` block.
pub fn render_state_index_fn() -> String {
    "\
pub fn state_index(s: &'static State) -> i32 {
    ((s as *const State as usize - &STATES[0] as *const State as usize)
        / std::mem::size_of::<State>()) as i32
}"
    .to_string()
}

/// Renders `states[]` against the real corpus at `corpus_dir`. `Err` names
/// the first entry/field this can't render, rather than emitting a
/// partial or wrong table.
pub fn render_states_table(corpus_dir: &Path) -> Result<StatesTable, String> {
    let info_items = rough_scan(&corpus_dir.join("info.h"));
    let enum_typedefs = collect_enum_typedef_names(&info_items);
    let state_fields = find_typedef_struct(&info_items, "state_t").ok_or("state_t not found")?;
    let mapped = map_struct_fields(state_fields, &enum_typedefs)?;
    let field_names: Vec<&str> = mapped.iter().map(|f| f.name.as_str()).collect();

    let constants = build_constant_index(corpus_dir, CONSTANT_SOURCES);
    let actions = build_action_function_index(corpus_dir);

    let c_path = corpus_dir.join("info.c");
    let (_, unit) = parse_full(c_path.to_str().unwrap())?;
    let mut states_initializer = None;
    'outer: for item in &unit.items {
        if let ExternalDecl::Declaration(Declaration { declarators, .. }) = item {
            for d in declarators {
                if declarator_name(&d.declarator).as_deref() == Some("states") {
                    states_initializer = d.initializer.clone();
                    break 'outer;
                }
            }
        }
    }
    let Some(Initializer::List(entries)) = states_initializer else {
        return Err("states' own initializer not found, or not a list".to_string());
    };

    let mut imports: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut action_names: BTreeSet<String> = BTreeSet::new();
    let mut rendered_entries = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let Initializer::List(values) = entry else {
            return Err(format!("states[{i}]: expected a nested {{...}} entry"));
        };
        if values.len() != field_names.len() {
            return Err(format!(
                "states[{i}]: expected {} values, got {}",
                field_names.len(),
                values.len()
            ));
        }
        let mut field_strs = Vec::with_capacity(field_names.len());
        for (name, value) in field_names.iter().zip(values) {
            let rendered = if *name == "action" {
                render_action_field(value, &actions, &mut action_names).ok_or_else(|| {
                    format!("states[{i}].action: no rendering for this value shape")
                })?
            } else {
                let Initializer::Expr(e) = value else {
                    return Err(format!(
                        "states[{i}].{name}: nested initializer, not a plain value"
                    ));
                };
                render_value_expr(e, "info", &constants, &mut imports).ok_or_else(|| {
                    format!("states[{i}].{name}: no rendering for this value shape")
                })?
            };
            field_strs.push(format!("{name}: {rendered}"));
        }
        rendered_entries.push(format!("    State {{ {} }},", field_strs.join(", ")));
    }

    let rendered = format!(
        "pub static STATES: [State; {}] = [\n{}\n];",
        entries.len(),
        rendered_entries.join("\n")
    );

    Ok(StatesTable {
        rendered,
        imports,
        action_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_renders_all_967_entries() {
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        assert!(
            table
                .rendered
                .starts_with("pub static STATES: [State; 967] = [")
        );
        assert_eq!(table.rendered.matches("State {").count(), 967);
    }

    #[test]
    fn test_first_entry_s_null_has_no_action() {
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        let first_line = table
            .rendered
            .lines()
            .find(|l| l.contains("State {"))
            .unwrap();
        assert_eq!(
            first_line,
            "    State { sprite: SPR_TROO, frame: 0, tics: -1, action: None, \
             nextstate: S_NULL, misc1: 0, misc2: 0 },"
        );
    }

    #[test]
    fn test_mobj_shaped_action_renders_as_bare_tag() {
        // S_POSS_RUN1: {SPR_POSS,0,4,{A_Chase},S_POSS_RUN2,0,0} -- A_Chase
        // (p_enemy.c) is fn(mobj_t*) shaped, the classic monster chase-AI
        // action. Round 19: no more `Mobj(..)` wrapper, just the bare tag.
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        let line = table
            .rendered
            .lines()
            .find(|l| l.contains("ActionFn::AChase"))
            .expect("ActionFn::AChase not found in rendered output");
        assert!(
            line.contains("action: Some(ActionFn::AChase)"),
            "expected AChase tag, got: {line}"
        );
    }

    #[test]
    fn test_weapon_shaped_action_renders_as_bare_tag() {
        // S_LIGHTDONE (index 1): {SPR_SHTG,4,0,{A_Light0},S_NULL,0,0} --
        // A_Light0 is fn(player_t*, pspdef_t*) shaped, but round 19's
        // tagged enum doesn't distinguish shape at the STATES-table level
        // at all any more -- the dispatcher alone knows.
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        let line = table
            .rendered
            .lines()
            .find(|l| l.contains("ActionFn::ALight0"))
            .expect("ActionFn::ALight0 not found in rendered output");
        assert!(
            line.contains("action: Some(ActionFn::ALight0)"),
            "expected ALight0 tag, got: {line}"
        );
    }

    #[test]
    fn test_action_defined_in_p_enemy_still_resolves_by_name() {
        // A_OpenShotgun2 is *defined* in p_enemy.c despite being
        // genuinely fn(player_t*, pspdef_t*)-shaped (the double-barrel
        // shotgun's own weapon state chain) -- round 18 confirmed shape
        // comes from real parameter count, not file location. Round 19:
        // that distinction no longer matters for the STATES table at
        // all, only that the name resolves to a real corpus action.
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        let line = table
            .rendered
            .lines()
            .find(|l| l.contains("ActionFn::AOpenShotgun2"))
            .expect("ActionFn::AOpenShotgun2 not found in rendered output");
        assert!(
            line.contains("action: Some(ActionFn::AOpenShotgun2)"),
            "expected AOpenShotgun2 tag, got: {line}"
        );
    }

    #[test]
    fn test_no_action_imports_needed_any_more() {
        // Round 19: a bare tag like `ActionFn::AChase` needs no
        // cross-module `use A_Chase;` the way the old `Mobj(A_Chase)`
        // fn-pointer literal did -- only the dispatcher's own module
        // needs to import the real functions it calls by name.
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        assert!(!table.imports.contains_key("p_enemy"));
        assert!(!table.imports.contains_key("p_pspr"));
    }

    #[test]
    fn test_action_names_is_the_74_real_state_dispatched_functions() {
        // Corpus-verified: `grep -oP '\{A_\w+\}' info.c | sort -u` finds
        // exactly 74 distinct names actually referenced from some
        // states[] entry's own action field -- out of 132 real `A_*`
        // definitions total (`p_enemy.c`/`p_pspr.c`). `A_PainShootSkull`
        // is the confirmed counterexample: a real, defined action
        // function that's a plain helper (`A_PainAttack`/`A_PainDie`
        // call it directly with an extra `angle` argument), never a
        // `states[]` entry's own action value, so it must NOT appear
        // here even though `build_action_function_index` (scanning
        // p_enemy.c/p_pspr.c directly) knows about it.
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        assert_eq!(table.action_names.len(), 74);
        assert!(table.action_names.contains("A_Chase"));
        assert!(table.action_names.contains("A_Light0"));
        assert!(table.action_names.contains("A_OpenShotgun2"));
        assert!(!table.action_names.contains("A_PainShootSkull"));
    }

    #[test]
    fn test_action_variant_name_strips_single_underscore() {
        assert_eq!(action_variant_name("A_Chase"), "AChase");
        assert_eq!(action_variant_name("A_FaceTarget"), "AFaceTarget");
        assert_eq!(action_variant_name("A_SPosAttack"), "ASPosAttack");
        assert_eq!(action_variant_name("A_BFGsound"), "ABFGsound");
    }

    #[test]
    fn test_build_action_function_index_resolves_known_shapes() {
        let index = build_action_function_index(&corpus_dir());
        assert_eq!(
            index.get("A_Chase").map(|(m, c)| (m.as_str(), *c)),
            Some(("p_enemy", 1))
        );
        assert_eq!(
            index.get("A_Light0").map(|(m, c)| (m.as_str(), *c)),
            Some(("p_pspr", 2))
        );
        assert_eq!(
            index.get("A_OpenShotgun2").map(|(m, c)| (m.as_str(), *c)),
            Some(("p_enemy", 2))
        );
    }

    #[test]
    fn test_renders_state_index_fn() {
        let rendered = render_state_index_fn();
        assert_eq!(
            rendered,
            "pub fn state_index(s: &'static State) -> i32 {\n    \
             ((s as *const State as usize - &STATES[0] as *const State as usize)\n        \
             / std::mem::size_of::<State>()) as i32\n\
             }"
        );
    }
}
