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
//! **Resolving which `ActionFn` variant a name needs**: `info.c`'s own
//! forward declarations of the 132 action functions (`void A_Foo();`) use
//! C89's "unspecified arguments" `()` shape, which can't tell arity apart
//! -- so `build_action_function_index` reads the *real* definitions in
//! `p_enemy.c`/`p_pspr.c` instead, via `parse_full` (both parse cleanly
//! standalone, same as `info.c` did for `mobjinfo_data.rs`). Confirmed
//! empirically, not assumed from file location: a function's home file
//! doesn't predict its shape -- `A_OpenShotgun2`/`A_LoadShotgun2`/
//! `A_CloseShotgun2` are defined in `p_enemy.c` but are genuinely
//! `fn(player_t*, pspdef_t*)`-shaped (the double-barrel shotgun's weapon
//! state chain), so shape is decided per-function by its own real
//! parameter count, not by which file happened to define it.

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

/// Renders `state_t.action`'s own nested union-initializer value
/// (`{NULL}`/`{A_Function}`) as an `Option<ActionFn>` literal, tracking
/// any cross-module import it needs.
fn render_action_field(
    init: &Initializer,
    home_module: &str,
    actions: &HashMap<String, (String, usize)>,
    imports: &mut BTreeMap<String, BTreeSet<String>>,
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
            let (module, param_count) = actions.get(name)?;
            if module != home_module {
                imports
                    .entry(module.clone())
                    .or_default()
                    .insert(name.clone());
            }
            let variant = match param_count {
                1 => "Mobj",
                2 => "Weapon",
                _ => return None,
            };
            Some(format!("Some(ActionFn::{variant}({name}))"))
        }
        _ => None,
    }
}

/// `states[]`'s rendered Rust text, plus the cross-module `use` imports it
/// needs.
pub struct StatesTable {
    pub rendered: String,
    pub imports: BTreeMap<String, BTreeSet<String>>,
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
                render_action_field(value, "info", &actions, &mut imports).ok_or_else(|| {
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

    Ok(StatesTable { rendered, imports })
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
    fn test_mobj_shaped_action_renders_correctly() {
        // S_POSS_RUN1: {SPR_POSS,0,4,{A_Chase},S_POSS_RUN2,0,0} -- A_Chase
        // (p_enemy.c) is fn(mobj_t*) shaped, the classic monster chase-AI
        // action.
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        let line = table
            .rendered
            .lines()
            .find(|l| l.contains("A_Chase"))
            .expect("A_Chase not found in rendered output");
        assert!(
            line.contains("action: Some(ActionFn::Mobj(A_Chase))"),
            "expected Mobj variant, got: {line}"
        );
    }

    #[test]
    fn test_weapon_shaped_action_renders_correctly() {
        // S_LIGHTDONE (index 1): {SPR_SHTG,4,0,{A_Light0},S_NULL,0,0} --
        // A_Light0 is fn(player_t*, pspdef_t*) shaped.
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        let line = table
            .rendered
            .lines()
            .find(|l| l.contains("A_Light0"))
            .expect("A_Light0 not found in rendered output");
        assert!(
            line.contains("action: Some(ActionFn::Weapon(A_Light0))"),
            "expected Weapon variant, got: {line}"
        );
    }

    #[test]
    fn test_cross_file_action_shape_resolved_by_signature_not_file() {
        // A_OpenShotgun2 is *defined* in p_enemy.c but is genuinely
        // fn(player_t*, pspdef_t*)-shaped (the double-barrel shotgun's own
        // weapon state chain) -- confirms shape comes from the real
        // parameter count, not from which file happened to define it.
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        let line = table
            .rendered
            .lines()
            .find(|l| l.contains("A_OpenShotgun2"))
            .expect("A_OpenShotgun2 not found in rendered output");
        assert!(
            line.contains("action: Some(ActionFn::Weapon(A_OpenShotgun2))"),
            "expected Weapon variant, got: {line}"
        );
        assert!(table.imports["p_enemy"].contains("A_OpenShotgun2"));
    }

    #[test]
    fn test_needs_p_enemy_and_p_pspr_imports_not_info() {
        let table = render_states_table(&corpus_dir()).expect("should render cleanly");
        assert!(!table.imports.contains_key("info"));
        assert!(table.imports.contains_key("p_enemy"));
        assert!(table.imports.contains_key("p_pspr"));
        assert!(table.imports["p_enemy"].contains("A_Chase"));
        assert!(table.imports["p_pspr"].contains("A_Light0"));
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
}
