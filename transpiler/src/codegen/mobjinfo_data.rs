//! Phase 3: `mobjinfo[]` Data Table
//!
//! `struct_fields.rs` translates `mobjinfo_t`'s *shape* (`MobjInfo`); this
//! renders the actual *data* -- `info.c`'s `mobjinfo_t mobjinfo[NUMMOBJTYPES]
//! = { {...}, {...}, ... };`, 137 positional struct-literal entries, 23
//! values each. `info.c` parses cleanly via `parse_full` (unlike the
//! header-only files elsewhere in this module, it's a `.c` file with a
//! complete typedef context via its own `#include` graph), giving a real
//! `Initializer::List` of nested `Initializer::List`s to walk directly --
//! no macro expansion or rough-scanning needed here.
//!
//! **Every value is one of three shapes**, confirmed by scanning the whole
//! section for any identifier that isn't already covered (only `FRACUNIT`
//! and the array-size `NUMMOBJTYPES` turned up outside `S_`/`sfx_`/`MF_`
//! constants):
//! - A plain integer literal (`-1`, `100`, `0`, ...), rendered as-is.
//! - An enum constant name (`S_PLAY`, `sfx_None`, `MF_SOLID`, ...) --
//!   resolved symbolically, not folded to its numeric value: a corpus-wide
//!   `name -> (home module, value)` index (`build_constant_index`, reusing
//!   `enum_values.rs`'s own `compute_enum_values`) says which module each
//!   name needs importing from, and the rendered field keeps the readable
//!   name rather than an opaque integer -- this is meant to read like the
//!   original, not like a disassembly of it.
//! - A `FRACUNIT`-scaled expression (`16*FRACUNIT`) -- `FRACUNIT` is a
//!   `#define`, not an enum constant, so it's special-cased by name to
//!   `runtime::FRACUNIT.0` (the field's declared type is plain `int`, per
//!   `mobjinfo_t`'s own C declaration, not `fixed_t`, so the value needs
//!   the wrapped `i32` out of the `FixedT` constant, not the constant
//!   itself).
//!
//! Everything else (a nested initializer where a plain value is expected,
//! a binary op this renderer doesn't recognize, an unresolvable
//! identifier) fails loudly, matching `struct_fields.rs`'s own practice.

use crate::codegen::enum_values::compute_enum_values;
use crate::codegen::struct_fields::{
    collect_enum_typedef_names, find_typedef_struct, map_struct_fields,
};
use crate::parser::ast::{
    BinaryOp, Declaration, Expr, ExternalDecl, Initializer, StorageClass, TypeSpecifier, UnaryOp,
};
use crate::parser::grammar::declarator_name;
use crate::parser::{attach_comments, lex_chunks, parse, parse_full};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

fn rough_scan(path: &Path) -> Vec<ExternalDecl> {
    let (_, resolved) = parse(path.to_str().unwrap()).unwrap_or_else(|e| panic!("{e}"));
    let entries = lex_chunks(&resolved).expect("lexing should succeed");
    let stream = attach_comments(entries);
    crate::parser::grammar::extract_top_level_decls(&stream)
}

/// Corpus-wide `name -> (home module, value)` index, over every
/// `typedef enum` in `sources`' files. Reuses `enum_values.rs`'s own
/// `compute_enum_values`, so a name whose value can't be folded (see that
/// module's own docs -- none of the enums this table needs are affected)
/// is simply absent, not guessed at.
pub fn build_constant_index(
    corpus_dir: &Path,
    sources: &[(&str, &str)],
) -> HashMap<String, (String, i64)> {
    let mut out = HashMap::new();
    for (file, module) in sources {
        let items = rough_scan(&corpus_dir.join(file));
        for item in &items {
            let ExternalDecl::Declaration(decl) = item else {
                continue;
            };
            if decl.specifiers.storage != Some(StorageClass::Typedef) {
                continue;
            }
            for ts in &decl.specifiers.type_specifiers {
                if let TypeSpecifier::Enum(spec) = ts {
                    for (name, value) in compute_enum_values(spec) {
                        if let Some(v) = value {
                            out.entry(name).or_insert((module.to_string(), v));
                        }
                    }
                }
            }
        }
    }
    out
}

/// Renders one initializer value expression as Rust source text, tracking
/// any cross-module `use` this rendering needs in `imports` (keyed by
/// module, keyed away from `home_module` itself). `None` for a shape this
/// renderer doesn't recognize (see module docs).
fn render_value_expr(
    e: &Expr,
    home_module: &str,
    constants: &HashMap<String, (String, i64)>,
    imports: &mut BTreeMap<String, BTreeSet<String>>,
) -> Option<String> {
    match e {
        Expr::IntLiteral(text) => Some(text.clone()),
        Expr::Unary {
            op: UnaryOp::Minus,
            expr,
        } => Some(format!(
            "-{}",
            render_value_expr(expr, home_module, constants, imports)?
        )),
        Expr::Ident(name) if name == "FRACUNIT" => {
            if home_module != "runtime" {
                imports
                    .entry("runtime".to_string())
                    .or_default()
                    .insert("FRACUNIT".to_string());
            }
            Some("FRACUNIT.0".to_string())
        }
        Expr::Ident(name) => {
            let (module, _) = constants.get(name)?;
            if module != home_module {
                imports
                    .entry(module.clone())
                    .or_default()
                    .insert(name.clone());
            }
            Some(name.clone())
        }
        Expr::Binary { op, lhs, rhs } => {
            let a = render_value_expr(lhs, home_module, constants, imports)?;
            let b = render_value_expr(rhs, home_module, constants, imports)?;
            let op_str = match op {
                BinaryOp::Mul => "*",
                BinaryOp::Div => "/",
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::BitOr => "|",
                BinaryOp::BitAnd => "&",
                BinaryOp::BitXor => "^",
                BinaryOp::Shl => "<<",
                BinaryOp::Shr => ">>",
                _ => return None,
            };
            Some(format!("{a} {op_str} {b}"))
        }
        _ => None,
    }
}

/// `mobjinfo[]`'s rendered Rust text (`pub static MOBJINFO: [MobjInfo; N]
/// = [...];`), plus the cross-module `use` imports it needs (keyed by
/// module).
pub struct MobjinfoTable {
    pub rendered: String,
    pub imports: BTreeMap<String, BTreeSet<String>>,
}

/// The three enums `mobjinfo[]`'s own values reference, and where each
/// lives -- `statenum_t`/`mobjtype_t` (`info.h`) share `mobjinfo_t`'s own
/// `info` module, so those need no import; `sfxenum_t`/`musicenum_t`
/// (`sounds.h`) and `mobjflag_t` (`p_mobj.h`) do.
const CONSTANT_SOURCES: &[(&str, &str)] = &[
    ("info.h", "info"),
    ("sounds.h", "sounds"),
    ("p_mobj.h", "p_mobj"),
];

/// Renders `mobjinfo[]` against the real corpus at `corpus_dir`. `Err`
/// names the first entry/field this can't render, rather than emitting a
/// partial or wrong table.
pub fn render_mobjinfo_table(corpus_dir: &Path) -> Result<MobjinfoTable, String> {
    let info_items = rough_scan(&corpus_dir.join("info.h"));
    let enum_typedefs = collect_enum_typedef_names(&info_items);
    let mobjinfo_fields =
        find_typedef_struct(&info_items, "mobjinfo_t").ok_or("mobjinfo_t not found")?;
    let mapped = map_struct_fields(mobjinfo_fields, &enum_typedefs)?;
    let field_names: Vec<&str> = mapped.iter().map(|f| f.name.as_str()).collect();

    let constants = build_constant_index(corpus_dir, CONSTANT_SOURCES);

    let c_path = corpus_dir.join("info.c");
    let (_, unit) = parse_full(c_path.to_str().unwrap())?;
    let mut mobjinfo_initializer = None;
    'outer: for item in &unit.items {
        if let ExternalDecl::Declaration(Declaration { declarators, .. }) = item {
            for d in declarators {
                if declarator_name(&d.declarator).as_deref() == Some("mobjinfo") {
                    mobjinfo_initializer = d.initializer.clone();
                    break 'outer;
                }
            }
        }
    }
    let Some(Initializer::List(entries)) = mobjinfo_initializer else {
        return Err("mobjinfo's own initializer not found, or not a list".to_string());
    };

    let mut imports: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut rendered_entries = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let Initializer::List(values) = entry else {
            return Err(format!("mobjinfo[{i}]: expected a nested {{...}} entry"));
        };
        if values.len() != field_names.len() {
            return Err(format!(
                "mobjinfo[{i}]: expected {} values, got {}",
                field_names.len(),
                values.len()
            ));
        }
        let mut field_strs = Vec::with_capacity(field_names.len());
        for (name, value) in field_names.iter().zip(values) {
            let Initializer::Expr(e) = value else {
                return Err(format!(
                    "mobjinfo[{i}].{name}: nested initializer, not a plain value"
                ));
            };
            let rendered =
                render_value_expr(e, "info", &constants, &mut imports).ok_or_else(|| {
                    format!("mobjinfo[{i}].{name}: no rendering for this value shape")
                })?;
            field_strs.push(format!("{name}: {rendered}"));
        }
        rendered_entries.push(format!("    MobjInfo {{ {} }},", field_strs.join(", ")));
    }

    let rendered = format!(
        "pub static MOBJINFO: [MobjInfo; {}] = [\n{}\n];",
        entries.len(),
        rendered_entries.join("\n")
    );

    Ok(MobjinfoTable { rendered, imports })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_renders_all_137_entries() {
        let table = render_mobjinfo_table(&corpus_dir()).expect("should render cleanly");
        assert!(
            table
                .rendered
                .starts_with("pub static MOBJINFO: [MobjInfo; 137] = [")
        );
        assert_eq!(table.rendered.matches("MobjInfo {").count(), 137);
    }

    #[test]
    fn test_first_entry_mt_player() {
        let table = render_mobjinfo_table(&corpus_dir()).expect("should render cleanly");
        let first_line = table
            .rendered
            .lines()
            .find(|l| l.contains("MobjInfo {"))
            .unwrap();
        assert_eq!(
            first_line,
            "    MobjInfo { doomednum: -1, spawnstate: S_PLAY, spawnhealth: 100, \
             seestate: S_PLAY_RUN1, seesound: sfx_None, reactiontime: 0, \
             attacksound: sfx_None, painstate: S_PLAY_PAIN, painchance: 255, \
             painsound: sfx_plpain, meleestate: S_NULL, missilestate: S_PLAY_ATK1, \
             deathstate: S_PLAY_DIE1, xdeathstate: S_PLAY_XDIE1, deathsound: sfx_pldeth, \
             speed: 0, radius: 16 * FRACUNIT.0, height: 56 * FRACUNIT.0, mass: 100, \
             damage: 0, activesound: sfx_None, \
             flags: MF_SOLID | MF_SHOOTABLE | MF_DROPOFF | MF_PICKUP | MF_NOTDMATCH, \
             raisestate: S_NULL },"
        );
    }

    #[test]
    fn test_needs_sounds_and_p_mobj_imports_not_info() {
        let table = render_mobjinfo_table(&corpus_dir()).expect("should render cleanly");
        // S_*/statenum_t constants live in `info` itself (mobjinfo's own
        // home module), so they must never show up as an import.
        assert!(!table.imports.contains_key("info"));
        assert!(table.imports.contains_key("sounds"));
        assert!(table.imports.contains_key("p_mobj"));
        assert!(table.imports["sounds"].contains("sfx_None"));
        assert!(table.imports["p_mobj"].contains("MF_SOLID"));
    }

    #[test]
    fn test_needs_runtime_fracunit_import() {
        let table = render_mobjinfo_table(&corpus_dir()).expect("should render cleanly");
        assert!(table.imports["runtime"].contains("FRACUNIT"));
    }

    #[test]
    fn test_build_constant_index_resolves_known_names() {
        let index = build_constant_index(&corpus_dir(), CONSTANT_SOURCES);
        assert_eq!(index.get("S_PLAY").map(|(m, _)| m.as_str()), Some("info"));
        assert_eq!(
            index.get("sfx_None").map(|(m, v)| (m.as_str(), *v)),
            Some(("sounds", 0))
        );
        assert_eq!(
            index.get("MF_SOLID").map(|(m, _)| m.as_str()),
            Some("p_mobj")
        );
    }
}
