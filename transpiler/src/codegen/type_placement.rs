//! Phase 3: Module Placement for Translated Types
//!
//! Every type this codebase's struct translation (`struct_fields.rs`,
//! `thinkers.rs`) produces needs a home Rust module before its `use`
//! imports can be generated -- two kinds: a corpus-translated
//! struct/enum, placed in whichever original C module (`Source` or
//! `HeaderOnly`, per `build_module_graph`) its C definition actually
//! lives in (confirmed against the real module graph, not assumed:
//! `p_spec.c` exists, so `p_spec.h`'s thinkers live in the `p_spec`
//! `Source` module, not `HeaderOnly`; `r_defs.h`/`doomdata.h` have no
//! matching `.c`, so they're `HeaderOnly`); and a runtime support type
//! (`runtime/*.rs`'s own copy, not a corpus module at all). `Thinker`
//! itself goes in `p_tick` -- where `P_InitThinkers`/`P_AddThinker`/
//! `P_RemoveThinker`/`P_RunThinkers` (the original's own thinker-list
//! management) already live, since the enum + dispatch replace exactly
//! that mechanism.
//!
//! A type not in this table at all (`MobjInfo`/`State` -- referenced by
//! `Mobj`'s `info`/`state` fields but not yet translated as real structs,
//! see `docs/03_TRANSPILER.md`) is simply not imported: a known,
//! documented gap, not something this step guesses at.
//!
//! **Imports are computed by scanning already-rendered field-type
//! strings**, not hand-written per struct: `render_imports_for` tokenizes
//! each `MappedField`'s `rust_type` (`Option<Handle<Thinker>>` ->
//! `["Option", "Handle", "Thinker"]`) and looks up every token in
//! `type_home_module`, so `Vec`/`Option`/primitives fall out on their own
//! (never in the table) without needing special-casing here. This scales
//! to every struct this module or a future one translates, rather than a
//! new hand-maintained import list per struct.

use crate::codegen::struct_fields::MappedField;
use std::collections::{BTreeMap, BTreeSet};

/// `name`'s home Rust module, if this codebase knows one.
pub fn type_home_module(name: &str) -> Option<&'static str> {
    match name {
        "FireFlicker" | "LightFlash" | "Strobe" | "Glow" | "Plat" | "VerticalDoor" | "Ceiling"
        | "FloorMove" => Some("p_spec"),
        "Mobj" => Some("p_mobj"),
        "Vertex" | "DegenMobj" | "Side" | "Subsector" | "Seg" | "Line" | "Sector" | "Node" => {
            Some("r_defs")
        }
        "MapThing" => Some("doomdata"),
        "Thinker" => Some("p_tick"),
        "FixedT" | "Handle" | "SectorId" | "SubsectorId" | "VertexId" | "SideId" | "LineId"
        | "PlayerId" | "Arena" => Some("runtime"),
        _ => None,
    }
}

/// Bare identifier-like tokens inside a rendered field-type string.
fn type_tokens(rust_type: &str) -> Vec<&str> {
    rust_type
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .collect()
}

/// The `use` block `home_module` needs for `fields`' own types: one line
/// per other module referenced, sorted, grouped, deduplicated -- excludes
/// anything already local to `home_module` itself (a struct referencing a
/// sibling in its own module needs no import for it).
pub fn render_imports_for(home_module: &str, fields: &[MappedField]) -> String {
    let mut by_module: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for f in fields {
        for tok in type_tokens(&f.rust_type) {
            if let Some(module) = type_home_module(tok)
                && module != home_module
            {
                by_module.entry(module).or_default().insert(tok);
            }
        }
    }
    by_module
        .into_iter()
        .map(|(module, names)| {
            let names: Vec<&str> = names.into_iter().collect();
            format!("use crate::{module}::{{{}}};", names.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::modules::{ModuleKind, build_module_graph};
    use crate::codegen::struct_fields::{
        collect_enum_typedef_names, find_typedef_struct, map_struct_fields,
    };
    use crate::parser::ast::ExternalDecl;
    use crate::parser::{attach_comments, lex_chunks, parse};
    use std::path::Path;

    fn corpus_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    fn parse_rough(corpus_file: &str) -> Vec<ExternalDecl> {
        let path = corpus_dir().join(corpus_file);
        let (_, resolved) = parse(path.to_str().unwrap()).unwrap();
        let entries = lex_chunks(&resolved).unwrap();
        let stream = attach_comments(entries);
        crate::parser::grammar::extract_top_level_decls(&stream)
    }

    /// Confirms every module this table claims actually has the `Source`/
    /// `HeaderOnly` kind this module's own docs assert, against the real
    /// module graph -- not just trusted from a one-time manual check.
    #[test]
    fn test_home_modules_match_the_real_module_graph() {
        let graph = build_module_graph(&corpus_dir());
        let kind_of = |name: &str| {
            graph
                .modules
                .iter()
                .find(|m| m.name == name)
                .unwrap_or_else(|| panic!("{name} not found in the module graph"))
                .kind
        };
        assert_eq!(kind_of("p_spec"), ModuleKind::Source);
        assert_eq!(kind_of("p_mobj"), ModuleKind::Source);
        assert_eq!(kind_of("p_tick"), ModuleKind::Source);
        assert_eq!(kind_of("r_defs"), ModuleKind::HeaderOnly);
        assert_eq!(kind_of("doomdata"), ModuleKind::HeaderOnly);
    }

    #[test]
    fn test_unknown_type_has_no_home_module() {
        // MobjInfo/State: referenced by Mobj's fields, not yet translated
        // as real structs -- a known gap, not guessed at.
        assert_eq!(type_home_module("MobjInfo"), None);
        assert_eq!(type_home_module("State"), None);
        assert_eq!(type_home_module("i32"), None);
    }

    #[test]
    fn test_mobj_imports() {
        let mut items = parse_rough("info.h");
        items.extend(parse_rough("p_mobj.h"));
        let enum_typedefs = collect_enum_typedef_names(&items);
        let fields = find_typedef_struct(&items, "mobj_t").unwrap();
        let mapped = map_struct_fields(fields, &enum_typedefs).unwrap();
        let rendered = render_imports_for("p_mobj", &mapped);
        assert_eq!(
            rendered,
            "use crate::doomdata::{MapThing};\n\
             use crate::p_tick::{Thinker};\n\
             use crate::runtime::{FixedT, Handle, PlayerId, SubsectorId};"
        );
    }

    #[test]
    fn test_sector_imports() {
        let items = parse_rough("r_defs.h");
        let enum_typedefs = collect_enum_typedef_names(&items);
        let fields = find_typedef_struct(&items, "sector_t").unwrap();
        let mapped = map_struct_fields(fields, &enum_typedefs).unwrap();
        // sector_t's own module is r_defs -- DegenMobj (also r_defs) needs
        // no import; only cross-module references do.
        let rendered = render_imports_for("r_defs", &mapped);
        assert_eq!(
            rendered,
            "use crate::p_tick::{Thinker};\nuse crate::runtime::{FixedT, Handle, LineId};"
        );
    }

    #[test]
    fn test_thinker_enum_imports() {
        // The enum itself, rendered in p_tick, needs every variant's own
        // struct imported from wherever it actually lives.
        let mapped: Vec<MappedField> = [
            "FireFlicker",
            "LightFlash",
            "Strobe",
            "Glow",
            "Plat",
            "VerticalDoor",
            "Ceiling",
            "FloorMove",
            "Mobj",
        ]
        .iter()
        .map(|n| MappedField {
            name: n.to_lowercase(),
            rust_type: n.to_string(),
        })
        .collect();
        let rendered = render_imports_for("p_tick", &mapped);
        assert_eq!(
            rendered,
            "use crate::p_mobj::{Mobj};\n\
             use crate::p_spec::{Ceiling, FireFlicker, FloorMove, Glow, LightFlash, Plat, Strobe, VerticalDoor};"
        );
    }
}
