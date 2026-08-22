//! Phase 3: Thinker Struct Field Translation
//!
//! Turns a `p_spec.h`-style thinker-subclass struct's C field list into
//! its Rust equivalent, per `docs/03_TRANSPILER.md`'s Memory Model
//! decisions: the embedded `thinker_t thinker;` header every such struct
//! starts with is dropped entirely (arena identity -- the `enum Thinker`
//! wrapper plus its `Handle<Thinker>` -- replaces its role: the C struct
//! needed that header to be walkable as a node in `thinkercap`'s list and
//! dispatchable via its `function` pointer; the Rust port gets both from
//! the arena instead), and a `sector_t*` field becomes a `SectorId` (level
//! geometry's plain index newtype, not an `Option` -- corpus-checked for
//! the structs this step covers: `p_lights.c`'s tick functions dereference
//! `flash->sector` unconditionally, and every spawn function sets it from
//! a required, non-null parameter).
//!
//! **Grown as real usage demands it, not ahead of it**: started mapping
//! only a bare `int` field and a single-level `sector_t*` pointer (the
//! four lighting-effect thinkers); now also `fixed_t` (-> `FixedT`),
//! `boolean` (-> Rust's native `bool` -- Doom's state/animation code never
//! does arithmetic on this *specific* enum the way it does on the
//! animation-sequence ones, so the Enums decision's "plain `i32`, not a
//! real enum" reasoning doesn't apply to it; see `enum_values.rs`'s own
//! `false`/`true`-keyword-collision fix, which this mapping makes moot),
//! `short` (-> `i16`), and any locally `typedef enum`'d name (-> `i32`,
//! resolved generically via `collect_enum_typedef_names` rather than
//! hardcoding each enum's name -- `plat_e`/`plattype_e`/`vldoor_e`/
//! `ceiling_e`/`floor_e` are all defined directly in `p_spec.h` itself,
//! reachable by the same scan that finds the structs). A field type still
//! outside all of this fails loudly (`map_struct_fields` returns `Err`)
//! rather than silently emitting something wrong -- e.g. `mobj_t`'s own
//! fields (self-referential `mobj_s*`, `player_s*`, `state_t*`, ...) are
//! well beyond this step's scope.

use crate::parser::ast::{DeclSpecifiers, ExternalDecl, FieldDecl, StorageClass, TypeSpecifier};
use crate::parser::grammar::declarator_name;
use std::collections::HashSet;

/// A single Rust struct field, already mapped.
#[derive(Debug, Clone, PartialEq)]
pub struct MappedField {
    pub name: String,
    pub rust_type: String,
}

/// Names of every enum `items` `typedef`s directly (`typedef enum {...}
/// name;`) -- lets an enum-typed field map to `i32` (the Enums decision)
/// generically, without hardcoding each enum's name one at a time.
pub fn collect_enum_typedef_names(items: &[ExternalDecl]) -> HashSet<String> {
    let mut out = HashSet::new();
    for item in items {
        let ExternalDecl::Declaration(decl) = item else {
            continue;
        };
        if decl.specifiers.storage != Some(StorageClass::Typedef) {
            continue;
        }
        let is_enum = decl
            .specifiers
            .type_specifiers
            .iter()
            .any(|ts| matches!(ts, TypeSpecifier::Enum(_)));
        if !is_enum {
            continue;
        }
        for d in &decl.declarators {
            if let Some(name) = declarator_name(&d.declarator) {
                out.insert(name);
            }
        }
    }
    out
}

/// The Rust type for a field with `pointer_depth` `*`s over `specs`.
/// `None` for anything this step doesn't handle yet (see module docs).
fn map_type(
    specs: &DeclSpecifiers,
    pointer_depth: usize,
    enum_typedefs: &HashSet<String>,
) -> Option<String> {
    if pointer_depth == 0 {
        return match specs.type_specifiers.as_slice() {
            [TypeSpecifier::Int] => Some("i32".to_string()),
            [TypeSpecifier::Short] => Some("i16".to_string()),
            [TypeSpecifier::TypedefName(name)] if name == "fixed_t" => Some("FixedT".to_string()),
            [TypeSpecifier::TypedefName(name)] if name == "boolean" => Some("bool".to_string()),
            // angle_t is `typedef unsigned angle_t;` (tables.h) -- a plain
            // BAM-angle integer, not an enum.
            [TypeSpecifier::TypedefName(name)] if name == "angle_t" => Some("u32".to_string()),
            // mapthing_t (doomdata.h) is a plain value struct (five `short`
            // fields, no pointers -- see its own translation in this
            // module's tests) embedded by value in mobj_t.spawnpoint, not
            // referenced by pointer -- "MapThing" is this step's own
            // translated name for it, same ad hoc naming as every other
            // corpus struct this module renders (there's no automatic
            // snake_case_t -> PascalCase converter yet; each caller has
            // picked a name explicitly so far).
            [TypeSpecifier::TypedefName(name)] if name == "mapthing_t" => {
                Some("MapThing".to_string())
            }
            [TypeSpecifier::TypedefName(name)] if enum_typedefs.contains(name) => {
                Some("i32".to_string())
            }
            _ => None,
        };
    }
    if pointer_depth == 1 {
        return match specs.type_specifiers.as_slice() {
            [TypeSpecifier::TypedefName(name)] if name == "sector_t" => {
                Some("SectorId".to_string())
            }
            // mobjinfo_t*/state_t* point into mobjinfo[]/states[] (info.h):
            // static, program-lifetime, read-only tables -- never freed or
            // per-level, unlike everything else mapped so far, so a real
            // `&'static` reference is simpler and more idiomatic than an
            // index newtype once those tables exist as translated Rust
            // statics (a separate, large undertaking -- translating
            // info.c's data -- not started by this). `info` is observably
            // redundant with `type` corpus-wide (always `&mobjinfo[type]`,
            // `p_mobj.c`/`p_saveg.c`), but kept as its own field for now --
            // preserving the original's struct shape 1:1 rather than
            // silently restructuring it away.
            [TypeSpecifier::TypedefName(name)] if name == "mobjinfo_t" => {
                Some("&'static MobjInfo".to_string())
            }
            [TypeSpecifier::TypedefName(name)] if name == "state_t" => {
                Some("&'static State".to_string())
            }
            // `struct subsector_s*` (a forward reference -- `fields: None`,
            // since it's never redefined at the point of use): subsectors
            // are level geometry, same "bulk-allocated once per level,
            // never individually freed" reasoning as SectorId. Corpus-
            // checked non-Option: P_SetThingPosition sets mobj_t.subsector
            // unconditionally, and every read dereferences it without a
            // null check (see runtime/geometry.rs's own docs).
            [TypeSpecifier::Struct(spec)]
                if spec.fields.is_none() && spec.name.as_deref() == Some("subsector_s") =>
            {
                Some("SubsectorId".to_string())
            }
            // `struct mobj_s*` -- self-referential (mobj_t's own snext/
            // sprev/bnext/bprev/target/tracer). Once mobj_t becomes a
            // variant of `enum Thinker`, one of these is a `Handle<Thinker>`
            // -- genuinely nullable throughout (list-end sentinels for the
            // sector/blockmap links, "no target/tracer yet" for the attack
            // fields), unlike SectorId/SubsectorId's always-set corpus
            // usage.
            [TypeSpecifier::Struct(spec)]
                if spec.fields.is_none() && spec.name.as_deref() == Some("mobj_s") =>
            {
                Some("Option<Handle<Thinker>>".to_string())
            }
            // `struct player_s*` -- `g_game.c`'s `player_t
            // players[MAXPLAYERS];` is a plain, fixed-size, never-resized
            // array (program lifetime, not per-level), so a plain
            // `PlayerId` index (`runtime/player.rs`) like `SectorId`, no
            // arena needed. Nullable: most mobjs (monsters, items,
            // decorations) never have a player attached, only actual
            // player pawns do.
            [TypeSpecifier::Struct(spec)]
                if spec.fields.is_none() && spec.name.as_deref() == Some("player_s") =>
            {
                Some("Option<PlayerId>".to_string())
            }
            _ => None,
        };
    }
    None
}

/// True for the embedded `thinker_t thinker;` header -- see module docs.
fn is_thinker_header(specs: &DeclSpecifiers) -> bool {
    matches!(specs.type_specifiers.as_slice(), [TypeSpecifier::TypedefName(name)] if name == "thinker_t")
}

/// Rust strict/reserved keywords, minus the four that raw-identifier
/// syntax still can't escape (`crate`/`self`/`super`/`Self`) and `true`/
/// `false` (literal tokens, not identifier-class keywords at all -- `r#true`
/// isn't valid syntax, same reasoning as `enum_values.rs`'s own
/// keyword-collision fix, just a different consequence: that module skips
/// the colliding item since an enum constant has no other name to fall
/// back to, but a field can be escaped via `r#name` for every *other*
/// keyword collision). Found via `plat_t`/`vldoor_t`/`ceiling_t`/
/// `floormove_t`'s own `type` field, the only one in the corpus's fields
/// this step has translated so far.
const RUST_ESCAPABLE_KEYWORDS: &[&str] = &[
    "as", "break", "const", "continue", "else", "enum", "extern", "fn", "for", "if", "impl", "in",
    "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "static", "struct",
    "trait", "type", "unsafe", "use", "where", "while", "dyn", "async", "await",
];

const RUST_NON_ESCAPABLE_KEYWORDS: &[&str] = &["crate", "self", "super", "Self", "true", "false"];

/// The Rust-safe identifier for a C field named `name`: unchanged if it's
/// not a keyword, `r#name` if it's an escapable one (`type` -> `r#type`),
/// or `Err` if it collides with one of the four keywords raw-identifier
/// syntax can't escape at all -- fails loudly rather than emitting
/// something that won't compile.
fn rust_field_name(name: &str) -> Result<String, String> {
    if RUST_NON_ESCAPABLE_KEYWORDS.contains(&name) {
        return Err(format!(
            "field name `{name}` collides with a Rust keyword that can't be raw-identifier-escaped"
        ));
    }
    if RUST_ESCAPABLE_KEYWORDS.contains(&name) {
        return Ok(format!("r#{name}"));
    }
    Ok(name.to_string())
}

/// Maps every field of a defining struct/union's field list to its Rust
/// equivalent, in source order, dropping the `thinker_t` header field.
/// `enum_typedefs` should come from `collect_enum_typedef_names`, scanned
/// over the same item list `fields` was found in. `Err` names the first
/// field this step can't map yet, rather than silently emitting a partial
/// or wrong struct.
pub fn map_struct_fields(
    fields: &[FieldDecl],
    enum_typedefs: &HashSet<String>,
) -> Result<Vec<MappedField>, String> {
    let mut out = Vec::new();
    for field in fields {
        if is_thinker_header(&field.specifiers) {
            continue;
        }
        for (declarator, bitwidth) in &field.declarators {
            if bitwidth.is_some() {
                return Err("bit-fields not supported yet".to_string());
            }
            let declarator = declarator
                .as_ref()
                .ok_or_else(|| "anonymous field has no name".to_string())?;
            let c_name = declarator_name(declarator)
                .ok_or_else(|| "field declarator has no name".to_string())?;
            let name = rust_field_name(&c_name)?;
            let pointer_depth = declarator.pointer_quals.len();
            let rust_type = map_type(&field.specifiers, pointer_depth, enum_typedefs)
                .ok_or_else(|| format!("no type mapping yet for field `{c_name}`"))?;
            out.push(MappedField { name, rust_type });
        }
    }
    Ok(out)
}

/// Renders `name`'s Rust struct definition from its already-mapped fields.
pub fn render_struct(name: &str, fields: &[MappedField]) -> String {
    let mut out = format!("pub struct {name} {{\n");
    for f in fields {
        out.push_str(&format!("    pub {}: {},\n", f.name, f.rust_type));
    }
    out.push('}');
    out
}

/// Finds a top-level `typedef struct { ... } name;`'s defining field list
/// (or a `typedef struct Tag { ... } name;`'s), by the typedef name --
/// these thinker-subclass structs are always anonymous-struct typedefs in
/// the corpus, only reachable this way, not by a struct tag.
///
/// Takes a raw `&[ExternalDecl]`, not a `TranslationUnit`, so it works
/// with the rough, typedef-table-free scan (`extract_top_level_decls`)
/// this module's own tests need for `p_spec.h`: like several headers this
/// codebase's other resolvers already document (e.g. `visibility.rs`'s
/// `own_matching_header_names`), it relies on `boolean` (`extern boolean
/// levelTimer;`, near its own top) being typedef'd by whatever `.c` file
/// includes it first, not by its own `#include` graph -- `parse_full`
/// fails on it standalone, and even parsing a `.c` file that includes it
/// doesn't splice its struct *definitions* into that file's own
/// `TranslationUnit.items` (`parse_full`'s header handling only seeds
/// typedef *names* for the parser, the same as Step 6b's own import
/// model). The rough scan needs no typedef table at all for reading off
/// field declarator shapes, so it works on `p_spec.h` directly.
pub fn find_typedef_struct<'a>(
    items: &'a [ExternalDecl],
    typedef_name: &str,
) -> Option<&'a [FieldDecl]> {
    for item in items {
        let ExternalDecl::Declaration(decl) = item else {
            continue;
        };
        if decl.specifiers.storage != Some(StorageClass::Typedef) {
            continue;
        }
        let names_this = decl
            .declarators
            .iter()
            .any(|d| declarator_name(&d.declarator).as_deref() == Some(typedef_name));
        if !names_this {
            continue;
        }
        for ts in &decl.specifiers.type_specifiers {
            if let TypeSpecifier::Struct(spec) = ts
                && let Some(fields) = &spec.fields
            {
                return Some(fields);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{attach_comments, lex_chunks, parse};
    use std::path::Path;

    fn corpus_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    /// The rough, typedef-table-free scan (see this module's own docs on
    /// `find_typedef_struct`) -- these headers can't go through
    /// `parse_full` standalone (they rely on typedefs seeded by whichever
    /// `.c` file includes them first, not their own `#include` graph), and
    /// even a `.c` file that does include one doesn't splice its struct
    /// *definitions* into that file's own `TranslationUnit.items`.
    fn parse_rough(corpus_file: &str) -> Vec<ExternalDecl> {
        let path = corpus_dir().join(corpus_file);
        let (_, resolved) = parse(path.to_str().unwrap()).unwrap();
        let entries = lex_chunks(&resolved).unwrap();
        let stream = attach_comments(entries);
        crate::parser::grammar::extract_top_level_decls(&stream)
    }

    fn parse_p_spec() -> Vec<ExternalDecl> {
        parse_rough("p_spec.h")
    }

    /// Same rough-scan pipeline, over a literal snippet rather than a
    /// corpus file -- for negative-case tests that shouldn't depend on
    /// which specific corpus struct happens to have an unmapped field.
    fn parse_items(src: &str) -> Vec<ExternalDecl> {
        let (_, chunks) = crate::parser::parse_chunks(src);
        let mut env = crate::parser::PreprocessorEnv::linux_doom_defaults();
        let resolved = crate::parser::resolve_conditionals(&chunks, &mut env).unwrap();
        let entries = lex_chunks(&resolved).unwrap();
        let stream = attach_comments(entries);
        crate::parser::grammar::extract_top_level_decls(&stream)
    }

    fn field(name: &str, rust_type: &str) -> MappedField {
        MappedField {
            name: name.to_string(),
            rust_type: rust_type.to_string(),
        }
    }

    #[test]
    fn test_finds_fireflicker_fields() {
        let items = parse_p_spec();
        let fields = find_typedef_struct(&items, "fireflicker_t").expect("fireflicker_t not found");
        // thinker_t thinker; sector_t* sector; int count; int maxlight; int minlight;
        assert_eq!(fields.len(), 5);
    }

    #[test]
    fn test_maps_and_renders_fireflicker() {
        let items = parse_p_spec();
        let fields = find_typedef_struct(&items, "fireflicker_t").unwrap();
        let enum_typedefs = collect_enum_typedef_names(&items);
        let mapped = map_struct_fields(fields, &enum_typedefs).expect("should map cleanly");
        assert_eq!(
            mapped,
            vec![
                field("sector", "SectorId"),
                field("count", "i32"),
                field("maxlight", "i32"),
                field("minlight", "i32"),
            ]
        );
        assert_eq!(
            render_struct("FireFlicker", &mapped),
            "pub struct FireFlicker {\n    \
             pub sector: SectorId,\n    \
             pub count: i32,\n    \
             pub maxlight: i32,\n    \
             pub minlight: i32,\n\
             }"
        );
    }

    #[test]
    fn test_maps_all_four_simple_light_thinkers() {
        let items = parse_p_spec();
        let enum_typedefs = collect_enum_typedef_names(&items);
        for (typedef_name, expected_field_count) in [
            ("fireflicker_t", 4),
            ("lightflash_t", 6),
            ("strobe_t", 6),
            ("glow_t", 4),
        ] {
            let fields = find_typedef_struct(&items, typedef_name)
                .unwrap_or_else(|| panic!("{typedef_name} not found"));
            let mapped = map_struct_fields(fields, &enum_typedefs)
                .unwrap_or_else(|e| panic!("{typedef_name} failed to map: {e}"));
            assert_eq!(
                mapped.len(),
                expected_field_count,
                "{typedef_name}: unexpected field count after dropping thinker_t header"
            );
            assert!(
                mapped.iter().all(|f| f.name != "thinker"),
                "{typedef_name}: thinker_t header should have been dropped"
            );
        }
    }

    #[test]
    fn test_maps_plat_t_exactly() {
        // Exercises fixed_t, boolean, and two different local enum
        // typedefs (plat_e, plattype_e) in one struct.
        let items = parse_p_spec();
        let enum_typedefs = collect_enum_typedef_names(&items);
        let fields = find_typedef_struct(&items, "plat_t").expect("plat_t not found");
        let mapped = map_struct_fields(fields, &enum_typedefs).expect("should map cleanly");
        assert_eq!(
            mapped,
            vec![
                field("sector", "SectorId"),
                field("speed", "FixedT"),
                field("low", "FixedT"),
                field("high", "FixedT"),
                field("wait", "i32"),
                field("count", "i32"),
                field("status", "i32"),
                field("oldstatus", "i32"),
                field("crush", "bool"),
                field("tag", "i32"),
                field("r#type", "i32"),
            ]
        );
    }

    #[test]
    fn test_maps_vldoor_t_exactly() {
        let items = parse_p_spec();
        let enum_typedefs = collect_enum_typedef_names(&items);
        let fields = find_typedef_struct(&items, "vldoor_t").expect("vldoor_t not found");
        let mapped = map_struct_fields(fields, &enum_typedefs).expect("should map cleanly");
        assert_eq!(
            mapped,
            vec![
                field("r#type", "i32"),
                field("sector", "SectorId"),
                field("topheight", "FixedT"),
                field("speed", "FixedT"),
                field("direction", "i32"),
                field("topwait", "i32"),
                field("topcountdown", "i32"),
            ]
        );
    }

    #[test]
    fn test_maps_ceiling_t_exactly() {
        let items = parse_p_spec();
        let enum_typedefs = collect_enum_typedef_names(&items);
        let fields = find_typedef_struct(&items, "ceiling_t").expect("ceiling_t not found");
        let mapped = map_struct_fields(fields, &enum_typedefs).expect("should map cleanly");
        assert_eq!(
            mapped,
            vec![
                field("r#type", "i32"),
                field("sector", "SectorId"),
                field("bottomheight", "FixedT"),
                field("topheight", "FixedT"),
                field("speed", "FixedT"),
                field("crush", "bool"),
                field("direction", "i32"),
                field("tag", "i32"),
                field("olddirection", "i32"),
            ]
        );
    }

    #[test]
    fn test_maps_floormove_t_exactly() {
        // Exercises `short` (texture), the last previously-unmapped kind.
        let items = parse_p_spec();
        let enum_typedefs = collect_enum_typedef_names(&items);
        let fields = find_typedef_struct(&items, "floormove_t").expect("floormove_t not found");
        let mapped = map_struct_fields(fields, &enum_typedefs).expect("should map cleanly");
        assert_eq!(
            mapped,
            vec![
                field("r#type", "i32"),
                field("crush", "bool"),
                field("sector", "SectorId"),
                field("direction", "i32"),
                field("newspecial", "i32"),
                field("texture", "i16"),
                field("floordestheight", "FixedT"),
                field("speed", "FixedT"),
            ]
        );
    }

    #[test]
    fn test_collect_enum_typedef_names_finds_all_p_spec_enums() {
        let items = parse_p_spec();
        let names = collect_enum_typedef_names(&items);
        for expected in ["plat_e", "plattype_e", "vldoor_e", "ceiling_e", "floor_e"] {
            assert!(names.contains(expected), "missing {expected}: {names:?}");
        }
    }

    #[test]
    fn test_unmapped_field_type_fails_loudly() {
        let items = parse_items("typedef struct { thinker_t thinker; double weird; } widget_t;");
        let fields = find_typedef_struct(&items, "widget_t").expect("widget_t not found");
        let enum_typedefs = collect_enum_typedef_names(&items);
        let result = map_struct_fields(fields, &enum_typedefs);
        assert!(
            result.is_err(),
            "double isn't mapped yet -- should fail, not guess"
        );
    }

    #[test]
    fn test_rust_field_name_escapes_type_keyword() {
        assert_eq!(rust_field_name("type").unwrap(), "r#type");
        assert_eq!(rust_field_name("sector").unwrap(), "sector");
    }

    #[test]
    fn test_rust_field_name_rejects_non_escapable_keywords() {
        for kw in ["crate", "self", "super", "Self", "true", "false"] {
            assert!(rust_field_name(kw).is_err(), "{kw} should not be escapable");
        }
    }

    #[test]
    fn test_maps_mapthing_t_exactly() {
        // doomdata.h's mapthing_t: a plain value struct, five `short`
        // fields, no pointers, not a thinker subclass at all (no embedded
        // thinker_t header to drop) -- used as mobj_t.spawnpoint (by
        // value) and as raw level-load data elsewhere.
        let items = parse_rough("doomdata.h");
        let enum_typedefs = collect_enum_typedef_names(&items);
        let fields = find_typedef_struct(&items, "mapthing_t").expect("mapthing_t not found");
        let mapped = map_struct_fields(fields, &enum_typedefs).expect("should map cleanly");
        assert_eq!(
            mapped,
            vec![
                field("x", "i16"),
                field("y", "i16"),
                field("angle", "i16"),
                field("r#type", "i16"),
                field("options", "i16"),
            ]
        );
    }

    #[test]
    fn test_mobj_t_maps_completely() {
        // mobj_t (p_mobj.h), the 9th and by far largest thinker subclass,
        // now maps end to end: mobjinfo_t*/state_t* as &'static references
        // into mobjinfo[]/states[] (info.h) -- genuinely static, program-
        // lifetime, read-only tables; `player: struct player_s*` as
        // `Option<PlayerId>` (`g_game.c`'s `player_t players[MAXPLAYERS];`
        // is a plain, fixed-size, never-resized array, program lifetime
        // not per-level -- the same "no generation counter needed"
        // reasoning as `SectorId`, so no separate arena turned out to be
        // needed); `spawnpoint: mapthing_t` as the embedded `MapThing`
        // value struct (`test_maps_mapthing_t_exactly`, above).
        //
        // spritenum_t/mobjtype_t are typedef'd enums in info.h, not
        // p_mobj.h itself -- collect_enum_typedef_names needs both files'
        // items, mirroring how a real caller would eventually need to
        // merge a struct's own #include closure, not just its own file.
        let mut items = parse_rough("info.h");
        items.extend(parse_rough("p_mobj.h"));
        let enum_typedefs = collect_enum_typedef_names(&items);
        let fields = find_typedef_struct(&items, "mobj_t").expect("mobj_t not found");
        let mapped = map_struct_fields(fields, &enum_typedefs).expect("mobj_t should map cleanly");

        assert_eq!(
            mapped,
            vec![
                field("x", "FixedT"),
                field("y", "FixedT"),
                field("z", "FixedT"),
                field("snext", "Option<Handle<Thinker>>"),
                field("sprev", "Option<Handle<Thinker>>"),
                field("angle", "u32"),
                field("sprite", "i32"),
                field("frame", "i32"),
                field("bnext", "Option<Handle<Thinker>>"),
                field("bprev", "Option<Handle<Thinker>>"),
                field("subsector", "SubsectorId"),
                field("floorz", "FixedT"),
                field("ceilingz", "FixedT"),
                field("radius", "FixedT"),
                field("height", "FixedT"),
                field("momx", "FixedT"),
                field("momy", "FixedT"),
                field("momz", "FixedT"),
                field("validcount", "i32"),
                field("r#type", "i32"),
                field("info", "&'static MobjInfo"),
                field("tics", "i32"),
                field("state", "&'static State"),
                field("flags", "i32"),
                field("health", "i32"),
                field("movedir", "i32"),
                field("movecount", "i32"),
                field("target", "Option<Handle<Thinker>>"),
                field("reactiontime", "i32"),
                field("threshold", "i32"),
                field("player", "Option<PlayerId>"),
                field("lastlook", "i32"),
                field("spawnpoint", "MapThing"),
                field("tracer", "Option<Handle<Thinker>>"),
            ]
        );
    }
}
