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
//! **Deliberately narrow for now**: only maps what the first four thinker
//! structs this step actually translates use -- a bare `int` field and a
//! single-level `sector_t*` pointer. `fixed_t`, `boolean`, and enum-typed
//! fields (needed by `plat_t`/`vldoor_t`/`ceiling_t`/`floormove_t`, not
//! yet translated) intentionally have no mapping yet, so a field this step
//! doesn't understand fails loudly (`map_struct_fields` returns `Err`)
//! rather than silently emitting something wrong -- matching this
//! project's "measure real usage before building the general case"
//! practice throughout.

use crate::parser::ast::{DeclSpecifiers, ExternalDecl, FieldDecl, StorageClass, TypeSpecifier};
use crate::parser::grammar::declarator_name;

/// A single Rust struct field, already mapped.
#[derive(Debug, Clone, PartialEq)]
pub struct MappedField {
    pub name: String,
    pub rust_type: String,
}

/// The Rust type for a field with `pointer_depth` `*`s over `specs`.
/// `None` for anything this step doesn't handle yet (see module docs).
fn map_type(specs: &DeclSpecifiers, pointer_depth: usize) -> Option<String> {
    match (pointer_depth, specs.type_specifiers.as_slice()) {
        (0, [TypeSpecifier::Int]) => Some("i32".to_string()),
        (1, [TypeSpecifier::TypedefName(name)]) if name == "sector_t" => {
            Some("SectorId".to_string())
        }
        _ => None,
    }
}

/// True for the embedded `thinker_t thinker;` header -- see module docs.
fn is_thinker_header(specs: &DeclSpecifiers) -> bool {
    matches!(specs.type_specifiers.as_slice(), [TypeSpecifier::TypedefName(name)] if name == "thinker_t")
}

/// Maps every field of a defining struct/union's field list to its Rust
/// equivalent, in source order, dropping the `thinker_t` header field.
/// `Err` names the first field this step can't map yet, rather than
/// silently emitting a partial or wrong struct.
pub fn map_struct_fields(fields: &[FieldDecl]) -> Result<Vec<MappedField>, String> {
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
            let name = declarator_name(declarator)
                .ok_or_else(|| "field declarator has no name".to_string())?;
            let pointer_depth = declarator.pointer_quals.len();
            let rust_type = map_type(&field.specifiers, pointer_depth)
                .ok_or_else(|| format!("no type mapping yet for field `{name}`"))?;
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
    /// `find_typedef_struct`) -- `p_spec.h` can't go through `parse_full`
    /// standalone (it relies on `boolean` being typedef'd by whichever
    /// `.c` file includes it first, not by its own `#include` graph), and
    /// even a `.c` file that does include it doesn't splice its struct
    /// *definitions* into that file's own `TranslationUnit.items`.
    fn parse_p_spec() -> Vec<ExternalDecl> {
        let path = corpus_dir().join("p_spec.h");
        let (_, resolved) = parse(path.to_str().unwrap()).unwrap();
        let entries = lex_chunks(&resolved).unwrap();
        let stream = attach_comments(entries);
        crate::parser::grammar::extract_top_level_decls(&stream)
    }

    #[test]
    fn test_finds_fireflicker_fields() {
        let unit = parse_p_spec();
        let fields = find_typedef_struct(&unit, "fireflicker_t").expect("fireflicker_t not found");
        // thinker_t thinker; sector_t* sector; int count; int maxlight; int minlight;
        assert_eq!(fields.len(), 5);
    }

    #[test]
    fn test_maps_and_renders_fireflicker() {
        let unit = parse_p_spec();
        let fields = find_typedef_struct(&unit, "fireflicker_t").unwrap();
        let mapped = map_struct_fields(fields).expect("should map cleanly");
        assert_eq!(
            mapped,
            vec![
                MappedField {
                    name: "sector".to_string(),
                    rust_type: "SectorId".to_string()
                },
                MappedField {
                    name: "count".to_string(),
                    rust_type: "i32".to_string()
                },
                MappedField {
                    name: "maxlight".to_string(),
                    rust_type: "i32".to_string()
                },
                MappedField {
                    name: "minlight".to_string(),
                    rust_type: "i32".to_string()
                },
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
        let unit = parse_p_spec();
        for (typedef_name, expected_field_count) in [
            ("fireflicker_t", 4),
            ("lightflash_t", 6),
            ("strobe_t", 6),
            ("glow_t", 4),
        ] {
            let fields = find_typedef_struct(&unit, typedef_name)
                .unwrap_or_else(|| panic!("{typedef_name} not found"));
            let mapped = map_struct_fields(fields)
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
    fn test_unmapped_field_type_fails_loudly() {
        // fixed_t isn't mapped yet (see module docs) -- plat_t has one.
        let unit = parse_p_spec();
        let fields = find_typedef_struct(&unit, "plat_t").expect("plat_t not found");
        let result = map_struct_fields(fields);
        assert!(
            result.is_err(),
            "plat_t has fixed_t/enum fields this step doesn't map yet -- should fail, not guess"
        );
    }
}
