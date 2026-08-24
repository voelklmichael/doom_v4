//! Phase 3: The `Thinker` Enum
//!
//! Every real thinker subclass (`struct_fields.rs`, all 9 now translated)
//! wraps into one closed `enum Thinker { Mobj(Mobj), Ceiling(Ceiling), ...
//! }`, per `docs/03_TRANSPILER.md`'s Memory Model decision -- a `match`
//! replacing the original's hand-rolled C vtable (`thinker_t.function`,
//! called as a function pointer), avoiding `Box<dyn Thinker>`'s per-item
//! heap allocation on a set that will never actually grow.
//!
//! **Shape only, not logic**: this renders the enum itself and a stub
//! `match` dispatch (`todo!()` in every arm), not real tick-function
//! bodies. Every step so far in Phase 3 -- module graph, `use` blocks,
//! enum constants, struct fields -- has been data *shape*; translating
//! `T_FireFlicker`'s actual random-timer logic (or any other tick
//! function's body) needs C-statement/expression-to-Rust transpilation,
//! which nothing in this codebase does yet. That's a large, separate
//! undertaking, deliberately not started here. The dispatch's own
//! signature (what context a real tick needs -- world state, the arena
//! itself, ...) is likewise left unsettled until real bodies force the
//! question -- guessing it now, before any body exists to check it
//! against, would just mean redoing it later.

use crate::codegen::struct_fields::{
    collect_enum_typedef_names, find_typedef_struct, map_struct_fields, render_struct,
};
use crate::parser::ast::ExternalDecl;
use crate::parser::{attach_comments, lex_chunks, parse};
use std::path::Path;

/// (C typedef name, Rust struct/variant name) for the 9 real thinker
/// subclasses, in wrapper-order -- confirmed by direct corpus read
/// (`slidedoor_t` is `#if 0`'d out dead code, not a 10th). No automatic
/// snake_case_t -> PascalCase converter exists, or would even help here:
/// `fireflicker_t` has no underscore to split on, so a naive converter
/// gives `Fireflicker`, not the more readable `FireFlicker`. Names are
/// picked explicitly, matching `struct_fields.rs`'s own precedent for
/// `mapthing_t` -> `MapThing`. `vldoor_t` -> `VerticalDoor`, not `VlDoor`,
/// matching the original's own function name (`T_VerticalDoor`) rather
/// than the struct's more cryptic one.
const THINKER_VARIANTS: &[(&str, &str)] = &[
    ("fireflicker_t", "FireFlicker"),
    ("lightflash_t", "LightFlash"),
    ("strobe_t", "Strobe"),
    ("glow_t", "Glow"),
    ("plat_t", "Plat"),
    ("vldoor_t", "VerticalDoor"),
    ("ceiling_t", "Ceiling"),
    ("floormove_t", "FloorMove"),
    ("mobj_t", "Mobj"),
];

fn rough_scan(path: &Path) -> Vec<ExternalDecl> {
    let (_, resolved) = parse(path.to_str().unwrap()).unwrap_or_else(|e| panic!("{e}"));
    let entries = lex_chunks(&resolved).expect("lexing should succeed");
    let stream = attach_comments(entries);
    crate::parser::grammar::extract_top_level_decls(&stream)
}

/// Renders every thinker subclass's own struct definition against the
/// real corpus at `corpus_dir`, in `THINKER_VARIANTS` order. All 8
/// non-`mobj_t` variants are found via `p_spec.h` alone; `mobj_t` needs
/// `info.h` merged with `p_mobj.h` (for `spritenum_t`/`mobjtype_t`'s enum
/// typedefs, defined in `info.h`, not `p_mobj.h` itself -- see
/// `struct_fields.rs`'s own `mobj_t` test).
pub fn render_thinker_structs(corpus_dir: &Path) -> Vec<String> {
    let p_spec = rough_scan(&corpus_dir.join("p_spec.h"));
    let mut mobj_items = rough_scan(&corpus_dir.join("info.h"));
    mobj_items.extend(rough_scan(&corpus_dir.join("p_mobj.h")));

    THINKER_VARIANTS
        .iter()
        .map(|(typedef_name, rust_name)| {
            let items: &[ExternalDecl] = if *typedef_name == "mobj_t" {
                &mobj_items
            } else {
                &p_spec
            };
            let enum_typedefs = collect_enum_typedef_names(items);
            let fields = find_typedef_struct(items, typedef_name)
                .unwrap_or_else(|| panic!("{typedef_name} not found in the corpus"));
            let mapped = map_struct_fields(fields, &enum_typedefs)
                .unwrap_or_else(|e| panic!("{typedef_name} failed to map: {e}"));
            render_struct(rust_name, &mapped)
        })
        .collect()
}

/// Renders the `enum Thinker { ... }` wrapper -- data shape only, no
/// dispatch logic (see this module's own docs on why).
pub fn render_thinker_enum() -> String {
    let mut out = "pub enum Thinker {\n".to_string();
    for (_, rust_name) in THINKER_VARIANTS {
        out.push_str(&format!("    {rust_name}({rust_name}),\n"));
    }
    out.push('}');
    out
}

/// A stub `match` dispatch: one arm per variant, `todo!()` body. Takes
/// `world: &mut World` -- confirmed necessary (not just anticipated) by
/// `function_body.rs`'s translation of `T_FireFlicker`, the first real
/// tick body: it needs `World` to resolve a `SectorId` cross-reference
/// field back to a real `&mut Sector`. Also takes `handle: Handle<Thinker>`
/// and `arena: &mut Arena<Thinker>`, matching `Arena::run`'s own closure
/// shape exactly (`FnMut(&mut T, Handle<T>, &mut Arena<T>)`) -- confirmed
/// necessary by `T_VerticalDoor`, the first complete real tick body that
/// removes itself (`P_RemoveThinker(&door->thinker)` -> `arena.
/// remove(handle)`); not every individual variant's own translated
/// function needs both (`render_fn` only adds them to a function's own
/// signature when its body actually self-removes), but the dispatch
/// itself needs them on hand to route through to whichever arm does.
/// Arms stay `todo!()` here since wiring a real body into this dispatch
/// also needs the function's own module placement settled (`p_lights.c`
/// vs `p_spec`'s own `Source` module -- not yet decided, see
/// `docs/03_TRANSPILER.md`), not just the signature.
pub fn render_thinker_dispatch_stub() -> String {
    let mut out = "impl Thinker {\n    pub fn tick(&mut self, world: &mut World, handle: Handle<Thinker>, arena: &mut Arena<Thinker>) {\n        match self {\n"
            .to_string();
    for (_, rust_name) in THINKER_VARIANTS {
        out.push_str(&format!(
            "            Thinker::{rust_name}(_) => todo!(\"{rust_name} tick logic\"),\n"
        ));
    }
    out.push_str("        }\n    }\n}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_renders_all_nine_thinker_structs() {
        let rendered = render_thinker_structs(&corpus_dir());
        assert_eq!(rendered.len(), 9);
        for (i, (_, rust_name)) in THINKER_VARIANTS.iter().enumerate() {
            assert!(
                rendered[i].starts_with(&format!("pub struct {rust_name} {{")),
                "expected {rust_name} at position {i}, got: {}",
                &rendered[i][..rendered[i].len().min(60)]
            );
        }
    }

    #[test]
    fn test_thinker_enum_shape() {
        let rendered = render_thinker_enum();
        assert_eq!(
            rendered,
            "pub enum Thinker {\n\
             \x20   FireFlicker(FireFlicker),\n\
             \x20   LightFlash(LightFlash),\n\
             \x20   Strobe(Strobe),\n\
             \x20   Glow(Glow),\n\
             \x20   Plat(Plat),\n\
             \x20   VerticalDoor(VerticalDoor),\n\
             \x20   Ceiling(Ceiling),\n\
             \x20   FloorMove(FloorMove),\n\
             \x20   Mobj(Mobj),\n\
             }"
        );
    }

    #[test]
    fn test_dispatch_stub_has_one_arm_per_variant() {
        let rendered = render_thinker_dispatch_stub();
        for (_, rust_name) in THINKER_VARIANTS {
            assert!(
                rendered.contains(&format!("Thinker::{rust_name}(_) => todo!(")),
                "missing dispatch arm for {rust_name} in:\n{rendered}"
            );
        }
    }

    #[test]
    fn test_variant_names_match_between_enum_and_dispatch() {
        let enum_rendered = render_thinker_enum();
        let dispatch_rendered = render_thinker_dispatch_stub();
        for (_, rust_name) in THINKER_VARIANTS {
            assert!(enum_rendered.contains(&format!("{rust_name}({rust_name})")));
            assert!(dispatch_rendered.contains(&format!("Thinker::{rust_name}(")));
        }
    }
}
