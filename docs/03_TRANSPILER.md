# Phase 3: Transpilation & Code Generation Plan

This phase translates the annotated Typed AST from Phase 2 into target programming languages, starting with Rust and extensible to .NET (C#).

---

## 🦀 Target 1: Rust Transpiler

### 1. Architectural Choices
- **Memory Model**:
  - Global game state encapsulation (e.g. `DoomState` struct or safe static modules).
  - Pointers & References: Determine safe Rust representations vs. index-based arena allocators for Doom's `thinker_t`, `mobj_t`, and sector references.
- **Fixed-Point Arithmetic**:
  - Map `fixed_t` directly to a dedicated Rust newtype (`FixedT(i32)`) or primitive with helper methods for `FixedMul` / `FixedDiv`.
- **Enums & Structs**:
  - Convert C enums into type-safe Rust enums with `#[repr(i32)]` or `#[repr(u8)]`.
  - Convert C structs into Rust structs with appropriate derives.
- **Doom Action Pointers**:
  - Map function pointers to Rust function pointers (`fn(...)`) or enum-based dispatch tables.
- **Module Structure (`.c`/`.h` -> Rust module)**:
  - Each `.c`/`.h` pair (e.g. `p_map.c` + `p_map.h`) merges into one Rust module. Items declared in the `.h` become `pub`; items that only exist in the `.c` stay private. This mirrors Rust's own visibility model and the original C author's intent -- the header already *is* their declaration of "here's my public API."
  - **Headers with no matching `.c`** (pure declaration headers: shared typedefs/structs/macros, no implementation) become their own module, with everything `pub`. Confirmed via corpus scan: 13 such headers exist -- `doomtype.h`, `doomdata.h`, `d_player.h`, `d_think.h`, `d_ticcmd.h`, `d_event.h`, `d_textur.h`, `d_englsh.h`, `d_french.h`, `p_local.h`, `r_defs.h`, `r_local.h`, `r_state.h`. **Implemented**: `transpiler/src/codegen/modules.rs` (`build_module_graph`) enumerates all 75 modules (62 Source, one per `.c` file, plus these 13 HeaderOnly) and, for every Source module, which of the 13 it needs to `use` items from -- following its own `#include` graph *transitively*, the same closure `ExportResolver`/`DeclaredTypesResolver`/`ImportResolver` already walk for their own purposes, reused here rather than reimplemented. 55 of the 62 Source modules need at least one; `doomtype` alone is needed by 53. **Not yet computed**: `.c`-to-`.c` module `use` edges (module A calling a function Step 0 found `pub` in module B) -- unlike header-only imports, a `.c` file never `#include`s another `.c` file's module in the source, so there's no `#include` graph to walk for it; that needs real identifier-usage resolution (which function bodies in A actually reference which cross-module symbol) instead, left for a follow-up step.
  - **Implemented -- true `extern` linkage vs. "declared in the matching header"**: "declared in this file's own `.h` => `pub`, else private" is a heuristic that follows C *convention*, not C's actual linkage rules. Measured with `transpiler/examples/extern_linkage_survey.rs` (see `docs/KNOWN_LIMITATIONS.md`): of 1360 externally-linked definitions corpus-wide, only 25% were both declared in their own matching header *and* genuinely visible to other files only through it. `transpiler/src/codegen/visibility.rs` (`resolve_module_visibility`) replaces it with four combined signals: (1) declared by any header reachable through the defining `.c` file's own `#include` graph (`ExportResolver::resolve_via_includes`, not just the name-matching header -- covers `doomstat.h`-style centralization and the 13 no-matching-header `.c` files, whose symbols attach to whichever shared header they actually include, e.g. `p_local.h`); (2) declared in the file's own matching header even when that file never `#include`s it itself (`r_bsp.c` never includes `r_bsp.h`, yet `r_bsp.h`'s declarations are still real); (3) redeclared as a bare prototype/`extern` directly at another `.c` file's own top level, bypassing headers on both ends (`RawDeclarationIndex`); private otherwise. Cross-checked against the survey's independent "used elsewhere" ground truth: 2 of 1360 definitions remain wrongly private (`rndindex`, `skyflatnum` -- both declared only in `doomstat.h`, which their defining files have no `#include` relationship to at all), left undone as not worth more machinery for a two-symbol gap.

---

## 🔷 Target 2: .NET / C# Transpiler

### 1. Architectural Choices
- Generate C# structs / classes.
- Use `unsafe` blocks for low-level memory and buffer manipulation if required, or translate to safe managed arrays / `Span<T>`.
- Map global state to static or singleton manager classes.

---

## 🧪 Validation & Execution Strategy
1. **Compilation Validation**: Ensure generated Rust code compiles cleanly via `cargo check` and `cargo build`.
2. **Deterministic Output Comparison**: Compare game tick execution and fixed-point math against native Doom to verify behavior equivalence.
