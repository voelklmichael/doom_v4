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
  - **Headers with no matching `.c`** (pure declaration headers: shared typedefs/structs/macros, no implementation) become their own module, with everything `pub`. Confirmed via corpus scan: 13 such headers exist -- `doomtype.h`, `doomdata.h`, `d_player.h`, `d_think.h`, `d_ticcmd.h`, `d_event.h`, `d_textur.h`, `d_englsh.h`, `d_french.h`, `p_local.h`, `r_defs.h`, `r_local.h`, `r_state.h`. These modules get imported by whichever `.c`+`.h` module pairs need them -- the same import graph Step 6b (`transpiler/src/parser/imports.rs`) already builds for typedef resolution can drive this directly.
  - **Resolved (investigated) -- true `extern` linkage vs. "declared in the matching header"**: "declared in this file's own `.h` => `pub`, else private" is a heuristic that follows C *convention*, not C's actual linkage rules. In real C, any non-`static` function or global has external linkage regardless of whether it appears in a header at all -- another `.c` file could call it via its own ad-hoc `extern` declaration, without it ever showing up in the "matching" header. `linuxdoom-1.10` does this: e.g. `doomstat.h` centralizes `extern` declarations for globals defined across several unrelated `.c` files, rather than each `.c` owning its own header for them. Measured with `transpiler/examples/extern_linkage_survey.rs` (see `docs/KNOWN_LIMITATIONS.md`): of 1360 externally-linked definitions corpus-wide, only 25% are both declared in their own matching header *and* genuinely visible to other files only through it; 16% are declared in some *other* header instead (spanning 40% of `.c` files, not a handful of outliers); another 16% come from one of the 13 `.c` files with no matching header at all (mostly `p_*.c` files meant to be declared through the shared `p_local.h`); the remaining 43% are externally linked but never actually used outside their own file, so "private" happens to be correct for them regardless. **Direction**: the merge rule must resolve a symbol's visibility from *every* header that declares it -- not just the name-matching one -- unioning across a `.c` file's real `#include` graph (the same one `ExportResolver`, Phase 2 Step 0, already walks), with the 13 no-matching-header files' symbols attached to whichever shared header actually declares them. Implementation still unwritten.

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
