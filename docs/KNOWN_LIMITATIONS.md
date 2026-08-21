# Known Limitations

Deviations from strict C89 semantics that the parser pipeline (`docs/01_PARSER.md`)
currently accepts as-is, because they don't (yet) affect `linuxdoom-1.10`'s actual
build. Each entry notes why it doesn't matter today and what would need to change
if it ever does.

---

## Wide literals (`L"..."`, `L'...'`) are not lexed as a single token

**Where**: Step 2 partitioning (`transpiler/src/parser/partitioner.rs`), Step 4 lexing
(`transpiler/src/parser/lexer.rs`).

Strict C89 grammar treats `L"foo"` / `L'x'` (a wide string/char literal) as one token.
Our partitioner starts a literal chunk the moment it sees `"` or `'`, without looking
one character back for a preceding `L`. So `L"foo"` would come through as a separate
`Identifier` token (`L`) immediately followed by a `StringLiteral` token (`"foo"`),
rather than one wide-string-literal token.

**Impact today**: none. `linuxdoom-1.10` never uses wide literals (verified: no real
`L"..."`/`L'...'` occurrences in the corpus; the only `L'` hits are apostrophes inside
French string content in `d_french.h`, e.g. `"DONJON DE L'ENFER"`, already consumed
correctly as part of that single string chunk).

**If it starts mattering**: teach the partitioner to check for an immediately
preceding `L` (with no whitespace) before starting a string/char literal chunk, and
fold it into that chunk.

---

## A comment trailing on the same line as a preprocessor directive is swallowed into the directive's raw text

**Where**: Step 2 partitioning (`transpiler/src/parser/partitioner.rs`).

The directive-scanning branch of `partition_source` reads to end-of-line
unconditionally:

```rust
if at_line_start && b == b'#' {
    ...
    while i < len && bytes[i] != b'\n' && bytes[i] != b'\r' {
        i += 1;   // doesn't stop at "//" or "/*"
    }
    ...
}
```

Unlike the code-scanning branch (which does check for `//`/`/*`), this one doesn't,
so a line like:

```c
#define NCMD_KILL	0x10000000	// kill game   // real example: d_net.c:40
```

becomes a single `SourceChunk::Preprocessor` whose parsed `body` is
`"0x10000000\t// kill game"` — the comment text ends up baked into the macro body
string, and no separate `Comment` chunk is ever created for it. By the time Step 4/5
run, that comment isn't a `Comment` token anywhere in the stream, so it can't be
attached to anything.

**Impact today**: none of the existing pipeline validation catches it, because
round-trip (Step 2) and comment-count (Step 5) tests only check that text is
preserved *somewhere*, not that it's classified correctly. It's cosmetic until
something downstream tries to use `PreprocessorDirective::Define.body` as literal
macro-replacement text (the comment would corrupt it) or wants the comment surfaced
as documentation on the `#define`. Real-world frequency: common — a quick corpus grep
found 7+ macros with trailing same-line comments in the first few hits alone (e.g.
`d_net.c:40`, `i_sound.c:99-100`, `hu_stuff.h:30-42`).

**If it starts mattering**: make the directive-scanning branch stop at an unescaped
`//`/`/*` the same way the code-scanning branch does, splitting off a real `Comment`
chunk (or chunks, for `/* ... */ ... // ...`) instead of folding it into the
directive's raw text.

---

## Step 6 resolves system headers (`#include <...>`) from the build machine, with a hand-seeded fallback

**Where**: Step 6b import resolution (`transpiler/src/parser/imports.rs`).

Step 6 treats `#include` as an import: a file's typedef table is its own top-level
typedefs unioned with everything transitively imported via `#include` (see
`docs/01_PARSER.md` Step 6). `#include "..."` (local) resolves relative to the
including file's directory, same as always. `#include <...>` (system) is resolved the
way a real preprocessor would: searched for across the build machine's actual system
include directories (`SYSTEM_INCLUDE_DIRS`, matching `gcc -E -Wp,-v`'s own search
order), and if found, processed through the same Steps 1-4 + Step 6a pipeline as any
local header -- including recursively following *its* `#include`s, which is how
`i_video.c`'s `Display`/`Window`/`GC`/... end up resolved via the real
`/usr/include/X11/Xlib.h` pulling in `/usr/include/X11/X.h` for `Window`, etc.

If a system header isn't present on the machine actually running this (or fails to
process cleanly), its typedefs are just missing from the result -- fails soft, not
hard. Two hardcoded fallback tables cover this so the result no longer depends on what
happens to be installed:

- `WELL_KNOWN_SYSTEM_TYPEDEFS` (`imports.rs`): `FILE`, `va_list` -- hand-picked, the two
  libc typedefs this corpus actually references.
- `xlib_typedefs::XLIB_TYPEDEFS` (`transpiler/src/parser/xlib_typedefs.rs`, **generated**):
  every one of the 108 typedef names transitively exported by this dev machine's real
  `/usr/include/X11/Xlib.h` (`Display`, `Window`, `GC`, `Visual`, `XEvent`, ...),
  captured by actually resolving it once and hardcoding the result. Regenerate with
  `cargo run --example update_xlib_typedefs` on a machine with X11 dev headers
  installed (e.g. after a distro upgrade changes Xlib.h) -- the example re-resolves
  Xlib.h via `ImportResolver` and overwrites the file. Never hand-edit it.

Both apply unconditionally (unioned in on every `resolve()` call, real headers or not),
so `i_video.c`/`i_system.c`/`z_zone.c` all resolve the same way regardless of what's
installed on the machine running the pipeline -- no longer environment-dependent.

**`d_main.c`/`g_game.c`/`m_menu.c` were a different issue entirely, not typedefs**: see
the macro-substitution entry below -- fixing `FILE` on `d_main.c` revealed its real
remaining blocker was a `#define`d string constant, unrelated to system headers. Now
fixed there too (Step 4b), so all 62 `.c` translation units pass Step 6.

**Impact today**: 0 of 62 `.c` translation units fail Step 6 for missing system types.

---

## Step 6 doesn't perform general `#define` macro expansion (fixed narrowly, for the one case that mattered)

**Where**: Step 3 (`transpiler/src/parser/preprocessor.rs`) only ever resolves
conditional compilation (`#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`); it never
substitutes an object-like or function-like macro's body into the token stream where
the macro is *used* in code (only into `#if`/`#elif` *expressions*, where
`evaluate_expr` needs the value).

**Root cause found via corpus analysis, not guesswork**: rather than assume this was a
big open-ended problem, the actual scope was measured by scanning the real lexed token
stream of every `.c` file for "an identifier sitting immediately next to a string/char
literal, where that identifier is genuinely `#define`d somewhere in the corpus" (814
macro names total). Result: exactly **4 distinct macros**, all in **3 files**:
`SAVEGAMENAME` (`d_main.c`, `g_game.c`, `m_menu.c`; defined in `dstrings.h`), `DEVDATA`
and `DEVMAPS` (`d_main.c`, defined in `d_main.c` itself), and `DOSY` (`m_menu.c`;
defined in `d_englsh.h`/`d_french.h`). All four are object-like macros whose body is
*just* a single string literal.

**Fixed**: added Step 4b (`docs/01_PARSER.md`), a narrow substitution pass rather than a
general macro expander:

- `transpiler/src/parser/macro_literals.rs` (`LiteralMacroResolver`) resolves every
  literal-bodied object-like `#define` visible to a file (its own, plus everything
  transitively `#include`d -- mirroring Step 6b's import treatment, reusing the same
  `system_headers.rs` include resolution).
- `transpiler/src/parser/macro_literal_subst.rs` walks the Step 4 token stream and
  substitutes each such macro's identifier with its literal token, but *only* where a
  string/char literal token sits immediately before or after it. Once substituted, it's
  a literal like any other, so Step 6c's existing adjacent-string-literal-concatenation
  handling (already exercised constantly, e.g. `d_main.c:820-830`) picks up the rest --
  no separate concatenation logic needed.
- A macro used anywhere *other* than directly touching a literal (assigned to a
  variable, passed as a lone argument, referenced inside another macro's body) is left
  as a plain, unexpanded identifier -- deliberately, this is not a general preprocessor.

**Impact today**: 0 of 62 `.c` translation units fail for this reason anymore -- all 62
now parse (see `transpiler/src/parser/mod.rs`'s corpus test).

**If general macro expansion starts mattering** (e.g. function-like macros used as
actual expressions, not just adjacent to literals): implement full expansion as a pass
between Steps 3 and 4 (object-like macros are a straightforward token-substitution;
function-like macros additionally need argument matching/substitution).

---

## Typechecker Step 1 (symbol resolution) still doesn't resolve every cross-header reference (partially fixed by Step 0)

**Where**: `transpiler/src/typecheck/exports.rs`, `transpiler/src/typecheck/resolve.rs`.

Step 1 originally built its `SymbolTable` from a single translation unit's own
top-level and block-scoped declarations only -- Step 6b's `ImportResolver`
(`transpiler/src/parser/imports.rs`) resolved *typedef names* transitively through
`#include`s, but nothing resolved *function or variable* declarations from headers,
so a call to a function declared in another header showed up as unresolved even
though it's perfectly valid C. Only 3 of 62 `.c` files fully resolved.

**Fixed (mostly)**: added Step 0 (`docs/02_TYPECHECKER.md`), generalizing Step 6b's
`#include`-as-import treatment beyond typedef names -- `ExportResolver`
(`transpiler/src/typecheck/exports.rs`) recursively collects a file's own top-level
function prototypes/definitions, `extern`/global variables, struct/union/enum tags,
and enum constants, unioned with everything transitively `#include`d (respecting
`static` linkage, unlike typedef export -- see the module's own docs), reusing Step
6a's rough top-level scan (`grammar::extract_top_level_decls`). `SymbolTable`'s global
scope is now seeded from this before Step 1 walks the file's own declarations
(`resolve_translation_unit_seeded`).

**Impact today**: 7 of 62 `.c` translation units now fully resolve (up from 3); total
unresolved identifier references dropped from 13735 to 4143 (see
`transpiler/src/typecheck/exports.rs`'s corpus test). The remaining gap is a mix of
real, separate wrinkles rather than one more missing resolver: implicit-`int`/K&R-style
calls to functions never actually declared anywhere in the corpus, libc functions
outside what Step 6b's system-header resolution reaches (a header not installed, or
one whose declarations Step 6a's rough scanner doesn't recognize as a plain top-level
declaration), and macro-generated identifiers Step 1 doesn't yet expand before
resolving. Still doesn't block Step 1 itself, same "measure, don't assume"
methodology as before.

**If it starts mattering further**: measure the actual remaining-unresolved set the
same way Step 4b/7 measured their own scope (corpus scan of *which* names are
unresolved, not just the count) before deciding what the next fix should target.
