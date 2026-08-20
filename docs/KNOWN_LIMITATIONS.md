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

## Step 6 can't resolve types from system headers not present in this corpus

**Where**: Step 6b import resolution (`transpiler/src/parser/imports.rs`).

Step 6 treats `#include` as an import: a file's typedef table is its own top-level
typedefs unioned with everything transitively imported via local `#include "..."`s
(see `docs/01_PARSER.md` Step 6). `#include <...>` (system headers) can't be resolved
this way, since there's no local file to read.

**Fixed for libc, via a small hand-seeded table**: `WELL_KNOWN_SYSTEM_TYPEDEFS` in
`imports.rs` hardcodes `FILE` (`<stdio.h>`) and `va_list` (`<stdarg.h>`), the two
libc typedefs this corpus actually references. This fully resolves `z_zone.c` and
`i_system.c`, which use nothing else external.

**Not fixed for X11 -- `i_video.c`**: tried the same approach for X11's `<Xlib.h>`
types, and it doesn't stay "small": `i_video.c` alone references `Display`, `Window`,
`Colormap`, `GC`, `XEvent`, `XVisualInfo`, `XShmSegmentInfo`, `Pixmap`, `XGCValues`,
`XColor`, `Cursor`, `XSetWindowAttributes`, `Visual`, and likely more beyond that --
this is most of a window-management API's surface, not a handful of names. Hand-seeding
it would mean hand-transcribing a meaningful chunk of Xlib.h's type declarations rather
than genuinely resolving them, so `i_video.c` stays a known failure.

**`d_main.c`/`g_game.c`/`m_menu.c` are a different issue entirely, not typedefs**: see
the macro-expansion entry below -- fixing `FILE` on `d_main.c` revealed its real
remaining blocker is a `#define`d string constant, unrelated to system headers.

**Impact today**: 4 of 62 `.c` translation units still fail Step 6: `i_video.c` (X11,
above) and `d_main.c`/`g_game.c`/`m_menu.c` (macro expansion, below). See
`transpiler/src/parser/mod.rs`'s `EXPECTED_FAILURES` list in the Step 6 corpus test,
which tracks this explicitly rather than silently ignoring it.

**If X11 starts mattering**: parse the actual `Xlib.h` on the build machine the way a
real preprocessor would, rather than hand-seeding names one failure at a time.

---

## Step 6 doesn't perform `#define` macro expansion

**Where**: Step 3 (`transpiler/src/parser/preprocessor.rs`) only ever resolves
conditional compilation (`#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`); it never
substitutes an object-like or function-like macro's body into the token stream where
the macro is *used* in code (only into `#if`/`#elif` *expressions*, where
`evaluate_expr` needs the value).

**Impact today**: 3 of 62 `.c` translation units fail to parse for this reason --
`g_game.c` and `m_menu.c` both write `sprintf(name, "..." SAVEGAMENAME "...", ...)`,
and `d_main.c` writes `D_AddFile(DEVDATA"doom1.wad")`, relying on `SAVEGAMENAME`/
`DEVDATA` (object-like `#define`d string constants) being substituted with their
literal text *before* parsing, so the result is adjacent string-literal tokens that
concatenate per C89 translation phase 6. Since we never substitute the macro, the
parser sees a plain identifier sitting where an expression can't have one, and fails.
This is a narrow case (found via exactly these 3 files across the whole corpus), not a
systemic problem with the adjacent-string-literal handling itself (which is otherwise
exercised constantly and works, e.g. `d_main.c:820-830`).

**If it starts mattering**: implement macro expansion as a pass between Steps 3 and 4
(object-like macros are a straightforward token-substitution; function-like macros
additionally need argument matching/substitution and are common enough elsewhere in
the corpus that a real transpiler would eventually need this anyway).
