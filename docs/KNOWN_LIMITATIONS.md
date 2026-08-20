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

**Impact today**: 3 of 62 `.c` translation units fail to parse for this reason --
`d_main.c` and `z_zone.c` use `FILE` (`<stdio.h>`), `i_video.c` uses `Display`
(`<X11/Xlib.h>`), and `i_system.c` uses `va_list` (`<stdarg.h>`). All three are
genuinely external: this repo doesn't (and can't sensibly) contain libc's or X11's
headers, so there's no source to import types from. See `transpiler/src/parser/mod.rs`'s
`EXPECTED_FAILURES` list in the Step 6 corpus test, which tracks this explicitly rather
than silently ignoring it.

**If it starts mattering**: either hand-seed a small table of well-known system
typedefs (`FILE`, `va_list`, common X11 types, ...) for the handful of files that need
them, or (more generally, much larger scope) parse the actual system headers on the
build machine the way a real preprocessor would.

---

## Step 6 doesn't perform `#define` macro expansion

**Where**: Step 3 (`transpiler/src/parser/preprocessor.rs`) only ever resolves
conditional compilation (`#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`); it never
substitutes an object-like or function-like macro's body into the token stream where
the macro is *used* in code (only into `#if`/`#elif` *expressions*, where
`evaluate_expr` needs the value).

**Impact today**: 2 of 62 `.c` translation units fail to parse for this reason --
`g_game.c` and `m_menu.c` both write `sprintf(name, "..." SAVEGAMENAME "...", ...)`,
relying on `SAVEGAMENAME` (an object-like `#define`d string constant) being substituted
with its literal text *before* parsing, so the result is three adjacent string-literal
tokens that concatenate per C89 translation phase 6. Since we never substitute the
macro, the parser sees `StringLiteral Identifier StringLiteral` -- an identifier
sitting where an expression can't have one -- and fails. This is a narrow case (found
via exactly these 2 files across the whole corpus), not a systemic problem with the
adjacent-string-literal handling itself (which is otherwise exercised constantly and
works, e.g. `d_main.c:820-830`).

**If it starts mattering**: implement macro expansion as a pass between Steps 3 and 4
(object-like macros are a straightforward token-substitution; function-like macros
additionally need argument matching/substitution and are common enough elsewhere in
the corpus that a real transpiler would eventually need this anyway).
