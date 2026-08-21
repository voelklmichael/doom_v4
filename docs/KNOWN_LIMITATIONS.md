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

**Measured, not guessed**: a corpus-wide breakdown of the 4143 remaining unresolved
names (not just the count) showed the "libc functions Step 6a's rough scanner doesn't
recognize" guess above was only part of the picture -- the *dominant* contributor was
actually macro references (`NULL`, `FRACUNIT`, `SCREENWIDTH`, `PU_CACHE`, ...), which
were never in scope for Step 1 at all (macro name resolution is Step 2's job, not
declared-symbol resolution -- see `docs/02_TYPECHECKER.md` Step 2, not yet
implemented). The libc-function gap was real too, though: `printf`, `memcpy`, `strcpy`,
etc. showed up unresolved because real glibc header declarations use GNU/C99 syntax
our C89-only grammar had no model for (`__restrict`, trailing `__THROW`/`__nonnull
((1))`/`__attribute__((...))` decorations, a leading `__extension__` prefix) --
Step 6a's rough scan (`skip_bodies` mode) failed to parse the declaration containing
them, and on error discarded *every* declaration already collected from that file
(one bad declaration losing a whole header's worth of good ones).

**Fixed further**: `grammar.rs`'s rough-scan mode now (a) recognizes and discards
`restrict`/`__restrict`/`__restrict__` as a pointer qualifier
(`parse_pointer_quals`), (b) skips a leading `__extension__` prefix and any run of
trailing decoration identifiers (each with an optional parenthesized argument list)
between a declarator and its terminating `;`/`{` (`skip_gnu_decorations`), and (c)
recovers from a top-level construct it still can't parse at all (e.g. inline `__asm__`)
by skipping to the next top-level boundary instead of discarding the whole file's scan
(`recover_to_next_top_level_boundary`) -- all three gated behind `skip_bodies` so Step
6c's strict, already-100%-passing corpus parse is untouched.

**Impact today**: 7 of 62 `.c` translation units fully resolve (up from 3 before Step
0); total unresolved identifier references 13735 -> 4143 (Step 0) -> 3771 (this fix)
(see `transpiler/src/typecheck/exports.rs`'s corpus test). `printf`/`sprintf`/
`fprintf`/`memcpy`/`memset`/`malloc`/`abs`/`toupper`/`atoi`/`access` and similar now
resolve. `string.h` itself still doesn't contribute anything -- see the next entry,
a different root cause entirely. Still doesn't block Step 1 itself, same "measure,
don't assume" methodology as before.

**If it starts mattering further**: implement Step 2 (macro typing) -- given the
measured breakdown, that would close far more of the remaining gap than any further
work on declaration parsing would.

**Update**: Step 2 is now implemented -- see the next entry. It doesn't feed back into
`SymbolTable`/`resolve_translation_unit_seeded` itself (macro names still aren't
declarations, so they still show up as `UnresolvedIdent`s here) -- it's a parallel
semantic layer, not a fix to this step. See `docs/02_TYPECHECKER.md` Step 2.

---

## Typechecker Step 2 (macro typing): ~91% of real-code macro references get a type; the rest wait on Step 3's declaration/struct types

**Where**: `transpiler/src/typecheck/types.rs`, `transpiler/src/typecheck/macro_types.rs`.

Step 2 introduces a `Type` representation (Goal 2 of `docs/02_TYPECHECKER.md`, deferred
until Step 2 actually needed one) and types every `#define`: an object-like macro's
`Expr` directly (memoized, cycle-guarded against a macro that references itself,
directly or through others); a function-like macro's body once per real call site, by
structurally substituting the actual argument expressions for its parameters and
typing the substituted tree (`MacroTyper::type_of_macro_call`/`substitute`) -- it has
no single fixed signature, so it isn't typed once in isolation. `collect_macro_uses`
finds every real-code reference to a known macro by walking the same
expression-bearing AST shapes Step 1's `Resolver` does (function bodies, initializers,
array-size/bit-field-width/enum-value expressions), without any scope bookkeeping --
unlike an ordinary identifier, a macro name is authoritative wherever it textually
appears, matching real preprocessor semantics.

**Measured**: across the 62-file corpus, 3587 real-code macro references were found;
3258 (91%) resolved to a concrete type, 329 (9%) came back `Type::Unknown` (logged,
not a hard error, matching this project's "measure, don't assume" policy). Breakdown
of the 329: 177 function-like-macro call sites where the substituted body's type
depended on an argument this step can't type (see below); 102 object-like macros whose
body itself contains such a dependency; 44 references to a macro whose body is a
statement sequence, not an expression (`MacroBody::Statements`, e.g. `m_swap.h`'s
`Z_ChangeTag`, 40 of the 44 alone); 6 references to an `Unparseable` body. No bare
(uncalled) function-like-macro references were found in the corpus.

**Root cause of the 329, confirmed by name (`SHORT`/`LONG` from `m_swap.h`, `FTOM`/
`MTOF`/`CXMTOF`/`CYMTOF` from `am_map.c`, and others dominate)**: these macros'
arguments are overwhelmingly struct-member accesses (`mobj->x`, `ld->dx`, ...) or plain
variable references -- and Step 2 deliberately doesn't model struct field layouts
(`Expr::Member` always types `Unknown`, since Step 0 only ever collected coarse tag
*kinds*, not member lists -- see `exports.rs`) or a plain variable/parameter's
declared type (that needs full declaration type-checking, Step 3 -- Step 2 only knows
enum-constant identifiers, via the seeded `SymbolTable`). `Type::Unknown` propagates
through every operator (a cast is the one exception: its result is the cast's own
target type regardless of its operand), so one `Unknown` argument taints everything
built from it -- confirmed concretely: `m_swap.h`'s active (little-endian) `SHORT(x)`
is just `(x)`, so `SHORT(mobj->x)`'s `Unknown` result is entirely inherited from the
untyped `Member` access, not a bug in the substitution/typing logic itself.

**Impact today**: none of Step 2's own validation criteria require 100% -- explicitly
scoped as "resolves to a type, or is explicitly logged for follow-up." The 9% gap is
real but expected, and traces to a single root cause (no struct-member/variable
declaration types yet) rather than many unrelated ones.

**If it starts mattering further**: implement Step 3 (type checking) -- once ordinary
declarations and struct members have real `Type`s, `Expr::Member` and plain
`Expr::Ident` variable/parameter lookups can return them instead of `Unknown`, closing
most of this gap without any further change to Step 2's own logic.

**Update**: Step 3 is now implemented -- see the next entry. It closed the gap this
entry predicted, though not "for free": `MacroTyper` itself is unchanged (still
`Unknown` on a struct-member macro argument in isolation), because Step 3 doesn't
reuse it -- see the next entry's note on why.

---

## Typechecker Step 3 (type checking): 93.1% of all expressions get a type; 0 assignment/call-argument incompatibilities found after fixing a real normalization bug

**Where**: `transpiler/src/typecheck/declared_types.rs`, `transpiler/src/typecheck/check.rs`,
`transpiler/src/typecheck/types.rs`.

Step 3 is "Step 0, but for full types": `declared_types.rs`'s `DeclaredTypesResolver`
mirrors `ExportResolver`'s exact recursive, cycle-guarded, `#include`-union shape, but
extracts a real `Type`/`FunctionSignature` per declaration (typedef, function,
variable, struct/union field) instead of a coarse `SymbolKind`. `check.rs`'s `Checker`
then walks a translation unit exactly like Step 1's `Resolver` (same scopes, same
declaration/statement/expression shapes) computing a `Type` for every expression, and
flags every declaration-initializer/`=`/call-argument site whose value type isn't
`types::is_assignment_compatible` with its target.

**Doesn't delegate macro typing to `MacroTyper`**: a macro's arguments at a real call
site can reference the calling function's locals and struct fields (`SHORT(mobj->x)`),
which `MacroTyper`'s own `type_of_expr` has no access to -- so `check.rs` reimplements
the same small substitute-then-type dance directly against its own richer
`type_of_expr`, rather than trying to inject Step 3's context into Step 2's
self-contained struct. This is what actually closed the previous entry's gap (verified:
`SHORT(mobj->x)` now types as `Int`, not `Unknown`, whenever `mobj_t`'s field layout is
in scope).

**A real bug, caught by measurement, not a design gap**: the first version of this
step normalized *nothing* before comparing types -- `Type::Named("fixed_t")` (an
unresolved typedef reference) was compared directly against `Type::Int`, and since
they're structurally different, every `fixed_t x = <int-typed-expr>;` in the corpus
(the single most common declaration shape in `linuxdoom-1.10`) came back "incompatible".
The corpus run reported 1310 diagnostics, dominated by `Named("fixed_t") <- Int` (137
assignments + 99 call-arguments) and `Named("boolean") <- Int` (266) alone -- not a
subtle long tail, an obviously-wrong headline number, exactly the kind of result this
project's "measure before trusting" methodology exists to catch. Root cause:
`DeclaredTypes::resolve_typedef` (added to do exactly this unwrapping) was never
actually called at any of the three comparison sites. Fixed by adding
`DeclaredTypes::normalize` (recurses through `Pointer`/`Array`/`Function` too, unlike
`resolve_typedef` alone -- needed for e.g. `Pointer(Named("mobj_t"))` vs.
`Pointer(Struct("mobj_s"))`, two spellings of the same type once `mobj_t`'s chain is
followed) and calling it on both sides at every check site. Diagnostics dropped
1310 -> 92.

**The remaining 92 were real modeling gaps, also fixed**: `Pointer(Unknown)` (address
of an untyped struct member) was being compared structurally against a known pointer
type and flagged, even though "unknown" isn't "confirmed different" -- fixed by
`is_assignment_compatible` withholding judgment (`contains_unknown`) when `Unknown`
appears anywhere inside either type, not just at the top level. A bare function value
passed where a function pointer (or `void*`) was expected was flagged too, missing
that a function decays to a pointer to itself the same way an array decays to a
pointer to its element -- both are core to Doom's action-pointer idiom (see this
doc's "Doom Idioms" section in `docs/02_TYPECHECKER.md`) and are now allowed
explicitly. After both fixes: **0 diagnostics** across the corpus.

**Measured**: across the 62-file corpus, 94301 expressions were typed; 6508 (6.9%)
came back `Unknown` (up from Step 2's 9% *macro-reference-only* gap -- this number
covers *every* expression, a much larger surface, and still resolves more of it).
0 assignment/call-argument compatibility diagnostics were found. That's a real,
positive result, not a placeholder: the two unit tests that construct a genuine
mismatch by hand (`test_pointer_to_unrelated_pointer_assignment_is_flagged`,
`test_call_argument_mismatch_is_flagged`) still correctly flag it, so the checker
isn't vacuous -- `linuxdoom-1.10` (once typedefs, function decay, and `void*`
looseness are modeled the way a real compiler treats them) simply doesn't trip the
approximated C89 rules this step checks.

**Scope note**: like Step 0, `declared_types.rs` only scans *top-level* declarations
(Step 6a's `skip_bodies`, non-recursive `scan_decl_specifiers`) -- a struct defined
inline inside another struct's field list isn't captured. Not yet measured to matter.
Aggregate (`{ ... }`) initializers are typed but not checked member-by-member (would
need walking them in lockstep with the target's field/element types). Return-type
compatibility (a `return` statement's value vs. the function's declared return type)
isn't checked in this pass either -- not in Step 3's stated "assignment/cast/call-
argument" list, so deliberately out of scope, not an oversight.

---

## Step 3's `#if` expression evaluator can't handle function-macro invocations, so some real system headers never resolve at all

**Where**: `transpiler/src/parser/preprocessor.rs` (`evaluate_expr`).

Found while investigating why `string.h`'s declarations (`strcpy`, `strlen`, `strcmp`,
...) still didn't resolve even after fixing Step 6a's rough-scan GNU-syntax handling
(previous entry): `ExportResolver::resolve_inner` calls
`system_headers::read_resolved_chunks_and_includes`, which runs Steps 1-3 on the
header -- and for `/usr/include/string.h` directly, Step 3 fails outright (returns
`None`, silently, same fail-soft policy as a header not being found at all). The real
cause: `string.h` contains `#if __GNUC_PREREQ (3,4)` -- `__GNUC_PREREQ(maj, min)` is
itself a `#define`d function-like macro (from `<features.h>`,
`((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))`), and Step 3 never
expands macros used *within* a `#if`/`#elif` expression (only `defined()` is special-
cased) -- `evaluate_expr` sees a bare identifier followed by a parenthesized,
comma-separated argument list where its grammar only expects a value, and errors.

**Impact today**: any header (or transitively-included header) containing a
function-like macro invocation in a `#if`/`#elif` condition fails Step 3 entirely,
contributing nothing to `ImportResolver`/`ExportResolver`/`LiteralMacroResolver`'s
results -- not just the declarations near the failing `#if`, the *whole file*. Real,
not hypothetical: this is why `string.h` contributes zero symbols to Step 0's export
set right now, despite being found and despite its declarations otherwise being
parseable after the previous entry's fix.

**If it starts mattering**: `evaluate_expr` would need to recognize a call-shaped
identifier in a `#if` expression, look it up as a function-like macro (reusing Step
7's `MacroBody::Function` parsing/argument-substitution machinery once Step 2 exists
to drive it, or a narrower purpose-built substitution just for `#if` context), and
evaluate the substituted expression -- plus seeding `PreprocessorEnv::linux_doom_defaults`
with compiler-identity macros (`__GNUC__`, `__GNUC_MINOR__`, ...) that
`__GNUC_PREREQ` and similar version-guard macros depend on.
