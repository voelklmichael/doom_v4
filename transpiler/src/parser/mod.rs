pub mod ast;
pub mod comment_attach;
pub mod grammar;
pub mod imports;
pub mod lexer;
pub mod partitioner;
pub mod preprocessor;
pub mod splicer;

pub use comment_attach::{Anchor, Commented, CommentedStream, attach_comments};
pub use grammar::{
    ParseError, extract_top_level_typedefs, parse_translation_unit, parse_translation_unit_seeded,
};
pub use imports::ImportResolver;
pub use lexer::{
    Keyword, LexEntry, LexError, LexItem, Punct, Token, TokenKind, lex_chunks, lex_code,
};
pub use partitioner::{
    CommentChunk, PreprocessorDirective, SourceChunk, parse_preprocessor_directive,
    partition_source,
};
pub use preprocessor::{PreprocessorEnv, PreprocessorError, evaluate_expr, resolve_conditionals};
pub use splicer::{SourceLocation, SplicedSource, splice};

/// Runs Step 1 (line splicing) followed by Step 2 (partitioning) on raw source text.
pub fn parse_chunks(source: &str) -> (SplicedSource, Vec<SourceChunk>) {
    let spliced = splice(source);
    let chunks = partition_source(&spliced.text);
    (spliced, chunks)
}

/// Runs Steps 1-3 on a source file: splicing, partitioning, and conditional resolution.
pub fn parse(path: &str) -> Result<(SplicedSource, Vec<SourceChunk>), String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let (spliced, chunks) = parse_chunks(&content);
    let mut env = PreprocessorEnv::linux_doom_defaults();
    let resolved = resolve_conditionals(&chunks, &mut env)
        .map_err(|e| format!("preprocessor error in {path}: {e}"))?;
    Ok((spliced, resolved))
}

/// Runs Steps 1-4 on a source file: splicing, partitioning, conditional resolution, and lexing.
pub fn parse_and_lex(path: &str) -> Result<(SplicedSource, Vec<LexEntry>), String> {
    let (spliced, resolved) = parse(path)?;
    let entries = lex_chunks(&resolved).map_err(|e| format!("lex error in {path}: {e}"))?;
    Ok((spliced, entries))
}

/// Runs Steps 1-5 on a source file: splicing, partitioning, conditional resolution,
/// lexing, and comment attaching.
pub fn parse_lex_and_attach(path: &str) -> Result<(SplicedSource, CommentedStream), String> {
    let (spliced, entries) = parse_and_lex(path)?;
    Ok((spliced, attach_comments(entries)))
}

/// Runs the full Steps 1-6 pipeline on a source file, producing a `TranslationUnit`.
/// Step 6's typedef table is seeded via Step 6b: `path`'s own top-level typedefs
/// unioned with everything transitively imported via local `#include`s.
pub fn parse_full(path: &str) -> Result<(SplicedSource, ast::TranslationUnit), String> {
    let (spliced, stream) = parse_lex_and_attach(path)?;
    let seed = ImportResolver::new().resolve(std::path::Path::new(path));
    let unit = parse_translation_unit_seeded(&stream, seed)
        .map_err(|e| format!("parse error in {path}: {e}"))?;
    Ok((spliced, unit))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Files that fail Step 6c for reasons outside the parser's control:
    /// `d_main.c`/`g_game.c`/`m_menu.c` need `#define` macro *expansion*
    /// (not just conditional resolution, which is all Step 3 does) to parse
    /// a string built from a macro constant; `i_video.c` needs a large
    /// chunk of X11's `<Xlib.h>` type vocabulary (`Display`, `Window`,
    /// `Colormap`, `Visual`, ... -- large enough that hand-seeding it isn't
    /// "small" the way `FILE`/`va_list` were). See docs/KNOWN_LIMITATIONS.md.
    const EXPECTED_FAILURES: &[&str] = &["d_main.c", "g_game.c", "i_video.c", "m_menu.c"];

    #[test]
    fn test_full_corpus_c_files_parse_except_known_limitations() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        files.sort();

        let mut unexpected_failures = Vec::new();
        let mut unexpected_successes = Vec::new();
        let mut ok_count = 0;

        for path in &files {
            let name = path.file_name().unwrap().to_str().unwrap();
            let result = parse_full(path.to_str().unwrap());
            let expected_to_fail = EXPECTED_FAILURES.contains(&name);
            match (result, expected_to_fail) {
                (Ok(_), false) => ok_count += 1,
                (Ok(_), true) => unexpected_successes.push(name.to_string()),
                (Err(_), true) => {}
                (Err(e), false) => unexpected_failures.push(format!("{name}: {e}")),
            }
        }

        assert!(
            unexpected_failures.is_empty(),
            "unexpected parse failures (not in the known-limitations list): {unexpected_failures:#?}"
        );
        assert!(
            unexpected_successes.is_empty(),
            "files expected to fail now parse successfully -- remove from EXPECTED_FAILURES: {unexpected_successes:#?}"
        );
        assert_eq!(ok_count, files.len() - EXPECTED_FAILURES.len());
        assert!(
            files.len() > 50,
            "expected to check the full Doom .c corpus, only found {}",
            files.len()
        );
    }
}
