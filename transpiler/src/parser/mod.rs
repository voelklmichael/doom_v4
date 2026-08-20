pub mod comment_attach;
pub mod lexer;
pub mod partitioner;
pub mod preprocessor;
pub mod splicer;

pub use comment_attach::{Anchor, Commented, CommentedStream, attach_comments};
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
