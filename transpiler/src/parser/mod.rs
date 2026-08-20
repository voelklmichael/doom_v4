pub mod partitioner;
pub mod preprocessor;
pub mod splicer;

pub use partitioner::{
    parse_preprocessor_directive, partition_source, CommentChunk, PreprocessorDirective,
    SourceChunk,
};
pub use preprocessor::{evaluate_expr, resolve_conditionals, PreprocessorEnv, PreprocessorError};
pub use splicer::{splice, SourceLocation, SplicedSource};

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
