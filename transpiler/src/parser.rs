pub mod partitioner;
pub mod preprocessor;
pub mod splicer;

pub use partitioner::{
    parse_chunks, partition_source, CommentChunk, PreprocessorDirective, SourceChunk,
};
pub use preprocessor::{
    evaluate_expr, resolve_conditionals, PreprocessorEnv, PreprocessorError,
};
pub use splicer::{splice, SourceLocation, SplicedSource};

pub fn parse(path: &str) -> Result<(SplicedSource, Vec<SourceChunk>), String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let (spliced, chunks) = parse_chunks(&content);
    let mut env = PreprocessorEnv::linux_doom_defaults();
    let resolved_chunks = resolve_conditionals(&chunks, &mut env)
        .map_err(|e| format!("Preprocessor error in {}: {}", path, e))?;
    Ok((spliced, resolved_chunks))
}
