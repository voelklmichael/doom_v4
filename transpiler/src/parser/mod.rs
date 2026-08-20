pub mod partitioner;
pub mod splicer;

pub use partitioner::{
    parse_preprocessor_directive, partition_source, CommentChunk, PreprocessorDirective,
    SourceChunk,
};
pub use splicer::{splice, SourceLocation, SplicedSource};

/// Runs Step 1 (line splicing) followed by Step 2 (partitioning) on raw source text.
pub fn parse_chunks(source: &str) -> (SplicedSource, Vec<SourceChunk>) {
    let spliced = splice(source);
    let chunks = partition_source(&spliced.text);
    (spliced, chunks)
}
