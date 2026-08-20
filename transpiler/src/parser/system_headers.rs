//! Shared `#include` resolution used by both Step 6b (`imports.rs`, typedef
//! names) and literal-macro resolution (`macro_literals.rs`): finding the
//! actual file an `#include` refers to, whether local (`"..."`) or system
//! (`<...>`).

use crate::parser::{PreprocessorEnv, SourceChunk, parse_chunks, resolve_conditionals};
use std::path::{Path, PathBuf};

/// Where a real preprocessor looks for `#include <...>` on a typical Linux
/// build machine, in search order (matches `gcc -E -Wp,-v -xc /dev/null`).
pub(crate) const SYSTEM_INCLUDE_DIRS: &[&str] = &[
    "/usr/lib/gcc/x86_64-linux-gnu/13/include",
    "/usr/local/include",
    "/usr/include/x86_64-linux-gnu",
    "/usr/include",
];

/// An `#include`'d path together with whether it was `<...>` (system) or
/// `"..."` (local).
pub(crate) struct IncludePath {
    pub text: String,
    pub is_system: bool,
}

/// Resolves an `#include` to an actual file on disk: local includes relative
/// to the including file's own directory; system includes by searching
/// `SYSTEM_INCLUDE_DIRS` in order. `None` if no such file exists anywhere
/// searched.
pub(crate) fn resolve_include_path(inc: &IncludePath, including_dir: &Path) -> Option<PathBuf> {
    if !inc.is_system {
        let candidate = including_dir.join(&inc.text);
        return candidate.is_file().then_some(candidate);
    }
    SYSTEM_INCLUDE_DIRS
        .iter()
        .map(|dir| Path::new(dir).join(&inc.text))
        .find(|p| p.is_file())
}

/// Reads `path` and runs it through Steps 1-3, returning the resolved chunks
/// and the paths of everything it `#include`s (local and system). `None` if
/// the file can't be read or fails Steps 1-3.
pub(crate) fn read_resolved_chunks_and_includes(
    path: &Path,
) -> Option<(Vec<SourceChunk>, Vec<IncludePath>)> {
    use crate::parser::partitioner::PreprocessorDirective;

    let content = std::fs::read_to_string(path).ok()?;
    let (_, chunks) = parse_chunks(&content);
    let mut env = PreprocessorEnv::linux_doom_defaults();
    let resolved = resolve_conditionals(&chunks, &mut env).ok()?;

    let includes = resolved
        .iter()
        .filter_map(|c| match c {
            SourceChunk::Preprocessor {
                directive: PreprocessorDirective::Include { path, is_system },
                ..
            } => Some(IncludePath {
                text: path.clone(),
                is_system: *is_system,
            }),
            _ => None,
        })
        .collect();

    Some((resolved, includes))
}
