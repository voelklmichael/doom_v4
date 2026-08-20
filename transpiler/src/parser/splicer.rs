//! Step 1: Line Splicing (Translation Phase 2)
//!
//! According to ISO C89/C90 §5.1.1.2:
//! "Each instance of a backslash character (\) immediately followed by a new-line character
//! is deleted, splicing physical source lines into logical source lines."

/// A location in the original, unspliced source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLocation {
    /// 1-based line number in original source
    pub line: usize,
    /// 1-based column number in original source (byte offset within line + 1)
    pub column: usize,
    /// 0-based byte offset in original source
    pub original_offset: usize,
}

/// Mapping entry to translate a byte offset in `spliced_text` back to the original source.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SpliceMapping {
    /// Byte offset in spliced text
    spliced_offset: usize,
    /// Byte offset in original text
    original_offset: usize,
}

/// Result of splicing a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplicedSource {
    /// The processed source code with backslash-newline sequences removed.
    pub text: String,
    /// Byte offsets of each line start in the *original* file.
    original_line_starts: Vec<usize>,
    /// Splicing mappings to translate offsets, ordered by `spliced_offset`.
    mappings: Vec<SpliceMapping>,
    /// Total count of line continuations found and removed.
    pub spliced_continuations_count: usize,
}

impl SplicedSource {
    /// Map a byte offset in `self.text` (spliced text) back to a location in the original file.
    pub fn original_location(&self, spliced_offset: usize) -> SourceLocation {
        let orig_offset = match self
            .mappings
            .binary_search_by_key(&spliced_offset, |m| m.spliced_offset)
        {
            Ok(i) => self.mappings[i].original_offset,
            Err(0) => spliced_offset,
            Err(i) => {
                let base = &self.mappings[i - 1];
                base.original_offset + (spliced_offset - base.spliced_offset)
            }
        };

        let (line, column) = self.offset_to_line_col_original(orig_offset);
        SourceLocation {
            line,
            column,
            original_offset: orig_offset,
        }
    }

    fn offset_to_line_col_original(&self, orig_offset: usize) -> (usize, usize) {
        let line_idx = match self.original_line_starts.binary_search(&orig_offset) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        let line_start = self.original_line_starts[line_idx];
        (line_idx + 1, orig_offset - line_start + 1)
    }
}

/// Process source text, splicing away all backslash-newline line continuations.
pub fn splice(source: &str) -> SplicedSource {
    let bytes = source.as_bytes();
    let len = bytes.len();

    let mut original_line_starts = vec![0];
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            original_line_starts.push(i + 1);
        }
    }

    let mut output: Vec<u8> = Vec::with_capacity(len);
    let mut mappings = Vec::new();
    let mut spliced_continuations_count = 0;

    let mut i = 0;
    while i < len {
        if bytes[i] == b'\\' {
            // Allow trailing spaces/tabs between the backslash and the newline
            // (common compiler extension for otherwise-clean source).
            let mut j = i + 1;
            while j < len && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }

            let continuation_end = if j < len && bytes[j] == b'\n' {
                Some(j + 1)
            } else if j + 1 < len && bytes[j] == b'\r' && bytes[j + 1] == b'\n' {
                Some(j + 2)
            } else {
                None
            };

            if let Some(next) = continuation_end {
                spliced_continuations_count += 1;
                i = next;
                mappings.push(SpliceMapping {
                    spliced_offset: output.len(),
                    original_offset: i,
                });
                continue;
            }
        }

        output.push(bytes[i]);
        i += 1;
    }

    SplicedSource {
        text: String::from_utf8(output)
            .expect("splicing only removes ASCII continuation sequences"),
        original_line_starts,
        mappings,
        spliced_continuations_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_splicing() {
        let input = "int a = 1;\nint b = 2;\n";
        let res = splice(input);
        assert_eq!(res.text, input);
        assert_eq!(res.spliced_continuations_count, 0);

        let loc = res.original_location(11); // start of line 2
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 1);
    }

    #[test]
    fn test_basic_splicing() {
        let input = "int a = \\\n10;\n";
        let res = splice(input);
        assert_eq!(res.text, "int a = 10;\n");
        assert_eq!(res.spliced_continuations_count, 1);

        // Location of '1' in '10' (spliced offset = 8)
        let loc = res.original_location(8);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 1);
    }

    #[test]
    fn test_crlf_splicing() {
        let input = "#define FOO(x) \\\r\n((x) * 2)\r\n";
        let res = splice(input);
        assert_eq!(res.text, "#define FOO(x) ((x) * 2)\r\n");
        assert_eq!(res.spliced_continuations_count, 1);
    }

    #[test]
    fn test_trailing_whitespace_splicing() {
        let input = "#define BAR \\\t \n123\n";
        let res = splice(input);
        assert_eq!(res.text, "#define BAR 123\n");
        assert_eq!(res.spliced_continuations_count, 1);
    }

    #[test]
    fn test_multiple_consecutive_splicing() {
        let input = "char *str = \\\n\"hello \" \\\n\"world\";\n";
        let res = splice(input);
        assert_eq!(res.text, "char *str = \"hello \" \"world\";\n");
        assert_eq!(res.spliced_continuations_count, 2);
    }

    #[test]
    fn test_trailing_backslash_without_newline_is_not_spliced() {
        // A backslash at end-of-file with no following newline is left untouched.
        let input = "int a = 1;\\";
        let res = splice(input);
        assert_eq!(res.text, input);
        assert_eq!(res.spliced_continuations_count, 0);
    }

    #[test]
    fn test_no_corruption_across_full_corpus() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("linuxdoom-1.10 directory should exist") {
            let path = entry.unwrap().path();
            let is_source = matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("c") | Some("h")
            );
            if !path.is_file() || !is_source {
                continue;
            }
            let content =
                std::fs::read_to_string(&path).expect("source file should be valid UTF-8");
            let res = splice(&content);

            // Every backslash-newline (or backslash-whitespace-newline) run in the
            // original must have been removed, and nothing else may have moved.
            let expected_len = content.len() - count_spliced_bytes(&content);
            assert_eq!(
                res.text.len(),
                expected_len,
                "unexpected output length for {}",
                path.display()
            );
            checked += 1;
        }
        assert!(
            checked > 100,
            "expected to check the full Doom corpus, only checked {checked}"
        );
    }

    /// Reference-counts the bytes that `splice` should remove, using a naive
    /// re-scan independent of the main implementation, to cross-check byte accounting.
    fn count_spliced_bytes(source: &str) -> usize {
        let bytes = source.as_bytes();
        let len = bytes.len();
        let mut removed = 0;
        let mut i = 0;
        while i < len {
            if bytes[i] == b'\\' {
                let mut j = i + 1;
                while j < len && (bytes[j] == b' ' || bytes[j] == b'\t') {
                    j += 1;
                }
                if j < len && bytes[j] == b'\n' {
                    removed += j + 1 - i;
                    i = j + 1;
                    continue;
                } else if j + 1 < len && bytes[j] == b'\r' && bytes[j + 1] == b'\n' {
                    removed += j + 2 - i;
                    i = j + 2;
                    continue;
                }
            }
            i += 1;
        }
        removed
    }
}
