//! Step 1: Line Splicing (Translation Phase 2)
//!
//! According to ISO C89/C90 §5.1.1.2:
//! "Each instance of a backslash character (\) immediately followed by a new-line character
//! is deleted, splicing physical source lines into logical source lines."

/// Represents a location in the original, unspliced source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLocation {
    /// 1-based line number in original source
    pub line: usize,
    /// 1-based column number in original source (byte offset within line + 1)
    pub column: usize,
    /// 0-based byte offset in original source
    pub original_offset: usize,
}

/// Splicing mapping entry to translate a byte offset in `spliced_text`
/// back to the original source location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpliceMapping {
    /// Byte offset in spliced text
    pub spliced_offset: usize,
    /// Byte offset in original text
    pub original_offset: usize,
    /// 1-based line in original text
    pub original_line: usize,
    /// 1-based column in original text
    pub original_col: usize,
}

/// Result of splicing a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplicedSource {
    /// The processed source code with backslash-newline sequences removed.
    pub text: String,
    /// Line index of the original file (byte offsets of each line start).
    original_line_starts: Vec<usize>,
    /// Splicing mappings to translate offsets.
    mappings: Vec<SpliceMapping>,
    /// Total count of spliced line continuations found and processed.
    pub spliced_continuations_count: usize,
}

impl SplicedSource {
    /// Map a byte offset in `self.text` (spliced text) back to the original source location.
    pub fn original_location(&self, spliced_offset: usize) -> SourceLocation {
        if self.mappings.is_empty() {
            let (line, col) = self.offset_to_line_col_original(spliced_offset);
            return SourceLocation {
                line,
                column: col,
                original_offset: spliced_offset,
            };
        }

        // Binary search for the largest mapping with spliced_offset <= requested offset
        let idx = match self.mappings.binary_search_by_key(&spliced_offset, |m| m.spliced_offset) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    0
                } else {
                    i - 1
                }
            }
        };

        let base = &self.mappings[idx];
        let delta = spliced_offset.saturating_sub(base.spliced_offset);
        let orig_offset = base.original_offset + delta;
        let (line, col) = self.offset_to_line_col_original(orig_offset);

        SourceLocation {
            line,
            column: col,
            original_offset: orig_offset,
        }
    }

    fn offset_to_line_col_original(&self, orig_offset: usize) -> (usize, usize) {
        if self.original_line_starts.is_empty() {
            return (1, 1);
        }

        let line_idx = match self.original_line_starts.binary_search(&orig_offset) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    0
                } else {
                    i - 1
                }
            }
        };

        let line_start = self.original_line_starts[line_idx];
        let line_num = line_idx + 1;
        let col_num = orig_offset.saturating_sub(line_start) + 1;
        (line_num, col_num)
    }
}

/// Splicer function to process source text and resolve all line continuations.
pub fn splice(source: &str) -> SplicedSource {
    let mut original_line_starts = Vec::new();
    original_line_starts.push(0);

    let bytes = source.as_bytes();
    let len = bytes.len();

    // First pass: compute original line starts for fast line/col calculations
    for i in 0..len {
        if bytes[i] == b'\n' {
            original_line_starts.push(i + 1);
        }
    }

    let mut output = String::with_capacity(len);
    let mut mappings = Vec::new();
    let mut spliced_continuations_count = 0;

    let mut i = 0;
    while i < len {
        if bytes[i] == b'\\' {
            // Check if this backslash is followed by optional whitespace and then newline
            let mut j = i + 1;
            // Allow trailing spaces or tabs before newline (common GCC extension / dirty source cleanup)
            while j < len && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }

            if j < len && bytes[j] == b'\n' {
                // Spliced \ \n
                spliced_continuations_count += 1;
                i = j + 1; // skip \ ... \n

                // Record mapping after splice
                mappings.push(SpliceMapping {
                    spliced_offset: output.len(),
                    original_offset: i,
                    original_line: 0, // computed lazily
                    original_col: 0,
                });
                continue;
            } else if j + 1 < len && bytes[j] == b'\r' && bytes[j + 1] == b'\n' {
                // Spliced \ \r\n
                spliced_continuations_count += 1;
                i = j + 2; // skip \ ... \r\n

                mappings.push(SpliceMapping {
                    spliced_offset: output.len(),
                    original_offset: i,
                    original_line: 0,
                    original_col: 0,
                });
                continue;
            }
        }

        // Regular character
        output.push(bytes[i] as char);
        i += 1;
    }

    SplicedSource {
        text: output,
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
}
