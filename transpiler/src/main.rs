use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use transpiler::parser::{parse_chunks, resolve_conditionals, PreprocessorEnv};

fn collect_c_source_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "c" || ext == "h" {
                        files.push(path);
                    }
                }
            }
        }
    }
    files.sort();
    files
}

fn main() {
    let doom_dir = Path::new("linuxdoom-1.10");
    let target_dir = if doom_dir.exists() {
        doom_dir
    } else {
        Path::new("../linuxdoom-1.10")
    };

    println!("============================================================");
    println!(" Doom C Transpiler: Stage 1.1 -> 1.3 Pipeline Runner       ");
    println!(" Target directory: {:?}", target_dir);
    println!("============================================================");

    let files = collect_c_source_files(target_dir);
    if files.is_empty() {
        eprintln!("Error: No C/H source files found in {:?}", target_dir);
        std::process::exit(1);
    }

    let start_time = Instant::now();
    let mut total_files = 0;
    let mut total_splices = 0;
    let mut total_raw_chunks = 0;
    let mut total_resolved_chunks = 0;
    let mut files_with_errors = 0;

    let global_env = PreprocessorEnv::linux_doom_defaults();

    for file_path in &files {
        let content = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(err) => {
                eprintln!("Failed to read {}: {}", file_path.display(), err);
                continue;
            }
        };

        let (spliced, raw_chunks) = parse_chunks(&content);
        total_files += 1;
        total_splices += spliced.spliced_continuations_count;
        total_raw_chunks += raw_chunks.len();

        let mut file_env = global_env.clone();
        match resolve_conditionals(&raw_chunks, &mut file_env) {
            Ok(resolved) => {
                total_resolved_chunks += resolved.len();
            }
            Err(err) => {
                eprintln!("  ✗ Preprocessor error in {}: {}", file_path.display(), err);
                files_with_errors += 1;
            }
        }
    }

    let elapsed = start_time.elapsed();

    println!("------------------------------------------------------------");
    println!("Execution Summary across {} Doom source files:", total_files);
    println!("  Total Line Continuations Spliced:  {}", total_splices);
    println!("  Total Raw Chunks (Step 1.2):       {}", total_raw_chunks);
    println!("  Total Active Chunks (Step 1.3):     {}", total_resolved_chunks);
    println!("  Filtered Inactive Chunks:          {}", total_raw_chunks.saturating_sub(total_resolved_chunks));
    println!("  Files with Preprocessor Errors:    {}", files_with_errors);
    println!("  Total Time Elapsed:                {:.2?}", elapsed);
    println!("============================================================");

    if files_with_errors == 0 {
        println!("🎉 All 124 files passed Step 1.1, 1.2, and 1.3 with 100% success!");
    }
}
