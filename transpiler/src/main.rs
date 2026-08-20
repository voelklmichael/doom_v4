use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use transpiler::parser::splice;

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
    println!(" Doom C Transpiler: Step 1 (Line Splicing) Pipeline Runner ");
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
    let mut files_with_errors = 0;

    for file_path in &files {
        let content = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(err) => {
                eprintln!("Failed to read {}: {}", file_path.display(), err);
                files_with_errors += 1;
                continue;
            }
        };

        let spliced = splice(&content);
        total_files += 1;
        total_splices += spliced.spliced_continuations_count;
    }

    let elapsed = start_time.elapsed();

    println!("------------------------------------------------------------");
    println!("Execution Summary across {} Doom source files:", total_files);
    println!("  Total Line Continuations Spliced:  {}", total_splices);
    println!("  Files with Errors:                 {}", files_with_errors);
    println!("  Total Time Elapsed:                {:.2?}", elapsed);
    println!("============================================================");

    if files_with_errors == 0 {
        println!("All {} files passed Step 1 (Line Splicing) with 100% success!", total_files);
    }
}
