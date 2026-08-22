//! Investigation for `docs/03_TRANSPILER.md`'s open issue: "declared in this
//! file's own `.h` => `pub`, else private" is a heuristic that follows C
//! *convention*, not C's actual linkage rules. This script measures how
//! often that heuristic would actually get it wrong across the corpus --
//! for every non-`static` (externally-linked) function/variable *defined*
//! in a `.c` file, checks whether its own matching header (`foo.c` <->
//! `foo.h`) declares it, and if not, whether any *other* `.c` file in the
//! corpus can actually see it (via `ExportResolver`, which already follows
//! the real `#include` graph). A "no" on the first and "yes" on the second
//! is exactly the case the doc worries about: the merge heuristic would
//! make something private that another module still needs to call.
//!
//! Also separately reports `.c` files with no matching header at all (the
//! doc's module-structure section doesn't say what happens to their
//! externally-linked symbols), since that's the same underlying question.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use transpiler::parser::ast::{
    Declaration, Declarator, DirectDeclarator, ExternalDecl, StorageClass,
};
use transpiler::parser::grammar::extract_top_level_decls;
use transpiler::parser::{attach_comments, lex_chunks, parse, parse_full};
use transpiler::typecheck::exports::ExportResolver;
use transpiler::typecheck::scope::SymbolKind;

/// Same shape check as `exports.rs`'s private `is_function_declarator`:
/// true for a plain function declarator (`name(...)`), false for a
/// variable of any other shape, including a function *pointer*.
fn is_function_declarator(d: &Declarator) -> bool {
    matches!(&d.direct, DirectDeclarator::Function(base, _) if matches!(**base, DirectDeclarator::Ident(_)))
}

/// `declarator_name` is `pub(crate)`-only in the library; reimplemented
/// here the same way the other corpus-breakdown examples do.
fn declarator_name(d: &Declarator) -> Option<String> {
    declarator_name_direct(&d.direct)
}

fn declarator_name_direct(d: &DirectDeclarator) -> Option<String> {
    match d {
        DirectDeclarator::Ident(name) => Some(name.clone()),
        DirectDeclarator::Paren(inner) => declarator_name(inner),
        DirectDeclarator::Array(base, _) | DirectDeclarator::Function(base, _) => {
            declarator_name_direct(base)
        }
    }
}

#[derive(Debug, Clone)]
struct Defined {
    name: String,
    kind: SymbolKind,
}

/// Externally-linked (non-`static`) symbols *defined* -- not merely
/// declared -- at `unit`'s own top level: functions with a body, and
/// top-level variables that aren't `extern` re-declarations or typedefs.
fn own_defined_symbols(items: &[ExternalDecl]) -> Vec<Defined> {
    let mut out = Vec::new();
    for item in items {
        match item {
            ExternalDecl::FunctionDef(f) => {
                if f.specifiers.storage == Some(StorageClass::Static) {
                    continue;
                }
                if let Some(name) = declarator_name(&f.declarator) {
                    out.push(Defined {
                        name,
                        kind: SymbolKind::Function,
                    });
                }
            }
            ExternalDecl::Declaration(decl) => out.extend(own_defined_from_declaration(decl)),
        }
    }
    out
}

fn own_defined_from_declaration(decl: &Declaration) -> Vec<Defined> {
    if matches!(
        decl.specifiers.storage,
        Some(StorageClass::Typedef) | Some(StorageClass::Extern) | Some(StorageClass::Static)
    ) {
        return Vec::new(); // not a definition in this file, or internal linkage
    }
    decl.declarators
        .iter()
        .filter_map(|init_decl| {
            let name = declarator_name(&init_decl.declarator)?;
            if is_function_declarator(&init_decl.declarator) {
                return None; // a bare top-level prototype, not a definition
            }
            Some(Defined {
                name,
                kind: SymbolKind::Variable,
            })
        })
        .collect()
}

/// Names declared (any storage class) at `path`'s own top level only --
/// deliberately *not* `ExportResolver`, which would also pull in whatever
/// `path` itself `#include`s, blurring "declared in this exact header".
///
/// Uses the Steps 1-3 + rough-scan pipeline (`parse`, not `parse_full`):
/// headers routinely reference typedefs (e.g. `boolean`) they never
/// `#include` themselves, relying on whatever `.c` file includes them to
/// have pulled those in first -- `parse_full`'s real grammar parse needs a
/// typedef table and fails standalone on exactly this shape, but the rough
/// top-level scan (same one `exports.rs`'s `ExportResolver` relies on) never
/// needed one to begin with.
fn own_top_level_names(path: &Path) -> Option<std::collections::HashSet<String>> {
    let (_, resolved) = parse(path.to_str()?).ok()?;
    let entries = lex_chunks(&resolved).ok()?;
    let stream = attach_comments(entries);
    let items = extract_top_level_decls(&stream);
    let mut names = std::collections::HashSet::new();
    for item in &items {
        match item {
            ExternalDecl::FunctionDef(f) => {
                if let Some(name) = declarator_name(&f.declarator) {
                    names.insert(name);
                }
            }
            ExternalDecl::Declaration(decl) => {
                for init_decl in &decl.declarators {
                    if let Some(name) = declarator_name(&init_decl.declarator) {
                        names.insert(name);
                    }
                }
            }
        }
    }
    Some(names)
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10");
    let mut c_files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
        .collect();
    c_files.sort();

    // Every other .c file's transitively-visible symbol set (follows the
    // real #include graph), keyed by filename -- the proxy for "some other
    // module can actually call this today".
    let mut resolver = ExportResolver::new();
    let mut visible_elsewhere: HashMap<String, Vec<String>> = HashMap::new();
    for path in &c_files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let exports = resolver.resolve(path);
        for name in exports.symbols.keys() {
            visible_elsewhere
                .entry(name.clone())
                .or_default()
                .push(fname.clone());
        }
    }

    let mut no_matching_header = Vec::new();
    let mut wrongly_private = Vec::new();
    let mut correctly_private = Vec::new();
    let mut header_covered = 0usize;

    for path in &c_files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let Ok((_, unit)) = parse_full(path.to_str().unwrap()) else {
            continue;
        };
        let defined = own_defined_symbols(&unit.items);
        if defined.is_empty() {
            continue;
        }

        let header_path = path.with_extension("h");
        let header_names = if header_path.exists() {
            own_top_level_names(&header_path)
        } else {
            None
        };

        for sym in &defined {
            let declared_in_matching_header = header_names
                .as_ref()
                .is_some_and(|names| names.contains(&sym.name));
            if declared_in_matching_header {
                header_covered += 1;
                continue;
            }

            // "elsewhere" excludes this symbol's own defining file --
            // ExportResolver includes a file's own top-level exports too.
            let users: Vec<&String> = visible_elsewhere
                .get(&sym.name)
                .map(|files| files.iter().filter(|f| **f != fname).collect())
                .unwrap_or_default();

            if header_path.exists() {
                if users.is_empty() {
                    correctly_private.push((fname.clone(), sym.clone()));
                } else {
                    wrongly_private.push((fname.clone(), sym.clone(), users.len()));
                }
            } else {
                no_matching_header.push((fname.clone(), sym.clone(), users.len()));
            }
        }
    }

    println!(
        "{} .c files, {header_covered} externally-linked definitions correctly \
         covered by their own matching header.\n",
        c_files.len()
    );

    println!(
        "--- WRONGLY PRIVATE: not in matching header, but used elsewhere ({}) ---",
        wrongly_private.len()
    );
    for (file, sym, n_users) in &wrongly_private {
        println!(
            "  {file}: {:?} `{}` -- visible to {n_users} other .c file(s)",
            sym.kind, sym.name
        );
    }

    println!(
        "\n--- NO MATCHING HEADER AT ALL, defined symbol ({}) ---",
        no_matching_header.len()
    );
    for (file, sym, n_users) in &no_matching_header {
        let usage = if *n_users > 0 {
            format!("used by {n_users} other .c file(s)")
        } else {
            "no cross-file use found".to_string()
        };
        println!("  {file}: {:?} `{}` -- {usage}", sym.kind, sym.name);
    }

    println!(
        "\n--- correctly private (not in matching header, no other user found): {} ---",
        correctly_private.len()
    );
}
