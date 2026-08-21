//! Step 3: Declared Types
//!
//! Step 0 (`exports.rs`) deliberately collected only a *coarse kind* for
//! each cross-header symbol/tag, matching how Step 1 itself deferred full
//! `Type` computation to this step (see `exports.rs`'s module docs). Step 3
//! is where that deferred work actually happens: this module is "Step 0,
//! but for full types" -- the exact same recursive, cycle-guarded,
//! memoized `#include`-union shape (reusing `system_headers.rs` and Step
//! 6a's rough top-level scan, `grammar::extract_top_level_decls`), just
//! extracting a real `Type`/`FunctionSignature` per declaration instead of
//! a bare `SymbolKind`.
//!
//! Four things are collected, all needed for Step 3's "every
//! assignment/cast/call-argument site is checked" validation criterion:
//! - **typedefs**: name -> underlying `Type` (possibly itself a
//!   `Type::Named` of another typedef -- chains are resolved at lookup
//!   time, not collection time, by `resolve_typedef`).
//! - **functions**: name -> `FunctionSignature`, for checking call
//!   arguments against declared parameter types.
//! - **variables**: name -> declared `Type`, for typing a plain identifier
//!   reference the way `MacroTyper` currently can't (Step 2 only knows
//!   enum-constant identifiers).
//! - **fields**: struct/union tag name -> `[(field name, Type)]`, for
//!   typing `Expr::Member` (Step 2 always returned `Unknown` here, since
//!   Step 0 never kept a member list -- see `types.rs`'s module docs).
//!
//! **Respects linkage** exactly like Step 0: a `static` function/variable
//! has internal linkage, so it's visible within its own defining file
//! (kept in that file's own cached result) but never unioned into whatever
//! `#include`s it. Typedefs and tags aren't subject to `static` (same
//! reasoning as Step 0 -- see `exports.rs`'s module docs), so those always
//! cross the `#include` boundary.
//!
//! **Scope note**: like Step 0, this only scans *top-level* declarations
//! (Step 6a's `skip_bodies` mode) -- a struct/union defined inline inside
//! another struct's field list isn't captured (matching Step 0's own
//! shallow, non-recursive `scan_decl_specifiers`, not Step 1's fully
//! recursive one). Real, but not yet measured to matter -- see
//! `docs/KNOWN_LIMITATIONS.md` if it does.

use crate::parser::ast::{
    DeclSpecifiers, Declaration, ExternalDecl, StorageClass, StructOrUnionSpec, TypeSpecifier,
};
use crate::parser::grammar::{declarator_name, extract_top_level_decls};
use crate::parser::system_headers::{read_resolved_chunks_and_includes, resolve_include_path};
use crate::parser::{attach_comments, lex_chunks};
use crate::typecheck::types::{
    FunctionSignature, Type, function_signature, type_from_declarator, type_from_specifiers,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A file's own declared types, keyed exactly like Step 0's `ExportedDecls`
/// -- functions/variables still carry their `StorageClass` so a caller
/// merging a *child* file's result can filter out `static` ones, the same
/// way `ExportResolver::resolve_inner` does.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeclaredTypes {
    pub typedefs: HashMap<String, Type>,
    pub functions: HashMap<String, (FunctionSignature, Option<StorageClass>)>,
    pub variables: HashMap<String, (Type, Option<StorageClass>)>,
    pub fields: HashMap<String, Vec<(String, Type)>>,
}

impl DeclaredTypes {
    /// Follows a possibly-chained typedef name down to its first
    /// non-`Named` underlying type (Step 3's "resolve down to underlying
    /// representation for conversion purposes" -- the original `Type`
    /// still has the name for diagnostics/codegen, this is only for
    /// compatibility checking). Cycle-guarded; an unknown or cyclic name
    /// resolves to `Type::Unknown`.
    pub fn resolve_typedef(&self, ty: &Type) -> Type {
        let mut current = ty.clone();
        let mut seen = HashSet::new();
        loop {
            match current {
                Type::Named(name) => {
                    if !seen.insert(name.clone()) {
                        return Type::Unknown; // cyclic typedef chain
                    }
                    match self.typedefs.get(&name) {
                        Some(next) => current = next.clone(),
                        None => return Type::Unknown,
                    }
                }
                other => return other,
            }
        }
    }

    /// `resolve_typedef`, but recursively into `Pointer`/`Array`/`Function`
    /// wrappers too -- `resolve_typedef` alone only unwraps when `ty`
    /// itself is directly `Type::Named` at the top level, so e.g.
    /// `Pointer(Named("mobj_t"))` would pass through unchanged even though
    /// `mobj_t` is itself a typedef of `Struct("mobj_s")`. Compatibility
    /// checks (`types::is_assignment_compatible`) need this deeper form --
    /// comparing `Pointer(Named("mobj_t"))` against `Pointer(Struct("mobj_s"))`
    /// structurally would otherwise flag two spellings of the same type as
    /// incompatible.
    pub fn normalize(&self, ty: &Type) -> Type {
        match self.resolve_typedef(ty) {
            Type::Pointer(inner) => Type::Pointer(Box::new(self.normalize(&inner))),
            Type::Array(inner) => Type::Array(Box::new(self.normalize(&inner))),
            Type::Function(inner) => Type::Function(Box::new(self.normalize(&inner))),
            other => other,
        }
    }
}

fn scan_struct_or_union_fields(spec: &StructOrUnionSpec, out: &mut DeclaredTypes) {
    let (Some(name), Some(fields)) = (&spec.name, &spec.fields) else {
        return;
    };
    out.fields.entry(name.clone()).or_insert_with(|| {
        fields
            .iter()
            .flat_map(|field| {
                let base = type_from_specifiers(&field.specifiers);
                field.declarators.iter().filter_map(move |(d, _bitwidth)| {
                    let d = d.as_ref()?;
                    let field_name = declarator_name(d)?;
                    Some((field_name, type_from_declarator(base.clone(), d)))
                })
            })
            .collect()
    });
}

fn scan_decl_specifiers(specs: &DeclSpecifiers, out: &mut DeclaredTypes) {
    for ts in &specs.type_specifiers {
        match ts {
            TypeSpecifier::Struct(spec) | TypeSpecifier::Union(spec) => {
                scan_struct_or_union_fields(spec, out);
            }
            _ => {}
        }
    }
}

fn scan_declaration(decl: &Declaration, out: &mut DeclaredTypes) {
    scan_decl_specifiers(&decl.specifiers, out);
    let base = type_from_specifiers(&decl.specifiers);
    if decl.specifiers.storage == Some(StorageClass::Typedef) {
        for init_decl in &decl.declarators {
            if let Some(name) = declarator_name(&init_decl.declarator) {
                let ty = type_from_declarator(base.clone(), &init_decl.declarator);
                out.typedefs.entry(name).or_insert(ty);
            }
        }
        return;
    }
    for init_decl in &decl.declarators {
        let Some(name) = declarator_name(&init_decl.declarator) else {
            continue;
        };
        if let Some(sig) = function_signature(base.clone(), &init_decl.declarator) {
            out.functions
                .entry(name)
                .or_insert((sig, decl.specifiers.storage));
        } else {
            let ty = type_from_declarator(base.clone(), &init_decl.declarator);
            out.variables
                .entry(name)
                .or_insert((ty, decl.specifiers.storage));
        }
    }
}

fn scan_top_level_declared_types(items: &[ExternalDecl]) -> DeclaredTypes {
    let mut out = DeclaredTypes::default();
    for item in items {
        match item {
            ExternalDecl::FunctionDef(f) => {
                scan_decl_specifiers(&f.specifiers, &mut out);
                let base = type_from_specifiers(&f.specifiers);
                if let (Some(sig), Some(name)) = (
                    function_signature(base, &f.declarator),
                    declarator_name(&f.declarator),
                ) {
                    out.functions
                        .entry(name)
                        .or_insert((sig, f.specifiers.storage));
                }
            }
            ExternalDecl::Declaration(decl) => scan_declaration(decl, &mut out),
        }
    }
    out
}

/// Resolves and caches each file's transitively-imported declared types,
/// exactly mirroring `ExportResolver`'s shape.
#[derive(Default)]
pub struct DeclaredTypesResolver {
    cache: HashMap<PathBuf, DeclaredTypes>,
}

impl DeclaredTypesResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resolve(&mut self, path: &Path) -> DeclaredTypes {
        let mut visiting = HashSet::new();
        self.resolve_inner(path, &mut visiting)
    }

    fn resolve_inner(&mut self, path: &Path, visiting: &mut HashSet<PathBuf>) -> DeclaredTypes {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }
        if !visiting.insert(key.clone()) {
            return DeclaredTypes::default();
        }

        let mut result = DeclaredTypes::default();
        if let Some((resolved, includes)) = read_resolved_chunks_and_includes(&key) {
            if let Ok(entries) = lex_chunks(&resolved) {
                let stream = attach_comments(entries);
                result = scan_top_level_declared_types(&extract_top_level_decls(&stream));
            }
            let dir = key.parent().unwrap_or_else(|| Path::new("."));
            for inc in includes {
                if let Some(resolved_path) = resolve_include_path(&inc, dir) {
                    let included = self.resolve_inner(&resolved_path, visiting);
                    for (name, ty) in included.typedefs {
                        result.typedefs.entry(name).or_insert(ty);
                    }
                    for (name, (sig, storage)) in included.functions {
                        if storage != Some(StorageClass::Static) {
                            result.functions.entry(name).or_insert((sig, storage));
                        }
                    }
                    for (name, (ty, storage)) in included.variables {
                        if storage != Some(StorageClass::Static) {
                            result.variables.entry(name).or_insert((ty, storage));
                        }
                    }
                    for (name, fields) in included.fields {
                        result.fields.entry(name).or_insert(fields);
                    }
                }
            }
        }

        visiting.remove(&key);
        self.cache.insert(key, result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../linuxdoom-1.10")
    }

    #[test]
    fn test_resolve_finds_fixed_t_typedef() {
        // `typedef int fixed_t;` in m_fixed.h.
        let mut resolver = DeclaredTypesResolver::new();
        let types = resolver.resolve(&corpus_dir().join("m_fixed.c"));
        assert_eq!(types.typedefs.get("fixed_t"), Some(&Type::Int));
    }

    #[test]
    fn test_resolve_finds_function_signature_from_header() {
        // `void *Z_Malloc(int size, int tag, void *user);` in z_zone.h.
        let mut resolver = DeclaredTypesResolver::new();
        let types = resolver.resolve(&corpus_dir().join("m_misc.c"));
        let sig = types
            .functions
            .get("Z_Malloc")
            .expect("expected Z_Malloc's signature");
        assert_eq!(sig.0.ret, Type::Pointer(Box::new(Type::Void)));
        assert_eq!(sig.0.params.len(), 3);
    }

    #[test]
    fn test_resolve_finds_extern_variable_type() {
        // `extern int gamemap;` in doomstat.h.
        let mut resolver = DeclaredTypesResolver::new();
        let types = resolver.resolve(&corpus_dir().join("m_misc.c"));
        assert_eq!(
            types.variables.get("gamemap").map(|(t, _)| t),
            Some(&Type::Int)
        );
    }

    #[test]
    fn test_static_function_not_exported_to_includers() {
        use crate::parser::{attach_comments, lex_chunks, parse_chunks};
        let (_, chunks) = parse_chunks("static int helper(void) { return 0; }\nint used(int x);\n");
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        let items = extract_top_level_decls(&stream);
        let scanned = scan_top_level_declared_types(&items);
        assert_eq!(
            scanned.functions.get("helper").map(|(_, s)| *s),
            Some(Some(StorageClass::Static))
        );
        assert!(scanned.functions.contains_key("used"));
    }

    #[test]
    fn test_typedef_chain_resolves_to_underlying() {
        let mut types = DeclaredTypes::default();
        types
            .typedefs
            .insert("a_t".to_string(), Type::Named("b_t".to_string()));
        types.typedefs.insert("b_t".to_string(), Type::Int);
        assert_eq!(
            types.resolve_typedef(&Type::Named("a_t".to_string())),
            Type::Int
        );
    }

    #[test]
    fn test_cyclic_typedef_chain_yields_unknown() {
        let mut types = DeclaredTypes::default();
        types
            .typedefs
            .insert("a_t".to_string(), Type::Named("b_t".to_string()));
        types
            .typedefs
            .insert("b_t".to_string(), Type::Named("a_t".to_string()));
        assert_eq!(
            types.resolve_typedef(&Type::Named("a_t".to_string())),
            Type::Unknown
        );
    }

    #[test]
    fn test_struct_field_types_scanned() {
        use crate::parser::{attach_comments, lex_chunks, parse_chunks};
        let (_, chunks) = parse_chunks("struct point_t { int x; int y; char *label; };\n");
        let entries = lex_chunks(&chunks).unwrap();
        let stream = attach_comments(entries);
        let items = extract_top_level_decls(&stream);
        let scanned = scan_top_level_declared_types(&items);
        let fields = scanned
            .fields
            .get("point_t")
            .expect("expected point_t's fields");
        assert_eq!(
            fields.as_slice(),
            &[
                ("x".to_string(), Type::Int),
                ("y".to_string(), Type::Int),
                ("label".to_string(), Type::Pointer(Box::new(Type::Char))),
            ]
        );
    }

    #[test]
    fn test_resolve_across_full_corpus_reports_coverage() {
        // Not a pass/fail assertion (matching this project's "measure,
        // don't assume" methodology) -- confirms the resolver runs cleanly
        // over the whole corpus and reports how much it found.
        let dir = corpus_dir();
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("linuxdoom-1.10 directory should exist")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        files.sort();
        assert!(files.len() > 50, "expected the full Doom .c corpus");

        let mut resolver = DeclaredTypesResolver::new();
        let mut total_typedefs = 0;
        let mut total_functions = 0;
        let mut total_variables = 0;
        let mut total_tags_with_fields = 0;
        for path in &files {
            let types = resolver.resolve(path);
            total_typedefs += types.typedefs.len();
            total_functions += types.functions.len();
            total_variables += types.variables.len();
            total_tags_with_fields += types.fields.len();
        }
        eprintln!(
            "declared types visible across {} files (summed per-file, not deduped): \
             {total_typedefs} typedefs, {total_functions} functions, {total_variables} \
             variables, {total_tags_with_fields} struct/union tags with field types",
            files.len()
        );
        assert!(total_typedefs > 0 && total_functions > 0);
    }
}
