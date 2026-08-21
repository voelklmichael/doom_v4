//! Step 1: Scope & Symbol Tables (docs/02_TYPECHECKER.md Step 1)
//!
//! A stack of lexical scopes, each with its own ordinary-identifier
//! namespace and its own tag namespace (`struct`/`union`/`enum`), per C89's
//! rule that tags and ordinary identifiers never collide. Scope 0 is always
//! the translation unit's file/global scope; `enter_scope`/`exit_scope`
//! push/pop for function bodies and every nested `{ ... }` block.
//!
//! This table only distinguishes global vs. block scope structurally (a
//! plain stack) -- it doesn't model file-static linkage as a separate scope
//! layer, since within a single translation unit a top-level `static`
//! declaration resolves exactly like a non-static one; the difference only
//! matters for cross-translation-unit linking, which is out of scope here
//! (this pipeline processes one `.c` file at a time, matching Step 6b's
//! per-translation-unit import model). Each `Symbol` still carries its
//! `storage` class, so that distinction isn't lost, just not scope-shaped.

use crate::parser::ast::StorageClass;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Variable,
    Parameter,
    EnumConstant,
    Typedef,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub storage: Option<StorageClass>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    Struct,
    Union,
    Enum,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tag {
    pub name: String,
    pub kind: TagKind,
    /// Whether this occurrence actually defined the tag's members
    /// (`struct Foo { ... }`), as opposed to a forward reference / bare use
    /// (`struct Foo *p;`).
    pub defined: bool,
}

#[derive(Debug, Default)]
struct Scope {
    idents: HashMap<String, Symbol>,
    tags: HashMap<String, Tag>,
}

/// A stack of lexical scopes: index 0 is always the file/global scope.
#[derive(Debug)]
pub struct SymbolTable {
    scopes: Vec<Scope>,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self {
            scopes: vec![Scope::default()],
        }
    }
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    pub fn exit_scope(&mut self) {
        assert!(self.scopes.len() > 1, "cannot pop the file/global scope");
        self.scopes.pop();
    }

    pub fn is_global(&self) -> bool {
        self.scopes.len() == 1
    }

    /// Declares `symbol` in the innermost scope. A redeclaration in the same
    /// scope (e.g. a forward function prototype followed by its
    /// definition) simply overwrites the earlier entry -- this table
    /// doesn't check prototype/definition compatibility, only visibility.
    pub fn declare(&mut self, symbol: Symbol) {
        self.scopes
            .last_mut()
            .unwrap()
            .idents
            .insert(symbol.name.clone(), symbol);
    }

    /// Declares `tag` in the innermost scope, unless it would blank out an
    /// already-defined tag with a later bare forward reference (`struct Foo
    /// *p;` after `struct Foo { ... };` has already been seen).
    pub fn declare_tag(&mut self, tag: Tag) {
        let scope = self.scopes.last_mut().unwrap();
        match scope.tags.get(&tag.name) {
            Some(existing) if existing.defined && !tag.defined => {}
            _ => {
                scope.tags.insert(tag.name.clone(), tag);
            }
        }
    }

    /// Looks up `name` in the ordinary identifier namespace, innermost
    /// scope outward.
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.scopes.iter().rev().find_map(|s| s.idents.get(name))
    }

    /// Looks up `name` in the tag namespace, innermost scope outward.
    pub fn lookup_tag(&self, name: &str) -> Option<&Tag> {
        self.scopes.iter().rev().find_map(|s| s.tags.get(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_declare_and_lookup() {
        let mut table = SymbolTable::new();
        table.declare(Symbol {
            name: "x".to_string(),
            kind: SymbolKind::Variable,
            storage: None,
        });
        assert!(table.lookup("x").is_some());
        assert!(table.lookup("y").is_none());
    }

    #[test]
    fn test_inner_scope_shadows_outer() {
        let mut table = SymbolTable::new();
        table.declare(Symbol {
            name: "x".to_string(),
            kind: SymbolKind::Variable,
            storage: Some(StorageClass::Static),
        });
        table.enter_scope();
        table.declare(Symbol {
            name: "x".to_string(),
            kind: SymbolKind::Parameter,
            storage: None,
        });
        assert_eq!(table.lookup("x").unwrap().kind, SymbolKind::Parameter);
        table.exit_scope();
        assert_eq!(table.lookup("x").unwrap().kind, SymbolKind::Variable);
    }

    #[test]
    fn test_exited_scope_is_invisible() {
        let mut table = SymbolTable::new();
        table.enter_scope();
        table.declare(Symbol {
            name: "local".to_string(),
            kind: SymbolKind::Variable,
            storage: None,
        });
        table.exit_scope();
        assert!(table.lookup("local").is_none());
    }

    #[test]
    fn test_tags_and_idents_dont_collide() {
        // `struct foo` and a variable `foo` are different namespaces.
        let mut table = SymbolTable::new();
        table.declare(Symbol {
            name: "foo".to_string(),
            kind: SymbolKind::Variable,
            storage: None,
        });
        table.declare_tag(Tag {
            name: "foo".to_string(),
            kind: TagKind::Struct,
            defined: true,
        });
        assert!(table.lookup("foo").is_some());
        assert!(table.lookup_tag("foo").is_some());
    }

    #[test]
    fn test_forward_reference_does_not_erase_definition() {
        let mut table = SymbolTable::new();
        table.declare_tag(Tag {
            name: "foo".to_string(),
            kind: TagKind::Struct,
            defined: true,
        });
        table.declare_tag(Tag {
            name: "foo".to_string(),
            kind: TagKind::Struct,
            defined: false,
        });
        assert!(table.lookup_tag("foo").unwrap().defined);
    }
}
