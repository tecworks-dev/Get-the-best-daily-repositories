#![allow(dead_code)]

use std::collections::HashMap;

use ahash::RandomState;
use serde::Deserialize;
use serde::Serialize;
use strum::Display;

use fennec_ast::ast::*;
use fennec_interner::StringIdentifier;
use fennec_interner::ThreadedInterner;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum NameKind {
    Default,
    Function,
    Constant,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct NameResolutionContext {
    namespace_name: String,
    default_aliases: HashMap<String, String, RandomState>,
    function_aliases: HashMap<String, String, RandomState>,
    constant_aliases: HashMap<String, String, RandomState>,
}

#[derive(Debug)]
pub struct NameContext<'a> {
    pub interner: &'a ThreadedInterner,
    name_resolution_contexts: Vec<NameResolutionContext>,
    namespace_name: Option<String>,
}

impl<'a> NameContext<'a> {
    pub fn new(interner: &'a ThreadedInterner) -> Self {
        NameContext {
            interner,
            name_resolution_contexts: vec![NameResolutionContext::default()],
            namespace_name: Default::default(),
        }
    }

    pub fn get_namespace_name(&self) -> Option<String> {
        self.namespace_name.clone()
    }

    pub fn get_namespaced_identifier(&mut self, identifier: &LocalIdentifier) -> StringIdentifier {
        if let Some(mut namespaced) = self.get_namespace_name() {
            namespaced.push('\\');
            namespaced.extend(self.interner.lookup(identifier.value).chars());

            self.interner.intern(namespaced)
        } else {
            identifier.value
        }
    }

    pub fn enter_namespace(&mut self, namespace: StringIdentifier) {
        let previous_context =
            self.name_resolution_contexts.last().expect("expected there to be at least one name resolution context");

        let namespace_name = self.interner.lookup(namespace);

        self.name_resolution_contexts.push(NameResolutionContext {
            namespace_name: namespace_name.to_string(),
            default_aliases: previous_context.default_aliases.clone(),
            function_aliases: previous_context.function_aliases.clone(),
            constant_aliases: previous_context.constant_aliases.clone(),
        });

        self.namespace_name = Some(if let Some(mut previous_namespace) = self.namespace_name.clone() {
            previous_namespace.push('\\');
            previous_namespace.extend(namespace_name.chars());

            previous_namespace
        } else {
            namespace_name.to_string()
        });
    }

    pub fn exit_namespace(&mut self) {
        if self.name_resolution_contexts.len() <= 1 {
            return;
        }

        self.name_resolution_contexts.pop();
        self.namespace_name = if let Some(last_context) = self.name_resolution_contexts.last() {
            Some(last_context.namespace_name.clone())
        } else {
            None
        };
    }

    pub fn add_name(&mut self, kind: NameKind, name_id: StringIdentifier, alias_id: Option<StringIdentifier>) {
        let name = self.interner.lookup(name_id);

        let alias = match alias_id {
            Some(alias_id) => self.interner.lookup(alias_id).to_string(),
            None => {
                if let Some(last_backslash_pos) = name.rfind(|c| c == '\\') {
                    name[last_backslash_pos + 1..].to_string()
                } else {
                    name.to_string()
                }
            }
        };

        let context = self
            .name_resolution_contexts
            .last_mut()
            .expect("expected there to be at least one resolution context in the context");

        match kind {
            NameKind::Default => context.default_aliases.insert(alias, name.to_string()),
            NameKind::Function => context.function_aliases.insert(alias, name.to_string()),
            NameKind::Constant => context.constant_aliases.insert(alias, name.to_string()),
        };
    }

    pub fn resolve_name(&mut self, kind: NameKind, name_id: StringIdentifier) -> (StringIdentifier, bool) {
        let name = self.interner.lookup(name_id).to_string();

        if name.starts_with('\\') {
            let resolve = name[1..].to_string();

            return (self.interner.intern(resolve), true);
        }

        if let Some(alias) = self.resolve_alias(kind, name.as_str()) {
            return (alias, true);
        }

        match self.get_namespace_name() {
            Some(namespace_name) => {
                let mut resolved = namespace_name.clone();
                resolved.push('\\');
                resolved.push_str(name.as_str());

                (self.interner.intern(resolved), false)
            }
            None => (name_id, false),
        }
    }

    fn resolve_alias(&mut self, kind: NameKind, name: &str) -> Option<StringIdentifier> {
        let context = self
            .name_resolution_contexts
            .last()
            .expect("expected there to be at least one resolution context in the context");

        let parts = name.split(|c| c == '\\').collect::<Vec<_>>();
        let first_part = parts.first().expect("expected there to be at least one part in the name");

        if parts.len() > 1 {
            let suffix = parts[1..].join("\\");
            let first_part_lower = first_part.to_ascii_lowercase();

            let alias = if first_part_lower == "namespace" {
                if let Some(namespace_name) = &self.namespace_name {
                    let mut resolved = namespace_name.clone();
                    resolved.push('\\');
                    resolved.push_str(&suffix);

                    return Some(self.interner.intern(resolved));
                }

                return Some(self.interner.intern(suffix));
            } else {
                context.default_aliases.get(*first_part)
            };

            if let Some(alias) = alias {
                let mut resolved = alias.clone();
                resolved.push('\\');
                resolved.push_str(&suffix);

                return Some(self.interner.intern(alias));
            }
        } else {
            let alias = match kind {
                NameKind::Default => context.default_aliases.get(*first_part),
                NameKind::Function => context.function_aliases.get(*first_part),
                NameKind::Constant => context.constant_aliases.get(*first_part),
            };

            if let Some(resolved) = alias {
                return Some(self.interner.intern(resolved));
            }
        }

        None
    }
}
