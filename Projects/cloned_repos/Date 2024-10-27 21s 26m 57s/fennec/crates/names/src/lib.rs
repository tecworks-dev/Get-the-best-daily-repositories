use ahash::HashMap;
use fennec_span::HasPosition;
use serde::Deserialize;
use serde::Serialize;

use fennec_ast::Program;
use fennec_interner::StringIdentifier;
use fennec_interner::ThreadedInterner;
use fennec_span::Position;
use fennec_walker::MutWalker;

use crate::internal::context::NameContext;
use crate::internal::resolver::NameResolver;

mod internal;

/// Represents a collection of resolved names in a program.
///
/// This struct stores a mapping of positions (represented as byte offsets)
/// to resolved names (represented as `StringIdentifier`s).
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Names {
    pub(crate) names: HashMap<usize, (StringIdentifier, bool)>,
}

impl Names {
    /// Resolves names in the given program.
    ///
    /// This method traverses the AST of the program, resolves names using a `NameResolver` and `NameContext`,
    /// and returns a `Names` instance containing the resolved names.
    ///
    /// # Arguments
    ///
    /// * `interner` - A `ThreadedInterner` used for string interning.
    /// * `program` - A reference to the `Program` AST node.
    ///
    /// # Returns
    ///
    /// A `Names` instance containing the resolved names.
    pub fn resolve<'ast, 'a>(interner: &'a ThreadedInterner, program: &'ast Program) -> Self {
        let mut resolver: NameResolver = NameResolver::new();
        let mut context: NameContext = NameContext::new(interner);

        resolver.walk_program(program, &mut context);

        resolver.resolved_names
    }

    /// Checks if a name is resolved at the given position.
    ///
    /// # Arguments
    ///
    /// * `position` - A reference to the `Position` in the code.
    ///
    /// # Returns
    ///
    /// `true` if a name is resolved at the given position, `false` otherwise.
    pub fn contains(&self, position: &Position) -> bool {
        return self.names.contains_key(&position.offset);
    }

    /// Gets the resolved name at the given position.
    ///
    /// # Arguments
    ///
    /// * `position` - A reference to a type that implements `HasPosition`.
    ///
    /// # Returns
    ///
    /// The `StringIdentifier` of the resolved name.
    ///
    /// # Panics
    ///
    /// Panics if the name is not found at the given position.
    pub fn get(&self, position: &impl HasPosition) -> StringIdentifier {
        return self
            .names
            .get(&position.position().offset)
            .map(|(name, _)| name.clone())
            .expect("name not found at position");
    }

    /// Returns whether the name at the given position was explicitly imported.
    ///
    /// # Arguments
    ///
    /// * `position` - A reference to the `Position` in the code.
    ///
    /// # Returns
    ///
    /// `true` if the name was imported, `false` otherwise.
    pub fn is_imported(&self, position: &impl HasPosition) -> bool {
        return self.names.get(&position.position().offset).map(|(_, imported)| *imported).unwrap_or(false);
    }

    /// Inserts a resolved name at the given position.
    ///
    /// This method is intended for internal use within the crate.
    ///
    /// # Arguments
    ///
    /// * `position` - The position (as a byte offset) where the name is resolved.
    /// * `name` - The `StringIdentifier` of the resolved name.
    pub(crate) fn insert_at<P: Into<usize>>(&mut self, position: P, name: StringIdentifier, imported: bool) {
        self.names.insert(position.into(), (name, imported));
    }
}
