use std::vec::IntoIter;

use fennec_source::HasSource;
use serde::Deserialize;
use serde::Serialize;

use crate::symbol::Symbol;
use crate::symbol::SymbolKind;

/// Represents a table of symbols, which can be functions, classes, variables, etc.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Default)]
pub struct SymbolTable {
    /// The list of symbols in the table.
    pub symbols: Vec<Symbol>,
}

impl SymbolTable {
    /// Creates a new, empty symbol table.
    pub fn new() -> Self {
        Self { symbols: Vec::new() }
    }

    /// Creates a new symbol table from an iterator of symbols.
    pub fn from_symbols(symbols: impl IntoIterator<Item = Symbol>) -> Self {
        Self { symbols: symbols.into_iter().collect() }
    }

    /// Returns the number of symbols in the table.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Returns `true` if the symbol table is empty.
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Adds a symbol to the table.
    pub fn add_symbol(&mut self, symbol: Symbol) {
        self.symbols.push(symbol);
    }

    /// Adds multiple symbols to the table.
    pub fn add_symbols(&mut self, symbols: impl IntoIterator<Item = Symbol>) {
        self.symbols.extend(symbols);
    }

    /// Merges another symbol table into this one.
    pub fn merge(&mut self, other: SymbolTable) {
        self.symbols.extend(other.symbols);
    }

    /// Returns a vector of references to the functions in the table.
    pub fn get_functions(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::Function).collect()
    }

    /// Returns a new symbol table containing only the functions from this table.
    pub fn only_functions(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::Function).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the closures in the table.
    pub fn get_closures(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::Closure).collect()
    }

    /// Returns a new symbol table containing only the closures from this table.
    pub fn only_closures(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::Closure).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the arrow functions in the table.
    pub fn get_arrow_functions(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::ArrowFunction).collect()
    }

    /// Returns a new symbol table containing only the arrow functions from this table.
    pub fn only_arrow_functions(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::ArrowFunction).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the constants in the table.
    pub fn get_constants(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::Constant).collect()
    }

    /// Returns a new symbol table containing only the constants from this table.
    pub fn only_constants(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::Constant).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the classes in the table.
    pub fn get_classes(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::Class).collect()
    }

    /// Returns a new symbol table containing only the classes from this table.
    pub fn only_classes(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::Class).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the anonymous classes in the table.
    pub fn get_anonymous_classes(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::AnonymousClass).collect()
    }

    /// Returns a new symbol table containing only the anonymous classes from this table.
    pub fn only_anonymous_classes(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::AnonymousClass).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the traits in the table.
    pub fn get_traits(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::Trait).collect()
    }

    /// Returns a new symbol table containing only the traits from this table.
    pub fn only_traits(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::Trait).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the enums in the table.
    pub fn get_enums(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::Enum).collect()
    }

    /// Returns a new symbol table containing only the enums from this table.
    pub fn only_enums(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::Enum).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the interfaces in the table.
    pub fn get_interfaces(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::Interface).collect()
    }

    /// Returns a new symbol table containing only the interfaces from this table.
    pub fn only_interfaces(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::Interface).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the class-like constants in the table.
    pub fn get_class_like_constants(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::ClassLikeConstant).collect()
    }

    /// Returns a new symbol table containing only the class-like constants from this table.
    pub fn only_class_like_constants(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::ClassLikeConstant).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the methods in the table.
    pub fn get_methods(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::Method).collect()
    }

    /// Returns a new symbol table containing only the methods from this table.
    pub fn only_methods(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::Method).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the enum cases in the table.
    pub fn get_enum_cases(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::EnumCase).collect()
    }

    /// Returns a new symbol table containing only the enum cases from this table.
    pub fn only_enum_cases(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::EnumCase).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the properties in the table.
    pub fn get_properties(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind == SymbolKind::Property).collect()
    }

    /// Returns a new symbol table containing only the properties from this table.
    pub fn only_properties(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind == SymbolKind::Property).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the class-like symbols in the table.
    pub fn get_class_like(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind.is_class_like()).collect()
    }

    /// Returns a new symbol table containing only the class-like symbols from this table.
    pub fn only_class_like(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind.is_class_like()).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the function-like symbols in the table.
    pub fn get_function_like(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind.is_function_like()).collect()
    }

    /// Returns a new symbol table containing only the function-like symbols from this table.
    pub fn only_function_like(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind.is_function_like()).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the anonymous symbols in the table.
    pub fn get_anonymous(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| symbol.kind.is_anonymous()).collect()
    }

    /// Returns a new symbol table containing only the anonymous symbols from this table.
    pub fn only_anonymous(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| symbol.kind.is_anonymous()).collect();

        Self { symbols }
    }

    /// Returns a vector of references to the symbols in the table that are not anonymous.
    pub fn get_non_anonymous(&self) -> Vec<&Symbol> {
        self.symbols.iter().filter(|symbol| !symbol.kind.is_anonymous()).collect()
    }

    /// Returns a new symbol table containing only the symbols from this table that are not anonymous.
    pub fn only_non_anonymous(self) -> Self {
        let symbols = self.symbols.into_iter().filter(|symbol| !symbol.kind.is_anonymous()).collect();

        Self { symbols }
    }

    /// Returns the function-like symbol (function, method, etc.) that encloses the given offset.
    ///
    /// This method iterates through the symbols in the table, filtering for function-like symbols
    /// that contain the given offset in their definition range. It returns the symbol with the
    /// largest starting offset, effectively finding the innermost function-like symbol containing
    /// the offset.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset to search for.
    ///
    /// # Returns
    ///
    /// * `Option<&Symbol>` - The enclosing function-like symbol, if found.
    pub fn get_enclosing_function_like(&self, offset: usize) -> Option<&Symbol> {
        self.symbols
            .iter()
            .filter(|symbol| symbol.kind.is_function_like())
            .filter(|symbol| symbol.span.has_offset(offset))
            .max_by_key(|symbol| symbol.span.start.offset)
    }

    /// Returns the class-like symbol (class, trait, etc.) that encloses the given offset.
    ///
    /// This method iterates through the symbols in the table, filtering for class-like symbols
    /// that contain the given offset in their definition range. It returns the symbol with the
    /// largest starting offset, effectively finding the innermost class-like symbol containing
    /// the offset.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset to search for.
    ///
    /// # Returns
    ///
    /// * `Option<&Symbol>` - The enclosing class-like symbol, if found.
    pub fn get_enclosing_class_like(&self, offset: usize) -> Option<&Symbol> {
        self.symbols
            .iter()
            .filter(|symbol| symbol.kind.is_class_like())
            .filter(|symbol| symbol.span.has_offset(offset))
            .max_by_key(|symbol| symbol.span.start.offset)
    }

    /// Sorts the symbols in the table by source and starting offset.
    ///
    /// This method sorts the symbols in the table by their source file and starting offset. This
    /// is useful for ensuring that the symbols are in a consistent order, which can be helpful for
    /// testing and debugging.
    pub fn sort(&mut self) {
        self.symbols
            .sort_by(|a, b| a.source().cmp(&b.source()).then_with(|| a.span.start.offset.cmp(&b.span.start.offset)));
    }

    /// Returns an iterator over the symbols in the table.
    pub fn iter(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols.iter()
    }

    /// Returns a mutable iterator over the symbols in the table.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Symbol> {
        self.symbols.iter_mut()
    }
}

impl IntoIterator for SymbolTable {
    type Item = Symbol;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.symbols.into_iter()
    }
}
