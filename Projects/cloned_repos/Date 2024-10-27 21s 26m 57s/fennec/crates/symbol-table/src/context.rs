use fennec_interner::ThreadedInterner;

use crate::symbol::Symbol;

#[derive(Debug)]
pub struct Context<'a> {
    pub interner: &'a ThreadedInterner,
    scope: Vec<Symbol>,
    namespaces: Vec<String>,
}

impl<'a> Context<'a> {
    pub fn new(interner: &'a ThreadedInterner) -> Self {
        Self { interner, scope: Vec::new(), namespaces: Vec::new() }
    }

    pub fn enter_namespace(&mut self, namespace: String) {
        self.namespaces.push(namespace);
    }

    pub fn exit_namespace(&mut self) {
        self.namespaces.pop();
    }

    pub fn get_namespace(&self) -> Option<&String> {
        self.namespaces.last()
    }

    pub fn enter_scope(&mut self, symbol: Symbol) {
        self.scope.push(symbol);
    }

    pub fn exit_scope(&mut self) -> Option<Symbol> {
        self.scope.pop()
    }

    pub fn get_scope(&self) -> Option<&Symbol> {
        self.scope.last()
    }
}
