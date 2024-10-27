#![allow(unused_variables)]

use fennec_ast::Node;

/// Traverse a given node using the given visitor.
///
/// # Parameters
///
/// - `node`: The node that is traversed.
/// - `visitor`: The visitor that is used to traverse the node.
/// - `context`: The context that is passed to the visitor.
///
/// # Lifetimes
///
/// - `'ast`: The lifetime of the AST nodes that are visited.
pub fn traverse<'ast, TContext>(node: Node<'ast>, visitor: &dyn NodeVisitor<TContext>, context: &mut TContext) {
    visitor.enter(&node, context);

    for child in node.children() {
        traverse(child, visitor, context);
    }

    visitor.exit(&node, context);
}

/// Traverse a given node using the given mutable visitor.
///
/// # Parameters
///
/// - `node`: The node that is traversed.
/// - `visitor`: The visitor that is used to traverse the node.
/// - `context`: The context that is passed to the visitor.
///
/// # Lifetimes
///
/// - `'ast`: The lifetime of the AST nodes that are visited.
pub fn traverse_mut<'ast, TContext>(
    node: Node<'ast>,
    visitor: &mut dyn MutNodeVisitor<TContext>,
    context: &mut TContext,
) {
    visitor.enter_mut(&node, context);

    for child in node.children() {
        traverse_mut(child, visitor, context);
    }

    visitor.enter_mut(&node, context);
}

/// A visitor that can be used to traverse an AST.
///
/// # Type Parameters
///
/// - `TContext`: The type of the context that is passed to the visitor.
pub trait NodeVisitor<TContext>: Sync + Send {
    /// Called when the visitor enters a node.
    ///
    /// # Parameters
    ///
    /// - `node`: The node that is entered.
    ///
    /// # Lifetimes
    ///
    /// - `'ast`: The lifetime of the AST nodes that are visited.
    fn enter<'ast>(&self, node: &Node<'ast>, context: &mut TContext);

    /// Called when the visitor exits a node.
    ///
    /// # Parameters
    ///
    /// - `node`: The node that is exited.
    ///
    /// # Lifetimes
    ///
    /// - `'ast`: The lifetime of the AST nodes that are visited.
    fn exit<'ast>(&self, node: &Node<'ast>, context: &mut TContext) {
        // Do nothing by default.
    }
}

/// A visitor that can be used to traverse an AST and modify the nodes.
///
/// # Type Parameters
///
/// - `TContext`: The type of the context that is passed to the visitor.
pub trait MutNodeVisitor<TContext>: Sync + Send {
    /// Called when the visitor enters a node.
    ///
    /// # Parameters
    ///
    /// - `node`: The node that is entered.
    ///
    /// # Lifetimes
    ///
    /// - `'ast`: The lifetime of the AST nodes that are visited.
    fn enter_mut<'ast>(&mut self, node: &Node<'ast>, context: &mut TContext);

    /// Called when the visitor exits a node.
    ///
    /// # Parameters
    ///
    /// - `node`: The node that is exited.
    ///
    /// # Lifetimes
    ///
    /// - `'ast`: The lifetime of the AST nodes that are visited.
    fn exit_mut<'ast>(&mut self, node: &Node<'ast>, context: &mut TContext) {
        // Do nothing by default.
    }
}

/// Implement the `MutNodeVisitor` trait for any type that implements the `NodeVisitor` trait.
impl<T> MutNodeVisitor<T> for dyn NodeVisitor<T> {
    fn enter_mut<'ast>(&mut self, node: &Node<'ast>, context: &mut T) {
        self.enter(node, context);
    }

    fn exit_mut<'ast>(&mut self, node: &Node<'ast>, context: &mut T) {
        self.exit(node, context);
    }
}
