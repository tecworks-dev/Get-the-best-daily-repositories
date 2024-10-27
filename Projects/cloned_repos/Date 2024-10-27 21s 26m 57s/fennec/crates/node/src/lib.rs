use std::any::Any;
use std::any::TypeId;
use std::fmt::Debug;

use fennec_span::HasSpan;

/// Represents an AST node.
pub trait Node: Any + Sync + Send + HasSpan + Debug {
    /// Retrieve the children of the node, which are the
    /// nodes that are contained within this node.
    fn children(&self) -> Vec<&dyn Node>;
}

/// Downcast a node to a concrete type.
///
/// This function is used to downcast a trait object to a concrete type.
///
/// # Type Parameters
///
/// - `TNode`: The type to downcast the node to.
///
/// # Parameters
///
/// - `node`: The node that is downcasted.
///
/// # Returns
///
/// The downcasted node if it is of the specified type.
pub fn downcast<'ast, TNode>(node: &'ast dyn Node) -> Option<&'ast TNode>
where
    TNode: Node + 'static,
{
    // Get `TypeId` of the type this function is instantiated with.
    let t = TypeId::of::<TNode>();

    // Get `TypeId` of the node we want to downcast.
    let concrete = node.type_id();

    // Compare both `TypeId`s on equality.
    if t == concrete {
        // Get the concrete type pointer from the trait object.
        let concrete = node as *const dyn Node as *const TNode;

        // Convert it to a reference and return it.
        //
        // SAFETY: This is safe because we know for sure that the pointer
        // is valid and references are only handed out for the lifetime
        // of the function ( 'ast ).
        let concrete: &'ast TNode = unsafe {
            // Return a reference to the concrete type.
            &*concrete
        };

        Some(concrete)
    } else {
        None
    }
}

/// Check if a node is of a specific type.
///
/// This function is used to check if a trait object is of a specific type at runtime,
/// without downcasting it to the concrete type.
///
/// # Type Parameters
///
/// - `TNode`: The type to check the node against.
///
/// # Parameters
///
/// - `node`: The node that is checked.
///
/// # Returns
///
/// `true` if the node is of the specified type, `false` otherwise.
pub fn is<'ast, TNode>(node: &'ast dyn Node) -> bool
where
    TNode: Node + 'static,
{
    // Get `TypeId` of the type this function is instantiated with.
    let t = TypeId::of::<TNode>();

    // Get `TypeId` of the node we want to check.
    let concrete = node.type_id();

    // Compare both `TypeId`s on equality.
    t == concrete
}

/// Find all nodes of a specific type in a node.
///
/// This function is used to find all nodes of a specific type in a node,
/// including all children of the node.
///
/// # Type Parameters
///
/// - `TNode`: The type to find in the node.
///
/// # Parameters
///
/// - `node`: The node to search for nodes of the specified type.
///
/// # Returns
///
/// A vector containing all nodes of the specified type.
pub fn find_all<'ast, TNode>(node: &'ast dyn Node) -> Vec<&'ast TNode>
where
    TNode: Node + 'static,
{
    let mut result = Vec::new();
    let mut stack = vec![node];
    let t = TypeId::of::<TNode>();

    while let Some(node) = stack.pop() {
        if node.type_id() == t {
            // SAFETY: The cast is safe because we've confirmed the type matches
            let concrete = unsafe { &*(node as *const dyn Node as *const TNode) };

            result.push(concrete);
        }

        stack.extend(node.children());
    }

    result
}

/// Macro to simplify downcasting.
#[macro_export]
macro_rules! downcast {
    // Simple downcast: Attempts to downcast `$node` to `$type` and returns an `Option`.
    ($node:expr, $type:ty) => {
        $crate::downcast::<$type>($node)
    };
    // Downcast with pattern matching and code block:
    // If the downcast to `$type` is successful, the value is bound to `$pattern` and `$block` is executed.
    ($node:expr, $type:ty as $pattern:pat, $block:block $(,)?) => {
        if let Some($pattern) = $crate::downcast::<$type>($node) $block
    };
    ($node:expr, $type:ty as $pattern:pat if $cond:expr, $block:block $(,)?) => {
        if let Some($pattern) = $crate::downcast::<$type>($node) {
            if $cond $block
        }
    };
    // Downcast with pattern matching, code block, and else block:
    // If the downcast succeeds, the value is bound to `$pattern` and `$block` is executed.
    // Otherwise, `$else` is executed.
    ($node:expr, $type:ty as $pattern:pat, $block:block, $else:block $(,)?) => {
        if let Some($pattern) = $crate::downcast::<$type>($node) $block else $else
    };
}

/// Macro to simplify type checking.
#[macro_export]
macro_rules! is {
    // Type check: Returns `true` if `$node` is any of the provided types, otherwise `false`.
    ($node:expr, $($type:ty),+) => {
        $($crate::is::<$type>($node) ||)+ false
    };
    // Type check with code block:
    // If `$node` is any of the specified types, `$block` is executed.
    ($node:expr, $($type:ty),+, $block:block) => {
        if $($crate::is::<$type>($node) ||)+ false $block
    };
    // Type check with code block and else block:
    // If `$node` is one of the specified types, `$block` is executed.
    // Otherwise, `$else` is executed.
    ($node:expr, $($type:ty),+, $block:block, $else:block) => {
        if $($crate::is::<$type>($node) ||)+ false $block else $else
    };
}
