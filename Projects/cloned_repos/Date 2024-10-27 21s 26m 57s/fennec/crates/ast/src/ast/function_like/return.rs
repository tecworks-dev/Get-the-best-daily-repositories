use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::type_hint::Hint;

/// Represents a function-like return type hint in PHP.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct FunctionLikeReturnTypeHint {
    pub colon: Span,
    pub hint: Hint,
}

impl HasSpan for FunctionLikeReturnTypeHint {
    fn span(&self) -> Span {
        Span::between(self.colon, self.hint.span())
    }
}
