use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConcatOperation {
    pub lhs: Expression,
    pub dot: Span,
    pub rhs: Expression,
}

impl HasSpan for ConcatOperation {
    fn span(&self) -> Span {
        self.lhs.span().join(self.rhs.span())
    }
}
