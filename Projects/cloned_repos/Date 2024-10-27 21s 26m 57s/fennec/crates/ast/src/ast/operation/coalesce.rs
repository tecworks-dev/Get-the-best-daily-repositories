use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct CoalesceOperation {
    pub lhs: Expression,
    pub double_question_mark: Span,
    pub rhs: Expression,
}

impl HasSpan for CoalesceOperation {
    fn span(&self) -> Span {
        self.lhs.span().join(self.rhs.span())
    }
}
