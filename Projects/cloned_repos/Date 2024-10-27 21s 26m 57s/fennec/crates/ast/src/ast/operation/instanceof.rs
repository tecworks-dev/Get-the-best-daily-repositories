use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;
use crate::ast::keyword::Keyword;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct InstanceofOperation {
    pub lhs: Expression,
    pub instanceof: Keyword,
    pub rhs: Expression,
}

impl HasSpan for InstanceofOperation {
    fn span(&self) -> Span {
        self.lhs.span().join(self.rhs.span())
    }
}
