use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;
use crate::ast::keyword::Keyword;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Throw {
    pub throw: Keyword,
    pub exception: Expression,
}

impl HasSpan for Throw {
    fn span(&self) -> Span {
        self.throw.span().join(self.exception.span())
    }
}
