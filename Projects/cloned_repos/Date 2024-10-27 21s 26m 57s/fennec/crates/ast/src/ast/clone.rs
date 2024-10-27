use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;
use crate::ast::keyword::Keyword;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Clone {
    pub clone: Keyword,
    pub object: Expression,
}

impl HasSpan for Clone {
    fn span(&self) -> Span {
        self.clone.span().join(self.object.span())
    }
}
