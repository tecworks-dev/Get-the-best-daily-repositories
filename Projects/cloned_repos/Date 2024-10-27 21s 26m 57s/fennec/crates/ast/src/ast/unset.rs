use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;
use crate::ast::keyword::Keyword;
use crate::ast::terminator::Terminator;
use crate::sequence::TokenSeparatedSequence;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Unset {
    pub unset: Keyword,
    pub left_parenthesis: Span,
    pub values: TokenSeparatedSequence<Expression>,
    pub right_parenthesis: Span,
    pub terminator: Terminator,
}

impl HasSpan for Unset {
    fn span(&self) -> Span {
        self.unset.span().join(self.terminator.span())
    }
}
