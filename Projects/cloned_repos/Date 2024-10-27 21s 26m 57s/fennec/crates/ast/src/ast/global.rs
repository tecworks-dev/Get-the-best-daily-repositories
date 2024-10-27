use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::keyword::Keyword;
use crate::ast::terminator::Terminator;
use crate::ast::variable::Variable;
use crate::sequence::TokenSeparatedSequence;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Global {
    pub global: Keyword,
    pub variables: TokenSeparatedSequence<Variable>,
    pub terminator: Terminator,
}

impl HasSpan for Global {
    fn span(&self) -> Span {
        self.global.span().join(self.terminator.span())
    }
}
