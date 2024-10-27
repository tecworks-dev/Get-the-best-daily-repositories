use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::keyword::Keyword;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct HaltCompiler {
    pub halt_compiler: Keyword,
    pub left_parenthesis: Span,
    pub right_parenthesis: Span,
    pub semicolon: Span,
}

impl HasSpan for HaltCompiler {
    fn span(&self) -> Span {
        self.halt_compiler.span().join(self.semicolon)
    }
}
