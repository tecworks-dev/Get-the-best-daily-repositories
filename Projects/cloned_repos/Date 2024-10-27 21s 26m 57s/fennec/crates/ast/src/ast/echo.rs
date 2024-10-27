use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;
use crate::ast::keyword::Keyword;
use crate::ast::terminator::Terminator;
use crate::sequence::TokenSeparatedSequence;

/// Represents a PHP `echo` statement.
///
/// # Examples
///
/// ```php
/// <?php
///
/// echo "Hello, World!";
/// echo $a, $b, $c;
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Echo {
    pub echo: Keyword,
    pub values: TokenSeparatedSequence<Expression>,
    pub terminator: Terminator,
}

impl HasSpan for Echo {
    fn span(&self) -> Span {
        self.echo.span().join(self.terminator.span())
    }
}
