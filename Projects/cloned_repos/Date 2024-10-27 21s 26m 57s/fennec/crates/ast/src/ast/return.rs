use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;
use crate::ast::keyword::Keyword;
use crate::ast::terminator::Terminator;

/// Represents a PHP `return` statement.
///
/// # Examples
///
/// ```php
/// <?php
///
/// function example(): int {
///     return 1;
/// }
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Return {
    pub r#return: Keyword,
    pub value: Option<Expression>,
    pub terminator: Terminator,
}

impl HasSpan for Return {
    fn span(&self) -> Span {
        self.r#return.span().join(self.terminator.span())
    }
}
