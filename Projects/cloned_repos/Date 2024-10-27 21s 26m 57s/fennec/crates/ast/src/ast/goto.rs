use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::identifier::LocalIdentifier;
use crate::ast::keyword::Keyword;
use crate::ast::terminator::Terminator;

/// Represents a `goto` statement in PHP.
///
/// Example:
///
/// ```php
/// <?php
///
/// goto foo;
/// ```
///
/// or
///
/// ```php
/// <?php
///
/// goto foo
///
/// ?>
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Goto {
    pub goto: Keyword,
    pub label: LocalIdentifier,
    pub terminator: Terminator,
}

/// Represents a Go-To label statement in PHP.
///
/// Example:
///
/// ```php
/// <?php
///
/// foo:
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Label {
    pub name: LocalIdentifier,
    pub colon: Span,
}

impl HasSpan for Goto {
    fn span(&self) -> Span {
        Span::between(self.goto.span(), self.terminator.span())
    }
}

impl HasSpan for Label {
    fn span(&self) -> Span {
        Span::between(self.name.span(), self.colon)
    }
}
