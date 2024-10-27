use serde::Deserialize;
use serde::Serialize;
use strum::Display;

use fennec_interner::StringIdentifier;
use fennec_span::HasSpan;
use fennec_span::Span;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
pub enum InlineKind {
    Text,
    Shebang,
}

/// Represents inline text within a PHP script.
///
/// # Example:
///
/// ```php
/// This is an inline text.
/// <?php
///   // PHP code
/// ?>
/// This is another inline text.
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Inline {
    pub kind: InlineKind,
    pub span: Span,
    pub value: StringIdentifier,
}

impl HasSpan for Inline {
    fn span(&self) -> Span {
        self.span
    }
}
