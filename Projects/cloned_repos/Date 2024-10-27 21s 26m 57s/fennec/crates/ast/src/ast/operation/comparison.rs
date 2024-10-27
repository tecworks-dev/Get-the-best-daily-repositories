use serde::Deserialize;
use serde::Serialize;
use strum::Display;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;

/// Represents a PHP comparison operator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum ComparisonOperator {
    Equal(Span),
    NotEqual(Span),
    Identical(Span),
    NotIdentical(Span),
    AngledNotEqual(Span),
    LessThan(Span),
    LessThanOrEqual(Span),
    GreaterThan(Span),
    GreaterThanOrEqual(Span),
    Spaceship(Span),
}

/// Represents a PHP comparison operation
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ComparisonOperation {
    pub lhs: Expression,
    pub operator: ComparisonOperator,
    pub rhs: Expression,
}

impl HasSpan for ComparisonOperator {
    fn span(&self) -> Span {
        match self {
            Self::Equal(span) => *span,
            Self::NotEqual(span) => *span,
            Self::Identical(span) => *span,
            Self::NotIdentical(span) => *span,
            Self::AngledNotEqual(span) => *span,
            Self::LessThan(span) => *span,
            Self::LessThanOrEqual(span) => *span,
            Self::GreaterThan(span) => *span,
            Self::GreaterThanOrEqual(span) => *span,
            Self::Spaceship(span) => *span,
        }
    }
}

impl HasSpan for ComparisonOperation {
    fn span(&self) -> Span {
        self.lhs.span().join(self.rhs.span())
    }
}
