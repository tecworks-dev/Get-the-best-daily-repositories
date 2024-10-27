use serde::Deserialize;
use serde::Serialize;
use strum::Display;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;

/// Represents a PHP bitwise operation.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum BitwiseOperation {
    Prefix(BitwisePrefixOperation),
    Infix(BitwiseInfixOperation),
}

/// Represents a PHP bitwise prefix operator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum BitwisePrefixOperator {
    Not(Span),
}

/// Represents a PHP bitwise prefix operation.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct BitwisePrefixOperation {
    pub operator: BitwisePrefixOperator,
    pub value: Expression,
}

/// Represents a PHP bitwise infix operator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum BitwiseInfixOperator {
    And(Span),
    Or(Span),
    Xor(Span),
    LeftShift(Span),
    RightShift(Span),
}

/// Represents a PHP bitwise infix operation.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct BitwiseInfixOperation {
    pub lhs: Expression,
    pub operator: BitwiseInfixOperator,
    pub rhs: Expression,
}

impl HasSpan for BitwiseOperation {
    fn span(&self) -> Span {
        match self {
            BitwiseOperation::Prefix(operation) => operation.span(),
            BitwiseOperation::Infix(operation) => operation.span(),
        }
    }
}

impl HasSpan for BitwisePrefixOperator {
    fn span(&self) -> Span {
        match self {
            BitwisePrefixOperator::Not(span) => *span,
        }
    }
}

impl HasSpan for BitwisePrefixOperation {
    fn span(&self) -> Span {
        self.operator.span().join(self.value.span())
    }
}

impl HasSpan for BitwiseInfixOperator {
    fn span(&self) -> Span {
        match self {
            BitwiseInfixOperator::And(span) => *span,
            BitwiseInfixOperator::Or(span) => *span,
            BitwiseInfixOperator::Xor(span) => *span,
            BitwiseInfixOperator::LeftShift(span) => *span,
            BitwiseInfixOperator::RightShift(span) => *span,
        }
    }
}

impl HasSpan for BitwiseInfixOperation {
    fn span(&self) -> Span {
        self.lhs.span().join(self.rhs.span())
    }
}
