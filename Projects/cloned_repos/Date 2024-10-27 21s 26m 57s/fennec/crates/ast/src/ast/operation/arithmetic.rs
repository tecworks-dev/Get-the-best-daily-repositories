use serde::Deserialize;
use serde::Serialize;
use strum::Display;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;

/// Represents a PHP arithmetic operation.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum ArithmeticOperation {
    Prefix(ArithmeticPrefixOperation),
    Infix(ArithmeticInfixOperation),
    Postfix(ArithmeticPostfixOperation),
}

/// Represents a PHP arithmetic prefix operator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum ArithmeticPrefixOperator {
    Increment(Span),
    Decrement(Span),
    Plus(Span),
    Minus(Span),
}

/// Represents a PHP arithmetic prefix operation.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ArithmeticPrefixOperation {
    pub operator: ArithmeticPrefixOperator,
    pub value: Expression,
}

/// Represents a PHP arithmetic infix operator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum ArithmeticInfixOperator {
    Addition(Span),
    Subtraction(Span),
    Multiplication(Span),
    Division(Span),
    Modulo(Span),
    Exponentiation(Span),
}

/// Represents a PHP arithmetic infix operation.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ArithmeticInfixOperation {
    pub lhs: Expression,
    pub operator: ArithmeticInfixOperator,
    pub rhs: Expression,
}

/// Represents a PHP arithmetic postfix operator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum ArithmeticPostfixOperator {
    Increment(Span),
    Decrement(Span),
}

/// Represents a PHP arithmetic postfix operation.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ArithmeticPostfixOperation {
    pub value: Expression,
    pub operator: ArithmeticPostfixOperator,
}

impl HasSpan for ArithmeticOperation {
    fn span(&self) -> Span {
        match self {
            ArithmeticOperation::Prefix(operation) => operation.span(),
            ArithmeticOperation::Infix(operation) => operation.span(),
            ArithmeticOperation::Postfix(operation) => operation.span(),
        }
    }
}

impl HasSpan for ArithmeticPrefixOperator {
    fn span(&self) -> Span {
        match self {
            ArithmeticPrefixOperator::Increment(span) => *span,
            ArithmeticPrefixOperator::Decrement(span) => *span,
            ArithmeticPrefixOperator::Plus(span) => *span,
            ArithmeticPrefixOperator::Minus(span) => *span,
        }
    }
}

impl HasSpan for ArithmeticPrefixOperation {
    fn span(&self) -> Span {
        self.operator.span().join(self.value.span())
    }
}

impl HasSpan for ArithmeticInfixOperator {
    fn span(&self) -> Span {
        match self {
            ArithmeticInfixOperator::Addition(span) => *span,
            ArithmeticInfixOperator::Subtraction(span) => *span,
            ArithmeticInfixOperator::Multiplication(span) => *span,
            ArithmeticInfixOperator::Division(span) => *span,
            ArithmeticInfixOperator::Modulo(span) => *span,
            ArithmeticInfixOperator::Exponentiation(span) => *span,
        }
    }
}

impl HasSpan for ArithmeticInfixOperation {
    fn span(&self) -> Span {
        self.lhs.span().join(self.rhs.span())
    }
}

impl HasSpan for ArithmeticPostfixOperator {
    fn span(&self) -> Span {
        match self {
            ArithmeticPostfixOperator::Increment(span) => *span,
            ArithmeticPostfixOperator::Decrement(span) => *span,
        }
    }
}

impl HasSpan for ArithmeticPostfixOperation {
    fn span(&self) -> Span {
        self.value.span().join(self.operator.span())
    }
}
