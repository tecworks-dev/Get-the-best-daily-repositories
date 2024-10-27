use serde::Deserialize;
use serde::Serialize;
use strum::Display;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;
use crate::ast::keyword::Keyword;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum LogicalOperation {
    Prefix(LogicalPrefixOperation),
    Infix(LogicalInfixOperation),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum LogicalPrefixOperator {
    Not(Span),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct LogicalPrefixOperation {
    pub operator: LogicalPrefixOperator,
    pub value: Expression,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum LogicalInfixOperator {
    And(Span),
    Or(Span),
    LowPrecedenceAnd(Keyword),
    LowPrecedenceOr(Keyword),
    LowPrecedenceXor(Keyword),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct LogicalInfixOperation {
    pub lhs: Expression,
    pub operator: LogicalInfixOperator,
    pub rhs: Expression,
}

impl HasSpan for LogicalOperation {
    fn span(&self) -> Span {
        match self {
            LogicalOperation::Prefix(operation) => operation.span(),
            LogicalOperation::Infix(operation) => operation.span(),
        }
    }
}

impl HasSpan for LogicalPrefixOperator {
    fn span(&self) -> Span {
        match self {
            LogicalPrefixOperator::Not(span) => *span,
        }
    }
}

impl HasSpan for LogicalPrefixOperation {
    fn span(&self) -> Span {
        self.operator.span().join(self.value.span())
    }
}

impl HasSpan for LogicalInfixOperator {
    fn span(&self) -> Span {
        match self {
            LogicalInfixOperator::And(span) => *span,
            LogicalInfixOperator::Or(span) => *span,
            LogicalInfixOperator::LowPrecedenceAnd(keyword) => keyword.span(),
            LogicalInfixOperator::LowPrecedenceOr(keyword) => keyword.span(),
            LogicalInfixOperator::LowPrecedenceXor(keyword) => keyword.span(),
        }
    }
}

impl HasSpan for LogicalInfixOperation {
    fn span(&self) -> Span {
        self.lhs.span().join(self.rhs.span())
    }
}
