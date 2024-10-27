use serde::Deserialize;
use serde::Serialize;
use strum::Display;

use fennec_interner::StringIdentifier;
use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::expression::Expression;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum CastOperator {
    Array(Span, StringIdentifier),
    Bool(Span, StringIdentifier),
    Boolean(Span, StringIdentifier),
    Double(Span, StringIdentifier),
    Real(Span, StringIdentifier),
    Float(Span, StringIdentifier),
    Int(Span, StringIdentifier),
    Integer(Span, StringIdentifier),
    Object(Span, StringIdentifier),
    Unset(Span, StringIdentifier),
    String(Span, StringIdentifier),
    Binary(Span, StringIdentifier),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct CastOperation {
    pub operator: CastOperator,
    pub value: Expression,
}

impl HasSpan for CastOperator {
    fn span(&self) -> Span {
        match self {
            CastOperator::Array(span, ..) => *span,
            CastOperator::Bool(span, ..) => *span,
            CastOperator::Boolean(span, ..) => *span,
            CastOperator::Double(span, ..) => *span,
            CastOperator::Real(span, ..) => *span,
            CastOperator::Float(span, ..) => *span,
            CastOperator::Int(span, ..) => *span,
            CastOperator::Integer(span, ..) => *span,
            CastOperator::Object(span, ..) => *span,
            CastOperator::Unset(span, ..) => *span,
            CastOperator::String(span, ..) => *span,
            CastOperator::Binary(span, ..) => *span,
        }
    }
}

impl HasSpan for CastOperation {
    fn span(&self) -> Span {
        self.operator.span().join(self.value.span())
    }
}
