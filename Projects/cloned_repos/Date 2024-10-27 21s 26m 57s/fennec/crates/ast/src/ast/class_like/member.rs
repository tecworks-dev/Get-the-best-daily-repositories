use serde::Deserialize;
use serde::Serialize;
use strum::Display;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::class_like::constant::ClassLikeConstant;
use crate::ast::class_like::enum_case::EnumCase;
use crate::ast::class_like::method::Method;
use crate::ast::class_like::property::Property;
use crate::ast::class_like::trait_use::TraitUse;
use crate::ast::expression::Expression;
use crate::ast::identifier::LocalIdentifier;
use crate::ast::variable::Variable;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
pub enum ClassLikeMember {
    TraitUse(TraitUse),
    Constant(ClassLikeConstant),
    Property(Property),
    EnumCase(EnumCase),
    Method(Method),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum ClassLikeMemberSelector {
    Identifier(LocalIdentifier),
    Variable(Variable),
    Expression(ClassLikeMemberExpressionSelector),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord, Display)]
#[serde(tag = "type", content = "value")]
pub enum ClassLikeConstantSelector {
    Identifier(LocalIdentifier),
    Expression(ClassLikeMemberExpressionSelector),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ClassLikeMemberExpressionSelector {
    pub left_brace: Span,
    pub expression: Box<Expression>,
    pub right_brace: Span,
}

impl HasSpan for ClassLikeMember {
    fn span(&self) -> Span {
        match self {
            ClassLikeMember::TraitUse(trait_use) => trait_use.span(),
            ClassLikeMember::Constant(constant) => constant.span(),
            ClassLikeMember::Property(property) => property.span(),
            ClassLikeMember::EnumCase(enum_case) => enum_case.span(),
            ClassLikeMember::Method(method) => method.span(),
        }
    }
}

impl HasSpan for ClassLikeMemberSelector {
    fn span(&self) -> Span {
        match self {
            ClassLikeMemberSelector::Identifier(i) => i.span(),
            ClassLikeMemberSelector::Variable(v) => v.span(),
            ClassLikeMemberSelector::Expression(e) => e.span(),
        }
    }
}

impl HasSpan for ClassLikeConstantSelector {
    fn span(&self) -> Span {
        match self {
            ClassLikeConstantSelector::Identifier(i) => i.span(),
            ClassLikeConstantSelector::Expression(e) => e.span(),
        }
    }
}

impl HasSpan for ClassLikeMemberExpressionSelector {
    fn span(&self) -> Span {
        self.left_brace.join(self.right_brace)
    }
}
