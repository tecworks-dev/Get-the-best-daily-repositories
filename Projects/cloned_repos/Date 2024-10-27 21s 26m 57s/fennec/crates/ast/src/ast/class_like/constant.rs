use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::attribute::AttributeList;
use crate::ast::expression::Expression;
use crate::ast::identifier::LocalIdentifier;
use crate::ast::keyword::Keyword;
use crate::ast::modifier::Modifier;
use crate::ast::terminator::Terminator;
use crate::ast::type_hint::Hint;
use crate::sequence::Sequence;
use crate::sequence::TokenSeparatedSequence;

/// Represents a class-like constant in PHP.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ClassLikeConstant {
    pub attributes: Sequence<AttributeList>,
    pub modifiers: Sequence<Modifier>,
    pub r#const: Keyword,
    pub hint: Option<Hint>,
    pub items: TokenSeparatedSequence<ClassLikeConstantItem>,
    pub terminator: Terminator,
}

/// Represents a single name-value pair within a constant statement.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ClassLikeConstantItem {
    pub name: LocalIdentifier,
    pub equals: Span,
    pub value: Expression,
}

impl ClassLikeConstant {
    pub fn first_item(&self) -> &ClassLikeConstantItem {
        self.items
            .first()
            .expect("expected class-like constant to have at least 1 item. this is a bug in fennec. please report it.")
    }
}

impl HasSpan for ClassLikeConstant {
    fn span(&self) -> Span {
        self.r#const.span().join(self.terminator.span())
    }
}

impl HasSpan for ClassLikeConstantItem {
    fn span(&self) -> Span {
        self.name.span().join(self.value.span())
    }
}
