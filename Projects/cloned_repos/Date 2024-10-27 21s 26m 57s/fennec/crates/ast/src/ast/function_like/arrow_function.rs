use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

use crate::ast::attribute::AttributeList;
use crate::ast::expression::Expression;
use crate::ast::function_like::parameter::FunctionLikeParameterList;
use crate::ast::function_like::r#return::FunctionLikeReturnTypeHint;
use crate::ast::keyword::Keyword;
use crate::sequence::Sequence;

/// Represents an arrow function in PHP.
///
/// Example:
///
/// ```php
/// <?php
///
/// $fn = fn($x) => $x * 2;
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ArrowFunction {
    pub attributes: Sequence<AttributeList>,
    pub r#static: Option<Keyword>,
    pub r#fn: Keyword,
    pub ampersand: Option<Span>,
    pub parameters: FunctionLikeParameterList,
    pub return_type_hint: Option<FunctionLikeReturnTypeHint>,
    pub arrow: Span,
    pub expression: Expression,
}

impl HasSpan for ArrowFunction {
    fn span(&self) -> Span {
        if let Some(attribute_list) = self.attributes.first() {
            return attribute_list.span().join(self.expression.span());
        }

        if let Some(r#static) = &self.r#static {
            return r#static.span().join(self.expression.span());
        }

        self.r#fn.span().join(self.expression.span())
    }
}
