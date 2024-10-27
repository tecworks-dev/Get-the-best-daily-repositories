use fennec_ast::ast::*;
use fennec_ast::sequence::Sequence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::expression::parse_expression;
use crate::internal::function_like::parameter::parse_function_like_parameter_list;
use crate::internal::function_like::r#return::parse_optional_function_like_return_type_hint;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_arrow_function_with_attributes<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    attributes: Sequence<AttributeList>,
) -> Result<ArrowFunction, ParseError> {
    Ok(ArrowFunction {
        attributes,
        r#static: utils::maybe_expect_keyword(stream, T!["static"])?,
        r#fn: utils::expect_keyword(stream, T!["fn"])?,
        ampersand: utils::maybe_expect(stream, T!["&"])?.map(|t| t.span),
        parameters: parse_function_like_parameter_list(stream)?,
        return_type_hint: parse_optional_function_like_return_type_hint(stream)?,
        arrow: utils::expect_span(stream, T!["=>"])?,
        expression: parse_expression(stream)?,
    })
}
