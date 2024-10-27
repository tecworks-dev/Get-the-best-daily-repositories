use fennec_ast::ast::*;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::expression;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_variable<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Variable, ParseError> {
    let token = utils::peek(stream)?;

    Ok(match &token.kind {
        T!["$variable"] => Variable::Direct(parse_direct_variable(stream)?),
        T!["${"] => Variable::Indirect(parse_indirect_variable(stream)?),
        T!["$"] => Variable::Nested(parse_nested_variable(stream)?),
        _ => return Err(utils::unexpected(stream, Some(token), T!["$variable", "${", "$"])),
    })
}

pub fn parse_direct_variable<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<DirectVariable, ParseError> {
    let token = utils::expect(stream, T!["$variable"])?;

    Ok(DirectVariable { span: token.span, name: token.value })
}

pub fn parse_indirect_variable<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<IndirectVariable, ParseError> {
    Ok(IndirectVariable {
        dollar_left_brace: utils::expect_span(stream, T!["${"])?,
        expression: Box::new(expression::parse_expression(stream)?),
        right_brace: utils::expect_span(stream, T!["}"])?,
    })
}

pub fn parse_nested_variable<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<NestedVariable, ParseError> {
    Ok(NestedVariable { dollar: utils::expect_span(stream, T!["$"])?, variable: Box::new(parse_variable(stream)?) })
}
