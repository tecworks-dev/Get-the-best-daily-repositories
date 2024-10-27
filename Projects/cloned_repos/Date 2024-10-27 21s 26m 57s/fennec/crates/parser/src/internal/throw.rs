use fennec_ast::ast::*;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::expression::parse_expression;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_throw<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Throw, ParseError> {
    Ok(Throw { throw: utils::expect_keyword(stream, T!["throw"])?, exception: parse_expression(stream)? })
}
