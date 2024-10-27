use fennec_ast::ast::*;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::expression::parse_expression;
use crate::internal::statement::parse_statement;
use crate::internal::terminator::parse_terminator;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_do_while<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<DoWhile, ParseError> {
    Ok(DoWhile {
        r#do: utils::expect_keyword(stream, T!["do"])?,
        statement: parse_statement(stream)?,
        r#while: utils::expect_keyword(stream, T!["while"])?,
        left_parenthesis: utils::expect_span(stream, T!["("])?,
        condition: parse_expression(stream)?,
        right_parenthesis: utils::expect_span(stream, T![")"])?,
        terminator: parse_terminator(stream)?,
    })
}
