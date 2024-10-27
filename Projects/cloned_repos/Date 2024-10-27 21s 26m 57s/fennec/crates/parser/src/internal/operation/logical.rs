use fennec_ast::ast::*;
use fennec_token::Precedence;

use crate::error::ParseError;
use crate::internal::expression::parse_expression_with_precedence;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_logical_not<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<LogicalOperation, ParseError> {
    Ok(LogicalOperation::Prefix(LogicalPrefixOperation {
        operator: LogicalPrefixOperator::Not(utils::expect_any(stream)?.span),
        value: parse_expression_with_precedence(stream, Precedence::Bang)?,
    }))
}
