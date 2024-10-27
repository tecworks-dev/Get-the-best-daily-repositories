use fennec_ast::ast::*;
use fennec_token::Precedence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::expression::parse_expression_with_precedence;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_prefix_operation<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
) -> Result<ArithmeticPrefixOperation, ParseError> {
    let token = utils::expect_any(stream)?;
    let operator = match token.kind {
        T!["-"] => ArithmeticPrefixOperator::Minus(token.span),
        T!["+"] => ArithmeticPrefixOperator::Plus(token.span),
        T!["--"] => ArithmeticPrefixOperator::Decrement(token.span),
        T!["++"] => ArithmeticPrefixOperator::Increment(token.span),
        _ => {
            return Err(utils::unexpected(stream, Some(token), T!["-", "+", "--", "++"]));
        }
    };

    Ok(ArithmeticPrefixOperation { operator, value: parse_expression_with_precedence(stream, Precedence::Prefix)? })
}
