use fennec_ast::ast::*;
use fennec_token::Precedence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::expression::parse_expression_with_precedence;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_cast<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<CastOperation, ParseError> {
    let token = utils::expect_any(stream)?;

    let operator = match token.kind {
        T!["(array)"] => CastOperator::Array(token.span, token.value),
        T!["(bool)"] => CastOperator::Bool(token.span, token.value),
        T!["(boolean)"] => CastOperator::Boolean(token.span, token.value),
        T!["(double)"] => CastOperator::Double(token.span, token.value),
        T!["(real)"] => CastOperator::Real(token.span, token.value),
        T!["(float)"] => CastOperator::Float(token.span, token.value),
        T!["(int)"] => CastOperator::Int(token.span, token.value),
        T!["(integer)"] => CastOperator::Integer(token.span, token.value),
        T!["(object)"] => CastOperator::Object(token.span, token.value),
        T!["(unset)"] => CastOperator::Unset(token.span, token.value),
        T!["(binary)"] => CastOperator::Binary(token.span, token.value),
        T!["(string)"] => CastOperator::String(token.span, token.value),
        _ => {
            return Err(utils::unexpected(
                stream,
                Some(token),
                T![
                    "(array)",
                    "(bool)",
                    "(boolean)",
                    "(double)",
                    "(real)",
                    "(float)",
                    "(int)",
                    "(integer)",
                    "(object)",
                    "(unset)",
                    "(binary)",
                    "(string)",
                ],
            ));
        }
    };

    Ok(CastOperation { operator, value: parse_expression_with_precedence(stream, Precedence::Prefix)? })
}
