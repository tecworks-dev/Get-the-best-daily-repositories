use fennec_ast::ast::*;
use fennec_ast::sequence::TokenSeparatedSequence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::expression::parse_expression;
use crate::internal::terminator::parse_terminator;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_unset<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Unset, ParseError> {
    let unset = utils::expect_keyword(stream, T!["unset"])?;
    let left_parenthesis = utils::expect_span(stream, T!["("])?;

    let mut values = vec![];
    let mut commas = vec![];
    loop {
        if matches!(utils::peek(stream)?.kind, T![")"]) {
            break;
        }

        values.push(parse_expression(stream)?);

        match utils::peek(stream)?.kind {
            T![","] => {
                commas.push(utils::expect_any(stream)?);
            }
            _ => {
                break;
            }
        }
    }

    let right_parenthesis = utils::expect_span(stream, T![")"])?;
    let terminator = parse_terminator(stream)?;

    Ok(Unset {
        unset,
        left_parenthesis,
        values: TokenSeparatedSequence::new(values, commas),
        right_parenthesis,
        terminator,
    })
}
