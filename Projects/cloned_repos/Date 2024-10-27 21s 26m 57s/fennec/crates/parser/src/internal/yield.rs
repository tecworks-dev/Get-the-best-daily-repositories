use fennec_ast::ast::*;
use fennec_token::Precedence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::expression::parse_expression_with_precedence;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_yield<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Yield, ParseError> {
    let r#yield = utils::expect_keyword(stream, T!["yield"])?;

    Ok(match utils::peek(stream)?.kind {
        T![";" | "?>"] => Yield::Value(YieldValue { r#yield, value: None }),
        T!["from"] => Yield::From(YieldFrom {
            r#yield,
            from: utils::expect_keyword(stream, T!["from"])?,
            iterator: parse_expression_with_precedence(stream, Precedence::YieldFrom)?,
        }),
        _ => {
            let key_or_value = parse_expression_with_precedence(stream, Precedence::Yield)?;

            if matches!(utils::maybe_peek(stream)?.map(|t| t.kind), Some(T!["=>"])) {
                let key = key_or_value;
                let arrow = utils::expect_span(stream, T!["=>"])?;
                let value = parse_expression_with_precedence(stream, Precedence::Yield)?;

                Yield::Pair(YieldPair { r#yield, key, arrow, value })
            } else {
                Yield::Value(YieldValue { r#yield, value: Some(key_or_value) })
            }
        }
    })
}
