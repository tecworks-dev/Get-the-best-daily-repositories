use fennec_ast::ast::*;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_opening_tag<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<OpeningTag, ParseError> {
    let token = utils::expect_one_of(stream, &[T!["<?php"], T!["<?="], T!["<?"]])?;

    Ok(match token.kind {
        T!["<?php"] => OpeningTag::Full(FullOpeningTag { span: token.span, value: token.value }),
        T!["<?="] => OpeningTag::Echo(EchoOpeningTag { span: token.span }),
        T!["<?"] => OpeningTag::Short(ShortOpeningTag { span: token.span }),
        _ => unreachable!(),
    })
}

pub fn parse_closing_tag<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<ClosingTag, ParseError> {
    let span = utils::expect_span(stream, T!["?>"])?;

    Ok(ClosingTag { span })
}
