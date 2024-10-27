use fennec_ast::ast::*;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_identifier<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Identifier, ParseError> {
    let token = utils::peek(stream)?;

    Ok(match &token.kind {
        T![QualifiedIdentifier] => Identifier::Qualified(parse_qualified_identifier(stream)?),
        T![FullyQualifiedIdentifier] => Identifier::FullyQualified(parse_fully_qualified_identifier(stream)?),
        _ => Identifier::Local(parse_local_identifier(stream)?),
    })
}

pub fn parse_local_identifier<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<LocalIdentifier, ParseError> {
    let token = utils::expect_any(stream)?;

    if !token.kind.is_identifier_maybe_reserved() {
        return Err(utils::unexpected(stream, Some(token), &[T![Identifier]]));
    }

    Ok(LocalIdentifier { span: token.span, value: token.value })
}

pub fn parse_qualified_identifier<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<QualifiedIdentifier, ParseError> {
    let token = utils::expect(stream, T![QualifiedIdentifier])?;

    Ok(QualifiedIdentifier { span: token.span, value: token.value })
}

pub fn parse_fully_qualified_identifier<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
) -> Result<FullyQualifiedIdentifier, ParseError> {
    let token = utils::expect(stream, T![FullyQualifiedIdentifier])?;

    Ok(FullyQualifiedIdentifier { span: token.span, value: token.value })
}
