use fennec_ast::ast::*;
use fennec_ast::sequence::Sequence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::identifier::parse_identifier;
use crate::internal::statement::parse_statement;
use crate::internal::terminator::parse_terminator;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

use super::block::parse_block;

pub fn parse_namespace<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Namespace, ParseError> {
    let namespace = utils::expect_keyword(stream, T!["namespace"])?;
    let name = match utils::peek(stream)?.kind {
        T![";" | "?>" | "{"] => None,
        _ => Some(parse_identifier(stream)?),
    };
    let body = parse_namespace_body(stream)?;

    Ok(Namespace { namespace, name, body })
}

pub fn parse_namespace_body<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<NamespaceBody, ParseError> {
    let next = utils::peek(stream)?;
    match next.kind {
        T!["{"] => Ok(NamespaceBody::BraceDelimited(parse_block(stream)?)),
        _ => Ok(NamespaceBody::Implicit(parse_namespace_implicit_body(stream)?)),
    }
}

pub fn parse_namespace_implicit_body<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
) -> Result<NamespaceImplicitBody, ParseError> {
    let terminator = parse_terminator(stream)?;
    let mut statements = Vec::new();
    loop {
        let next = utils::maybe_peek(stream)?.map(|t| t.kind);
        if matches!(next, None | Some(T!["namespace"])) {
            break;
        }

        statements.push(parse_statement(stream)?);
    }

    Ok(NamespaceImplicitBody { terminator, statements: Sequence::new(statements) })
}
