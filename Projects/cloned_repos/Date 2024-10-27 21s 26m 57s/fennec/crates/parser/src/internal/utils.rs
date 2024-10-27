use fennec_ast::ast::*;
use fennec_span::Span;
use fennec_token::Token;
use fennec_token::TokenKind;

use crate::error::ParseError;
use crate::internal::token_stream::TokenStream;

pub fn peek<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Token, ParseError> {
    match stream.peek() {
        Some(Ok(token)) => Ok(token),
        Some(Err(error)) => Err(error.into()),
        None => Err(unexpected(stream, None, &[])),
    }
}

pub fn maybe_peek<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Option<Token>, ParseError> {
    match stream.peek() {
        Some(Ok(token)) => Ok(Some(token)),
        Some(Err(error)) => Err(error.into()),
        None => Ok(None),
    }
}

pub fn peek_nth<'a, 'i>(stream: &mut TokenStream<'a, 'i>, n: usize) -> Result<Token, ParseError> {
    match stream.peek_nth(n) {
        Some(Ok(token)) => Ok(token),
        Some(Err(error)) => Err(error.into()),
        None => Err(unexpected(stream, None, &[])),
    }
}

pub fn maybe_peek_nth<'a, 'i>(stream: &mut TokenStream<'a, 'i>, n: usize) -> Result<Option<Token>, ParseError> {
    match stream.peek_nth(n) {
        Some(Ok(token)) => Ok(Some(token)),
        Some(Err(error)) => Err(error.into()),
        None => Ok(None),
    }
}

pub fn expect_any<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Token, ParseError> {
    match stream.advance() {
        Some(Ok(token)) => Ok(token),
        Some(Err(error)) => Err(error.into()),
        None => Err(unexpected(stream, None, &[])),
    }
}

pub fn expect<'a, 'i>(stream: &mut TokenStream<'a, 'i>, kind: TokenKind) -> Result<Token, ParseError> {
    let token = expect_any(stream)?;

    if kind == token.kind {
        Ok(token)
    } else {
        Err(unexpected(stream, Some(token), &[kind]))
    }
}

pub fn expect_one_of<'a, 'i>(stream: &mut TokenStream<'a, 'i>, one_of: &[TokenKind]) -> Result<Token, ParseError> {
    let token = expect_any(stream)?;

    if one_of.contains(&token.kind) {
        Ok(token)
    } else {
        Err(unexpected(stream, Some(token), one_of))
    }
}

pub fn maybe_expect<'a, 'i>(stream: &mut TokenStream<'a, 'i>, kind: TokenKind) -> Result<Option<Token>, ParseError> {
    let next = match stream.peek() {
        Some(Ok(token)) => token,
        Some(Err(error)) => return Err(error.into()),
        None => return Ok(None),
    };

    if kind == next.kind {
        let token = match stream.advance() {
            Some(result) => result?,
            None => unreachable!("the token was peeked, so it should be available"),
        };

        Ok(Some(token))
    } else {
        Ok(None)
    }
}

pub fn expect_span<'a, 'i>(stream: &mut TokenStream<'a, 'i>, kind: TokenKind) -> Result<Span, ParseError> {
    expect(stream, kind).map(|token| token.span)
}

pub fn expect_one_of_keyword<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    one_of: &[TokenKind],
) -> Result<Keyword, ParseError> {
    expect_one_of(stream, one_of).map(|token| to_keyword(token))
}

pub fn maybe_expect_keyword<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    kind: TokenKind,
) -> Result<Option<Keyword>, ParseError> {
    maybe_expect(stream, kind).map(|maybe_token| maybe_token.map(|token| to_keyword(token)))
}

pub fn expect_keyword<'a, 'i>(stream: &mut TokenStream<'a, 'i>, kind: TokenKind) -> Result<Keyword, ParseError> {
    expect(stream, kind).map(|token| to_keyword(token))
}

pub fn expect_any_keyword<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Keyword, ParseError> {
    expect_any(stream).map(|token| to_keyword(token))
}

pub fn to_keyword<'a, 'i>(token: Token) -> Keyword {
    Keyword { span: token.span, value: token.value }
}

pub fn unexpected<'a, 'i>(stream: &mut TokenStream<'a, 'i>, token: Option<Token>, one_of: &[TokenKind]) -> ParseError {
    if let Some(token) = token {
        ParseError::UnexpectedToken(one_of.to_vec(), token.kind, token.span)
    } else {
        ParseError::UnexpectedEndOfFile(one_of.to_vec(), stream.get_position())
    }
}
