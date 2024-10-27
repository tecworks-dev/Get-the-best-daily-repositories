use fennec_ast::ast::*;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_inline<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Inline, ParseError> {
    let token = utils::expect_one_of(stream, T![InlineText, InlineShebang])?;

    Ok(Inline {
        kind: if token.kind == T![InlineShebang] { InlineKind::Shebang } else { InlineKind::Text },
        span: token.span,
        value: token.value,
    })
}
