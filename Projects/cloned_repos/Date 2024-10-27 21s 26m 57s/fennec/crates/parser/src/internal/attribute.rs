use fennec_ast::ast::*;
use fennec_ast::sequence::Sequence;
use fennec_ast::sequence::TokenSeparatedSequence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::argument;
use crate::internal::identifier;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;

pub fn parse_attribute_list_sequence<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
) -> Result<Sequence<AttributeList>, ParseError> {
    let mut inner = Vec::new();
    loop {
        let next = utils::peek(stream)?;
        if next.kind == T!["#["] {
            inner.push(parse_attribute_list(stream)?);
        } else {
            break;
        }
    }

    Ok(Sequence::new(inner))
}

pub fn parse_attribute_list<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<AttributeList, ParseError> {
    Ok(AttributeList {
        hash_left_bracket: utils::expect_span(stream, T!["#["])?,
        attributes: {
            let mut attributes = Vec::new();
            let mut commas = Vec::new();
            loop {
                let next = utils::peek(stream)?;
                if next.kind == T!["]"] {
                    break;
                }

                attributes.push(parse_attribute(stream)?);

                let next = utils::peek(stream)?;
                if next.kind == T![","] {
                    commas.push(utils::expect_any(stream)?);
                } else {
                    break;
                }
            }

            TokenSeparatedSequence::new(attributes, commas)
        },
        right_bracket: utils::expect_span(stream, T!["]"])?,
    })
}

pub fn parse_attribute<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Attribute, ParseError> {
    Ok(Attribute {
        name: identifier::parse_identifier(stream)?,
        arguments: argument::parse_optional_argument_list(stream)?,
    })
}
