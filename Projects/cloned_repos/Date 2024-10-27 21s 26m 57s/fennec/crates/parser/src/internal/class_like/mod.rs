use fennec_ast::ast::*;
use fennec_ast::sequence::Sequence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::argument::parse_optional_argument_list;
use crate::internal::attribute::parse_attribute_list_sequence;
use crate::internal::class_like::inheritance::parse_optional_extends;
use crate::internal::class_like::inheritance::parse_optional_implements;
use crate::internal::class_like::member::parse_classlike_memeber;
use crate::internal::identifier::parse_local_identifier;
use crate::internal::modifier::parse_modifier_sequence;
use crate::internal::token_stream::TokenStream;
use crate::internal::type_hint::parse_type_hint;
use crate::internal::utils;

pub mod constant;
pub mod enum_case;
pub mod inheritance;
pub mod member;
pub mod method;
pub mod property;
pub mod trait_use;

pub fn parse_interface_with_attributes<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    attributes: Sequence<AttributeList>,
) -> Result<Interface, ParseError> {
    Ok(Interface {
        attributes,
        interface: utils::expect_keyword(stream, T!["interface"])?,
        name: parse_local_identifier(stream)?,
        extends: parse_optional_extends(stream)?,
        left_brace: utils::expect_span(stream, T!["{"])?,
        members: {
            let mut members = Vec::new();
            loop {
                if matches!(utils::maybe_peek(stream)?.map(|t| t.kind), Some(T!["}"])) {
                    break;
                }

                members.push(parse_classlike_memeber(stream)?);
            }

            Sequence::new(members)
        },
        right_brace: utils::expect_span(stream, T!["}"])?,
    })
}

pub fn parse_class_with_attributes<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    attributes: Sequence<AttributeList>,
) -> Result<Class, ParseError> {
    let modifiers = parse_modifier_sequence(stream)?;

    parse_class_with_attributes_and_modifiers(stream, attributes, modifiers)
}

pub fn parse_class_with_attributes_and_modifiers<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    attributes: Sequence<AttributeList>,
    modifiers: Sequence<Modifier>,
) -> Result<Class, ParseError> {
    Ok(Class {
        attributes,
        modifiers,
        class: utils::expect_keyword(stream, T!["class"])?,
        name: parse_local_identifier(stream)?,
        extends: parse_optional_extends(stream)?,
        implements: parse_optional_implements(stream)?,
        left_brace: utils::expect_span(stream, T!["{"])?,
        members: {
            let mut members = Vec::new();
            loop {
                if matches!(utils::maybe_peek(stream)?.map(|t| t.kind), Some(T!["}"])) {
                    break;
                }

                members.push(parse_classlike_memeber(stream)?);
            }

            Sequence::new(members)
        },
        right_brace: utils::expect_span(stream, T!["}"])?,
    })
}

pub fn parse_anonymous_class<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<AnonymousClass, ParseError> {
    Ok(AnonymousClass {
        new: utils::expect_keyword(stream, T!["new"])?,
        attributes: parse_attribute_list_sequence(stream)?,
        modifiers: parse_modifier_sequence(stream)?,
        class: utils::expect_keyword(stream, T!["class"])?,
        arguments: parse_optional_argument_list(stream)?,
        extends: parse_optional_extends(stream)?,
        implements: parse_optional_implements(stream)?,
        left_brace: utils::expect_span(stream, T!["{"])?,
        members: {
            let mut members = Vec::new();
            loop {
                if matches!(utils::maybe_peek(stream)?.map(|t| t.kind), Some(T!["}"])) {
                    break;
                }

                members.push(parse_classlike_memeber(stream)?);
            }

            Sequence::new(members)
        },
        right_brace: utils::expect_span(stream, T!["}"])?,
    })
}

pub fn parse_trait_with_attributes<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    attributes: Sequence<AttributeList>,
) -> Result<Trait, ParseError> {
    Ok(Trait {
        attributes,
        r#trait: utils::expect_keyword(stream, T!["trait"])?,
        name: parse_local_identifier(stream)?,
        left_brace: utils::expect_span(stream, T!["{"])?,
        members: {
            let mut members = Vec::new();
            loop {
                if matches!(utils::maybe_peek(stream)?.map(|t| t.kind), Some(T!["}"])) {
                    break;
                }

                members.push(parse_classlike_memeber(stream)?);
            }
            Sequence::new(members)
        },
        right_brace: utils::expect_span(stream, T!["}"])?,
    })
}

pub fn parse_enum_with_attributes<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    attributes: Sequence<AttributeList>,
) -> Result<Enum, ParseError> {
    Ok(Enum {
        attributes,
        r#enum: utils::expect_keyword(stream, T!["enum"])?,
        name: parse_local_identifier(stream)?,
        backing_type_hint: parse_optional_enum_backing_type_hint(stream)?,
        implements: parse_optional_implements(stream)?,
        left_brace: utils::expect_span(stream, T!["{"])?,
        members: {
            let mut members = Vec::new();
            loop {
                if matches!(utils::maybe_peek(stream)?.map(|t| t.kind), Some(T!["}"])) {
                    break;
                }

                members.push(parse_classlike_memeber(stream)?);
            }
            Sequence::new(members)
        },
        right_brace: utils::expect_span(stream, T!["}"])?,
    })
}

pub fn parse_optional_enum_backing_type_hint<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
) -> Result<Option<EnumBackingTypeHint>, ParseError> {
    Ok(match utils::maybe_peek(stream)?.map(|t| t.kind) {
        Some(T![":"]) => {
            Some(EnumBackingTypeHint { colon: utils::expect_any(stream)?.span, hint: parse_type_hint(stream)? })
        }
        _ => None,
    })
}
