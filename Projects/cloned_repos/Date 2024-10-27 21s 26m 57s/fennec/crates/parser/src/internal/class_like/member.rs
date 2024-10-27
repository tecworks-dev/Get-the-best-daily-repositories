use either::Either;

use fennec_ast::ast::*;
use fennec_ast::sequence::Sequence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::attribute::parse_attribute_list_sequence;
use crate::internal::class_like::constant::parse_class_like_constant_with_attributes_and_modifiers;
use crate::internal::class_like::enum_case::parse_enum_case_with_attributes;
use crate::internal::class_like::method::parse_method_with_attributes_and_modifiers;
use crate::internal::class_like::property::parse_property_with_attributes_and_modifiers;
use crate::internal::class_like::trait_use::parse_trait_use;
use crate::internal::expression;
use crate::internal::identifier;
use crate::internal::modifier::parse_modifier_sequence;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;
use crate::internal::variable;

pub fn parse_classlike_memeber<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<ClassLikeMember, ParseError> {
    Ok(match utils::peek(stream)?.kind {
        T!["#["] => {
            let attributes = parse_attribute_list_sequence(stream)?;

            parse_classlike_memeber_with_attributes(stream, attributes)?
        }
        k if k.is_modifier() => {
            let modifiers = parse_modifier_sequence(stream)?;

            parse_classlike_memeber_with_attributes_and_modifiers(stream, Sequence::empty(), modifiers)?
        }
        T!["const"] => ClassLikeMember::Constant(parse_class_like_constant_with_attributes_and_modifiers(
            stream,
            Sequence::empty(),
            Sequence::empty(),
        )?),
        T!["function"] => ClassLikeMember::Method(parse_method_with_attributes_and_modifiers(
            stream,
            Sequence::empty(),
            Sequence::empty(),
        )?),
        T!["case"] => ClassLikeMember::EnumCase(parse_enum_case_with_attributes(stream, Sequence::empty())?),
        T!["use"] => ClassLikeMember::TraitUse(parse_trait_use(stream)?),
        _ => ClassLikeMember::Property(parse_property_with_attributes_and_modifiers(
            stream,
            Sequence::empty(),
            Sequence::empty(),
        )?),
    })
}

pub fn parse_classlike_memeber_with_attributes<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    attributes: Sequence<AttributeList>,
) -> Result<ClassLikeMember, ParseError> {
    Ok(match utils::peek(stream)?.kind {
        k if k.is_modifier() => {
            let modifiers = parse_modifier_sequence(stream)?;

            parse_classlike_memeber_with_attributes_and_modifiers(stream, attributes, modifiers)?
        }
        T!["case"] => ClassLikeMember::EnumCase(parse_enum_case_with_attributes(stream, attributes)?),
        T!["const"] => ClassLikeMember::Constant(parse_class_like_constant_with_attributes_and_modifiers(
            stream,
            attributes,
            Sequence::empty(),
        )?),
        T!["function"] => {
            ClassLikeMember::Method(parse_method_with_attributes_and_modifiers(stream, attributes, Sequence::empty())?)
        }
        _ => ClassLikeMember::Property(parse_property_with_attributes_and_modifiers(
            stream,
            attributes,
            Sequence::empty(),
        )?),
    })
}

pub fn parse_classlike_memeber_with_attributes_and_modifiers<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
    attributes: Sequence<AttributeList>,
    modifiers: Sequence<Modifier>,
) -> Result<ClassLikeMember, ParseError> {
    Ok(match utils::peek(stream)?.kind {
        T!["const"] => ClassLikeMember::Constant(parse_class_like_constant_with_attributes_and_modifiers(
            stream, attributes, modifiers,
        )?),
        T!["function"] => {
            ClassLikeMember::Method(parse_method_with_attributes_and_modifiers(stream, attributes, modifiers)?)
        }
        _ => ClassLikeMember::Property(parse_property_with_attributes_and_modifiers(stream, attributes, modifiers)?),
    })
}

pub fn parse_classlike_memeber_selector<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
) -> Result<ClassLikeMemberSelector, ParseError> {
    let token = utils::peek(stream)?;

    Ok(match token.kind {
        T!["$"] | T!["${"] | T!["$variable"] => ClassLikeMemberSelector::Variable(variable::parse_variable(stream)?),
        T!["{"] => ClassLikeMemberSelector::Expression(ClassLikeMemberExpressionSelector {
            left_brace: utils::expect_span(stream, T!["{"])?,
            expression: Box::new(expression::parse_expression(stream)?),
            right_brace: utils::expect_span(stream, T!["}"])?,
        }),
        kind @ _ if kind.is_identifier_maybe_reserved() => {
            ClassLikeMemberSelector::Identifier(identifier::parse_local_identifier(stream)?)
        }
        _ => return Err(utils::unexpected(stream, Some(token), T!["$variable", "${", "$", "{", Identifier])),
    })
}

pub fn parse_classlike_constant_selector_or_variable<'a, 'i>(
    stream: &mut TokenStream<'a, 'i>,
) -> Result<Either<ClassLikeConstantSelector, Variable>, ParseError> {
    let token = utils::peek(stream)?;

    Ok(match token.kind {
        T!["$"] | T!["${"] | T!["$variable"] => Either::Right(variable::parse_variable(stream)?),
        T!["{"] => Either::Left(ClassLikeConstantSelector::Expression(ClassLikeMemberExpressionSelector {
            left_brace: utils::expect_span(stream, T!["{"])?,
            expression: Box::new(expression::parse_expression(stream)?),
            right_brace: utils::expect_span(stream, T!["}"])?,
        })),
        kind @ _ if kind.is_identifier_maybe_reserved() => {
            Either::Left(ClassLikeConstantSelector::Identifier(identifier::parse_local_identifier(stream)?))
        }
        _ => return Err(utils::unexpected(stream, Some(token), T!["$variable", "${", "$", "{", Identifier])),
    })
}
