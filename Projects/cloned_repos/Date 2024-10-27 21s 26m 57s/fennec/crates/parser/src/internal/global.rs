use fennec_ast::ast::*;
use fennec_ast::sequence::TokenSeparatedSequence;
use fennec_token::T;

use crate::error::ParseError;
use crate::internal::terminator::parse_terminator;
use crate::internal::token_stream::TokenStream;
use crate::internal::utils;
use crate::internal::variable::parse_variable;

pub fn parse_global<'a, 'i>(stream: &mut TokenStream<'a, 'i>) -> Result<Global, ParseError> {
    Ok(Global {
        global: utils::expect_keyword(stream, T!["global"])?,
        variables: {
            let mut variables = vec![];
            let mut commas = vec![];

            loop {
                if matches!(utils::peek(stream)?.kind, T!["?>" | ";"]) {
                    break;
                }

                variables.push(parse_variable(stream)?);

                match utils::peek(stream)?.kind {
                    T![","] => {
                        commas.push(utils::expect_any(stream)?);
                    }
                    _ => {
                        break;
                    }
                }
            }

            TokenSeparatedSequence::new(variables, commas)
        },
        terminator: parse_terminator(stream)?,
    })
}
