use fennec_reporting::Annotation;
use fennec_reporting::Issue;
use serde::Deserialize;
use serde::Serialize;

use fennec_ast::ast::*;
use fennec_lexer::error::SyntaxError;
use fennec_span::HasSpan;
use fennec_span::Position;
use fennec_span::Span;
use fennec_token::TokenKind;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ParseError {
    SyntaxError(SyntaxError),
    UnexpectedEndOfFile(Vec<TokenKind>, Position),
    UnexpectedToken(Vec<TokenKind>, TokenKind, Span),
    UnclosedLiteralString(LiteralStringKind, Span),
}

impl HasSpan for ParseError {
    fn span(&self) -> Span {
        match &self {
            ParseError::SyntaxError(syntax_error) => syntax_error.span(),
            ParseError::UnexpectedEndOfFile(_, position) => Span::new(*position, *position),
            ParseError::UnexpectedToken(_, _, span) => *span,
            ParseError::UnclosedLiteralString(_, span) => *span,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            ParseError::SyntaxError(e) => {
                return write!(f, "{}", e);
            }
            ParseError::UnexpectedEndOfFile(expected, _) => {
                let expected = expected.iter().map(|kind| kind.to_string()).collect::<Vec<_>>().join("`, `");

                if expected.is_empty() {
                    format!("unexpected end of file")
                } else if expected.len() == 1 {
                    format!("expected `{}` before end of file", expected)
                } else {
                    format!("expected one of `{}` before end of file", expected)
                }
            }
            ParseError::UnexpectedToken(expected, found, _) => {
                let expected = expected.iter().map(|kind| kind.to_string()).collect::<Vec<_>>().join("`, `");

                let found = found.to_string();

                if expected.is_empty() {
                    format!("unexpected token `{}`", found)
                } else if expected.len() == 1 {
                    format!("expected `{}`, found `{}`", expected, found)
                } else {
                    format!("expected one of `{}`, found `{}`", expected, found)
                }
            }
            ParseError::UnclosedLiteralString(kind, _) => match kind {
                LiteralStringKind::SingleQuoted => "unclosed single-quoted string".to_string(),
                LiteralStringKind::DoubleQuoted => "unclosed double-quoted string".to_string(),
            },
        };

        write!(f, "parse error: {}", message)
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParseError::SyntaxError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SyntaxError> for ParseError {
    fn from(error: SyntaxError) -> Self {
        ParseError::SyntaxError(error)
    }
}

impl Into<Issue> for &ParseError {
    fn into(self) -> Issue {
        Issue::error(self.to_string()).with_annotation(Annotation::primary(self.span()))
    }
}
