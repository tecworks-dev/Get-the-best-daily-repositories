use serde::Deserialize;
use serde::Serialize;

use fennec_reporting::Annotation;
use fennec_reporting::Issue;
use fennec_span::HasPosition;
use fennec_span::HasSpan;
use fennec_span::Position;
use fennec_span::Span;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum SyntaxError {
    UnexpectedToken(u8, Position),
    UnrecognizedToken(u8, Position),
}

impl HasSpan for SyntaxError {
    fn span(&self) -> Span {
        let position = match self {
            Self::UnexpectedToken(_, p) => *p,
            Self::UnrecognizedToken(_, p) => *p,
        };

        Span::new(position, Position { offset: position.offset + 1, ..position })
    }
}

impl std::fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnexpectedToken(token, _) => &format!("unexpected token `{}` (0x{:02X})", *token as char, token),
            Self::UnrecognizedToken(token, _) => &format!("unrecognised token `{}` (0x{:02X})", *token as char, token),
        };

        write!(f, "syntax error: {}", message)
    }
}

impl std::error::Error for SyntaxError {}

impl Into<Issue> for SyntaxError {
    fn into(self) -> Issue {
        let position = self.position();
        let span = Span::new(position, Position { offset: position.offset + 1, ..position });

        Issue::error(self.to_string()).with_annotation(Annotation::primary(span))
    }
}
