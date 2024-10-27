use fennec_span::Span;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum Token<'a> {
    Line { content: &'a str, span: Span },
    EmptyLine { span: Span },
}

impl<'a> Token<'a> {
    pub fn span(&self) -> Span {
        match self {
            Token::Line { span, .. } => *span,
            Token::EmptyLine { span } => *span,
        }
    }
}
