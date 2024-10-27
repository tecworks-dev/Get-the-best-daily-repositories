use serde::Deserialize;
use serde::Serialize;

use fennec_span::HasSpan;
use fennec_span::Span;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ParseError {
    InvalidTrivia(Span),
    UnclosedInlineTag(Span),
    UnclosedInlineCode(Span),
    UnclosedCodeBlock(Span),
    InvalidTagName(Span),
    InvalidAnnotationName(Span),
    UnclosedAnnotationArguments(Span),
    MalformedCodeBlock(Span),
    InvalidComment(Span),
    InconsistentIndentation(Span, usize, usize),
    MissingAsterisk(Span),
    MissingWhitespaceAfterAsterisk(Span),
    MissingWhitespaceAfterOpeningAsterisk(Span),
    MissingWhitespaceBeforeClosingAsterisk(Span),
    ExpectedLine(Span),
}

impl HasSpan for ParseError {
    fn span(&self) -> Span {
        match self {
            ParseError::InvalidTrivia(span) => *span,
            ParseError::UnclosedInlineTag(span) => *span,
            ParseError::UnclosedInlineCode(span) => *span,
            ParseError::UnclosedCodeBlock(span) => *span,
            ParseError::InvalidTagName(span) => *span,
            ParseError::InvalidAnnotationName(span) => *span,
            ParseError::UnclosedAnnotationArguments(span) => *span,
            ParseError::MalformedCodeBlock(span) => *span,
            ParseError::InvalidComment(span) => *span,
            ParseError::InconsistentIndentation(span, _, _) => *span,
            ParseError::MissingAsterisk(span) => *span,
            ParseError::MissingWhitespaceAfterAsterisk(span) => *span,
            ParseError::MissingWhitespaceAfterOpeningAsterisk(span) => *span,
            ParseError::MissingWhitespaceBeforeClosingAsterisk(span) => *span,
            ParseError::ExpectedLine(span) => *span,
        }
    }
}

impl std::error::Error for ParseError {}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InvalidTrivia(_) => {
                write!(f, "invalid trivia")
            }
            ParseError::UnclosedInlineTag(_) => {
                write!(f, "unclosed inline tag")
            }
            ParseError::UnclosedInlineCode(_) => {
                write!(f, "unclosed inline code")
            }
            ParseError::UnclosedCodeBlock(_) => {
                write!(f, "unclosed code block")
            }
            ParseError::InvalidTagName(_) => {
                write!(f, "invalid tag name")
            }
            ParseError::InvalidAnnotationName(_) => {
                write!(f, "invalid annotation name")
            }
            ParseError::UnclosedAnnotationArguments(_) => {
                write!(f, "unclosed annotation arguments")
            }
            ParseError::MalformedCodeBlock(_) => {
                write!(f, "malformed code block")
            }
            ParseError::InvalidComment(_) => {
                write!(f, "invalid comment")
            }
            ParseError::InconsistentIndentation(_, _, _) => {
                write!(f, "inconsistent indentation")
            }
            ParseError::MissingAsterisk(_) => {
                write!(f, "missing `*` in a docblock comment")
            }
            ParseError::MissingWhitespaceAfterAsterisk(_) => {
                write!(f, "missing whitespace after `*`")
            }
            ParseError::MissingWhitespaceAfterOpeningAsterisk(_) => {
                write!(f, "missing whitespace after `/*` in a single-line docblock")
            }
            ParseError::MissingWhitespaceBeforeClosingAsterisk(_) => {
                write!(f, "missing whitespace before `*/` in a single-line docblock")
            }
            ParseError::ExpectedLine(_) => {
                write!(f, "missing expected line")
            }
        }
    }
}

impl ParseError {
    pub fn note(&self) -> &'static str {
        match self {
            ParseError::InvalidTrivia(_) => "the comment is not recognized as a docblock. It should start with '/**' and end with '*/'.",
            ParseError::UnclosedInlineTag(_) => "the inline tag is missing a closing '}'.",
            ParseError::UnclosedInlineCode(_) => "inline code is missing a closing backtick '`'.",
            ParseError::UnclosedCodeBlock(_) => "the code block is missing a closing delimiter '```'.",
            ParseError::InvalidTagName(_) => "the tag name contains invalid characters.",
            ParseError::InvalidAnnotationName(_) => "the annotation name is invalid. It must start with an uppercase letter, '_', or '\\', and contain only allowed characters.",
            ParseError::UnclosedAnnotationArguments(_) => "the annotation arguments are missing a closing parenthesis ')'.",
            ParseError::MalformedCodeBlock(_) => "the code block is malformed or incorrectly formatted.",
            ParseError::InvalidComment(_) => "the comment is not a valid docblock. It should start with '/**' and end with '*/'.",
            ParseError::InconsistentIndentation(_, _, _) => "the indentation in the docblock comment is inconsistent.",
            ParseError::MissingAsterisk(_) => "an asterisk '*' is missing at the beginning of a line in the docblock.",
            ParseError::MissingWhitespaceAfterAsterisk(_) => "missing whitespace after the asterisk '*' in the docblock.",
            ParseError::MissingWhitespaceAfterOpeningAsterisk(_) => "missing whitespace after the opening '/**' in a single-line docblock.",
            ParseError::MissingWhitespaceBeforeClosingAsterisk(_) => "missing whitespace before the closing '*/' in a single-line docblock.",
            ParseError::ExpectedLine(_) => "a line or tag was expected in the docblock but none was found.",
        }
    }

    pub fn help(&self) -> &'static str {
        match self {
            ParseError::InvalidTrivia(_) => "replace the comment with a proper docblock starting with '/**' and ending with '*/'.",
            ParseError::UnclosedInlineTag(_) => "add a closing '}' to complete the inline tag.",
            ParseError::UnclosedInlineCode(_) => "add a closing '`' to terminate the inline code.",
            ParseError::UnclosedCodeBlock(_) => "add a closing '```' to terminate the code block.",
            ParseError::InvalidTagName(_) => "use only letters, numbers, and hyphens in the tag name.",
            ParseError::InvalidAnnotationName(_) => "correct the annotation name to start with an uppercase letter, '_', or '\\', and use only letters, numbers, '_', '\\', or unicode characters.",
            ParseError::UnclosedAnnotationArguments(_) => "add a closing ')' to complete the annotation arguments.",
            ParseError::MalformedCodeBlock(_) => "ensure the code block starts with '```', optionally followed by a language identifier, and ends with a closing '```'.",
            ParseError::InvalidComment(_) => "replace the comment with a valid docblock starting with '/**' and ending with '*/'.",
            ParseError::InconsistentIndentation(_, _, _) => "adjust the indentation to be consistent across all lines in the docblock.",
            ParseError::MissingAsterisk(_) => "add an '*' at the beginning of each line in the docblock after the opening '/**'.",
            ParseError::MissingWhitespaceAfterAsterisk(_) => "insert a space after the '*' at the beginning of the line.",
            ParseError::MissingWhitespaceAfterOpeningAsterisk(_) => "insert a space between '/**' and the text in the single-line docblock.",
            ParseError::MissingWhitespaceBeforeClosingAsterisk(_) => "insert a space between the text and '*/' in the single-line docblock.",
            ParseError::ExpectedLine(_) => "ensure that the docblock contains at least one line of text or a tag.",
        }
    }
}
