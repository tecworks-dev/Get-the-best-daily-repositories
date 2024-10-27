use std::collections::VecDeque;
use std::fmt::Debug;

use fennec_ast::sequence::Sequence;
use fennec_ast::trivia::Trivia;
use fennec_ast::trivia::TriviaKind;
use fennec_interner::ThreadedInterner;
use fennec_lexer::error::SyntaxError;
use fennec_lexer::Lexer;
use fennec_span::Position;
use fennec_token::Token;
use fennec_token::TokenKind;

#[derive(Debug)]
pub struct TokenStream<'a, 'i> {
    interner: &'i ThreadedInterner,
    lexer: Lexer<'a, 'i>,
    buffer: VecDeque<Token>,
    trivia: Vec<Token>,
    position: Position,
}

impl<'a, 'i> TokenStream<'a, 'i> {
    pub fn new(interner: &'i ThreadedInterner, lexer: Lexer<'a, 'i>) -> TokenStream<'a, 'i> {
        let position = lexer.get_position();

        TokenStream { interner, lexer, buffer: VecDeque::new(), trivia: Vec::new(), position }
    }

    pub fn interner(&self) -> &'i ThreadedInterner {
        self.interner
    }

    /// Advances the stream to the next token in the input source code and returns it.
    ///
    /// If the stream has already read the entire input source code, this method will return `None`.
    ///
    /// # Returns
    ///
    /// The next token in the input source code, or `None` if the lexer has reached the end of the input.
    #[inline]
    pub fn advance(&mut self) -> Option<Result<Token, SyntaxError>> {
        match self.fill_buffer(1) {
            Ok(Some(_)) => {
                if let Some(token) = self.buffer.pop_front() {
                    self.position = token.span.end;

                    Some(Ok(token))
                } else {
                    None
                }
            }
            Ok(None) => None,
            Err(error) => {
                return Some(Err(error));
            }
        }
    }

    /// Return the current position of the stream in the input source code.
    #[inline]
    pub const fn get_position(&self) -> Position {
        self.position
    }

    #[inline]
    pub fn has_reached_eof(&mut self) -> Result<bool, SyntaxError> {
        return Ok(match self.fill_buffer(1)? {
            Some(_) => false,
            None => true,
        });
    }

    /// Peeks at the next token in the input source code without consuming it.
    ///
    /// This method returns the next token that the lexer would produce if `advance` were called.
    ///
    /// If the lexer has already read the entire input source code, this method will return `None`.
    #[inline]
    pub fn peek(&mut self) -> Option<Result<Token, SyntaxError>> {
        self.peek_nth(0)
    }

    /// Peeks at the `n`-th token in the input source code without consuming it.
    ///
    /// This method returns the `n`-th token that the lexer would produce if `advance` were called `n` times.
    ///
    /// If the lexer has already read the entire input source code, this method will return `None`.
    #[inline]
    pub fn peek_nth(&mut self, n: usize) -> Option<Result<Token, SyntaxError>> {
        let index = match self.fill_buffer(n + 1) {
            Ok(index) => index?,
            Err(error) => {
                return Some(Err(error));
            }
        };

        Some(Ok(self.buffer[index - 1]))
    }

    /// Consumes the comments collected by the lexer and returns them.
    #[inline]
    pub fn get_trivia(&mut self) -> Sequence<Trivia> {
        let mut tokens = Vec::new();

        std::mem::swap(&mut tokens, &mut self.trivia);

        tokens
            .into_iter()
            .map(|token| match token.kind {
                TokenKind::Whitespace => Trivia { kind: TriviaKind::WhiteSpace, span: token.span, value: token.value },
                TokenKind::HashComment => {
                    Trivia { kind: TriviaKind::HashComment, span: token.span, value: token.value }
                }
                TokenKind::SingleLineComment => {
                    Trivia { kind: TriviaKind::SingleLineComment, span: token.span, value: token.value }
                }
                TokenKind::MultiLineComment => {
                    Trivia { kind: TriviaKind::MultiLineComment, span: token.span, value: token.value }
                }
                TokenKind::DocBlockComment => {
                    Trivia { kind: TriviaKind::DocBlockComment, span: token.span, value: token.value }
                }
                _ => unreachable!(),
            })
            .collect()
    }

    /// Fills the token buffer with at least `n` tokens.
    #[inline]
    fn fill_buffer(&mut self, n: usize) -> Result<Option<usize>, SyntaxError> {
        loop {
            if self.buffer.len() >= n {
                break;
            }

            match self.lexer.advance() {
                Some(result) => match result {
                    Ok(token) => {
                        if token.kind.is_trivia() {
                            self.trivia.push(token);

                            continue;
                        }

                        self.buffer.push_back(token);
                    }
                    Err(error) => return Err(error),
                },
                None => return Ok(None),
            };
        }

        Ok(Some(n))
    }
}
