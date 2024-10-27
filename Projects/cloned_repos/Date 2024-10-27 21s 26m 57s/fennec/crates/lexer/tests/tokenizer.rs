use pretty_assertions::assert_eq;

use fennec_interner::ThreadedInterner;
use fennec_lexer::input::Input;
use fennec_source::SourceIdentifier;
use fennec_token::DocumentKind;
use fennec_token::TokenKind;

use fennec_lexer::error::SyntaxError;
use fennec_lexer::Lexer;

#[tokio::test]
async fn test_casts() -> Result<(), SyntaxError> {
    let code = b"hello <?= ( string ) + - / ??= ?-> ... ( int   ) (integer    ) (    double) &&  ?> world";
    let expected = vec![
        TokenKind::InlineText,
        TokenKind::EchoTag,
        TokenKind::Whitespace,
        TokenKind::StringCast,
        TokenKind::Whitespace,
        TokenKind::Plus,
        TokenKind::Whitespace,
        TokenKind::Minus,
        TokenKind::Whitespace,
        TokenKind::Slash,
        TokenKind::Whitespace,
        TokenKind::QuestionQuestionEqual,
        TokenKind::Whitespace,
        TokenKind::QuestionMinusGreaterThan,
        TokenKind::Whitespace,
        TokenKind::DotDotDot,
        TokenKind::Whitespace,
        TokenKind::IntCast,
        TokenKind::Whitespace,
        TokenKind::IntegerCast,
        TokenKind::Whitespace,
        TokenKind::DoubleCast,
        TokenKind::Whitespace,
        TokenKind::AmpersandAmpersand,
        TokenKind::Whitespace,
        TokenKind::CloseTag,
        TokenKind::InlineText,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_namespace() -> Result<(), SyntaxError> {
    let code = b"<?php use Foo\\{Bar, Baz}";
    let expected = vec![
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::Use,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::NamespaceSeparator,
        TokenKind::LeftBrace,
        TokenKind::Identifier,
        TokenKind::Comma,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::RightBrace,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_comments() -> Result<(), SyntaxError> {
    let code = b"
            Testing Comment Types:
            <?php
            // This is a single-line comment
            ++ /* This is a
            multi-line comment */
            /** This is a DocBlock comment */
            -- ; # Another single-line comment
            ?> hello
        ";

    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::SingleLineComment,
        TokenKind::Whitespace,
        TokenKind::PlusPlus,
        TokenKind::Whitespace,
        TokenKind::MultiLineComment,
        TokenKind::Whitespace,
        TokenKind::DocBlockComment,
        TokenKind::Whitespace,
        TokenKind::MinusMinus,
        TokenKind::Whitespace,
        TokenKind::Semicolon,
        TokenKind::Whitespace,
        TokenKind::HashComment,
        TokenKind::Whitespace,
        TokenKind::CloseTag,
        TokenKind::InlineText,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_single_line_comments() -> Result<(), SyntaxError> {
    let code = b"<?php // this is a single-line comment ?> hello <?php // another single-line comment ?> world";

    let expected = vec![
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::SingleLineComment,
        TokenKind::Whitespace,
        TokenKind::CloseTag,
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::SingleLineComment,
        TokenKind::Whitespace,
        TokenKind::CloseTag,
        TokenKind::InlineText,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_keywords() -> Result<(), SyntaxError> {
    let mut code: Vec<u8> = vec![b'<', b'?', b'p', b'h', b'p', b' '];
    let mut expected = vec![TokenKind::OpenTag, TokenKind::Whitespace];
    for (value, kind) in KEYWORD_TYPES.iter() {
        code.extend(value.to_vec());
        code.extend(b" ");

        expected.push(*kind);
        expected.push(TokenKind::Whitespace);
    }

    code.extend(b"();");

    expected.push(TokenKind::LeftParenthesis);
    expected.push(TokenKind::RightParenthesis);
    expected.push(TokenKind::Semicolon);

    test_lexer(code.as_slice(), expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_halt() -> Result<(), SyntaxError> {
    let code = b"hello <?= echo + __halt_compiler ( ) ;  echo 'unreachable';";
    let expected = vec![
        TokenKind::InlineText,
        TokenKind::EchoTag,
        TokenKind::Whitespace,
        TokenKind::Echo,
        TokenKind::Whitespace,
        TokenKind::Plus,
        TokenKind::Whitespace,
        TokenKind::HaltCompiler,
        TokenKind::Whitespace,
        TokenKind::LeftParenthesis,
        TokenKind::Whitespace,
        TokenKind::RightParenthesis,
        TokenKind::Whitespace,
        TokenKind::Semicolon,
        TokenKind::InlineText,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_identifiers() -> Result<(), SyntaxError> {
    let code = b"hello <?php FooBar Foo\\Bar Foo\\\\Bar::class;";
    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::Whitespace,
        TokenKind::QualifiedIdentifier,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::NamespaceSeparator,
        TokenKind::FullyQualifiedIdentifier,
        TokenKind::ColonColon,
        TokenKind::Class,
        TokenKind::Semicolon,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_nss() -> Result<(), SyntaxError> {
    let code = b"<?php use Foo\\{};";
    let expected = vec![
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::Use,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::NamespaceSeparator,
        TokenKind::LeftBrace,
        TokenKind::RightBrace,
        TokenKind::Semicolon,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_numbers() -> Result<(), SyntaxError> {
    let code = b"hello <?php 123 123.456 0x123 0b101 0o123 4e-2;";
    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::LiteralInteger,
        TokenKind::Whitespace,
        TokenKind::LiteralFloat,
        TokenKind::Whitespace,
        TokenKind::LiteralInteger,
        TokenKind::Whitespace,
        TokenKind::LiteralInteger,
        TokenKind::Whitespace,
        TokenKind::LiteralInteger,
        TokenKind::Whitespace,
        TokenKind::LiteralFloat,
        TokenKind::Semicolon,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_emojis() -> Result<(), SyntaxError> {
    let code = "hello <?php final readonly class 🐘 { const 🦀 = 🐱 + 🦊; }".as_bytes();
    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::Final,
        TokenKind::Whitespace,
        TokenKind::Readonly,
        TokenKind::Whitespace,
        TokenKind::Class,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::Whitespace,
        TokenKind::LeftBrace,
        TokenKind::Whitespace,
        TokenKind::Const,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::Whitespace,
        TokenKind::Equal,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::Whitespace,
        TokenKind::Plus,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::Semicolon,
        TokenKind::Whitespace,
        TokenKind::RightBrace,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_single_quote_literal_string() -> Result<(), SyntaxError> {
    let code = b"hello <?php 'hello world';";
    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::LiteralString,
        TokenKind::Semicolon,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_partial_single_quote_literal_string() -> Result<(), SyntaxError> {
    let code = b"hello <?php 'hello world";
    let expected =
        vec![TokenKind::InlineText, TokenKind::OpenTag, TokenKind::Whitespace, TokenKind::PartialLiteralString];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_double_quote_literal_string() -> Result<(), SyntaxError> {
    let code = b"hello <?php \"hello world\";";
    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::LiteralString,
        TokenKind::Semicolon,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_partial_double_quote_literal_string() -> Result<(), SyntaxError> {
    let code = b"hello <?php \"hello world";
    let expected =
        vec![TokenKind::InlineText, TokenKind::OpenTag, TokenKind::Whitespace, TokenKind::PartialLiteralString];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_variables() -> Result<(), SyntaxError> {
    let code = b"hello <?php $foo $foo_bar $fooBar $foo123 $foo_123 $foo_123_bar $$bar ${bar};";
    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Dollar,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::DollarLeftBrace,
        TokenKind::Identifier,
        TokenKind::RightBrace,
        TokenKind::Semicolon,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_literal_nowdoc_heredoc() -> Result<(), SyntaxError> {
    let code = b"
            hello
            <?php

            $foo = <<<'EOF'
                hello world
            EOF;

            $bar = <<<FOF
                hello world
            FOF;
        ";

    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Equal,
        TokenKind::Whitespace,
        TokenKind::DocumentStart(DocumentKind::Nowdoc),
        TokenKind::StringPart,
        TokenKind::DocumentEnd,
        TokenKind::Semicolon,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Equal,
        TokenKind::Whitespace,
        TokenKind::DocumentStart(DocumentKind::Heredoc),
        TokenKind::StringPart,
        TokenKind::DocumentEnd,
        TokenKind::Semicolon,
        TokenKind::Whitespace,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_heredoc() -> Result<(), SyntaxError> {
    let code = b"
                hello
                <?php

                $foo = <<<e
                    hello
                    {$bar[${'baz'}]->{'qux'}}
                    ${bar[${'baz'}]->{'qux'}}
                    $bar
                    $baz->qux
                    $baz?->qux
                    $baz[1+2]
                    \\${bar[1]->{'qux'}}
                    \\$bar
                    \\$baz->qux
                    \\$baz?->qux
                    \\$baz[1+2]
                    world
                e;
            ";

    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Equal,
        TokenKind::Whitespace,
        TokenKind::DocumentStart(DocumentKind::Heredoc),
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::LeftBrace,
        TokenKind::Variable,
        TokenKind::LeftBracket,
        TokenKind::DollarLeftBrace,
        TokenKind::LiteralString,
        TokenKind::RightBrace,
        TokenKind::RightBracket,
        TokenKind::MinusGreaterThan,
        TokenKind::LeftBrace,
        TokenKind::LiteralString,
        TokenKind::RightBrace,
        TokenKind::RightBrace,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::DollarLeftBrace,
        TokenKind::Identifier,
        TokenKind::LeftBracket,
        TokenKind::DollarLeftBrace,
        TokenKind::LiteralString,
        TokenKind::RightBrace,
        TokenKind::RightBracket,
        TokenKind::MinusGreaterThan,
        TokenKind::LeftBrace,
        TokenKind::LiteralString,
        TokenKind::RightBrace,
        TokenKind::RightBrace,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::MinusGreaterThan,
        TokenKind::Identifier,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::QuestionMinusGreaterThan,
        TokenKind::Identifier,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::LeftBracket,
        TokenKind::LiteralInteger,
        TokenKind::Plus,
        TokenKind::LiteralInteger,
        TokenKind::RightBracket,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::StringPart,
        TokenKind::DocumentEnd,
        TokenKind::Semicolon,
        TokenKind::Whitespace,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_double_quote_string() -> Result<(), SyntaxError> {
    let code = b"
            hello
            <?php

            $foo = \"hello
            {$bar[${'baz'}]->{'qux'}}
            ${bar[${'baz'}]->{'qux'}}
            $bar
            $baz->qux
            $baz?->qux
            $baz[1+2]
            \\${bar[1]->{'qux'}}
            \\$bar
            \\$baz->qux
            \\$baz?->qux
            \\$baz[1+2]
            world\";

            $bar = \"$foo\\\"\";
        ";

    let expected = vec![
        TokenKind::InlineText,
        TokenKind::OpenTag,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Equal,
        TokenKind::Whitespace,
        TokenKind::DoubleQuote,
        TokenKind::StringPart,
        TokenKind::LeftBrace,
        TokenKind::Variable,
        TokenKind::LeftBracket,
        TokenKind::DollarLeftBrace,
        TokenKind::LiteralString,
        TokenKind::RightBrace,
        TokenKind::RightBracket,
        TokenKind::MinusGreaterThan,
        TokenKind::LeftBrace,
        TokenKind::LiteralString,
        TokenKind::RightBrace,
        TokenKind::RightBrace,
        TokenKind::StringPart,
        TokenKind::DollarLeftBrace,
        TokenKind::Identifier,
        TokenKind::LeftBracket,
        TokenKind::DollarLeftBrace,
        TokenKind::LiteralString,
        TokenKind::RightBrace,
        TokenKind::RightBracket,
        TokenKind::MinusGreaterThan,
        TokenKind::LeftBrace,
        TokenKind::LiteralString,
        TokenKind::RightBrace,
        TokenKind::RightBrace,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::MinusGreaterThan,
        TokenKind::Identifier,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::QuestionMinusGreaterThan,
        TokenKind::Identifier,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::LeftBracket,
        TokenKind::LiteralInteger,
        TokenKind::Plus,
        TokenKind::LiteralInteger,
        TokenKind::RightBracket,
        TokenKind::StringPart,
        TokenKind::DoubleQuote,
        TokenKind::Semicolon,
        TokenKind::Whitespace,
        TokenKind::Variable,
        TokenKind::Whitespace,
        TokenKind::Equal,
        TokenKind::Whitespace,
        TokenKind::DoubleQuote,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::StringPart,
        TokenKind::DoubleQuote,
        TokenKind::Semicolon,
        TokenKind::Whitespace,
    ];

    test_lexer(code, expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_escape() -> Result<(), SyntaxError> {
    let code = r##"<?= "\033]8;;{$attr['href']}\033\\{$value}\033]8;;\033\\" . FOO;"##;

    let expected = vec![
        TokenKind::EchoTag,
        TokenKind::Whitespace,
        TokenKind::DoubleQuote,
        TokenKind::StringPart,
        TokenKind::LeftBrace,
        TokenKind::Variable,
        TokenKind::LeftBracket,
        TokenKind::LiteralString,
        TokenKind::RightBracket,
        TokenKind::RightBrace,
        TokenKind::StringPart,
        TokenKind::LeftBrace,
        TokenKind::Variable,
        TokenKind::RightBrace,
        TokenKind::StringPart,
        TokenKind::DoubleQuote,
        TokenKind::Whitespace,
        TokenKind::Dot,
        TokenKind::Whitespace,
        TokenKind::Identifier,
        TokenKind::Semicolon,
    ];

    test_lexer(code.as_bytes(), expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_sep_literal_num() -> Result<(), SyntaxError> {
    let code = r##"<?= 1_200;"##;

    let expected = vec![TokenKind::EchoTag, TokenKind::Whitespace, TokenKind::LiteralInteger, TokenKind::Semicolon];

    test_lexer(code.as_bytes(), expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

#[tokio::test]
async fn test_escape_in_string() -> Result<(), SyntaxError> {
    let code = r##"<?= "$foo->bar\nvar";"##;

    let expected = vec![
        TokenKind::EchoTag,
        TokenKind::Whitespace,
        TokenKind::DoubleQuote,
        TokenKind::StringPart,
        TokenKind::Variable,
        TokenKind::MinusGreaterThan,
        TokenKind::Identifier,
        TokenKind::StringPart,
        TokenKind::DoubleQuote,
        TokenKind::Semicolon,
    ];

    test_lexer(code.as_bytes(), expected).await.map_err(|err| {
        panic!("unexpected error: {}", err);
    })
}

async fn test_lexer(code: &[u8], expected_kinds: Vec<TokenKind>) -> Result<(), SyntaxError> {
    let interner = ThreadedInterner::new();
    let input = Input::new(SourceIdentifier::dummy(), code);
    let mut lexer = Lexer::new(&interner, input);

    let mut tokens = Vec::new();
    while let Some(result) = lexer.advance() {
        let token = result?;

        tokens.push(token);
    }

    assert_eq!(expected_kinds, tokens.iter().map(|t| t.kind).collect::<Vec<_>>());
    let mut found = String::new();
    for token in tokens.iter() {
        found.push_str(interner.lookup(token.value));
    }

    assert_eq!(code, found.as_bytes());

    Ok(())
}

pub const KEYWORD_TYPES: [(&[u8], TokenKind); 84] = [
    (b"eval", TokenKind::Eval),
    (b"die", TokenKind::Die),
    (b"empty", TokenKind::Empty),
    (b"isset", TokenKind::Isset),
    (b"unset", TokenKind::Unset),
    (b"exit", TokenKind::Exit),
    (b"enddeclare", TokenKind::EndDeclare),
    (b"endswitch", TokenKind::EndSwitch),
    (b"endwhile", TokenKind::EndWhile),
    (b"endforeach", TokenKind::EndForeach),
    (b"endfor", TokenKind::EndFor),
    (b"endif", TokenKind::EndIf),
    (b"from", TokenKind::From),
    (b"and", TokenKind::And),
    (b"or", TokenKind::Or),
    (b"xor", TokenKind::Xor),
    (b"print", TokenKind::Print),
    (b"readonly", TokenKind::Readonly),
    (b"global", TokenKind::Global),
    (b"match", TokenKind::Match),
    (b"abstract", TokenKind::Abstract),
    (b"array", TokenKind::Array),
    (b"as", TokenKind::As),
    (b"break", TokenKind::Break),
    (b"case", TokenKind::Case),
    (b"catch", TokenKind::Catch),
    (b"class", TokenKind::Class),
    (b"clone", TokenKind::Clone),
    (b"continue", TokenKind::Continue),
    (b"const", TokenKind::Const),
    (b"declare", TokenKind::Declare),
    (b"default", TokenKind::Default),
    (b"do", TokenKind::Do),
    (b"echo", TokenKind::Echo),
    (b"elseif", TokenKind::ElseIf),
    (b"else", TokenKind::Else),
    (b"enum", TokenKind::Enum),
    (b"extends", TokenKind::Extends),
    (b"false", TokenKind::False),
    (b"finally", TokenKind::Finally),
    (b"final", TokenKind::Final),
    (b"fn", TokenKind::Fn),
    (b"foreach", TokenKind::Foreach),
    (b"for", TokenKind::For),
    (b"function", TokenKind::Function),
    (b"goto", TokenKind::Goto),
    (b"if", TokenKind::If),
    (b"include_once", TokenKind::IncludeOnce),
    (b"include", TokenKind::Include),
    (b"implements", TokenKind::Implements),
    (b"interface", TokenKind::Interface),
    (b"instanceof", TokenKind::Instanceof),
    (b"namespace", TokenKind::Namespace),
    (b"new", TokenKind::New),
    (b"null", TokenKind::Null),
    (b"private", TokenKind::Private),
    (b"protected", TokenKind::Protected),
    (b"public", TokenKind::Public),
    (b"require_once", TokenKind::RequireOnce),
    (b"require", TokenKind::Require),
    (b"return", TokenKind::Return),
    (b"static", TokenKind::Static),
    (b"switch", TokenKind::Switch),
    (b"throw", TokenKind::Throw),
    (b"trait", TokenKind::Trait),
    (b"true", TokenKind::True),
    (b"try", TokenKind::Try),
    (b"use", TokenKind::Use),
    (b"var", TokenKind::Var),
    (b"yield", TokenKind::Yield),
    (b"while", TokenKind::While),
    (b"insteadof", TokenKind::Insteadof),
    (b"list", TokenKind::List),
    (b"self", TokenKind::Self_),
    (b"parent", TokenKind::Parent),
    (b"__dir__", TokenKind::DirConstant),
    (b"__file__", TokenKind::FileConstant),
    (b"__line__", TokenKind::LineConstant),
    (b"__function__", TokenKind::FunctionConstant),
    (b"__class__", TokenKind::ClassConstant),
    (b"__method__", TokenKind::MethodConstant),
    (b"__trait__", TokenKind::TraitConstant),
    (b"__namespace__", TokenKind::NamespaceConstant),
    (b"__halt_compiler", TokenKind::HaltCompiler),
];
