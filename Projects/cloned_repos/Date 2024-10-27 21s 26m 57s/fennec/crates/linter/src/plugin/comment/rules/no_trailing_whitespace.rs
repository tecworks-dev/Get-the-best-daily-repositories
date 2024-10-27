use fennec_ast::Program;
use fennec_fixer::FixPlan;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_source::*;
use fennec_span::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct NoTrailingWhitespaceRule;

impl Rule for NoTrailingWhitespaceRule {
    #[inline]
    fn get_name(&self) -> &'static str {
        "no-trailing-whitespace"
    }

    #[inline]
    fn get_default_level(&self) -> Option<Level> {
        Level::Note.into()
    }
}

impl<'a> Walker<LintContext<'a>> for NoTrailingWhitespaceRule {
    fn walk_program<'ast>(&self, program: &'ast Program, context: &mut LintContext<'a>) {
        let mut issues = vec![];
        for trivia in program.trivia.iter() {
            if trivia.kind.is_comment() {
                let comment_span = trivia.span();
                let value = context.lookup(trivia.value);
                let lines = value.lines().collect::<Vec<_>>();

                let mut offset = 0;

                for line in lines.iter() {
                    let trimmed = line.trim_end();
                    let trimmed_length = trimmed.len();
                    let trailing_whitespace_length = line.len() - trimmed_length;

                    if trailing_whitespace_length > 0 {
                        let whitespace_start = offset + trimmed_length;

                        let whitespace_span = Span::new(
                            comment_span.start.forward(whitespace_start),
                            comment_span.start.forward(whitespace_start + trailing_whitespace_length),
                        );

                        issues.push(
                            Issue::new(context.level(), "trailing whitespace detected in comment.")
                                .with_annotations([
                                    Annotation::primary(whitespace_span).with_message("trailing whitespace"),
                                    Annotation::secondary(comment_span)
                                        .with_message("comment containing trailing whitespace"),
                                ])
                                .with_note("trailing whitespaces can cause unnecessary diffs and formatting issues.")
                                .with_help("remove the extra whitespace.")
                                .with_suggestion(
                                    whitespace_span.source(),
                                    FixPlan::new().delete(whitespace_span.to_range(), SafetyClassification::Safe),
                                ),
                        );
                    }

                    offset += line.len() + 1;
                }
            }
        }

        for issue in issues {
            context.report(issue);
        }
    }
}
