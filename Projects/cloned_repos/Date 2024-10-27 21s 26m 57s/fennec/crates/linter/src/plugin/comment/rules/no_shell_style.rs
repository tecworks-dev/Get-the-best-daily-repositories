use fennec_ast::*;
use fennec_fixer::FixPlan;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct NoShellStyleRule;

impl Rule for NoShellStyleRule {
    #[inline]
    fn get_name(&self) -> &'static str {
        "no-shell-style"
    }

    #[inline]
    fn get_default_level(&self) -> Option<Level> {
        Level::Warning.into()
    }
}

impl<'a> Walker<LintContext<'a>> for NoShellStyleRule {
    fn walk_program<'ast>(&self, program: &'ast Program, context: &mut LintContext<'a>) {
        for trivia in program.trivia.iter() {
            if let TriviaKind::HashComment = trivia.kind {
                let comment_span = trivia.span();
                let comment_pos = comment_span.start;

                context.report(
                    Issue::new(context.level(), "shell-style comments ('#') are not allowed.")
                        .with_annotation(Annotation::primary(comment_span).with_message("shell-style comment here"))
                        .with_help("consider using double slash comments ('//') instead.")
                        .with_suggestion(
                            comment_pos.source,
                            FixPlan::new().replace(comment_pos.range_for(1), "//", SafetyClassification::Safe),
                        ),
                );
            }
        }
    }
}
