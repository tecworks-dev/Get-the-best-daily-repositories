use fennec_ast::*;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct NoGotoRule;

impl Rule for NoGotoRule {
    fn get_name(&self) -> &'static str {
        "no-goto"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Note)
    }
}

impl<'a> Walker<LintContext<'a>> for NoGotoRule {
    fn walk_in_goto<'ast>(&self, goto: &'ast Goto, context: &mut LintContext<'a>) {
        let issue = Issue::new(context.level(), "avoid using `goto`")
            .with_annotation(Annotation::primary(goto.goto.span()))
            .with_annotation(Annotation::secondary(goto.label.span()))
            .with_note("the `goto` statement can make code harder to read, understand, and maintain.")
            .with_note("it can lead to spaghetti code and make it difficult to follow the flow of execution.")
            .with_note(
                "consider using structured control flow statements like `if`, `else`, `for`, and `while` instead.",
            )
            .with_help("refactor your code to avoid using `goto`.");

        context.report(issue);
    }

    fn walk_in_label<'ast>(&self, label: &'ast Label, context: &mut LintContext<'a>) {
        let issue = Issue::new(context.level(), "avoid using labels")
            .with_annotation(Annotation::primary(label.span()))
            .with_note("labels are often used with `goto` statements, which can make code harder to read and maintain.")
            .with_note(
                "consider using structured control flow statements like `if`, `else`, `for`, and `while` instead.",
            )
            .with_help("refactor your code to avoid using labels.");

        context.report(issue);
    }
}
