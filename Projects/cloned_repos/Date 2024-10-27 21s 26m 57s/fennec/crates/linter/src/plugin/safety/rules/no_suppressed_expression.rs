use fennec_ast::*;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct NoSuppressedExpressionRule;

impl Rule for NoSuppressedExpressionRule {
    fn get_name(&self) -> &'static str {
        "no-suppressed-expression"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Error)
    }
}

impl<'a> Walker<LintContext<'a>> for NoSuppressedExpressionRule {
    fn walk_in_suppressed<'ast>(&self, suppressed: &'ast Suppressed, context: &mut LintContext<'a>) {
        let issue = Issue::new(context.level(), "unsafe use of suppressed expressions")
            .with_annotation(Annotation::primary(suppressed.at))
            .with_annotation(Annotation::secondary(suppressed.expression.span()))
            .with_note("suppressed expressions hide potential errors and make debugging more difficult.")
            .with_help("remove the `@` and use `set_error_handler` to handle errors instead.");

        context.report_with_fix(issue, |plan| plan.delete(suppressed.at.to_range(), SafetyClassification::Safe));
    }
}
