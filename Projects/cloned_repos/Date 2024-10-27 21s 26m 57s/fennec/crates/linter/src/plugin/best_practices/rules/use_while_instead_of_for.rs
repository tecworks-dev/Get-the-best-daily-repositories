use fennec_ast::*;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct UseWhileInsteadOfForRule;

impl Rule for UseWhileInsteadOfForRule {
    fn get_name(&self) -> &'static str {
        "use-while-instead-of-for"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Note)
    }
}

impl<'a> Walker<LintContext<'a>> for UseWhileInsteadOfForRule {
    fn walk_in_for<'ast>(&self, r#for: &'ast For, context: &mut LintContext<'a>) {
        if !r#for.initializations.is_empty() || !r#for.increments.is_empty() {
            return;
        }

        let issue = Issue::new(
            context.level(),
            "use `while` instead of `for`",
        )
        .with_annotation(Annotation::primary(r#for.span()))
        .with_note("this `for` loop can be simplified to a `while` loop since it doesn't have initializations or increments.")
        .with_help("use a `while` loop instead of a `for` loop.");

        context.report_with_fix(issue, |mut plan| {
            plan = plan
                .replace(
                    r#for.r#for.span.join(r#for.initializations_semicolon).to_range(),
                    "while (",
                    SafetyClassification::Safe,
                )
                .delete(r#for.conditions_semicolon.to_range(), SafetyClassification::Safe);

            for semicolon in r#for.conditions.tokens.iter() {
                plan = plan.replace(semicolon.span.to_range(), " && ", SafetyClassification::Safe);
            }

            match &r#for.body {
                ForBody::Statement(_) => plan,
                ForBody::ColonDelimited(for_colon_delimited_body) => plan.replace(
                    for_colon_delimited_body.end_for.span.to_range(),
                    "endwhile",
                    SafetyClassification::Safe,
                ),
            }
        });
    }
}
