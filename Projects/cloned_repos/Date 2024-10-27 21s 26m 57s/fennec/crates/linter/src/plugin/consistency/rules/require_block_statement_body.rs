use fennec_ast::*;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct RequireBlockStatementBodyRule;

impl RequireBlockStatementBodyRule {
    fn report<'ast>(&self, r#loop: &impl HasSpan, statement: &'ast Statement, context: &mut LintContext<'_>) {
        if matches!(statement, Statement::Block(_)) {
            return;
        }

        let issue = Issue::new(context.level(), "loop body should be enclosed in a block")
            .with_annotations([Annotation::primary(statement.span()), Annotation::secondary(r#loop.span())])
            .with_note(
                "enclosing the loop body in a block improves readability and prevents potential errors \
                when adding or modifying statements within the loop.",
            )
            .with_help("enclose the loop body in a block for clarity and error prevention.");

        context.report_with_fix(issue, |plan| {
            if matches!(statement, Statement::Noop(_)) {
                plan.replace(statement.span().to_range(), "{}", SafetyClassification::Safe)
            } else {
                plan.insert(statement.span().start.offset, "{", SafetyClassification::Safe).insert(
                    statement.span().end.offset,
                    "}",
                    SafetyClassification::Safe,
                )
            }
        });
    }
}

impl Rule for RequireBlockStatementBodyRule {
    fn get_name(&self) -> &'static str {
        "require-block-statement-body"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Note)
    }
}

impl<'a> Walker<LintContext<'a>> for RequireBlockStatementBodyRule {
    fn walk_in_for<'ast>(&self, r#for: &'ast For, context: &mut LintContext<'a>) {
        let ForBody::Statement(statement) = &r#for.body else {
            return;
        };

        self.report(r#for, statement, context);
    }

    fn walk_in_while<'ast>(&self, r#while: &'ast While, context: &mut LintContext<'a>) {
        let WhileBody::Statement(statement) = &r#while.body else {
            return;
        };

        self.report(r#while, statement, context);
    }

    fn walk_in_foreach<'ast>(&self, foreach: &'ast Foreach, context: &mut LintContext<'a>) {
        let ForeachBody::Statement(statement) = &foreach.body else {
            return;
        };

        self.report(foreach, statement, context);
    }
}
