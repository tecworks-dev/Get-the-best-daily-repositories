use fennec_ast::*;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct RedudnantClosingTagRule;

impl RedudnantClosingTagRule {
    fn report<'ast>(&self, sequence: &'ast Sequence<Statement>, context: &mut LintContext<'_>) {
        let Some(last_statement) = sequence.last() else {
            return;
        };

        if let Statement::ClosingTag(closing_tag) = last_statement {
            let issue = Issue::new(context.level(), "redundant closing tag")
                .with_annotation(Annotation::primary(closing_tag.span()))
                .with_help("remove the redundant closing tag.");

            context
                .report_with_fix(issue, |plan| plan.delete(closing_tag.span().to_range(), SafetyClassification::Safe));

            return;
        }

        if let Statement::Inline(inline) = last_statement {
            let value = context.interner.lookup(inline.value);
            if value.bytes().all(|b| b.is_ascii_whitespace()) {
                let Some(Statement::ClosingTag(tag)) = sequence.get(sequence.len() - 2) else {
                    return;
                };

                let issue = Issue::new(context.level(), "redundant closing tag")
                    .with_annotation(Annotation::primary(tag.span()))
                    .with_annotation(Annotation::secondary(inline.span()).with_message("trailing whitespaces"))
                    .with_help("remove the redundant closing tag.");

                context.report_with_fix(issue, |plan| {
                    plan.delete(inline.span().to_range(), SafetyClassification::Safe)
                        .delete(tag.span().to_range(), SafetyClassification::Safe)
                });
            }
        }

        if let Statement::Namespace(namespace) = last_statement {
            match &namespace.body {
                NamespaceBody::Implicit(namespace_implicit_body) => {
                    self.report(&namespace_implicit_body.statements, context);
                }
                NamespaceBody::BraceDelimited(_) => {}
            }
        }
    }
}

impl Rule for RedudnantClosingTagRule {
    fn get_name(&self) -> &'static str {
        "redundant-closing-tag"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Help)
    }
}

impl<'a> Walker<LintContext<'a>> for RedudnantClosingTagRule {
    fn walk_program<'ast>(&self, program: &'ast Program, context: &mut LintContext<'a>) {
        self.report(&program.statements, context);
    }
}
