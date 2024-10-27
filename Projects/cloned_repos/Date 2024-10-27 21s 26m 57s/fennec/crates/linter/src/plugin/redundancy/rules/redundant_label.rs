use fennec_ast::*;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct RedundantLabelRule;

impl Rule for RedundantLabelRule {
    fn get_name(&self) -> &'static str {
        "redundant-label"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Help)
    }
}

impl<'a> Walker<LintContext<'a>> for RedundantLabelRule {
    fn walk_program<'ast>(&self, program: &'ast Program, context: &mut LintContext<'a>) {
        let node = Node::Program(&program);

        let labels =
            node.filter_map(
                |node| {
                    if let Node::Label(label) = node {
                        Some((label.name.value, label.span()))
                    } else {
                        None
                    }
                },
            );

        let gotos = node.filter_map(|node| if let Node::Goto(goto) = node { Some(goto.label.value) } else { None });

        for (label_id, label_span) in labels.into_iter() {
            if gotos.contains(&label_id) {
                continue;
            }

            let label_name = context.interner.lookup(label_id);

            let issue = Issue::new(context.level(), format!("redundant label: `{}`", label_name))
                .with_annotation(Annotation::primary(label_span))
                .with_note("this label is declared but not used by any `goto` statement.")
                .with_help("remove the redundant label.");

            context.report_with_fix(issue, |plan| plan.delete(label_span.to_range(), SafetyClassification::Safe));
        }
    }
}
