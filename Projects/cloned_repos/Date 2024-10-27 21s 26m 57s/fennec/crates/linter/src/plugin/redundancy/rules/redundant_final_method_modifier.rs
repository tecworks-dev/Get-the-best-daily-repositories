use fennec_ast::*;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct RedundantFinalMethodModifierRule;

impl RedundantFinalMethodModifierRule {
    fn report<'ast>(&self, method: &'ast Method, context: &mut LintContext<'_>, in_enum: bool) {
        let Some(final_modifier) = method.modifiers.get_final() else {
            return;
        };

        let (class_like_kind, class_like_name, class_like_fqcn, class_like_span) =
            context.get_class_like_details(method);

        let method_name = context.interner.lookup(method.name.value);

        let message = if in_enum {
            format!(
                "the `final` modifier on method `{}` is redundant in enum `{}` as enums cannot be extended.",
                method_name, class_like_name
            )
        } else {
            format!(
                "the `final` modifier on method `{}` is redundant as the class `{}` is already final.",
                method_name, class_like_name
            )
        };

        let issue = Issue::new(context.level(), message)
            .with_annotations([
                Annotation::primary(final_modifier.span()),
                Annotation::secondary(class_like_span)
                    .with_message(format!("{} `{}` defined here", class_like_kind, class_like_fqcn)),
            ])
            .with_help("Remove the redundant `final` modifier.");

        context
            .report_with_fix(issue, |plan| plan.delete(final_modifier.span().to_range(), SafetyClassification::Safe));
    }
}

impl Rule for RedundantFinalMethodModifierRule {
    fn get_name(&self) -> &'static str {
        "redundant-final-method-modifier"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Help)
    }
}

impl<'a> Walker<LintContext<'a>> for RedundantFinalMethodModifierRule {
    fn walk_in_class<'ast>(&self, class: &'ast Class, context: &mut LintContext<'a>) {
        if !class.modifiers.contains_final() {
            return;
        }

        for member in class.members.iter() {
            if let ClassLikeMember::Method(method) = member {
                self.report(method, context, false);
            }
        }
    }

    fn walk_in_enum<'ast>(&self, r#enum: &'ast Enum, context: &mut LintContext<'a>) {
        for member in r#enum.members.iter() {
            if let ClassLikeMember::Method(method) = member {
                self.report(method, context, true);
            }
        }
    }
}
