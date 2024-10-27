use fennec_ast::*;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct ArraySyntaxRule;

impl Rule for ArraySyntaxRule {
    fn get_name(&self) -> &'static str {
        "array-syntax"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Note)
    }
}

impl<'a> Walker<LintContext<'a>> for ArraySyntaxRule {
    fn walk_in_legacy_array<'ast>(&self, legacy_array: &'ast LegacyArray, context: &mut LintContext<'a>) {
        if context.option("syntax").and_then(|o| o.as_str()).map(|v| v.to_lowercase().eq("long")).unwrap_or(false) {
            return;
        }

        let issue = Issue::new(context.level(), "short array syntax `[..]` is preferred over `array(..)`")
            .with_annotation(Annotation::primary(legacy_array.span()))
            .with_help("use the short array syntax `[..]` instead");

        context.report_with_fix(issue, |plan| {
            plan.replace(
                legacy_array.array.span.join(legacy_array.left_parenthesis).to_range(),
                "[",
                SafetyClassification::Safe,
            )
            .replace(legacy_array.right_parenthesis.to_range(), "]", SafetyClassification::Safe)
        });
    }

    fn walk_in_array<'ast>(&self, array: &'ast Array, context: &mut LintContext<'a>) {
        if !context.option("syntax").and_then(|o| o.as_str()).map(|v| v.to_lowercase().eq("long")).unwrap_or(false) {
            return;
        }

        let issue = Issue::new(context.level(), "long array syntax `array(..)` is preferred over `[..]`")
            .with_annotation(Annotation::primary(array.span()))
            .with_help("use the long array syntax `array(..)` instead");

        context.report_with_fix(issue, |plan| {
            plan.replace(array.left_bracket.to_range(), "array(", SafetyClassification::Safe).replace(
                array.right_bracket.to_range(),
                ")",
                SafetyClassification::Safe,
            )
        });
    }
}
