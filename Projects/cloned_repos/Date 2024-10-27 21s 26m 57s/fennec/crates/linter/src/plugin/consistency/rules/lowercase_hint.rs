use fennec_ast::ast::*;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Copy, Debug)]
pub struct LowercaseHintRule;

impl Rule for LowercaseHintRule {
    fn get_name(&self) -> &'static str {
        "lowercase-hint"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Help)
    }
}

impl<'a> Walker<LintContext<'a>> for LowercaseHintRule {
    fn walk_in_hint<'ast>(&self, hint: &'ast Hint, context: &mut LintContext<'a>) {
        if let Hint::Void(identifier)
        | Hint::Never(identifier)
        | Hint::Float(identifier)
        | Hint::Bool(identifier)
        | Hint::Integer(identifier)
        | Hint::String(identifier)
        | Hint::Object(identifier)
        | Hint::Mixed(identifier)
        | Hint::Iterable(identifier) = hint
        {
            let name = context.interner.lookup(identifier.value);
            let lowered = name.to_ascii_lowercase();
            if !lowered.eq(&name) {
                let issue = Issue::new(context.level(), format!("type hint `{}` should be in lowercase", name))
                    .with_annotation(Annotation::primary(identifier.span()))
                    .with_help(format!("consider using `{}` instead of `{}`.", lowered, name));

                context.report_with_fix(issue, |p| {
                    p.replace(identifier.span.to_range(), lowered, SafetyClassification::Safe)
                });
            }
        }
    }
}
