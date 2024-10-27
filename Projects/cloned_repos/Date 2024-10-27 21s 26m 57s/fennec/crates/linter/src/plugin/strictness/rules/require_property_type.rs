use fennec_ast::ast::*;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct RequirePropertyTypeRule;

impl Rule for RequirePropertyTypeRule {
    #[inline]
    fn get_name(&self) -> &'static str {
        "require-property-type"
    }

    #[inline]
    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Warning)
    }
}

impl<'a> Walker<LintContext<'a>> for RequirePropertyTypeRule {
    fn walk_in_property<'ast>(&self, property: &'ast Property, context: &mut LintContext<'a>) {
        if property.hint().is_some() {
            return;
        }

        let (class_like_kind, class_like_name, class_like_fqcn, class_like_span) =
            context.get_class_like_details(property);

        let variable = context.lookup(property.first_variable().name);

        context.report(
            Issue::new(
                context.level(),
                format!("{} property `{}::{}` is missing a type hint", class_like_kind, class_like_name, variable),
            )
            .with_annotation(Annotation::primary(property.span()))
            .with_annotation(
                Annotation::secondary(class_like_span)
                    .with_message(format!("{} `{}` declared here", class_like_kind, class_like_fqcn)),
            )
            .with_note(format!(
                "adding a type hint to {} properties improves code readability and helps prevent type errors.",
                class_like_kind
            ))
            .with_help(format!("consider specifying a type hint for `{}::{}`.", class_like_name, variable)),
        );
    }
}
