use fennec_ast::ast::*;
use fennec_reporting::*;
use fennec_span::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct RequireConstantTypeRule;

impl Rule for RequireConstantTypeRule {
    #[inline]
    fn get_name(&self) -> &'static str {
        "require-constant-type"
    }

    #[inline]
    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Warning)
    }
}

impl<'a> Walker<LintContext<'a>> for RequireConstantTypeRule {
    fn walk_class_like_constant<'ast>(
        &self,
        class_like_constant: &'ast ClassLikeConstant,
        context: &mut LintContext<'a>,
    ) {
        if class_like_constant.hint.is_some() {
            return;
        }

        let (class_like_kind, class_like_name, class_like_fqcn, class_like_span) =
            context.get_class_like_details(class_like_constant);

        for item in class_like_constant.items.iter() {
            let constant_name = context.lookup(item.name.value);

            context.report(
                Issue::new(
                    context.level(),
                    format!(
                        "{} constant `{}::{}` is missing a type hint",
                        class_like_kind, class_like_name, constant_name
                    ),
                )
                .with_annotations([
                    Annotation::primary(class_like_constant.span()),
                    Annotation::secondary(class_like_span)
                        .with_message(format!("{} `{}` declared here", class_like_kind, class_like_fqcn)),
                ])
                .with_note("adding a type hint to constants improves code readability and helps prevent type errors.")
                .with_help(format!("consider specifying a type hint for `{}::{}`.", class_like_name, constant_name)),
            );
        }
    }
}
