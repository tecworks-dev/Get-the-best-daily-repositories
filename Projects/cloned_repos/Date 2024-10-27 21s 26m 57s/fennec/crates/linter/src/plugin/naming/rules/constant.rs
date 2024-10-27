use fennec_ast::ast::*;
use fennec_reporting::*;
use fennec_span::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Copy, Debug)]
pub struct ConstantRule;

impl Rule for ConstantRule {
    fn get_name(&self) -> &'static str {
        "constant"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Help)
    }
}

impl<'a> Walker<LintContext<'a>> for ConstantRule {
    fn walk_in_constant<'ast>(&self, constant: &'ast Constant, context: &mut LintContext<'a>) {
        for item in constant.items.iter() {
            let name = context.lookup(item.name.value);
            let fqcn = context.lookup_name(&item.name);
            if !fennec_casing::is_constant_case(&name) {
                context.report(
                    Issue::new(context.level(), format!("constant name `{}` should be in constant case", name))
                        .with_annotations([
                            Annotation::primary(item.name.span()),
                            Annotation::secondary(constant.span())
                                .with_message(format!("constant `{}` is declared here", fqcn)),
                        ])
                        .with_note(format!("the constant name `{}` does not follow constant naming convention.", name))
                        .with_help(format!(
                            "consider renaming it to `{}` to adhere to the naming convention.",
                            fennec_casing::to_constant_case(&name)
                        )),
                );
            }
        }
    }

    fn walk_in_class_like_constant<'ast>(
        &self,
        class_like_constant: &'ast ClassLikeConstant,
        context: &mut LintContext<'a>,
    ) {
        let (class_like_kind, class_like_name, class_like_fqcn, class_like_span) =
            context.get_class_like_details(class_like_constant);

        for item in class_like_constant.items.iter() {
            let name = context.lookup(item.name.value);

            if !fennec_casing::is_constant_case(&name) {
                context.report(
                    Issue::new(
                        context.level(),
                        format!(
                            "{} constant name `{}::{}` should be in constant case",
                            class_like_kind, class_like_name, name
                        ),
                    )
                    .with_annotations([
                        Annotation::primary(item.name.span()),
                        Annotation::secondary(class_like_constant.span()).with_message(format!(
                            "{} constant `{}::{}` is declared here",
                            class_like_kind, class_like_name, name
                        )),
                        Annotation::secondary(class_like_span)
                            .with_message(format!("{} `{}` is declared here", class_like_kind, class_like_fqcn)),
                    ])
                    .with_note(format!(
                        "the {} constant name `{}::{}` does not follow constant naming convention.",
                        class_like_kind, class_like_name, name
                    ))
                    .with_help(format!(
                        "consider renaming it to `{}` to adhere to the naming convention.",
                        fennec_casing::to_constant_case(&name)
                    )),
                );
            }
        }
    }
}
