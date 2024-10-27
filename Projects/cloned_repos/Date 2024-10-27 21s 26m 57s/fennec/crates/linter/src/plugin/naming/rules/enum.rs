use fennec_ast::ast::*;
use fennec_reporting::*;
use fennec_span::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Copy, Debug)]
pub struct EnumRule;

impl Rule for EnumRule {
    fn get_name(&self) -> &'static str {
        "enum"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Help)
    }
}

impl<'a> Walker<LintContext<'a>> for EnumRule {
    fn walk_in_enum<'ast>(&self, r#enum: &'ast Enum, context: &mut LintContext<'a>) {
        let name = context.lookup(r#enum.name.value);
        let fqcn = context.lookup_name(&r#enum.name);

        if !fennec_casing::is_class_case(&name) {
            context.report(
                Issue::new(context.level(), format!("enum name `{}` should be in class case", name))
                    .with_annotations([
                        Annotation::primary(r#enum.name.span()),
                        Annotation::secondary(r#enum.span()).with_message(format!("enum `{}` is declared here", fqcn)),
                    ])
                    .with_note(format!("the enum name `{}` does not follow class naming convention.", name))
                    .with_help(format!(
                        "consider renaming it to `{}` to adhere to the naming convention.",
                        fennec_casing::to_class_case(&name)
                    )),
            );
        }
    }
}
