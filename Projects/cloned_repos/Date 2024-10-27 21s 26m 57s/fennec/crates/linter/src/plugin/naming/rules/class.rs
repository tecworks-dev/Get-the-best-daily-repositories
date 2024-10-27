use fennec_ast::ast::*;
use fennec_reporting::*;
use fennec_span::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Copy, Debug)]
pub struct ClassRule;

impl Rule for ClassRule {
    fn get_name(&self) -> &'static str {
        "class"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Help)
    }
}

impl<'a> Walker<LintContext<'a>> for ClassRule {
    fn walk_in_class<'ast>(&self, class: &'ast Class, context: &mut LintContext<'a>) {
        let mut issues = vec![];

        let name = context.lookup(class.name.value);
        let fqcn = context.lookup_name(&class.name);

        if !fennec_casing::is_class_case(&name) {
            let issue = Issue::new(context.level(), format!("class name `{}` should be in class case", name))
                .with_annotations([
                    Annotation::primary(class.name.span()),
                    Annotation::secondary(class.span()).with_message(format!("class `{}` is declared here", fqcn)),
                ])
                .with_note(format!("the class name `{}` does not follow class naming convention.", name))
                .with_help(format!(
                    "consider renaming it to `{}` to adhere to the naming convention.",
                    fennec_casing::to_class_case(&name)
                ));

            issues.push(issue);
        }

        if class.modifiers.contains_abstract() && context.option("psr").and_then(|o| o.as_bool()).unwrap_or(true) {
            if !name.starts_with("Abstract") {
                issues.push(
                    Issue::new(
                        context.level(),
                        format!("abstract class name `{}` should be prefixed with `Abstract`", name),
                    )
                    .with_annotations([
                        Annotation::primary(class.name.span),
                        Annotation::secondary(class.span())
                            .with_message(format!("abstract class `{}` is declared here", fqcn)),
                    ])
                    .with_note(format!("the abstract class name `{}` does not follow PSR naming convention.", name))
                    .with_help(format!(
                        "consider renaming it to `Abstract{}` to adhere to the naming convention.",
                        name
                    )),
                );
            }
        }

        for issue in issues {
            context.report(issue);
        }
    }
}
