use fennec_ast::ast::*;
use fennec_reporting::*;
use fennec_span::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Copy, Debug)]
pub struct InterfaceRule;

impl Rule for InterfaceRule {
    fn get_name(&self) -> &'static str {
        "interface"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Help)
    }
}

impl<'a> Walker<LintContext<'a>> for InterfaceRule {
    fn walk_in_interface<'ast>(&self, interface: &'ast Interface, context: &mut LintContext<'a>) {
        let mut issues = vec![];

        let name = context.lookup(interface.name.value);
        let fqcn = context.lookup_name(&interface.name);

        if !fennec_casing::is_class_case(&name) {
            issues.push(
                Issue::new(context.level(), format!("interface name `{}` should be in class case", name))
                    .with_annotations([
                        Annotation::primary(interface.name.span()),
                        Annotation::secondary(interface.span())
                            .with_message(format!("interface `{}` is declared here", fqcn)),
                    ])
                    .with_note(format!("the interface name `{}` does not follow class naming convention.", name))
                    .with_help(format!(
                        "consider renaming it to `{}` to adhere to the naming convention.",
                        fennec_casing::to_class_case(&name)
                    )),
            );
        }

        if context.option("psr").and_then(|o| o.as_bool()).unwrap_or(true) {
            if !name.ends_with("Interface") {
                issues.push(
                    Issue::new(
                        context.level(),
                        format!("interface name `{}` should be suffixed with `Interface`", name),
                    )
                    .with_annotations([
                        Annotation::primary(interface.name.span()),
                        Annotation::secondary(interface.span())
                            .with_message(format!("interface `{}` is declared here", fqcn)),
                    ])
                    .with_note(format!("the interface name `{}` does not follow PSR naming convention.", name))
                    .with_help(format!(
                        "consider renaming it to `{}Interface` to adhere to the naming convention.",
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
