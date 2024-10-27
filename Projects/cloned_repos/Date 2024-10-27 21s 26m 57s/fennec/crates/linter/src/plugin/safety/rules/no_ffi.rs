use fennec_ast::*;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

const FFI_CLASSES: [&'static str; 3] = ["FFI", "FFI\\Cdata", "FFI\\Ctype"];

#[derive(Clone, Debug)]
pub struct NoFFIRule;

impl NoFFIRule {
    fn report<'ast>(&self, identifier: &'ast Identifier, node: Option<&impl HasSpan>, context: &mut LintContext<'_>) {
        let class_name = context.lookup_name(identifier);

        if FFI_CLASSES.iter().any(|ffi| ffi.eq_ignore_ascii_case(class_name)) {
            let mut issue = Issue::new(
               context.level(),
               format!("potentionally unsafe use of FFI class `{}`", class_name),
            )
            .with_annotation(Annotation::primary(identifier.span()))
            .with_note("FFI (Foreign Function Interface) allows interaction with code written in other languages.")
            .with_note("this can introduce potential security risks and stability issues if not handled carefully.")
            .with_note("make sure you understand the implications and potential vulnerabilities before using FFI in production.")
            .with_help("if possible, consider using alternative solutions within PHP to avoid relying on FFI.");

            if let Some(node) = node {
                issue = issue.with_annotation(Annotation::secondary(node.span()));
            }

            context.report(issue);
        }
    }
}

impl Rule for NoFFIRule {
    fn get_name(&self) -> &'static str {
        "no-ffi"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Error)
    }
}

impl<'a> Walker<LintContext<'a>> for NoFFIRule {
    fn walk_in_static_method_call<'ast>(
        &self,
        static_method_call: &'ast StaticMethodCall,
        context: &mut LintContext<'a>,
    ) {
        let Expression::Identifier(class_identifier) = static_method_call.class.as_ref() else {
            return;
        };

        self.report(class_identifier, Some(static_method_call), context);
    }

    fn walk_in_class_constant_access<'ast>(
        &self,
        class_constant_access: &'ast ClassConstantAccess,
        context: &mut LintContext<'a>,
    ) {
        let Expression::Identifier(class_identifier) = &class_constant_access.class else {
            return;
        };

        self.report(class_identifier, Some(class_constant_access), context);
    }

    fn walk_in_instantiation<'ast>(&self, instantiation: &'ast Instantiation, context: &mut LintContext<'a>) {
        let Expression::Identifier(class_identifier) = &instantiation.class else {
            return;
        };

        self.report(class_identifier, Some(instantiation), context);
    }

    fn walk_in_hint<'ast>(&self, hint: &'ast Hint, context: &mut LintContext<'a>) {
        let Hint::Identifier(identifier) = &hint else {
            return;
        };

        self.report(identifier, Option::<&Hint>::None, context);
    }
}
