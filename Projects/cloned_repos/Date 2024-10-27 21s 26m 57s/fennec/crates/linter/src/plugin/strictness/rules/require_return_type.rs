use fennec_ast::ast::*;
use fennec_reporting::*;
use fennec_span::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct RequireReturnTypeRule;

impl Rule for RequireReturnTypeRule {
    #[inline]
    fn get_name(&self) -> &'static str {
        "require-return-type"
    }

    #[inline]
    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Warning)
    }
}

impl<'a> Walker<LintContext<'a>> for RequireReturnTypeRule {
    fn walk_in_function<'ast>(&self, function: &'ast Function, context: &mut LintContext<'a>) {
        if function.return_type_hint.is_some() {
            return;
        }

        let function_name = context.lookup(function.name.value);
        let function_fqn = context.lookup_name(&function.name);

        context.report(
            Issue::new(context.level(), format!("function `{}` is missing a return type hint", function_name))
                .with_annotation(Annotation::primary(function.name.span()))
                .with_annotation(
                    Annotation::secondary(function.span())
                        .with_message(format!("function `{}` defined here", function_fqn)),
                )
                .with_note("type hints improve code readability and help prevent type-related errors.")
                .with_help(format!("consider adding a return type hint to function `{}`.", function_name)),
        );
    }

    fn walk_in_closure<'ast>(&self, closure: &'ast Closure, context: &mut LintContext<'a>) {
        if closure.return_type_hint.is_some() {
            return;
        }

        context.report(
            Issue::new(context.level(), "closure is missing a return type hint")
                .with_annotation(Annotation::primary(closure.function.span()))
                .with_annotation(Annotation::secondary(closure.span()).with_message("closure defined here"))
                .with_note("type hints improve code readability and help prevent type-related errors.")
                .with_help("consider adding a return type hint to the closure."),
        );
    }

    fn walk_in_arrow_function<'ast>(&self, arrow_function: &'ast ArrowFunction, context: &mut LintContext<'a>) {
        if arrow_function.return_type_hint.is_some() {
            return;
        }

        context.report(
            Issue::new(context.level(), "arrow function is missing a return type hint")
                .with_annotation(Annotation::primary(arrow_function.r#fn.span()))
                .with_annotation(
                    Annotation::secondary(arrow_function.span()).with_message("arrow function defined here"),
                )
                .with_note("type hints improve code readability and help prevent type-related errors.")
                .with_help("consider adding a return type hint to the arrow function."),
        );
    }

    fn walk_in_method<'ast>(&self, method: &'ast Method, context: &mut LintContext<'a>) {
        if method.return_type_hint.is_some() {
            return;
        }

        let method_name = context.lookup(method.name.value);
        if "__construct" == method_name || "__destruct" == method_name {
            // constructors and destructors cannot have return types.
            return;
        }

        let (class_like_kind, class_like_name, class_like_fqcn, class_like_span) =
            context.get_class_like_details(method);

        context.report(
            Issue::new(
                context.level(),
                format!(
                    "{} method `{}::{}` is missing a return type hint",
                    class_like_kind, class_like_name, method_name
                ),
            )
            .with_annotation(Annotation::primary(method.name.span()))
            .with_annotations([
                Annotation::secondary(method.span())
                    .with_message(format!("{} method `{}` defined here", class_like_kind, method_name)),
                Annotation::secondary(class_like_span)
                    .with_message(format!("{} `{}` defined here", class_like_kind, class_like_fqcn)),
            ])
            .with_note("type hints improve code readability and help prevent type-related errors.")
            .with_help(format!("consider adding a return type hint to {} method `{}`.", class_like_kind, method_name)),
        );
    }
}
