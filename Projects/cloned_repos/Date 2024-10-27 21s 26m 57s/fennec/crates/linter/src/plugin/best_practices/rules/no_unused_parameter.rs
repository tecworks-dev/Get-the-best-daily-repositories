use fennec_ast::*;
use fennec_fixer::SafetyClassification;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::plugin::best_practices::rules::utils::expression_potentially_contains_function_call;
use crate::plugin::best_practices::rules::utils::get_foreign_variable_names;
use crate::plugin::best_practices::rules::utils::is_variable_used_in_expression;
use crate::plugin::best_practices::rules::utils::potentially_contains_function_call;
use crate::rule::Rule;

const FUNC_GET_ARGS: &str = "func_get_args";

#[derive(Clone, Debug)]
pub struct NoUnusedParameterRule;

impl NoUnusedParameterRule {
    fn report<'ast>(
        &self,
        parameter: &'ast FunctionLikeParameter,
        function_like: &impl HasSpan,
        context: &mut LintContext,
        kind: &'static str,
    ) {
        if parameter.ampersand.is_some() {
            return;
        }

        let parameter_name = context.interner.lookup(parameter.variable.name);
        if parameter_name.starts_with("$_") {
            return;
        }

        let issue = Issue::new(context.level(), format!("unused parameter: `{}`", parameter_name))
            .with_annotations([
                Annotation::primary(parameter.span()),
                Annotation::secondary(function_like.span()),
            ])
            .with_note(format!("this parameter is declared but not used within the {}", kind))
            .with_help("consider prefixing the parameter with an underscore (`_`) to indicate that it is intentionally unused, or remove it if it is not needed");

        context.report_with_fix(issue, |plan| {
            plan.insert(
                parameter.variable.span().start.offset + 1, // skip the leading `$`
                "_",
                SafetyClassification::PotentiallyUnsafe,
            )
        });
    }
}

impl Rule for NoUnusedParameterRule {
    fn get_name(&self) -> &'static str {
        "no-unused-parameter"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Note)
    }
}

impl<'a> Walker<LintContext<'a>> for NoUnusedParameterRule {
    fn walk_in_function<'ast>(&self, function: &'ast Function, context: &mut LintContext<'a>) {
        if potentially_contains_function_call(&function.body, FUNC_GET_ARGS, context) {
            // `func_get_args` is potentially used, so we can't determine if the parameters are unused
            // in this case

            return;
        }

        let foreign_variables = get_foreign_variable_names(&function.body, context);

        for parameter in function.parameters.parameters.iter() {
            if foreign_variables.contains(&parameter.variable.name) {
                continue;
            }

            self.report(parameter, function, context, "function");
        }
    }

    fn walk_in_closure<'ast>(&self, closure: &'ast Closure, context: &mut LintContext<'a>) {
        if potentially_contains_function_call(&closure.body, FUNC_GET_ARGS, context) {
            // `func_get_args` is potentially used, so we can't determine if the parameters are unused
            // in this case

            return;
        }

        let foreign_variables = get_foreign_variable_names(&closure.body, context);

        for parameter in closure.parameters.parameters.iter() {
            if foreign_variables.contains(&parameter.variable.name) {
                continue;
            }

            self.report(parameter, closure, context, "closure");
        }
    }

    fn walk_in_method<'ast>(&self, method: &'ast Method, context: &mut LintContext<'a>) {
        let MethodBody::Concrete(block) = &method.body else {
            return;
        };

        if !context.option("methods").and_then(|o| o.as_bool()).unwrap_or(false) {
            tracing::trace!("skipping checking parameters, methods are disabled");

            return;
        }

        if potentially_contains_function_call(block, FUNC_GET_ARGS, context) {
            // `func_get_args` is potentially used, so we can't determine if the parameters are unused
            // in this case

            return;
        }

        let foreign_variables = get_foreign_variable_names(block, context);

        for parameter in method.parameters.parameters.iter() {
            // Skip promoted properties
            if parameter.is_promoted_property() {
                continue;
            }

            if foreign_variables.contains(&parameter.variable.name) {
                continue;
            }

            self.report(parameter, method, context, "method");
        }
    }

    fn walk_in_arrow_function<'ast>(&self, arrow_function: &'ast ArrowFunction, context: &mut LintContext<'a>) {
        if expression_potentially_contains_function_call(&arrow_function.expression, FUNC_GET_ARGS, context) {
            // `func_get_args` is potentially used, so we can't determine if the parameters are unused
            // in this case

            return;
        }

        for parameter in arrow_function.parameters.parameters.iter() {
            if !is_variable_used_in_expression(&arrow_function.expression, context, parameter.variable.name) {
                self.report(parameter, arrow_function, context, "arrow function");
            }
        }
    }
}
