use fennec_ast::*;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct NoEvalRule;

impl Rule for NoEvalRule {
    fn get_name(&self) -> &'static str {
        "no-eval"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Error)
    }
}

impl<'a> Walker<LintContext<'a>> for NoEvalRule {
    fn walk_in_eval_construct<'ast>(&self, eval_construct: &'ast EvalConstruct, context: &mut LintContext<'a>) {
        let issue = Issue::new(context.level(), "unsafe use of `eval`")
            .with_annotation(Annotation::primary(eval_construct.eval.span))
            .with_annotation(Annotation::secondary(eval_construct.value.span()))
            .with_note("the `eval` construct executes arbitrary code, which can be a major security risk if not used carefully.")
            .with_note("it can potentially lead to remote code execution vulnerabilities if the evaluated code is not properly sanitized.")
            .with_note("consider using safer alternatives whenever possible.")
            .with_help("avoid using `eval` unless absolutely necessary, and ensure that any dynamically generated code is properly validated and sanitized before execution.")
        ;

        context.report(issue);
    }
}
