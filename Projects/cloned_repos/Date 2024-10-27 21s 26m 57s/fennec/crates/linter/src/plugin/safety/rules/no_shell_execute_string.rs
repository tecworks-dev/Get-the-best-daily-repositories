use fennec_ast::*;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

#[derive(Clone, Debug)]
pub struct NoShellExecuteStringRule;

impl Rule for NoShellExecuteStringRule {
    fn get_name(&self) -> &'static str {
        "no-shell-execute-string"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Error)
    }
}

impl<'a> Walker<LintContext<'a>> for NoShellExecuteStringRule {
    fn walk_in_shell_execute_string<'ast>(
        &self,
        shell_execute_string: &'ast ShellExecuteString,
        context: &mut LintContext<'a>,
    ) {
        let mut is_interpolated = false;
        for part in shell_execute_string.parts.iter() {
            if !matches!(part, StringPart::Literal(..)) {
                is_interpolated = true;

                break;
            }
        }

        let issue = if is_interpolated {
            Issue::new(context.level(), "unsafe use of interpolated shell execute string")
                    .with_annotation(Annotation::primary(shell_execute_string.span()))
                    .with_note("interpolating shell execute strings (`...`) is a potential security vulnerability, as it allows executing arbitrary shell commands.")
                    .with_help(
                        "consider using `shell_exec()` along with `escapeshellarg()` or `escapeshellcmd()` to escape arguments instead."
                    )
        } else {
            Issue::new(context.level(), "potentilly unsafe use of shell execute string")
                .with_annotation(Annotation::primary(shell_execute_string.span()))
                .with_note("shell execute strings (`...`) can often be replaced with safer alternatives.")
                .with_help("consider using `shell_exec()` instead.")
        };

        context.report(issue);
    }
}
