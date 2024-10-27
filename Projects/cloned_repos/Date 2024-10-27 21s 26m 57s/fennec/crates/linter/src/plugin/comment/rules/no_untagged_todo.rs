use std::sync::LazyLock;

use regex::Regex;

use fennec_ast::Program;
use fennec_reporting::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::plugin::comment::rules::utils::comment_content;
use crate::rule::Rule;

static TAGGED_TODO_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"todo\((#|@)?\S+\)").unwrap());

#[derive(Clone, Debug)]
pub struct NoUntaggedTodoRule;

impl Rule for NoUntaggedTodoRule {
    #[inline]
    fn get_name(&self) -> &'static str {
        "no-untagged-todo"
    }

    #[inline]
    fn get_default_level(&self) -> Option<Level> {
        Level::Warning.into()
    }
}

impl<'a> Walker<LintContext<'a>> for NoUntaggedTodoRule {
    fn walk_program<'ast>(&self, program: &'ast Program, context: &mut LintContext<'a>) {
        for trivia in program.trivia.iter() {
            if let Some(content) = comment_content(trivia, context) {
                let content = content.to_ascii_lowercase();
                if !content.contains("todo") {
                    continue;
                }

                for line in content.lines() {
                    let trimmied = line.trim_start();
                    if !trimmied.starts_with("todo") {
                        continue;
                    }

                    if (*TAGGED_TODO_REGEX).is_match(&trimmied) {
                        continue;
                    }

                    context.report(
                        Issue::new(context.level(), "TODO should be tagged with (@username) or (#issue)")
                            .with_annotation(Annotation::primary(trivia.span))
                            .with_help(
                                "add a user tag or issue reference to the TODO comment, e.g. TODO(@azjezz), TODO(azjezz), TODO(#123)",
                            )
                    );

                    break;
                }
            }
        }
    }
}
