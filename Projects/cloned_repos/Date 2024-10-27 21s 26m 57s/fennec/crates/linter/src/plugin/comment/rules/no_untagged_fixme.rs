use std::sync::LazyLock;

use regex::Regex;

use fennec_ast::Program;
use fennec_reporting::*;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::plugin::comment::rules::utils::comment_content;
use crate::rule::Rule;

static TAGGED_FIXME_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"fixme\((#|@)?\S+\)").unwrap());

#[derive(Clone, Debug)]
pub struct NoUntaggedFixmeRule;

impl Rule for NoUntaggedFixmeRule {
    #[inline]
    fn get_name(&self) -> &'static str {
        "no-untagged-fixme"
    }

    #[inline]
    fn get_default_level(&self) -> Option<Level> {
        Level::Warning.into()
    }
}

impl<'a> Walker<LintContext<'a>> for NoUntaggedFixmeRule {
    fn walk_program<'ast>(&self, program: &'ast Program, context: &mut LintContext<'a>) {
        for trivia in program.trivia.iter() {
            if let Some(content) = comment_content(trivia, context) {
                let content = content.to_lowercase();
                if !content.contains("fixme") {
                    continue;
                }

                for line in content.lines() {
                    let trimmied = line.trim_start();
                    if !trimmied.starts_with("fixme") {
                        continue;
                    }

                    if (*TAGGED_FIXME_REGEX).is_match(&trimmied) {
                        continue;
                    }

                    context.report(
                        Issue::new(context.level(), "FIXME should be tagged with (@username) or (#issue)")
                            .with_annotation(Annotation::primary(trivia.span))
                            .with_help(
                                "add a user tag or issue reference to the FIXME comment, e.g. FIXME(@azjezz), FIXME(azjezz), FIXME(#123)",
                            )
                    );

                    break;
                }
            }
        }
    }
}
