use fennec_ast::*;
use fennec_reporting::*;
use fennec_span::HasSpan;
use fennec_walker::walk_block_mut;
use fennec_walker::MutWalker;
use fennec_walker::Walker;

use crate::context::LintContext;
use crate::rule::Rule;

const DEFAULT_THRESHOLD: i64 = 7;

#[derive(Clone, Debug)]
pub struct ExcessiveNesting;

impl Rule for ExcessiveNesting {
    fn get_name(&self) -> &'static str {
        "excessive-nesting"
    }

    fn get_default_level(&self) -> Option<Level> {
        Some(Level::Warning)
    }
}

impl<'a> Walker<LintContext<'a>> for ExcessiveNesting {
    fn walk_program<'ast>(&self, program: &'ast Program, context: &mut LintContext<'a>) {
        let threshold = context.option("threshold").and_then(|value| value.as_integer()).unwrap_or(DEFAULT_THRESHOLD);

        let mut walker = NestingWalker { threshold: threshold as usize, level: 0 };

        walker.walk_program(program, context);
    }
}

struct NestingWalker {
    threshold: usize,
    level: usize,
}

impl NestingWalker {
    fn check_indent<'ast>(&self, block: &'ast Block, context: &mut LintContext) -> bool {
        if self.level > self.threshold {
            let issue = Issue::new(context.level(), "excessive block nesting")
                .with_annotation(Annotation::primary(block.span()))
                .with_note(format!(
                    "this block has a nesting level of {} which exceeds the threshold of {}",
                    self.level, self.threshold
                ))
                .with_note("excessive nesting can make code harder to read, understand, and maintain.")
                .with_help("refactor your code to reduce the level of nesting.");

            context.report(issue);

            return true;
        }

        false
    }
}

impl<'a> MutWalker<LintContext<'a>> for NestingWalker {
    fn walk_block<'ast>(&mut self, block: &'ast Block, context: &mut LintContext<'a>) {
        self.level += 1;

        if !self.check_indent(block, context) {
            walk_block_mut(self, block, context);
        }

        self.level -= 1;
    }
}
