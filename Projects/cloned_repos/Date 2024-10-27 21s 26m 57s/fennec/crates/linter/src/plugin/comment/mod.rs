use crate::plugin::comment::rules::docblock_syntax::DocblockSyntaxRule;
use crate::plugin::comment::rules::no_shell_style::NoShellStyleRule;
use crate::plugin::comment::rules::no_trailing_whitespace::NoTrailingWhitespaceRule;
use crate::plugin::comment::rules::no_untagged_fixme::NoUntaggedFixmeRule;
use crate::plugin::comment::rules::no_untagged_todo::NoUntaggedTodoRule;

use crate::plugin::Plugin;
use crate::rule::Rule;

pub mod rules;

#[derive(Debug)]
pub struct CommentPlugin;

impl Plugin for CommentPlugin {
    fn get_name(&self) -> &'static str {
        "comment"
    }

    fn is_enabled_by_default(&self) -> bool {
        true
    }

    fn get_rules(&self) -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(NoUntaggedTodoRule),
            Box::new(NoUntaggedFixmeRule),
            Box::new(NoShellStyleRule),
            Box::new(NoTrailingWhitespaceRule),
            Box::new(DocblockSyntaxRule),
        ]
    }
}
