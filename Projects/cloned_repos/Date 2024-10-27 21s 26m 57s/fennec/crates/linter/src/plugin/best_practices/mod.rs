use crate::plugin::best_practices::rules::disallowed_functions::DisallowedFunctionsRule;
use crate::plugin::best_practices::rules::excessive_nesting::ExcessiveNesting;
use crate::plugin::best_practices::rules::no_debug_symbols::NoDebugSymbolsRule;
use crate::plugin::best_practices::rules::no_goto::NoGotoRule;
use crate::plugin::best_practices::rules::no_unused_parameter::NoUnusedParameterRule;
use crate::plugin::best_practices::rules::use_while_instead_of_for::UseWhileInsteadOfForRule;

use crate::plugin::Plugin;
use crate::rule::Rule;

pub mod rules;

#[derive(Debug)]
pub struct BestPracticesPlugin;

impl Plugin for BestPracticesPlugin {
    fn get_name(&self) -> &'static str {
        "best-practices"
    }

    fn is_enabled_by_default(&self) -> bool {
        true
    }

    fn get_rules(&self) -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(DisallowedFunctionsRule),
            Box::new(NoUnusedParameterRule),
            Box::new(ExcessiveNesting),
            Box::new(NoGotoRule),
            Box::new(NoDebugSymbolsRule),
            Box::new(UseWhileInsteadOfForRule),
        ]
    }
}
