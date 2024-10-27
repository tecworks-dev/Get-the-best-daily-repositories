use crate::plugin::symfony::rules::quality::interface_should_be_used::InterfaceShouldBeUsed;

use crate::plugin::Plugin;
use crate::rule::Rule;

pub mod rules;

#[derive(Debug)]
pub struct SymfonyPlugin;

impl Plugin for SymfonyPlugin {
    fn get_name(&self) -> &'static str {
        "symfony"
    }

    fn is_enabled_by_default(&self) -> bool {
        false
    }

    fn get_rules(&self) -> Vec<Box<dyn Rule>> {
        vec![Box::new(InterfaceShouldBeUsed)]
    }
}
