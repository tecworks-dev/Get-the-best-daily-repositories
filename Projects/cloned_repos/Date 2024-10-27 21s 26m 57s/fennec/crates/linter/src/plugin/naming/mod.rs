use crate::plugin::naming::rules::class::ClassRule;
use crate::plugin::naming::rules::constant::ConstantRule;
use crate::plugin::naming::rules::function::FunctionRule;
use crate::plugin::naming::rules::interface::InterfaceRule;
use crate::plugin::naming::rules::r#enum::EnumRule;
use crate::plugin::naming::rules::r#trait::TraitRule;

use crate::plugin::Plugin;
use crate::rule::Rule;

pub mod rules;

#[derive(Debug)]
pub struct NamingPlugin;

impl Plugin for NamingPlugin {
    fn get_name(&self) -> &'static str {
        "naming"
    }

    fn is_enabled_by_default(&self) -> bool {
        true
    }

    fn get_rules(&self) -> Vec<Box<dyn Rule>> {
        vec![
            Box::new(ClassRule),
            Box::new(ConstantRule),
            Box::new(EnumRule),
            Box::new(FunctionRule),
            Box::new(InterfaceRule),
            Box::new(TraitRule),
        ]
    }
}
