use ahash::HashMap;
use config::builder::BuilderState;
use config::ConfigBuilder;
use serde::Deserialize;
use serde::Serialize;
use toml::value::Value;

use crate::error::ConfigurationError;
use crate::Entry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LinterLevel {
    Off,
    Help,
    Note,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinterConfiguration {
    pub level: Option<LinterLevel>,
    pub external: Option<bool>,
    pub default_plugins: Option<bool>,
    pub plugins: Vec<String>,
    pub rules: Vec<LinterRuleConfiguration>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LinterRuleConfiguration {
    pub name: String,
    pub level: Option<LinterLevel>,
    pub options: Option<HashMap<String, Value>>,
}

impl Default for LinterConfiguration {
    fn default() -> Self {
        Self { level: None, external: None, default_plugins: None, plugins: Vec::new(), rules: Vec::new() }
    }
}

impl Entry for LinterConfiguration {
    fn configure<St: BuilderState>(self, builder: ConfigBuilder<St>) -> Result<ConfigBuilder<St>, ConfigurationError> {
        use config::Value;
        use config::ValueKind;

        let builder = builder
            .set_default("linter.level", Value::new(None, ValueKind::Nil))?
            .set_default("linter.default_plugins", Value::new(None, ValueKind::Nil))?
            .set_default("linter.external", Value::new(None, ValueKind::Nil))?
            .set_default("linter.plugins", Value::new(None, ValueKind::Array(vec![])))?
            .set_default("linter.rules", Value::new(None, ValueKind::Array(vec![])))?;

        Ok(builder)
    }
}
