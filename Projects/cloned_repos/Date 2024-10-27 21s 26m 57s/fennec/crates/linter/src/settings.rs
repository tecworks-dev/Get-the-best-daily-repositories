use ahash::HashMap;
use serde::Deserialize;
use serde::Serialize;
use toml::value::Value;

use fennec_reporting::Level;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub level: Option<Level>,
    pub external: bool,
    pub default_plugins: bool,
    pub plugins: Vec<String>,
    pub rules: HashMap<String, RuleSettings>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RuleSettings {
    pub enabled: bool,
    pub level: Option<Level>,
    pub options: HashMap<String, Value>,
}

impl Settings {
    pub fn new() -> Self {
        Self {
            level: Some(Level::Error),
            external: false,
            default_plugins: true,
            plugins: Vec::new(),
            rules: HashMap::default(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.level.is_some()
    }

    pub fn get_rule_settings(&self, rule_name: &str) -> Option<&RuleSettings> {
        self.rules.get(rule_name)
    }

    pub fn off(mut self) -> Self {
        self.level = None;
        self
    }

    pub fn with_level(mut self, level: Level) -> Self {
        self.level = Some(level);
        self
    }

    pub fn with_external(mut self, external: bool) -> Self {
        self.external = external;
        self
    }

    pub fn with_default_plugins(mut self, default_plugins: bool) -> Self {
        self.default_plugins = default_plugins;
        self
    }

    pub fn with_plugins(mut self, plugins: Vec<String>) -> Self {
        self.plugins = plugins;
        self
    }

    pub fn with_rules(mut self, rules: HashMap<String, RuleSettings>) -> Self {
        self.rules = rules;
        self
    }

    pub fn with_rule(mut self, rule: impl Into<String>, settings: RuleSettings) -> Self {
        self.rules.insert(rule.into(), settings);
        self
    }
}

impl RuleSettings {
    pub fn enabled() -> Self {
        Self { enabled: true, level: None, options: Default::default() }
    }

    pub fn disabled() -> Self {
        Self { enabled: false, level: None, options: Default::default() }
    }

    pub fn from_level(level: Option<Level>) -> Self {
        Self { enabled: true, level, options: Default::default() }
    }

    pub fn with_options(mut self, options: HashMap<String, Value>) -> Self {
        self.options = options;

        self
    }

    pub fn is_enabled(&self) -> bool {
        self.level.is_some()
    }

    pub fn get_option<'c>(&'c self, option_name: &str) -> Option<&'c Value> {
        self.options.get(option_name)
    }
}
