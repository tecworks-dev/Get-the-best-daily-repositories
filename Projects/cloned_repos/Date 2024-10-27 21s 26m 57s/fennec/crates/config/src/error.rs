#[derive(Debug)]
pub enum ConfigurationError {
    Building(config::ConfigError),
    DeserializingToml(toml::de::Error),
    SerializingToml(toml::ser::Error),
    CanonicalizingRootPath(std::path::PathBuf, std::io::Error),
    CanonicalizingSourcePath(std::path::PathBuf, std::io::Error),
    CanonicalizingIncludePath(std::path::PathBuf, std::io::Error),
    CanonicalizingExcludePath(std::path::PathBuf, std::io::Error),
}

impl std::fmt::Display for ConfigurationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigurationError::Building(error) => {
                write!(f, "failed to build configuration: {}", error)
            }
            ConfigurationError::DeserializingToml(error) => {
                write!(f, "failed to deserialize TOML: {}", error)
            }
            ConfigurationError::SerializingToml(error) => {
                write!(f, "failed to serialize TOML: {}", error)
            }
            ConfigurationError::CanonicalizingRootPath(path, error) => {
                write!(f, "failed to canonicalize root path '{}': {}", path.display(), error)
            }
            ConfigurationError::CanonicalizingSourcePath(path, error) => {
                write!(f, "failed to canonicalize source path '{}': {}", path.display(), error)
            }
            ConfigurationError::CanonicalizingIncludePath(path, error) => {
                write!(f, "failed to canonicalize include path '{}': {}", path.display(), error)
            }
            ConfigurationError::CanonicalizingExcludePath(path, error) => {
                write!(f, "failed to canonicalize exclude path '{}': {}", path.display(), error)
            }
        }
    }
}

impl std::error::Error for ConfigurationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigurationError::Building(error) => Some(error),
            ConfigurationError::DeserializingToml(error) => Some(error),
            ConfigurationError::SerializingToml(error) => Some(error),
            ConfigurationError::CanonicalizingRootPath(_, error) => Some(error),
            ConfigurationError::CanonicalizingSourcePath(_, error) => Some(error),
            ConfigurationError::CanonicalizingIncludePath(_, error) => Some(error),
            ConfigurationError::CanonicalizingExcludePath(_, error) => Some(error),
        }
    }
}

impl From<config::ConfigError> for ConfigurationError {
    fn from(error: config::ConfigError) -> Self {
        ConfigurationError::Building(error)
    }
}

impl From<toml::de::Error> for ConfigurationError {
    fn from(error: toml::de::Error) -> Self {
        ConfigurationError::DeserializingToml(error)
    }
}

impl From<toml::ser::Error> for ConfigurationError {
    fn from(error: toml::ser::Error) -> Self {
        ConfigurationError::SerializingToml(error)
    }
}
