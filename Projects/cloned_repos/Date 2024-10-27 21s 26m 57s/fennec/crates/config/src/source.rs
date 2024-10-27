use std::path::PathBuf;
use std::sync::LazyLock;

use config::builder::BuilderState;
use config::ConfigBuilder;
use serde::Deserialize;
use serde::Serialize;

use crate::error::ConfigurationError;
use crate::Entry;

/// The current working directory.
const CURRENT_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    std::env::current_dir().unwrap_or_else(|e| {
        tracing::error!("failed to get the current working directory: {}", e);
        tracing::error!(
            "this might occur if the directory has been deleted or if the process lacks the necessary permissions"
        );
        tracing::error!(
            "please ensure that the directory exists and that you have the required permissions to access it"
        );

        std::process::exit(1);
    })
});

/// Configuration options for source discovery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceConfiguration {
    /// The root directory from which to start scanning.
    ///
    /// Defaults to the current working directory.
    pub root: PathBuf,

    /// Paths to user defined source files.
    ///
    /// If empty, all files in the root directory are included.
    ///
    /// Defaults to `[]`.
    pub paths: Vec<PathBuf>,

    /// Paths to non-user defined files to include in the scan.
    ///
    /// Defaults to `[]`.
    pub includes: Vec<PathBuf>,

    /// Patterns to exclude from the scan.
    ///
    /// Defaults to `[]`.
    pub excludes: Vec<String>,

    /// File extensions to filter by.
    ///
    /// Defaults to `[".php"]`.
    pub extensions: Vec<String>,
}

impl SourceConfiguration {
    /// Creates a new `SourceConfiguration` with the given root directory.
    ///
    /// # Arguments
    ///
    /// * `root` - The root directory from which to start scanning.
    ///
    /// # Returns
    ///
    /// A new `SourceConfiguration` with the given root directory.
    pub fn from_root(root: PathBuf) -> Self {
        Self { root, ..Default::default() }
    }
}

impl Default for SourceConfiguration {
    fn default() -> Self {
        Self {
            root: (*CURRENT_DIR).clone(),
            paths: Vec::new(),
            includes: Vec::new(),
            excludes: Vec::new(),
            extensions: vec!["php".to_string()],
        }
    }
}

impl Entry for SourceConfiguration {
    fn configure<St: BuilderState>(self, builder: ConfigBuilder<St>) -> Result<ConfigBuilder<St>, ConfigurationError> {
        use config::Value;
        use config::ValueKind;

        builder
            .set_default("source.root", Value::new(None, ValueKind::String(self.root.to_string_lossy().to_string())))?
            .set_default("source.paths", Value::new(None, ValueKind::Array(vec![])))?
            .set_default("source.includes", Value::new(None, ValueKind::Array(vec![])))?
            .set_default("source.excludes", Value::new(None, ValueKind::Array(vec![])))?
            .set_default(
                "source.extensions",
                Value::new(None, ValueKind::Array(vec![Value::new(None, ValueKind::String("php".to_string()))])),
            )
            .map_err(|error| ConfigurationError::from(error))
    }

    fn normalize(&mut self) -> Result<(), ConfigurationError> {
        // Make root absolute if not already
        let root = if !self.root.is_absolute() { (*CURRENT_DIR).join(&self.root) } else { self.root.clone() };

        self.root = root.canonicalize().map_err(|e| ConfigurationError::CanonicalizingRootPath(root, e))?;

        // Normalize source paths
        self.paths = self
            .paths
            .iter()
            .map(|p| {
                let path = if p.is_absolute() { p.clone() } else { self.root.join(p) };

                path.canonicalize().map_err(|e| ConfigurationError::CanonicalizingSourcePath(p.clone(), e))
            })
            .collect::<Result<Vec<PathBuf>, ConfigurationError>>()?;

        // Normalize include paths
        self.includes = self
            .includes
            .iter()
            .map(|p| {
                let path = if p.is_absolute() { p.clone() } else { self.root.join(p) };

                path.canonicalize().map_err(|e| ConfigurationError::CanonicalizingIncludePath(p.clone(), e))
            })
            .collect::<Result<Vec<PathBuf>, ConfigurationError>>()?;

        Ok(())
    }
}
