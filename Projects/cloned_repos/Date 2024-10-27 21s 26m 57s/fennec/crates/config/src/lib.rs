use std::sync::LazyLock;

use config::builder::BuilderState;
use config::ConfigBuilder;
use num_cpus::get as get_logical_cpus;
use serde::Deserialize;
use serde::Serialize;

pub use toml::Value;

use crate::error::ConfigurationError;
use crate::linter::LinterConfiguration;
use crate::source::SourceConfiguration;

pub mod error;
pub mod linter;
pub mod source;

/// The name of the environment variable prefix for fennec.
const ENVIRONMENT_PREFIX: &str = "FENNEC";
/// The name of the configuration file for fennec.
const CONFIGURATION_FILE: &str = "fennec";
/// The minimum stack size for each thread.
const MINIMUM_STACK_SIZE: usize = 8 * 1024 * 1024;
/// The default stack size for each thread.
const DEFAULT_STACK_SIZE: usize = 36 * 1024 * 1024;
/// The maximum stack size for each thread.
const MAXIMUM_STACK_SIZE: usize = 256 * 1024 * 1024;
/// The number of logical CPUs on the system.
const LOGICAL_CPUS: LazyLock<usize> = LazyLock::new(get_logical_cpus);

/// Configuration options for fennec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Configuration {
    /// The number of threads to use.
    pub threads: usize,

    /// The size of the stack for each thread.
    pub stack_size: usize,

    /// Configuration options for source discovery.
    pub source: SourceConfiguration,

    /// Configuration options for the linter.
    pub linter: LinterConfiguration,
}

impl Configuration {
    /// Creates a new `Configuration` with the given source and linter configurations.
    ///
    /// # Arguments
    ///
    /// * `threads` - The number of threads to use.
    /// * `stack_size` - The size of the stack for each thread.
    /// * `source` - Configuration options for source discovery.
    /// * `linter` - Configuration options for the linter.
    ///
    /// # Returns
    ///
    /// A new `Configuration` with the given source and linter configurations.
    pub fn new(threads: usize, stack_size: usize, source: SourceConfiguration, linter: LinterConfiguration) -> Self {
        Self { threads, stack_size, source, linter }
    }

    pub fn load() -> Result<Self, ConfigurationError> {
        use config::Config;
        use config::Environment;
        use config::File;
        use config::FileFormat;

        let builder = Config::builder()
            .add_source(File::with_name(CONFIGURATION_FILE).required(false).format(FileFormat::Toml))
            .add_source(Environment::with_prefix(ENVIRONMENT_PREFIX));

        tracing::debug!("loading configuration from sources");

        let mut this = Configuration::default().configure(builder)?.build()?.try_deserialize::<Configuration>()?;

        this.normalize()?;

        Ok(this)
    }

    /// Creates a new `Configuration` with the given root directory.
    ///
    /// # Arguments
    ///
    /// * `root` - The root directory from which to start scanning.
    ///
    /// # Returns
    ///
    /// A new `Configuration` with the given root directory.
    pub fn from_root(root: std::path::PathBuf) -> Self {
        Self { source: SourceConfiguration::from_root(root), ..Default::default() }
    }
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            threads: *LOGICAL_CPUS,
            stack_size: DEFAULT_STACK_SIZE,
            source: SourceConfiguration::default(),
            linter: LinterConfiguration::default(),
        }
    }
}

/// Internal trait for configuration entries.
trait Entry {
    /// Configures the builder with the entry.
    fn configure<St: BuilderState>(self, builder: ConfigBuilder<St>) -> Result<ConfigBuilder<St>, ConfigurationError>;

    fn normalize(&mut self) -> Result<(), ConfigurationError> {
        Ok(())
    }
}

impl Entry for Configuration {
    fn configure<St: BuilderState>(self, builder: ConfigBuilder<St>) -> Result<ConfigBuilder<St>, ConfigurationError> {
        use config::Value;
        use config::ValueKind;

        tracing::trace!("configuring configuration entry");

        let mut builder = builder
            .set_default("threads", Value::new(None, ValueKind::U64(self.threads as u64)))?
            .set_default("stack_size", Value::new(None, ValueKind::U64(self.stack_size as u64)))?;

        tracing::trace!("configuring source entry");
        builder = self.source.configure(builder)?;

        tracing::trace!("configuring linter entry");
        builder = self.linter.configure(builder)?;

        Ok(builder)
    }

    fn normalize(&mut self) -> Result<(), ConfigurationError> {
        if self.threads == 0 {
            tracing::info!("thread configuration is zero, using the number of logical CPUs: {}", *LOGICAL_CPUS);

            self.threads = *LOGICAL_CPUS;
        } else {
            tracing::debug!("configuration specifies {} threads", self.threads);
        }

        if self.stack_size == 0 {
            tracing::info!("stack size configuration is zero, using the maximum size of {} bytes", MAXIMUM_STACK_SIZE);

            self.stack_size = MAXIMUM_STACK_SIZE;
        } else if self.stack_size > MAXIMUM_STACK_SIZE {
            tracing::warn!(
                "stack size configuration is too large, reducing to maximum size of {} bytes",
                MAXIMUM_STACK_SIZE
            );

            self.stack_size = MAXIMUM_STACK_SIZE;
        } else if self.stack_size < MINIMUM_STACK_SIZE {
            tracing::warn!(
                "stack size configuration is too small, increasing to minimum size of {} bytes",
                MINIMUM_STACK_SIZE
            );

            self.stack_size = MINIMUM_STACK_SIZE;
        } else {
            tracing::debug!("configuration specifies a stack size of {} bytes", self.stack_size);
        }

        self.source.normalize()?;
        self.linter.normalize()?;

        Ok(())
    }
}
