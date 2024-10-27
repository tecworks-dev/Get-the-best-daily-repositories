pub use indicatif::ProgressBar;
pub use tracing::level_filters::*;
pub use tracing::*;

pub use crate::logger::initialize_logger;
pub use crate::progress::create_progress_bar;
pub use crate::progress::remove_progress_bar;
pub use crate::progress::ProgressBarTheme;

/// The `logger` module handles setting up and configuring logging for the application.
pub mod logger;

/// The `progress` module manages progress bars and provides utilities for creating and removing them.
pub mod progress;
