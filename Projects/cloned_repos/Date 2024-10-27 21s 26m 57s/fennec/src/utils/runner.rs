use std::future::Future;
use std::process::exit;

use tokio::runtime::Builder;

use fennec_config::Configuration;
use fennec_feedback::initialize_logger;
use fennec_feedback::LevelFilter;

use crate::utils::error::bail;

#[inline(always)]
pub fn run_with_configuration<F>(task: impl FnOnce(Configuration) -> F) -> !
where
    F: Future<Output = i32>,
{
    // Set up the logger.
    initialize_logger(LevelFilter::WARN, "FENNEC_LOG");

    // Load the configuration.
    let configuration = Configuration::load().unwrap_or_else(bail);

    // Run the task.
    Builder::new_multi_thread()
        .worker_threads(configuration.threads)
        .thread_stack_size(configuration.stack_size)
        .enable_all()
        .build()
        .unwrap_or_else(bail)
        .block_on(async move { exit(task(configuration).await) })
}
