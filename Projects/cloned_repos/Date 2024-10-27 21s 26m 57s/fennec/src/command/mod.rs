use fennec_linter::settings::RuleSettings;
use fennec_semantics::Semantics;

use fennec_config::linter::LinterConfiguration;
use fennec_config::linter::LinterLevel;
use fennec_feedback::create_progress_bar;
use fennec_feedback::remove_progress_bar;
use fennec_feedback::ProgressBarTheme;
use fennec_interner::ThreadedInterner;
use fennec_linter::plugin::best_practices::BestPracticesPlugin;
use fennec_linter::plugin::comment::CommentPlugin;
use fennec_linter::plugin::consistency::ConsistencyPlugin;
use fennec_linter::plugin::naming::NamingPlugin;
use fennec_linter::plugin::redundancy::RedundancyPlugin;
use fennec_linter::plugin::safety::SafetyPlugin;
use fennec_linter::plugin::strictness::StrictnessPlugin;
use fennec_linter::plugin::symfony::SymfonyPlugin;
use fennec_linter::settings::Settings;
use fennec_linter::Linter;
use fennec_reporting::*;
use fennec_semantics::build;
use fennec_source::error::SourceError;
use fennec_source::SourceManager;
use fennec_symbol_table::table::SymbolTable;

pub mod fix;
pub mod lint;
pub mod symbols;

/// Create a linter with the given configuration.
///
/// This function is responsible for creating a linter with the given configuration.
///
/// # Parameters
///
/// - `interner`: The interner to use for the linter.
/// - `configuration`: The configuration to use for the linter.
///
/// # Returns
///
/// The created linter.
pub fn create_linter(interner: &ThreadedInterner, configuration: LinterConfiguration) -> Linter {
    let mut settings = Settings::new();

    if let Some(level) = configuration.level {
        settings = match &level {
            LinterLevel::Off => settings.off(),
            LinterLevel::Help => settings.with_level(Level::Help),
            LinterLevel::Note => settings.with_level(Level::Note),
            LinterLevel::Warning => settings.with_level(Level::Warning),
            LinterLevel::Error => settings.with_level(Level::Error),
        };
    }

    if let Some(external) = configuration.external {
        settings = settings.with_external(external);
    }

    if let Some(default_plugins) = configuration.default_plugins {
        settings = settings.with_default_plugins(default_plugins);
    }

    settings = settings.with_plugins(configuration.plugins);

    for rule in configuration.rules {
        let mut rule_settings = match rule.level {
            Some(linter_level) => match linter_level {
                LinterLevel::Off => RuleSettings::disabled(),
                LinterLevel::Help => RuleSettings::from_level(Some(Level::Help)),
                LinterLevel::Note => RuleSettings::from_level(Some(Level::Note)),
                LinterLevel::Warning => RuleSettings::from_level(Some(Level::Warning)),
                LinterLevel::Error => RuleSettings::from_level(Some(Level::Error)),
            },
            None => RuleSettings::enabled(),
        };

        if let Some(options) = rule.options {
            rule_settings = rule_settings.with_options(options);
        }

        settings = settings.with_rule(rule.name, rule_settings)
    }

    let mut linter = Linter::new(settings, interner.clone());

    linter.add_plugin(BestPracticesPlugin);
    linter.add_plugin(CommentPlugin);
    linter.add_plugin(ConsistencyPlugin);
    linter.add_plugin(NamingPlugin);
    linter.add_plugin(RedundancyPlugin);
    linter.add_plugin(SafetyPlugin);
    linter.add_plugin(StrictnessPlugin);
    linter.add_plugin(SymfonyPlugin);

    linter
}

/// A type alias for a processed source file with linting.
///
/// This type alias is used to represent the result of processing and linting a single source file.
///
/// The tuple contains the following:
///
/// - The parsed program.
/// - An optional parse error.
/// - The resolved names.
/// - The issues found during semantic analysis.
/// - The issues found during linting.
pub type ProcessedSourceWithLint = (Semantics, IssueCollection);

/// Process all source files concurrently and lint them.
///
/// This function is responsible for processing and linting all source files in the source manager.
///
/// # Parameters
///
/// - `manager`: The source manager to load the source files from.
/// - `interner`: The interner to use for parsing and resolving names.
/// - `linter`: The linter to use for linting the source files.
/// - `include_external`: Whether to include external sources in the processing.
///
/// # Returns
///
/// A vector of results, where each result is the result of processing and linting a single source file.
async fn process_and_lint_all(
    manager: &SourceManager,
    interner: &ThreadedInterner,
    linter: &Linter,
    include_external: bool,
) -> Vec<Result<ProcessedSourceWithLint, SourceError>> {
    let mut source_ids = manager.source_ids().collect::<Vec<_>>();
    if !include_external {
        source_ids.retain(|id| id.is_user_defined());
    }
    let total_sources = source_ids.len();

    let source_pb = create_progress_bar(source_ids.len(), "📂  Loading", ProgressBarTheme::Red);
    let semantics_pb = create_progress_bar(source_ids.len(), "🔬  Building", ProgressBarTheme::Blue);
    let lint_pb = create_progress_bar(source_ids.len(), "🧹  Linting", ProgressBarTheme::Cyan);

    let mut handles = vec![];
    for source_id in source_ids {
        let interner = interner.clone();
        let manager = manager.clone();
        let linter = linter.clone();

        let handle = tokio::spawn({
            let source_pb = source_pb.clone();
            let semantics_pb = semantics_pb.clone();
            let lint_pb = lint_pb.clone();

            async move {
                // Step 1: Load source
                let source = manager.load(source_id).await?;
                source_pb.inc(1);

                // Step 2: Build Semantic
                let semantics = build(&interner, source);
                semantics_pb.inc(1);

                // Step 3: Lint the source
                let lint_result = linter.lint(&semantics);
                lint_pb.inc(1);

                // Step 4: Create the processed source
                Result::<ProcessedSourceWithLint, SourceError>::Ok((semantics, lint_result))
            }
        });

        handles.push(handle);
    }

    let mut results = Vec::with_capacity(total_sources);
    for handle in handles.into_iter() {
        let result =
            handle.await.expect("failed to process and lint the sources. this is a bug in fennec. please report it.");

        results.push(result);
    }

    remove_progress_bar(source_pb);
    remove_progress_bar(semantics_pb);
    remove_progress_bar(lint_pb);

    results
}

/// Process all source files.
///
/// This function is responsible for processing all source files in the source manager.
///
/// # Parameters
///
/// - `manager`: The source manager to load the source files from.
/// - `interner`: The interner to use for parsing and resolving names.
/// - `include_external`: Whether to include external source files in the processing.
async fn process_all(
    manager: &SourceManager,
    interner: &ThreadedInterner,
    include_external: bool,
) -> Vec<Result<Semantics, SourceError>> {
    let mut source_ids = manager.source_ids().collect::<Vec<_>>();
    if !include_external {
        source_ids.retain(|id| id.is_user_defined());
    }

    let total_sources = source_ids.len();
    let source_pb = create_progress_bar(source_ids.len(), "📂  Loading", ProgressBarTheme::Red);
    let semantics_pb = create_progress_bar(source_ids.len(), "🔬  Building", ProgressBarTheme::Blue);

    let mut handles = vec![];
    for source_id in source_ids {
        let interner = interner.clone();
        let manager = manager.clone();

        let handle = tokio::spawn({
            let source_pb = source_pb.clone();
            let semantics_pb = semantics_pb.clone();

            async move {
                // Step 1: Load source
                let source = manager.load(source_id).await?;
                source_pb.inc(1);

                // Step 2: Build Semantic
                let semantics = build(&interner, source);
                semantics_pb.inc(1);

                // Step 4: Create the processed source
                Result::<Semantics, SourceError>::Ok(semantics)
            }
        });

        handles.push(handle);
    }

    let mut results = Vec::with_capacity(total_sources);
    for handle in handles.into_iter() {
        let result =
            handle.await.expect("failed to process and lint the sources. this is a bug in fennec. please report it.");

        results.push(result);
    }

    remove_progress_bar(source_pb);
    remove_progress_bar(semantics_pb);

    results
}

/// Get the symbol table for all source files.
///
/// This function is responsible for processing all source files in the source manager,
/// and collecting the symbol table for each source file and merging them into a single symbol table.
///
/// # Parameters
///
/// - `manager`: The source manager to load the source files from.
/// - `interner`: The interner to use for parsing and resolving names.
/// - `include_external`: Whether to include external sources in the processing.
///
/// # Returns
///
/// The symbol table for all source files.
async fn get_symbol_table(
    manager: &SourceManager,
    interner: &ThreadedInterner,
    include_external: bool,
) -> Result<SymbolTable, SourceError> {
    let semantics = process_all(manager, interner, include_external).await;

    let mut symbol_table = SymbolTable::new();
    for semantic in semantics {
        let semantic = semantic?;

        symbol_table.merge(semantic.symbols);
    }

    Ok(symbol_table)
}
