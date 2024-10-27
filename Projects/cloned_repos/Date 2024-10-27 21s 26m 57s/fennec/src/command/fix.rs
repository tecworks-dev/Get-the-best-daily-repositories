use std::collections::hash_map::Entry;

use ahash::HashMap;

use fennec_config::Configuration;
use fennec_feedback::create_progress_bar;
use fennec_feedback::ProgressBarTheme;
use fennec_fixer::FixPlan;
use fennec_fixer::SafetyClassification;
use fennec_interner::ThreadedInterner;
use fennec_source::SourceIdentifier;
use fennec_source::SourceManager;

use crate::command::create_linter;
use crate::command::process_and_lint_all;
use crate::utils::error::bail;

pub async fn execute(
    configuration: Configuration,
    interner: ThreadedInterner,
    source_manager: SourceManager,
    safe_classification: SafetyClassification,
    dry_run: bool,
) -> i32 {
    let linter = create_linter(&interner, configuration.linter);

    let results = process_and_lint_all(&source_manager, &interner, &linter, linter.settings.external).await;

    let mut issues = vec![];
    for result in results {
        let (semantics, lint_issues) = result.unwrap_or_else(bail);

        issues.extend(semantics.issues);
        issues.extend(lint_issues);
    }

    let progress_bar = create_progress_bar(issues.len(), "fixing issues", ProgressBarTheme::Red);

    let mut plans: HashMap<SourceIdentifier, Vec<FixPlan>> = HashMap::default();
    for issue in issues.into_iter() {
        for suggestion in issue.suggestions.into_iter() {
            match plans.entry(suggestion.0) {
                Entry::Occupied(occupied_entry) => {
                    occupied_entry.into_mut().push(suggestion.1);
                }
                Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(vec![suggestion.1]);
                }
            }
        }
    }

    let mut handles = vec![];
    for mut source_plans in plans.into_iter() {
        handles.push(tokio::spawn({
            let source_manager = source_manager.clone();
            let interner = interner.clone();
            let progress_bar = progress_bar.clone();

            let source = source_plans.0;
            let plan = source_plans.1.drain(..).collect::<FixPlan>();

            async move {
                let source = source_manager.load(source).await.unwrap_or_else(bail);
                let source_name = interner.lookup(source.identifier.value());
                let source_content = interner.lookup(source.content);

                if plan.get_minimum_safety_classification() > safe_classification {
                    let required = safe_classification.to_string();
                    let cuurent = plan.get_minimum_safety_classification().to_string();

                    fennec_feedback::debug!(
                        "skipping fix for `{}` because it requires a higher safety level ({} > {})",
                        source_name,
                        required,
                        cuurent
                    );

                    progress_bar.inc(1);
                } else {
                    fennec_feedback::info!("fixing issue in `{}` ( {} fix operations )", source_name, plan.len());

                    let code = plan.execute(source_content, safe_classification);

                    if dry_run {
                        // todo, print the diff in a pretty way
                        println!("TOO LAZY TO PRETTY PRINT: {:#?}", code);

                        progress_bar.inc(1);
                    } else if let Some(path) = source.path {
                        std::fs::write(path, code.get_fixed())?;

                        fennec_feedback::info!("fixed issue in `{}`", source_name);

                        progress_bar.inc(1);
                    } else {
                        unreachable!();
                    }
                }

                Ok::<(), std::io::Error>(())
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap_or_else(bail).unwrap_or_else(bail);
    }

    progress_bar.finish_and_clear();

    0
}
