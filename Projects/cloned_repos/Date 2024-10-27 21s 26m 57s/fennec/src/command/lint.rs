use std::io::stdout;

use ahash::HashMap;
use serde_json::json;

use fennec_config::Configuration;
use fennec_interner::ThreadedInterner;
use fennec_reporting::reporter::Reporter;
use fennec_reporting::*;
use fennec_source::SourceManager;

use crate::command::create_linter;
use crate::command::process_and_lint_all;
use crate::utils::error::bail;

pub async fn execute(
    configuration: Configuration,
    interner: ThreadedInterner,
    source_manager: SourceManager,
    reporter: Reporter,
    only_fixable: bool,
    json: bool,
) -> i32 {
    let linter = create_linter(&interner, configuration.linter);

    let results = process_and_lint_all(&source_manager, &interner, &linter, linter.settings.external).await;

    let mut issues = IssueCollection::new();
    for result in results {
        let (semantics, lint_issues) = result.unwrap_or_else(bail);

        if !only_fixable {
            if let Some(error) = &semantics.parse_error {
                issues.push(Into::<Issue>::into(error));
            }

            issues.extend(semantics.issues);
            issues.extend(lint_issues);
        } else {
            issues.extend(semantics.issues.into_iter().filter(|issue| !issue.suggestions.is_empty()));

            issues.extend(lint_issues.into_iter().filter(|issue| !issue.suggestions.is_empty()));
        }
    }

    let highest_level = issues.get_highest_level();

    if json {
        let strings = HashMap::from_iter(
            issues
                .iter()
                .flat_map(|issue| {
                    let mut ids = Vec::new();
                    for suggestion in &issue.suggestions {
                        ids.push(suggestion.0.value());
                    }

                    for annotations in &issue.annotations {
                        ids.push(annotations.span.start.source.value());
                    }

                    ids
                })
                .map(|id| {
                    let name = interner.lookup(id);

                    (name.to_string(), id.value())
                }),
        );

        // return a JSON object
        let json_output = json!({
            "symbols": issues.iter().collect::<Vec<_>>(),
            "strings": strings,
        });

        serde_json::to_writer_pretty(stdout(), &json_output).unwrap_or_else(bail)
    } else {
        reporter.report_all(issues.into_iter()).await;
    }

    if let Some(level) = highest_level {
        if level == Level::Error {
            return 1;
        }

        0
    } else {
        0
    }
}
