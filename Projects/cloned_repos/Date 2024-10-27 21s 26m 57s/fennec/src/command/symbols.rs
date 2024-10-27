use std::io::stdout;

use ahash::HashMap;
use ahash::HashSet;
use clap::ValueEnum;
use fennec_symbol_table::symbol::SymbolReference;
use serde_json::json;

use fennec_interner::ThreadedInterner;
use fennec_reporting::reporter::Reporter;
use fennec_reporting::*;
use fennec_source::HasSource;
use fennec_source::SourceManager;
use fennec_symbol_table::table::SymbolTable;

use crate::command::get_symbol_table;
use crate::utils::error::bail;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum SymbolKindArgument {
    // All symbols
    All,
    // Class-like symbols
    ClassLike,
    Class,
    Trait,
    Enum,
    Interface,
    AnonymousClass,
    // Function-like symbols
    FunctionLike,
    Function,
    ArrowFunction,
    Closure,
    Method,
    // Constants
    Constant,
    // Class-like members
    ClassLikeConstant,
    EnumCase,
    Property,
    // Special
    Anonymous,
    NonAnonymous,
}

pub async fn execute(
    interner: ThreadedInterner,
    source_manager: SourceManager,
    reporter: Reporter,
    include_external: bool,
    query: Option<String>,
    kind: SymbolKindArgument,
    json: bool,
    sort: bool,
) -> i32 {
    let table = get_symbol_table(&source_manager, &interner, include_external).await.unwrap_or_else(bail);

    let mut table = match &kind {
        SymbolKindArgument::All => table,
        SymbolKindArgument::Function => table.only_functions(),
        SymbolKindArgument::Constant => table.only_constants(),
        SymbolKindArgument::Class => table.only_classes(),
        SymbolKindArgument::Interface => table.only_interfaces(),
        SymbolKindArgument::Trait => table.only_traits(),
        SymbolKindArgument::Enum => table.only_enums(),
        SymbolKindArgument::EnumCase => table.only_enum_cases(),
        SymbolKindArgument::Method => table.only_methods(),
        SymbolKindArgument::Property => table.only_properties(),
        SymbolKindArgument::ClassLikeConstant => table.only_class_like_constants(),
        SymbolKindArgument::AnonymousClass => table.only_anonymous_classes(),
        SymbolKindArgument::Closure => table.only_closures(),
        SymbolKindArgument::ArrowFunction => table.only_arrow_functions(),
        SymbolKindArgument::ClassLike => table.only_class_like(),
        SymbolKindArgument::FunctionLike => table.only_function_like(),
        SymbolKindArgument::Anonymous => table.only_anonymous(),
        SymbolKindArgument::NonAnonymous => table.only_non_anonymous(),
    };

    if let Some(query) = query {
        let query = query.trim().to_lowercase();

        table = SymbolTable::from_symbols(table.into_iter().filter(|symbol| {
            let Some(identifier) = symbol.identifier else {
                if let Some(SymbolReference { identifier: Some(scope_identifier), .. }) = symbol.scope {
                    let name = interner.lookup(scope_identifier.fully_qualified_name);

                    if name.to_lowercase().contains(&query) {
                        return true;
                    }
                }

                return false;
            };

            let name = interner.lookup(identifier.fully_qualified_name);

            name.to_lowercase().contains(&query)
        }));
    }

    if sort {
        table.sort();
    }

    if json {
        // collect all interned strings
        let mut ids = HashSet::default();
        for symbol in table.iter() {
            if let Some(namespace) = symbol.namespace {
                ids.insert(namespace);
            }

            if let Some(identifier) = &symbol.identifier {
                ids.insert(identifier.name);
                ids.insert(identifier.fully_qualified_name);
            }

            if let Some(scope) = &symbol.scope {
                if let Some(scope_identifier) = &scope.identifier {
                    ids.insert(scope_identifier.name);
                    ids.insert(scope_identifier.fully_qualified_name);
                }
            }

            ids.insert(symbol.span.source().value());
        }

        // convert the interned strings to a hashmap,
        // where the key is the string and the value is the interned id
        let strings = HashMap::from_iter(ids.into_iter().map(|id| {
            let name = interner.lookup(id);

            (name.to_string(), id.value())
        }));

        // return a JSON object
        let json_output = json!({
            "symbols": table.iter().collect::<Vec<_>>(),
            "strings": strings,
        });

        serde_json::to_writer_pretty(stdout(), &json_output).unwrap_or_else(bail);

        return 0;
    }

    for symbol in table.iter() {
        let (name, fqn) = if let Some(identifier) = &symbol.identifier {
            let name = interner.lookup(identifier.name);

            let fqn = interner.lookup(identifier.fully_qualified_name);

            (name, Some(fqn))
        } else {
            ("<anonymous>", None)
        };

        let mut issue = Issue::note(format!("{} `{}`", symbol.kind, name));
        if let Some(identifier) = &symbol.identifier {
            issue = issue.with_annotations([
                Annotation::primary(identifier.span),
                Annotation::secondary(symbol.span).with_message(format!(
                    "{} `{}` defined here",
                    symbol.kind,
                    fqn.unwrap_or(name)
                )),
            ]);
        } else {
            issue = issue.with_annotations([Annotation::primary(symbol.span)]);
        }

        if let Some(scope) = symbol.scope {
            issue = issue.with_annotation(Annotation::secondary(scope.span).with_message(format!(
                    "{} `{}` is defined within {} `{}`",
                    symbol.kind,
                    name,
                    scope.kind,
                    scope.identifier
                        .map(|identifier| {
                            interner
                                .lookup(identifier.fully_qualified_name)
                        })
                        .unwrap_or("<anonymous>")
                )));
        }

        if let Some(namespace) = symbol.namespace {
            issue =
                issue.with_note(format!("{} `{}` is in namespace `{}`", symbol.kind, name, interner.lookup(namespace)));
        } else {
            issue = issue.with_note(format!("{} `{}` is in the global namespace", symbol.kind, name));
        }

        reporter.report(issue).await;
    }

    0
}
