use crate::config::SessionConfig;
use crate::types::DevelopmentType;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{prelude::*, rng};
use std::{thread, time::Duration};

pub fn display_boot_sequence(config: &SessionConfig) {
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("##-"),
    );

    println!(
        "{}",
        "\nINITIALIZING DEVELOPMENT ENVIRONMENT"
            .bold()
            .bright_cyan()
    );

    let project_display = if config.minimal_output {
        config.project_name.clone()
    } else {
        config
            .project_name
            .to_uppercase()
            .bold()
            .bright_yellow()
            .to_string()
    };

    println!("Project: {}", project_display);

    let dev_type_str = format!("{:?}", config.dev_type).to_string();
    println!(
        "Environment: {} Development",
        if config.minimal_output {
            dev_type_str
        } else {
            dev_type_str.bright_green().to_string()
        }
    );

    if !config.framework.is_empty() {
        let framework_display = if config.minimal_output {
            config.framework.clone()
        } else {
            config.framework.bright_blue().to_string()
        };
        println!("Framework: {}", framework_display);
    }

    println!();

    for i in 0..=100 {
        pb.set_position(i);

        if i % 20 == 0 {
            let message = match i {
                0 => "Loading configuration files...",
                20 => "Establishing secure connections...",
                40 => "Initializing development modules...",
                60 => "Syncing with repository...",
                80 => "Analyzing code dependencies...",
                100 => "Environment ready!",
                _ => "",
            };

            if !message.is_empty() {
                pb.println(format!("  {}", message));
            }
        }

        thread::sleep(Duration::from_millis(rng().random_range(50..100)));
    }

    pb.finish_and_clear();
    println!(
        "\n{}\n",
        "✅ DEVELOPMENT ENVIRONMENT INITIALIZED"
            .bold()
            .bright_green()
    );
    thread::sleep(Duration::from_millis(500));
}

pub fn display_random_alert(config: &SessionConfig) {
    let alert_types = [
        "SECURITY",
        "PERFORMANCE",
        "RESOURCE",
        "DEPLOYMENT",
        "COMPLIANCE",
    ];

    let alert_type = alert_types.choose(&mut rng()).unwrap();
    let severity = if rng().random_ratio(1, 4) {
        "CRITICAL"
    } else if rng().random_ratio(1, 3) {
        "HIGH"
    } else {
        "MEDIUM"
    };

    let alert_message = match *alert_type {
        "SECURITY" => match config.dev_type {
            DevelopmentType::Security => {
                "Potential intrusion attempt detected on production server"
            }
            DevelopmentType::Backend => "API authentication token expiration approaching",
            DevelopmentType::Frontend => {
                "Cross-site scripting vulnerability detected in form input"
            }
            DevelopmentType::Blockchain => {
                "Smart contract privilege escalation vulnerability detected"
            }
            _ => "Unusual login pattern detected in production environment",
        },
        "PERFORMANCE" => match config.dev_type {
            DevelopmentType::Backend => {
                "API response time degradation detected in payment endpoint"
            }
            DevelopmentType::Frontend => "Rendering performance issue detected in main dashboard",
            DevelopmentType::DataScience => "Data processing pipeline throughput reduced by 25%",
            DevelopmentType::MachineLearning => "Model inference latency exceeding threshold",
            _ => "Performance regression detected in latest deployment",
        },
        "RESOURCE" => match config.dev_type {
            DevelopmentType::DevOps => "Kubernetes cluster resource allocation approaching limit",
            DevelopmentType::Backend => "Database connection pool nearing capacity",
            DevelopmentType::DataScience => "Data processing job memory usage exceeding allocation",
            _ => "System resource utilization approaching threshold",
        },
        "DEPLOYMENT" => match config.dev_type {
            DevelopmentType::DevOps => "Canary deployment showing increased error rate",
            DevelopmentType::Backend => "Service deployment incomplete on 3 nodes",
            DevelopmentType::Frontend => "Asset optimization failed in production build",
            _ => "CI/CD pipeline failure detected in release branch",
        },
        "COMPLIANCE" => match config.dev_type {
            DevelopmentType::Security => "Potential data handling policy violation detected",
            DevelopmentType::Backend => "API endpoint missing required audit logging",
            DevelopmentType::Blockchain => "Smart contract failing regulatory compliance check",
            _ => "Code scan detected potential compliance issue",
        },
        _ => "System alert condition detected",
    };

    let severity_color = match severity {
        "CRITICAL" => "bright_red",
        "HIGH" => "bright_yellow",
        "MEDIUM" => "bright_cyan",
        _ => "normal",
    };

    let alert_display = format!("🚨 {} ALERT [{}]: {}", alert_type, severity, alert_message);

    if config.minimal_output {
        println!("{}", alert_display);
    } else {
        match severity_color {
            "bright_red" => println!("{}", alert_display.bright_red().bold()),
            "bright_yellow" => println!("{}", alert_display.bright_yellow().bold()),
            "bright_cyan" => println!("{}", alert_display.bright_cyan().bold()),
            _ => println!("{}", alert_display),
        }
    }

    // Show automated response action
    let response_action = match *alert_type {
        "SECURITY" => "Initiating security protocol and notifying security team",
        "PERFORMANCE" => "Analyzing performance metrics and scaling resources",
        "RESOURCE" => "Optimizing resource allocation and preparing scaling plan",
        "DEPLOYMENT" => "Running deployment recovery procedure and notifying DevOps",
        "COMPLIANCE" => "Documenting issue and preparing compliance report",
        _ => "Initiating standard recovery procedure",
    };

    println!("  ↳ AUTOMATED RESPONSE: {}", response_action);
    println!();

    // Pause for dramatic effect
    thread::sleep(Duration::from_millis(1000));
}

pub fn display_team_activity(config: &SessionConfig) {
    let team_names = [
        "Alice", "Bob", "Carlos", "Diana", "Eva", "Felix", "Grace", "Hector", "Irene", "Jack",
    ];
    let team_member = team_names.choose(&mut rng()).unwrap();

    let activities = match config.dev_type {
        DevelopmentType::Backend => [
            "pushed new API endpoint implementation",
            "requested code review on service layer refactoring",
            "merged database optimization pull request",
            "commented on your API authentication PR",
            "resolved 3 high-priority backend bugs",
        ],
        DevelopmentType::Frontend => [
            "updated UI component library",
            "pushed new responsive design implementation",
            "fixed cross-browser compatibility issue",
            "requested review on animation performance PR",
            "updated design system documentation",
        ],
        DevelopmentType::Fullstack => [
            "implemented end-to-end feature integration",
            "fixed client-server sync issue",
            "updated full-stack deployment pipeline",
            "refactored shared validation logic",
            "documented API integration patterns",
        ],
        DevelopmentType::DataScience => [
            "updated data transformation pipeline",
            "shared new analysis notebook",
            "optimized data aggregation queries",
            "updated visualization dashboard",
            "documented new data metrics",
        ],
        DevelopmentType::DevOps => [
            "updated Kubernetes configuration",
            "improved CI/CD pipeline performance",
            "added new monitoring alerts",
            "fixed auto-scaling configuration",
            "updated infrastructure documentation",
        ],
        DevelopmentType::Blockchain => [
            "optimized smart contract gas usage",
            "implemented new transaction validation",
            "updated consensus algorithm implementation",
            "fixed wallet integration issue",
            "documented token economics model",
        ],
        DevelopmentType::MachineLearning => [
            "shared improved model accuracy results",
            "optimized model training pipeline",
            "added new feature extraction method",
            "implemented model versioning system",
            "documented model evaluation metrics",
        ],
        DevelopmentType::SystemsProgramming => [
            "optimized memory allocation strategy",
            "reduced thread contention in core module",
            "implemented lock-free data structure",
            "fixed race condition in scheduler",
            "documented concurrency pattern usage",
        ],
        DevelopmentType::GameDevelopment => [
            "optimized rendering pipeline",
            "fixed physics collision detection issue",
            "implemented new particle effect system",
            "reduced loading time by 30%",
            "documented game engine architecture",
        ],
        DevelopmentType::Security => [
            "implemented additional encryption layer",
            "fixed authentication bypass vulnerability",
            "updated security scanning rules",
            "implemented improved access control",
            "documented security compliance requirements",
        ],
    };

    let activity = activities.choose(&mut rng()).unwrap();
    let minutes_ago = rng().random_range(1..30);
    let notification = format!(
        "👥 TEAM: {} {} ({} minutes ago)",
        team_member, activity, minutes_ago
    );

    println!(
        "{}",
        if config.minimal_output {
            notification
        } else {
            notification.bright_cyan().to_string()
        }
    );

    // Sometimes add a requested action
    if rng().random_ratio(1, 2) {
        let actions = [
            "Review requested on PR #342",
            "Mentioned you in a comment",
            "Assigned ticket DEV-867 to you",
            "Requested your input on design decision",
            "Shared documentation for your review",
        ];

        let action = actions.choose(&mut rng()).unwrap();
        println!("  ↳ ACTION NEEDED: {}", action);
    }

    println!();

    // Short pause to notice the team activity
    thread::sleep(Duration::from_millis(800));
}
