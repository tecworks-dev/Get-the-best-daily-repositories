use clap::Parser;
use colored::*;
use console::Term;
use rand::prelude::*;
use rand::rng;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

mod activities;
mod config;
mod display;
mod generators;
mod types;
use types::{Complexity, DevelopmentType, JargonLevel};

/// A CLI tool that generates impressive-looking terminal output when stakeholders walk by
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Type of development activity to simulate
    #[arg(short, long, value_enum, default_value_t = DevelopmentType::Backend)]
    dev_type: DevelopmentType,

    /// Level of technical jargon in output
    #[arg(short, long, value_enum, default_value_t = JargonLevel::Medium)]
    jargon: JargonLevel,

    /// How busy and complex the output should appear
    #[arg(short, long, value_enum, default_value_t = Complexity::Medium)]
    complexity: Complexity,

    /// Duration in seconds to run (0 = run until interrupted)
    #[arg(short = 'T', long, default_value_t = 0)]
    duration: u64,

    /// Show critical system alerts or issues
    #[arg(short, long, default_value_t = false)]
    alerts: bool,

    /// Simulate a specific project
    #[arg(short, long, default_value = "distributed-cluster")]
    project: String,

    /// Use less colorful output
    #[arg(long, default_value_t = false)]
    minimal: bool,

    /// Show team collaboration activity
    #[arg(short, long, default_value_t = false)]
    team: bool,

    /// Simulate a specific framework usage
    #[arg(short = 'F', long, default_value = "")]
    framework: String,
}

fn main() {
    let args = Args::parse();

    let config = config::SessionConfig {
        dev_type: args.dev_type,
        jargon_level: args.jargon,
        complexity: args.complexity,
        alerts_enabled: args.alerts,
        project_name: args.project,
        minimal_output: args.minimal,
        team_activity: args.team,
        framework: args.framework,
    };

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let term = Term::stdout();
    let _ = term.clear_screen();

    // Display an initial "system boot" to set the mood
    display::display_boot_sequence(&config);

    let start_time = Instant::now();
    let target_duration = if args.duration > 0 {
        Some(Duration::from_secs(args.duration))
    } else {
        None
    };

    while running.load(Ordering::SeqCst) {
        if let Some(duration) = target_duration {
            if start_time.elapsed() >= duration {
                break;
            }
        }

        // Based on complexity, determine how many activities to show simultaneously
        let activities_count = match config.complexity {
            Complexity::Low => 1,
            Complexity::Medium => 2,
            Complexity::High => 3,
            Complexity::Extreme => 4,
        };

        // Randomly select and run activities
        let mut activities: Vec<fn(&config::SessionConfig)> = vec![
            activities::run_code_analysis,
            activities::run_performance_metrics,
            activities::run_system_monitoring,
            activities::run_data_processing,
            activities::run_network_activity,
        ];
        activities.shuffle(&mut rng());

        for activity in activities.iter().take(activities_count) {
            activity(&config);

            // Random short pause between activities
            let pause_time = rng().random_range(100..500);
            thread::sleep(Duration::from_millis(pause_time));

            // Check if we should exit
            if !running.load(Ordering::SeqCst)
                || (target_duration.is_some() && start_time.elapsed() >= target_duration.unwrap())
            {
                break;
            }
        }

        if config.alerts_enabled && rng().random_ratio(1, 10) {
            display::display_random_alert(&config);
        }

        if config.team_activity && rng().random_ratio(1, 5) {
            display::display_team_activity(&config);
        }
    }

    let _ = term.clear_screen();
    println!("{}", "Session terminated.".bright_green());
}
