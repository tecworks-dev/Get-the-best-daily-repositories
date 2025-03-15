use crate::config::SessionConfig;
use crate::generators::{
    code_analyzer, data_processing, jargon, metrics, network_activity, system_monitoring,
};
use crate::types::{DevelopmentType, JargonLevel};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{prelude::*, rng};
use std::{thread, time::Duration};

pub fn run_code_analysis(config: &SessionConfig) {
    let files_to_analyze = rng().random_range(5..25);
    let total_lines = rng().random_range(1000..10000);

    let framework_specific = if !config.framework.is_empty() {
        format!(" ({} specific)", config.framework)
    } else {
        String::new()
    };

    let title = match config.dev_type {
        DevelopmentType::Backend => format!(
            "🔍 Running Code Analysis on API Components{}",
            framework_specific
        ),
        DevelopmentType::Frontend => format!("🔍 Analyzing UI Components{}", framework_specific),
        DevelopmentType::Fullstack => "🔍 Analyzing Full-Stack Integration Points".to_string(),
        DevelopmentType::DataScience => "🔍 Analyzing Data Pipeline Components".to_string(),
        DevelopmentType::DevOps => "🔍 Analyzing Infrastructure Configuration".to_string(),
        DevelopmentType::Blockchain => "🔍 Analyzing Smart Contract Security".to_string(),
        DevelopmentType::MachineLearning => "🔍 Analyzing Model Prediction Accuracy".to_string(),
        DevelopmentType::SystemsProgramming => "🔍 Analyzing Memory Safety Patterns".to_string(),
        DevelopmentType::GameDevelopment => "🔍 Analyzing Game Physics Components".to_string(),
        DevelopmentType::Security => "🔍 Running Security Vulnerability Scan".to_string(),
    };

    println!(
        "{}",
        if config.minimal_output {
            title.clone()
        } else {
            title.bold().bright_blue().to_string()
        }
    );

    let pb = ProgressBar::new(files_to_analyze);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({eta})")
        .unwrap()
        .progress_chars("▰▰▱"));

    for i in 0..files_to_analyze {
        pb.set_position(i);

        if rng().random_ratio(1, 3) {
            let file_name = code_analyzer::generate_filename(config.dev_type);
            let issue_type = code_analyzer::generate_code_issue(config.dev_type);
            let complexity = code_analyzer::generate_complexity_metric();

            let message = if rng().random_ratio(1, 4) {
                format!("  ⚠️ {} - {}: {}", file_name, issue_type, complexity)
            } else {
                format!("  ✓ {} - {}", file_name, complexity)
            };

            pb.println(message);
        }

        thread::sleep(Duration::from_millis(rng().random_range(100..300)));
    }

    pb.finish();

    // Final analysis summary
    let issues_found = rng().random_range(0..5);
    let code_quality = rng().random_range(85..99);
    let tech_debt = rng().random_range(1..15);

    println!(
        "📊 Analysis Complete: {} files, {} lines of code",
        files_to_analyze, total_lines
    );
    println!("  - Issues found: {}", issues_found);
    println!("  - Code quality score: {}%", code_quality);
    println!("  - Technical debt: {}%", tech_debt);

    if config.jargon_level >= JargonLevel::Medium {
        println!(
            "  - {}",
            jargon::generate_code_jargon(config.dev_type, config.jargon_level)
        );
    }

    println!();
}

pub fn run_performance_metrics(config: &SessionConfig) {
    let title = match config.dev_type {
        DevelopmentType::Backend => "⚡ Analyzing API Response Time".to_string(),
        DevelopmentType::Frontend => "⚡ Measuring UI Rendering Performance".to_string(),
        DevelopmentType::Fullstack => "⚡ Evaluating End-to-End Performance".to_string(),
        DevelopmentType::DataScience => "⚡ Benchmarking Data Processing Pipeline".to_string(),
        DevelopmentType::DevOps => "⚡ Evaluating Infrastructure Scalability".to_string(),
        DevelopmentType::Blockchain => "⚡ Measuring Transaction Throughput".to_string(),
        DevelopmentType::MachineLearning => "⚡ Benchmarking Model Training Speed".to_string(),
        DevelopmentType::SystemsProgramming => {
            "⚡ Measuring Memory Allocation Efficiency".to_string()
        }
        DevelopmentType::GameDevelopment => "⚡ Analyzing Frame Rate Optimization".to_string(),
        DevelopmentType::Security => "⚡ Benchmarking Encryption Performance".to_string(),
    };

    println!(
        "{}",
        if config.minimal_output {
            title.clone()
        } else {
            title.bold().bright_yellow().to_string()
        }
    );

    let iterations = rng().random_range(50..200);
    let pb = ProgressBar::new(iterations);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.yellow} [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} samples ({eta})")
        .unwrap()
        .progress_chars("▰▱▱"));

    let mut performance_data: Vec<f64> = Vec::new();

    for i in 0..iterations {
        pb.set_position(i);

        // Generate realistic-looking performance numbers
        let base_perf = match config.dev_type {
            DevelopmentType::Backend => rng().random_range(20.0..80.0), // ms
            DevelopmentType::Frontend => rng().random_range(5.0..30.0), // ms
            DevelopmentType::DataScience => rng().random_range(100.0..500.0), // ms
            DevelopmentType::Blockchain => rng().random_range(200.0..800.0), // ms
            DevelopmentType::MachineLearning => rng().random_range(300.0..900.0), // ms
            _ => rng().random_range(10.0..100.0),                       // ms
        };

        // Add some variation but keep it somewhat consistent
        let jitter = rng().random_range(-5.0..5.0);
        let perf_value = f64::max(base_perf + jitter, 1.0);
        performance_data.push(perf_value);

        if i % 10 == 0 && rng().random_ratio(1, 3) {
            let metric_name = metrics::generate_performance_metric(config.dev_type);
            let metric_value = rng().random_range(10..999);
            let metric_unit = metrics::generate_metric_unit(config.dev_type);

            pb.println(format!(
                "  📊 {}: {} {}",
                metric_name, metric_value, metric_unit
            ));
        }

        thread::sleep(Duration::from_millis(rng().random_range(50..100)));
    }

    pb.finish();

    // Calculate and display metrics
    performance_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = performance_data.iter().sum::<f64>() / performance_data.len() as f64;
    let median = performance_data[performance_data.len() / 2];
    let p95 = performance_data[(performance_data.len() as f64 * 0.95) as usize];
    let p99 = performance_data[(performance_data.len() as f64 * 0.99) as usize];

    let unit = match config.dev_type {
        DevelopmentType::DataScience | DevelopmentType::MachineLearning => "seconds",
        _ => "milliseconds",
    };

    println!("📈 Performance Results:");
    println!("  - Average: {:.2} {}", avg, unit);
    println!("  - Median: {:.2} {}", median, unit);
    println!("  - P95: {:.2} {}", p95, unit);
    println!("  - P99: {:.2} {}", p99, unit);

    // Add optimization recommendations based on dev type
    let rec = metrics::generate_optimization_recommendation(config.dev_type);
    println!("💡 Recommendation: {}", rec);

    if config.jargon_level >= JargonLevel::Medium {
        println!(
            "  - {}",
            jargon::generate_performance_jargon(config.dev_type, config.jargon_level)
        );
    }

    println!();
}

pub fn run_system_monitoring(config: &SessionConfig) {
    let title = "🖥️ System Resource Monitoring".to_string();
    println!(
        "{}",
        if config.minimal_output {
            title.clone()
        } else {
            title.bold().bright_green().to_string()
        }
    );

    let duration = rng().random_range(5..15);
    let pb = ProgressBar::new(duration);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} seconds",
            )
            .unwrap()
            .progress_chars("▰▱▱"),
    );

    let cpu_base = rng().random_range(10..60);
    let memory_base = rng().random_range(30..70);
    let network_base = rng().random_range(1..20);
    let disk_base = rng().random_range(5..40);

    for i in 0..duration {
        pb.set_position(i);

        // Generate slightly varied metrics for realistic fluctuation
        let cpu = cpu_base + rng().random_range(-5..10);
        let memory = memory_base + rng().random_range(-3..5);
        let network = network_base + rng().random_range(-1..3);
        let disk = disk_base + rng().random_range(-2..4);

        let processes = rng().random_range(80..200);

        let cpu_str = if cpu > 80 {
            format!("{}% (!)", cpu).red().to_string()
        } else if cpu > 60 {
            format!("{}% (!)", cpu).yellow().to_string()
        } else {
            format!("{}%", cpu).normal().to_string()
        };

        let mem_str = if memory > 85 {
            format!("{}%", memory).red().to_string()
        } else if memory > 70 {
            format!("{}%", memory).yellow().to_string()
        } else {
            format!("{}%", memory).normal().to_string()
        };

        let stats = format!(
            "  CPU: {}  |  RAM: {}  |  Network: {} MB/s  |  Disk I/O: {} MB/s  |  Processes: {}",
            cpu_str, mem_str, network, disk, processes
        );

        pb.println(if config.minimal_output {
            stats.clone()
        } else {
            stats
        });

        if i % 3 == 0 && rng().random_ratio(1, 3) {
            let system_event = system_monitoring::generate_system_event();
            pb.println(format!("  🔄 {}", system_event));
        }

        thread::sleep(Duration::from_millis(rng().random_range(200..500)));
    }

    pb.finish();

    // Display summary
    println!("📊 Resource Utilization Summary:");
    println!("  - Peak CPU: {}%", cpu_base + rng().random_range(5..15));
    println!(
        "  - Peak Memory: {}%",
        memory_base + rng().random_range(5..15)
    );
    println!(
        "  - Network Throughput: {} MB/s",
        network_base + rng().random_range(5..10)
    );
    println!(
        "  - Disk Throughput: {} MB/s",
        disk_base + rng().random_range(2..8)
    );
    println!(
        "  - {}",
        system_monitoring::generate_system_recommendation()
    );
    println!();
}

pub fn run_data_processing(config: &SessionConfig) {
    let operations = rng().random_range(5..20);

    let title = match config.dev_type {
        DevelopmentType::Backend => "🔄 Processing API Data Streams".to_string(),
        DevelopmentType::Frontend => "🔄 Processing User Interaction Data".to_string(),
        DevelopmentType::Fullstack => "🔄 Synchronizing Client-Server Data".to_string(),
        DevelopmentType::DataScience => "🔄 Running Data Transformation Pipeline".to_string(),
        DevelopmentType::DevOps => "🔄 Analyzing System Logs".to_string(),
        DevelopmentType::Blockchain => "🔄 Validating Transaction Blocks".to_string(),
        DevelopmentType::MachineLearning => "🔄 Processing Training Data Batches".to_string(),
        DevelopmentType::SystemsProgramming => "🔄 Optimizing Memory Access Patterns".to_string(),
        DevelopmentType::GameDevelopment => "🔄 Processing Game Asset Pipeline".to_string(),
        DevelopmentType::Security => "🔄 Analyzing Security Event Logs".to_string(),
    };

    println!(
        "{}",
        if config.minimal_output {
            title.clone()
        } else {
            title.bold().bright_cyan().to_string()
        }
    );

    for _ in 0..operations {
        let operation = data_processing::generate_data_operation(config.dev_type);
        let records = rng().random_range(100..10000);
        let size = rng().random_range(1..100);
        let size_unit = if rng().random_ratio(1, 4) { "GB" } else { "MB" };

        println!(
            "  🔄 {} {} records ({} {})",
            operation, records, size, size_unit
        );

        // Sometimes add sub-tasks with progress bars
        if rng().random_ratio(1, 3) {
            let subtasks = rng().random_range(10..30);
            let pb = ProgressBar::new(subtasks);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "     {spinner:.blue} [{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len}",
                    )
                    .unwrap()
                    .progress_chars("▰▱▱"),
            );

            for i in 0..subtasks {
                pb.set_position(i);
                thread::sleep(Duration::from_millis(rng().random_range(20..100)));

                if rng().random_ratio(1, 8) {
                    let sub_operation =
                        data_processing::generate_data_sub_operation(config.dev_type);
                    pb.println(format!("       - {}", sub_operation));
                }
            }

            pb.finish_and_clear();
        } else {
            thread::sleep(Duration::from_millis(rng().random_range(300..800)));
        }

        // Add some details about the operation
        if rng().random_ratio(1, 2) {
            let details = data_processing::generate_data_details(config.dev_type);
            println!("     ✓ {}", details);
        }
    }

    // Add a summary
    let processed_records = rng().random_range(10000..1000000);
    let processing_rate = rng().random_range(1000..10000);
    let total_size = rng().random_range(10..500);
    let time_saved = rng().random_range(10..60);

    println!("📊 Data Processing Summary:");
    println!("  - Records processed: {}", processed_records);
    println!("  - Processing rate: {} records/sec", processing_rate);
    println!("  - Total data size: {} GB", total_size);
    println!("  - Estimated time saved: {} minutes", time_saved);

    if config.jargon_level >= JargonLevel::Medium {
        println!(
            "  - {}",
            jargon::generate_data_jargon(config.dev_type, config.jargon_level)
        );
    }

    println!();
}

pub fn run_network_activity(config: &SessionConfig) {
    let title = match config.dev_type {
        DevelopmentType::Backend => "🌐 Monitoring API Network Traffic".to_string(),
        DevelopmentType::Frontend => "🌐 Analyzing Client-Side Network Requests".to_string(),
        DevelopmentType::Fullstack => "🌐 Optimizing Client-Server Communication".to_string(),
        DevelopmentType::DataScience => "🌐 Synchronizing Distributed Data Nodes".to_string(),
        DevelopmentType::DevOps => "🌐 Monitoring Infrastructure Network".to_string(),
        DevelopmentType::Blockchain => "🌐 Monitoring Blockchain Network".to_string(),
        DevelopmentType::MachineLearning => "🌐 Distributing Model Training".to_string(),
        DevelopmentType::SystemsProgramming => {
            "🌐 Analyzing Network Protocol Efficiency".to_string()
        }
        DevelopmentType::GameDevelopment => {
            "🌐 Simulating Multiplayer Network Conditions".to_string()
        }
        DevelopmentType::Security => "🌐 Analyzing Network Security Patterns".to_string(),
    };

    println!(
        "{}",
        if config.minimal_output {
            title.clone()
        } else {
            title.bold().bright_magenta().to_string()
        }
    );

    let requests = rng().random_range(5..15);

    for _ in 0..requests {
        let endpoint = network_activity::generate_endpoint(config.dev_type);
        let method = network_activity::generate_method();
        let status = network_activity::generate_status();
        let size = rng().random_range(1..1000);
        let time = rng().random_range(10..500);

        let method_colored = match method.as_str() {
            "GET" => method.green(),
            "POST" => method.blue(),
            "PUT" => method.yellow(),
            "DELETE" => method.red(),
            _ => method.normal(),
        };

        let status_colored = if (200..300).contains(&status) {
            status.to_string().green()
        } else if (300..400).contains(&status) {
            status.to_string().yellow()
        } else {
            status.to_string().red()
        };

        let request_line = format!(
            "  {} {} → {} | {} ms | {} KB",
            if config.minimal_output {
                method.to_string()
            } else {
                method_colored.to_string()
            },
            endpoint,
            if config.minimal_output {
                status.to_string()
            } else {
                status_colored.to_string()
            },
            time,
            size
        );

        println!("{}", request_line);

        // Sometimes add request details
        if rng().random_ratio(1, 3) {
            let details = network_activity::generate_request_details(config.dev_type);
            println!("     ↳ {}", details);
        }

        thread::sleep(Duration::from_millis(rng().random_range(100..400)));
    }

    // Add summary
    let total_requests = rng().random_range(1000..10000);
    let avg_response = rng().random_range(50..200);
    let success_rate = rng().random_range(95..100);
    let bandwidth = rng().random_range(10..100);

    println!("📊 Network Activity Summary:");
    println!("  - Total requests: {}", total_requests);
    println!("  - Average response time: {} ms", avg_response);
    println!("  - Success rate: {}%", success_rate);
    println!("  - Bandwidth utilization: {} MB/s", bandwidth);

    if config.jargon_level >= JargonLevel::Medium {
        println!(
            "  - {}",
            jargon::generate_network_jargon(config.dev_type, config.jargon_level)
        );
    }

    println!();
}
