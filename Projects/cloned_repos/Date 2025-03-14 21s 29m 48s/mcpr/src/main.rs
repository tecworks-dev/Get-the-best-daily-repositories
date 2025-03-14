//! MCP CLI tool for generating server and client stubs

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// MCP CLI tool for generating server and client stubs
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a server stub
    GenerateServer {
        /// Name of the server
        #[arg(short, long)]
        name: String,

        /// Output directory
        #[arg(short, long, default_value = ".")]
        output: String,
    },

    /// Generate a client stub
    GenerateClient {
        /// Name of the client
        #[arg(short, long)]
        name: String,

        /// Output directory
        #[arg(short, long, default_value = ".")]
        output: String,
    },

    /// Generate a complete "hello mcp" project with both client and server
    GenerateProject {
        /// Name of the project
        #[arg(short, long)]
        name: String,

        /// Output directory
        #[arg(short, long, default_value = ".")]
        output: String,

        /// Transport type to use (stdio, sse, websocket)
        #[arg(short, long, default_value = "stdio")]
        transport: String,
    },

    /// Run a server
    RunServer {
        /// Path to the server implementation
        #[arg(short, long)]
        path: String,
    },

    /// Connect to a server as a client
    Connect {
        /// URI of the server to connect to
        #[arg(short, long)]
        uri: String,
    },

    /// Validate an MCP message
    Validate {
        /// Path to the message file
        #[arg(short, long)]
        path: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::GenerateServer { name, output } => {
            println!("Generating server stub '{}' in '{}'", name, output);

            let output_path = PathBuf::from(output);
            match mcpr::generator::generate_server(name, &output_path) {
                Ok(_) => {
                    println!("Server stub generated successfully!");
                }
                Err(e) => {
                    eprintln!("Error generating server stub: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::GenerateClient { name, output } => {
            println!("Generating client stub '{}' in '{}'", name, output);

            let output_path = PathBuf::from(output);
            match mcpr::generator::generate_client(name, &output_path) {
                Ok(_) => {
                    println!("Client stub generated successfully!");
                }
                Err(e) => {
                    eprintln!("Error generating client stub: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::GenerateProject {
            name,
            output,
            transport,
        } => {
            println!(
                "Generating complete 'hello mcp' project '{}' in '{}'",
                name, output
            );
            println!("Using transport type: {}", transport);

            let output_path = PathBuf::from(output);
            match mcpr::generator::generate_project(name, &output_path, transport) {
                Ok(_) => {
                    println!("Complete 'hello mcp' project generated successfully!");
                }
                Err(e) => {
                    eprintln!("Error generating complete 'hello mcp' project: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::RunServer { path } => {
            println!("Running server from '{}'", path);
            // TODO: Implement server runner
            println!("Server runner not yet implemented");
        }
        Commands::Connect { uri } => {
            println!("Connecting to server at '{}'", uri);
            // TODO: Implement client connection
            println!("Client connection not yet implemented");
        }
        Commands::Validate { path } => {
            println!("Validating message from '{}'", path);
            // TODO: Implement message validation
            println!("Message validation not yet implemented");
        }
    }
}
