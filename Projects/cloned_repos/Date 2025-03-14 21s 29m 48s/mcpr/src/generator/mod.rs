//! Module for generating MCP server and client stubs

use std::fs;
use std::io;
use std::path::Path;

mod templates;

/// Error type for generator operations
#[derive(Debug, thiserror::Error)]
pub enum GeneratorError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Template error: {0}")]
    Template(String),
    #[error("Invalid name: {0}")]
    InvalidName(String),
}

/// Generate a server stub
pub fn generate_server(name: &str, output_dir: &Path) -> Result<(), GeneratorError> {
    // Validate name
    if !is_valid_name(name) {
        return Err(GeneratorError::InvalidName(format!(
            "Invalid server name: {}",
            name
        )));
    }

    // Create output directory if it doesn't exist
    let server_dir = output_dir.join(name);
    fs::create_dir_all(&server_dir)?;

    // Generate server files
    generate_server_main(&server_dir, name)?;
    generate_server_cargo_toml(&server_dir, name)?;
    generate_server_readme(&server_dir, name)?;

    println!(
        "Server stub '{}' generated successfully in '{}'",
        name,
        server_dir.display()
    );
    println!("To run the server:");
    println!("  cd {}", server_dir.display());
    println!("  cargo run");

    Ok(())
}

/// Generate a client stub
pub fn generate_client(name: &str, output_dir: &Path) -> Result<(), GeneratorError> {
    // Validate name
    if !is_valid_name(name) {
        return Err(GeneratorError::InvalidName(format!(
            "Invalid client name: {}",
            name
        )));
    }

    // Create output directory if it doesn't exist
    let client_dir = output_dir.join(name);
    fs::create_dir_all(&client_dir)?;

    // Generate client files
    generate_client_main(&client_dir, name)?;
    generate_client_cargo_toml(&client_dir, name)?;
    generate_client_readme(&client_dir, name)?;

    println!(
        "Client stub '{}' generated successfully in '{}'",
        name,
        client_dir.display()
    );
    println!("To run the client:");
    println!("  cd {}", client_dir.display());
    println!("  cargo run -- --uri <server_uri>");

    Ok(())
}

/// Generate a complete "hello mcp" project with both client and server
pub fn generate_project(
    name: &str,
    output_dir: &Path,
    transport_type: &str,
) -> Result<(), GeneratorError> {
    // Validate name
    if !is_valid_name(name) {
        return Err(GeneratorError::InvalidName(format!(
            "Invalid project name: {}",
            name
        )));
    }

    // Validate transport type
    if !["stdio", "sse", "websocket"].contains(&transport_type) {
        return Err(GeneratorError::InvalidName(format!(
            "Invalid transport type: {}. Must be one of: stdio, sse, websocket",
            transport_type
        )));
    }

    // Create output directory if it doesn't exist
    let project_dir = output_dir.join(name);
    fs::create_dir_all(&project_dir)?;

    // Generate project structure
    let server_dir = project_dir.join("server");
    let client_dir = project_dir.join("client");

    fs::create_dir_all(&server_dir)?;
    fs::create_dir_all(&client_dir)?;

    // Generate server files
    generate_project_server(&server_dir, name, transport_type)?;

    // Generate client files
    generate_project_client(&client_dir, name, transport_type)?;

    // Generate project README
    generate_project_readme(&project_dir, name, transport_type)?;

    // Generate test scripts
    generate_project_test_script(&project_dir, name, transport_type)?;

    println!(
        "Project '{}' generated successfully in '{}'",
        name,
        project_dir.display()
    );
    println!("To run the tests:");
    println!("  cd {}", project_dir.display());
    println!("  ./run_tests.sh    # Run all tests");
    println!("  ./test_server.sh  # Run server tests only");
    println!("  ./test_client.sh  # Run client tests only");
    println!("  ./test.sh         # Run legacy test");

    Ok(())
}

// Helper functions

fn is_valid_name(name: &str) -> bool {
    // Check if name is a valid Rust crate name
    if name.is_empty() {
        return false;
    }

    // Must start with a letter
    if !name.chars().next().unwrap().is_ascii_alphabetic() {
        return false;
    }

    // Can contain letters, numbers, underscores, and hyphens
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn generate_server_main(server_dir: &Path, name: &str) -> Result<(), GeneratorError> {
    let main_rs = server_dir.join("src").join("main.rs");
    fs::create_dir_all(main_rs.parent().unwrap())?;

    let content = templates::SERVER_MAIN_TEMPLATE.replace("{{name}}", name);

    fs::write(main_rs, content)?;
    Ok(())
}

fn generate_server_cargo_toml(server_dir: &Path, name: &str) -> Result<(), GeneratorError> {
    let cargo_toml = server_dir.join("Cargo.toml");

    let content = templates::SERVER_CARGO_TEMPLATE.replace("{{name}}", name);

    fs::write(cargo_toml, content)?;
    Ok(())
}

fn generate_server_readme(server_dir: &Path, name: &str) -> Result<(), GeneratorError> {
    let readme = server_dir.join("README.md");

    let content = templates::SERVER_README_TEMPLATE.replace("{{name}}", name);

    fs::write(readme, content)?;
    Ok(())
}

fn generate_client_main(client_dir: &Path, name: &str) -> Result<(), GeneratorError> {
    let main_rs = client_dir.join("src").join("main.rs");
    fs::create_dir_all(main_rs.parent().unwrap())?;

    let content = templates::CLIENT_MAIN_TEMPLATE.replace("{{name}}", name);

    fs::write(main_rs, content)?;
    Ok(())
}

fn generate_client_cargo_toml(client_dir: &Path, name: &str) -> Result<(), GeneratorError> {
    let cargo_toml = client_dir.join("Cargo.toml");

    let content = templates::CLIENT_CARGO_TEMPLATE.replace("{{name}}", name);

    fs::write(cargo_toml, content)?;
    Ok(())
}

fn generate_client_readme(client_dir: &Path, name: &str) -> Result<(), GeneratorError> {
    let readme = client_dir.join("README.md");

    let content = templates::CLIENT_README_TEMPLATE.replace("{{name}}", name);

    fs::write(readme, content)?;
    Ok(())
}

fn generate_project_server(
    server_dir: &Path,
    name: &str,
    transport_type: &str,
) -> Result<(), GeneratorError> {
    // Create src directory
    let src_dir = server_dir.join("src");
    fs::create_dir_all(&src_dir)?;

    // Generate main.rs
    let main_rs = src_dir.join("main.rs");

    // Read the template
    let template = templates::PROJECT_SERVER_TEMPLATE;

    // Process the template based on transport type
    let mut lines: Vec<String> = Vec::new();
    let mut skip_section = false;

    for line in template.lines() {
        if line.trim().starts_with("{{#if transport_type ==") {
            let condition = line.trim();
            if condition.contains(&format!("\"{}\"", transport_type)) {
                skip_section = false;
            } else {
                skip_section = true;
            }
            continue;
        } else if line.trim().starts_with("{{#if transport_type !=") {
            let condition = line.trim();
            if condition.contains(&format!("\"{}\"", transport_type)) {
                skip_section = true;
            } else {
                skip_section = false;
            }
            continue;
        } else if line.trim() == "{{/if}}" {
            skip_section = false;
            continue;
        }

        if !skip_section {
            lines.push(line.replace("{{name}}", name));
        }
    }

    // Write the processed template to the file
    let content = lines.join("\n");
    fs::write(main_rs, content)?;

    // Generate Cargo.toml
    let cargo_toml = server_dir.join("Cargo.toml");
    let content = templates::PROJECT_SERVER_CARGO_TEMPLATE
        .replace("{{name}}", &format!("{}-server", name))
        .replace(
            "{{transport_deps}}",
            match transport_type {
                "sse" => "reqwest = { version = \"0.11\", features = [\"blocking\"] }",
                "websocket" => "tungstenite = \"0.20\"",
                _ => "",
            },
        );
    fs::write(cargo_toml, content)?;

    Ok(())
}

fn generate_project_client(
    client_dir: &Path,
    name: &str,
    transport_type: &str,
) -> Result<(), GeneratorError> {
    // Create src directory
    let src_dir = client_dir.join("src");
    fs::create_dir_all(&src_dir)?;

    // Generate main.rs
    let main_rs = src_dir.join("main.rs");

    // Read the template
    let template = templates::PROJECT_CLIENT_TEMPLATE;

    // Process the template based on transport type
    let mut lines: Vec<String> = Vec::new();
    let mut skip_section = false;

    for line in template.lines() {
        if line.trim().starts_with("{{#if transport_type ==") {
            let condition = line.trim();
            if condition.contains(&format!("\"{}\"", transport_type)) {
                skip_section = false;
            } else {
                skip_section = true;
            }
            continue;
        } else if line.trim().starts_with("{{#if transport_type !=") {
            let condition = line.trim();
            if condition.contains(&format!("\"{}\"", transport_type)) {
                skip_section = true;
            } else {
                skip_section = false;
            }
            continue;
        } else if line.trim() == "{{/if}}" {
            skip_section = false;
            continue;
        }

        if !skip_section {
            lines.push(line.replace("{{name}}", name));
        }
    }

    // Write the processed template to the file
    let content = lines.join("\n");
    fs::write(main_rs, content)?;

    // Generate Cargo.toml
    let cargo_toml = client_dir.join("Cargo.toml");
    let content = templates::PROJECT_CLIENT_CARGO_TEMPLATE
        .replace("{{name}}", &format!("{}-client", name))
        .replace(
            "{{transport_deps}}",
            match transport_type {
                "sse" => "reqwest = { version = \"0.11\", features = [\"blocking\"] }",
                "websocket" => "tungstenite = \"0.20\"",
                _ => "",
            },
        );
    fs::write(cargo_toml, content)?;

    Ok(())
}

fn generate_project_readme(
    project_dir: &Path,
    name: &str,
    transport_type: &str,
) -> Result<(), GeneratorError> {
    let readme = project_dir.join("README.md");
    let content = templates::PROJECT_README_TEMPLATE
        .replace("{{name}}", name)
        .replace("{{transport_type}}", transport_type);
    fs::write(readme, content)?;

    Ok(())
}

fn generate_project_test_script(
    project_dir: &Path,
    name: &str,
    transport_type: &str,
) -> Result<(), GeneratorError> {
    // Generate server test script
    let server_test_script = project_dir.join("test_server.sh");
    let server_test_content = process_template(
        templates::PROJECT_SERVER_TEST_TEMPLATE,
        name,
        transport_type,
    )?;
    fs::write(&server_test_script, server_test_content)?;
    make_executable(&server_test_script)?;

    // Generate client test script
    let client_test_script = project_dir.join("test_client.sh");
    let client_test_content = process_template(
        templates::PROJECT_CLIENT_TEST_TEMPLATE,
        name,
        transport_type,
    )?;
    fs::write(&client_test_script, client_test_content)?;
    make_executable(&client_test_script)?;

    // Generate combined run_tests.sh script
    let run_tests_script = project_dir.join("run_tests.sh");
    let run_tests_content =
        process_template(templates::PROJECT_RUN_TESTS_TEMPLATE, name, transport_type)?;
    fs::write(&run_tests_script, run_tests_content)?;
    make_executable(&run_tests_script)?;

    // For backward compatibility, also generate the original test.sh
    let test_script = project_dir.join("test.sh");
    let test_content = process_template(
        templates::PROJECT_TEST_SCRIPT_TEMPLATE,
        name,
        transport_type,
    )?;
    fs::write(&test_script, test_content)?;
    make_executable(&test_script)?;

    Ok(())
}

/// Process a template by replacing variables and handling conditional sections
fn process_template(
    template: &str,
    name: &str,
    transport_type: &str,
) -> Result<String, GeneratorError> {
    let mut lines: Vec<String> = Vec::new();
    let mut skip_section = false;

    for line in template.lines() {
        if line.trim().starts_with("{{#if transport_type ==") {
            let condition = line.trim();
            if condition.contains(&format!("\"{}\"", transport_type)) {
                skip_section = false;
            } else {
                skip_section = true;
            }
            continue;
        } else if line.trim().starts_with("{{#if transport_type !=") {
            let condition = line.trim();
            if condition.contains(&format!("\"{}\"", transport_type)) {
                skip_section = true;
            } else {
                skip_section = false;
            }
            continue;
        } else if line.trim() == "{{/if}}" {
            skip_section = false;
            continue;
        }

        if !skip_section {
            lines.push(line.replace("{{name}}", name));
        }
    }

    Ok(lines.join("\n"))
}

/// Make a file executable on Unix systems
fn make_executable(file_path: &Path) -> Result<(), GeneratorError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(file_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(file_path, perms)?;
    }

    Ok(())
}
