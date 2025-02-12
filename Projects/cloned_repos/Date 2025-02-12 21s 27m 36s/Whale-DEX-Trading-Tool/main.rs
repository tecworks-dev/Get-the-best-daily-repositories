use serde::Deserialize;
use std::fs;
use reqwest::Error;

#[derive(Deserialize, Debug)]
struct Config {
    network: String,
    order_amount: u64,
    max_slippage: f64,
    splits: u64,
    aggregator_api: String,
}

// Asynchronous function to fetch liquidity from the aggregator API
async fn fetch_liquidity(api_url: &str) -> Result<f64, Error> {
    let resp = reqwest::get(api_url).await?;
    // Expecting JSON with a "liquidity" field
    let json: serde_json::Value = resp.json().await?;
    let liquidity = json["liquidity"].as_f64().unwrap_or(0.0);
    Ok(liquidity)
}

// Function to read the configuration file
fn read_config(file_path: &str) -> Config {
    let config_data = fs::read_to_string(file_path)
        .expect("Failed to read the configuration file");
    toml::from_str(&config_data)
        .expect("Failed to parse the configuration file")
}

#[tokio::main]
async fn main() {
    let config = read_config("config.toml");
    println!("Network: {}", config.network);
    println!("Order Amount: ${}", config.order_amount);
    println!("Maximum Slippage: {}%", config.max_slippage);
    println!("Order will be split into {} parts", config.splits);

    match fetch_liquidity(&config.aggregator_api).await {
        Ok(liquidity) => {
            println!("Fetched Liquidity: {}", liquidity);
            let split_amount = config.order_amount / config.splits;
            println!("Each order part: ${}", split_amount);
            if liquidity > split_amount as f64 {
                println!("Sufficient liquidity to execute each order part.");
            } else {
                println!("Warning: Insufficient liquidity for the orders!");
            }
        },
        Err(e) => println!("Error fetching liquidity: {:?}", e),
    }
}
