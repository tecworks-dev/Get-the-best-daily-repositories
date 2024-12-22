"""
Configuration utilities for MSTO
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

def load_config(config_path: str = ".env") -> Dict[str, Any]:
    """
    Load configuration from environment file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration values
    """
    # Load environment variables from file
    load_dotenv(config_path)
    
    # Required configuration keys and their default values
    config_keys = {
        "ENV": "dev",
        "TRADING_MODE": "paper",
        "DROP_LOOKBACK_DAYS": "60",
        "TRADINGVIEW_WEBHOOK_URL": None,
        "NEWS_API_KEY": None,
        "DB_CONNECTION_STRING": None,
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "json",
        "DEFAULT_MIN_IMPACT_THRESHOLD": "0.3",
        "MAX_PE_RATIO": "30.0",
        "MIN_DROP_THRESHOLD": "-5.0",
        "MAX_POSITION_SIZE": "1000",
        "MIN_POSITION_SIZE": "10",
        "BASE_POSITION_SIZE": "100"
    }
    
    # Build configuration dictionary
    config = {}
    for key, default in config_keys.items():
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Required configuration key '{key}' not found in {config_path}")
        config[key] = value
    
    return config 