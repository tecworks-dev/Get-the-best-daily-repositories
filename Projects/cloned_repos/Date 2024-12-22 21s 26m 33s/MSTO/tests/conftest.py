"""
Test configuration and fixtures
"""
import pytest
from typing import Dict, Any

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Fixture providing mock configuration."""
    return {
        "ENV": "test",
        "TRADING_MODE": "paper",
        "DROP_LOOKBACK_DAYS": "60",
        "TRADINGVIEW_WEBHOOK_URL": "http://test.com/webhook",
        "NEWS_API_KEY": "test_key",
        "DB_CONNECTION_STRING": "sqlite:///:memory:",
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "json",
        "DEFAULT_MIN_IMPACT_THRESHOLD": "0.3",
        "MAX_PE_RATIO": "30.0",
        "MIN_DROP_THRESHOLD": "-5.0",
        "MAX_POSITION_SIZE": "1000",
        "MIN_POSITION_SIZE": "10",
        "BASE_POSITION_SIZE": "100"
    }

@pytest.fixture
def mock_stock_data():
    """Fixture providing mock stock data."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    prices = np.random.normal(100, 10, len(dates))
    
    return pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": prices * 1.02,
        "Low": prices * 0.98,
        "Close": prices * 1.01,
        "Volume": np.random.randint(1000, 100000, len(dates))
    }).set_index("Date") 