import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "dev")  # "dev" or "prod"
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"
TRADINGVIEW_WEBHOOK_URL = os.getenv("TRADINGVIEW_WEBHOOK_URL", "https://example.com/tradingview_webhook")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "postgresql://user:password@localhost:5432/mydb")

DROP_LOOKBACK_DAYS = int(os.getenv("DROP_LOOKBACK_DAYS", 60))
