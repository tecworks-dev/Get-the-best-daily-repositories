import requests
import logging
import json
from config import ENV, TRADING_MODE, TRADINGVIEW_WEBHOOK_URL

class TradingViewIntegration:
    def __init__(self):
        self.env = ENV
        self.mode = TRADING_MODE
        self.webhook_url = TRADINGVIEW_WEBHOOK_URL

    def execute_signals(self, signals):
        for signal in signals:
            payload = {
                "symbol": signal["symbol"],
                "action": signal["action"],
                "quantity": signal["quantity"],
                "env": self.env,
                "mode": self.mode
            }
            self._send_to_tradingview(payload)

    def _send_to_tradingview(self, payload: dict):
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logging.info(json.dumps({"level": "INFO", "message": "Signal sent", "payload": payload}))
        except requests.HTTPError as e:
            logging.error(json.dumps({"level": "ERROR", "message": "Failed to send signal", "error": str(e), "payload": payload}))
