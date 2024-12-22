"""
Simple volatility-based trading strategy.
"""
from typing import List, Dict, Any, Optional
import logging
import json
from msto.strategies.base import Strategy

logger = logging.getLogger(__name__)


class SimpleVolatilityStrategy(Strategy):
    def __init__(self,
                 name: str = "SimpleVolatility",
                 min_impact_threshold: float = 0.3,
                 min_drop_threshold: float = -2.0,
                 min_sentiment_threshold: float = -0.5,
                 position_size: int = 10):
        """
        Initialize the simple volatility strategy.

        Args:
            name: Strategy name
            min_impact_threshold: Minimum impact threshold for signals
            min_drop_threshold: Minimum price drop threshold (negative value)
            min_sentiment_threshold: Minimum sentiment threshold (negative value)
            position_size: Number of shares to trade
        """
        super().__init__(name, min_impact_threshold)
        self.min_drop_threshold = min_drop_threshold
        self.min_sentiment_threshold = min_sentiment_threshold
        self.position_size = position_size
        
        logger.info(json.dumps({
            "level": "INFO",
            "message": "Initialized SimpleVolatilityStrategy",
            "strategy": name,
            "min_impact_threshold": min_impact_threshold,
            "min_drop_threshold": min_drop_threshold,
            "min_sentiment_threshold": min_sentiment_threshold,
            "position_size": position_size
        }))

    def process_data(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Process market data and generate trading signals based on volatility and sentiment.

        Args:
            data: Dictionary containing processed market data including:
                - ticker: str
                - drop: float
                - avg_sentiment: float
                - most_common_event: str
                - fundamentals: dict
                - impact: float

        Returns:
            List of trading signals or None if no signals generated
        """
        ticker = data.get('ticker', 'UNKNOWN')
        
        if not self._validate_data(data):
            logger.warning(json.dumps({
                "level": "WARNING",
                "message": "Invalid data for strategy",
                "strategy": self.name,
                "ticker": ticker
            }))
            return None

        drop = data['drop']
        sentiment = data['avg_sentiment']

        # Log the current market conditions
        logger.debug(json.dumps({
            "level": "DEBUG",
            "message": "Processing market data",
            "strategy": self.name,
            "ticker": ticker,
            "conditions": {
                "price_drop": drop,
                "sentiment": sentiment,
                "drop_threshold": self.min_drop_threshold,
                "sentiment_threshold": self.min_sentiment_threshold
            }
        }))

        # Generate buy signal if we detect a significant drop and negative sentiment
        # The idea is to buy when there's panic selling (contrarian strategy)
        if drop < self.min_drop_threshold and sentiment < self.min_sentiment_threshold:
            signal = self._generate_signal(ticker, "BUY", self.position_size)
            logger.info(json.dumps({
                "level": "INFO",
                "message": "Strategy triggered",
                "strategy": self.name,
                "ticker": ticker,
                "trigger_conditions": {
                    "price_drop": drop,
                    "sentiment": sentiment
                },
                "signal": signal
            }))
            return [signal]
        else:
            logger.debug(json.dumps({
                "level": "DEBUG",
                "message": "Strategy not triggered",
                "strategy": self.name,
                "ticker": ticker,
                "reason": (
                    "Price drop not significant"
                    if drop >= self.min_drop_threshold
                    else "Sentiment not negative enough"
                    if sentiment >= self.min_sentiment_threshold
                    else "Unknown"
                )
            }))
            return None 