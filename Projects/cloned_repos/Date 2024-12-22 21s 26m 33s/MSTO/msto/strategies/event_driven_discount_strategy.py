from msto.strategies.base import Strategy
from typing import Any, Dict, List, Optional


class EventDrivenDiscountStrategy(Strategy):
    """
    Buys when a negative event drives the price down, provided the event impact is moderate
    and fundamentals haven't changed drastically. The idea: the market overreacted to news
    that doesn't truly damage long-term value.
    """
    def __init__(self,
                 name: str = "EventDrivenDiscount",
                 min_impact_threshold: float = 0.3,
                 drop_threshold: float = -4.0,
                 sentiment_threshold: float = -0.2,
                 position_size: int = 30):
        super().__init__(name, min_impact_threshold)
        self.drop_threshold = drop_threshold
        self.sentiment_threshold = sentiment_threshold
        self.position_size = position_size

    def process_data(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if not self._validate_data(data):
            return None

        ticker = data['ticker']
        drop = data['drop']
        impact = data['impact']
        sentiment = data['avg_sentiment']

        # Conditions:
        # 1. Impact above threshold (some meaningful event)
        # 2. Price dropped significantly (market overreaction)
        # 3. Sentiment not extremely negative (i.e., sentiment_threshold to filter out catastrophes)
        if abs(impact) >= self.min_impact_threshold and drop < self.drop_threshold and sentiment > self.sentiment_threshold:
            return [self._generate_signal(ticker, "BUY", self.position_size)]
        return None