from typing import Any, Dict, List, Optional
from msto.strategies.base import Strategy


class ValueInvestingStrategy(Strategy):
    """
    Buys if the stock is trading at a discount relative to its fundamental value metrics.
    For example, if P/E ratio is low and there's been a recent price drop.
    """
    def __init__(self,
                 name: str = "ValueInvesting",
                 min_impact_threshold: float = 0.3,
                 max_pe_ratio: float = 15.0,
                 drop_threshold: float = -5.0,
                 position_size: int = 50):
        super().__init__(name, min_impact_threshold)
        self.max_pe_ratio = max_pe_ratio
        self.drop_threshold = drop_threshold
        self.position_size = position_size

    def process_data(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if not self._validate_data(data):
            return None

        ticker = data['ticker']
        drop = data['drop']
        fundamentals = data['fundamentals']
        pe_ratio = fundamentals.get('pe_ratio', None)

        # Buy if P/E is below threshold and price has dropped significantly
        if pe_ratio is not None and pe_ratio < self.max_pe_ratio and drop < self.drop_threshold:
            return [self._generate_signal(ticker, "BUY", self.position_size)]
        return None