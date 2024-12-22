from msto.strategies.base import Strategy
from typing import Any, Dict, List, Optional

class QARPStrategy(Strategy):
    """
    Quality At a Reasonable Price:
    Buys if the stock is fundamentally sound and has dropped enough to present a discount.
    Checks for reasonable P/E and a decent drop.
    """
    def __init__(self,
                 name: str = "QARP",
                 min_impact_threshold: float = 0.3,
                 max_pe_ratio: float = 20.0,
                 drop_threshold: float = -3.0,
                 position_size: int = 25):
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

        # Conditions:
        # - Fundamentals must be reasonable (pe_ratio < max_pe_ratio)
        # - Price has dropped (drop < drop_threshold)
        if pe_ratio is not None and pe_ratio < self.max_pe_ratio and drop < self.drop_threshold:
            return [self._generate_signal(ticker, "BUY", self.position_size)]
        return None