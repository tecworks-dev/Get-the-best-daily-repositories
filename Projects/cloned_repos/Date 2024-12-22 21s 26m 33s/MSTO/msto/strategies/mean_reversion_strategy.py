from msto.strategies.base import Strategy
from typing import Any, Dict, List, Optional

class MeanReversionStrategy(Strategy):
    """
    Buys if the stock has deviated significantly from its 'mean'—represented here by a large drop—
    assuming it will revert back.
    """
    def __init__(self,
                 name: str = "MeanReversion",
                 min_impact_threshold: float = 0.3,
                 drop_threshold: float = -3.0,
                 position_size: int = 20):
        super().__init__(name, min_impact_threshold)
        self.drop_threshold = drop_threshold
        self.position_size = position_size

    def process_data(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if not self._validate_data(data):
            return None
        
        ticker = data['ticker']
        drop = data['drop']

        # If the price has dropped significantly, we assume it's oversold and might rebound
        if drop < self.drop_threshold:
            return [self._generate_signal(ticker, "BUY", self.position_size)]
        return None