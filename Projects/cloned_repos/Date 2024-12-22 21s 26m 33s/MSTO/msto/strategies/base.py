from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import json

class Strategy(ABC):
    def __init__(self, name: str, min_impact_threshold: float = 0.3):
        self.name = name
        self.min_impact_threshold = min_impact_threshold

    @abstractmethod
    def process_data(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Process the data and generate trading signals.
        
        Args:
            data: Dictionary containing processed market data including:
                - ticker: str
                - drop: float
                - avg_sentiment: float
                - most_common_event: str
                - fundamentals: dict
                - impact: float
        
        Returns:
            List of signal dictionaries or None if no signals generated
        """
        pass

    def _generate_signal(self, ticker: str, action: str, quantity: int) -> Dict[str, Any]:
        """
        Generate a standardized signal dictionary.
        """
        signal = {
            "symbol": ticker,
            "action": action,
            "quantity": quantity,
            "strategy": self.name
        }
        logging.info(json.dumps({
            "level": "INFO",
            "message": "Signal generated",
            "signal": signal
        }))
        return signal

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate that all required data fields are present.
        """
        required_fields = ['ticker', 'drop', 'avg_sentiment', 'most_common_event', 'fundamentals', 'impact']
        return all(field in data for field in required_fields)