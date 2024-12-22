"""
Fundamental and event-driven trading strategy.
"""
from typing import List, Dict, Any, Optional
import logging
import json
from msto.strategies.base import Strategy

logger = logging.getLogger(__name__)

class FundamentalEventDrivenStrategy(Strategy):
    def __init__(self, 
                 name: str = "FundamentalEventDriven",
                 min_impact_threshold: float = 0.3,
                 max_pe_ratio: float = 30.0,
                 min_drop_threshold: float = -5.0):
        """
        Initialize the fundamental event-driven strategy.

        Args:
            name: Strategy name
            min_impact_threshold: Minimum impact threshold for signals
            max_pe_ratio: Maximum P/E ratio to consider for buying
            min_drop_threshold: Minimum price drop threshold (negative value)
        """
        super().__init__(name, min_impact_threshold)
        self.max_pe_ratio = max_pe_ratio
        self.min_drop_threshold = min_drop_threshold
        
        logger.info(json.dumps({
            "level": "INFO",
            "message": "Initialized FundamentalEventDrivenStrategy",
            "strategy": name,
            "min_impact_threshold": min_impact_threshold,
            "max_pe_ratio": max_pe_ratio,
            "min_drop_threshold": min_drop_threshold
        }))

    def process_data(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Process market data and generate trading signals based on fundamentals and events.

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
        impact = data['impact']
        fundamentals = data['fundamentals']
        pe_ratio = fundamentals.get('pe_ratio')

        # Log the current market conditions
        logger.debug(json.dumps({
            "level": "DEBUG",
            "message": "Processing market data",
            "strategy": self.name,
            "ticker": ticker,
            "conditions": {
                "price_drop": drop,
                "impact": impact,
                "pe_ratio": pe_ratio,
                "thresholds": {
                    "min_impact": self.min_impact_threshold,
                    "max_pe": self.max_pe_ratio,
                    "min_drop": self.min_drop_threshold
                }
            }
        }))

        # Check each condition and log if it fails
        if abs(impact) < self.min_impact_threshold:
            logger.debug(json.dumps({
                "level": "DEBUG",
                "message": "Strategy not triggered",
                "strategy": self.name,
                "ticker": ticker,
                "reason": "Impact not significant enough",
                "impact": impact,
                "threshold": self.min_impact_threshold
            }))
            return None

        if drop > self.min_drop_threshold:
            logger.debug(json.dumps({
                "level": "DEBUG",
                "message": "Strategy not triggered",
                "strategy": self.name,
                "ticker": ticker,
                "reason": "Price drop not significant enough",
                "drop": drop,
                "threshold": self.min_drop_threshold
            }))
            return None

        if pe_ratio is None or pe_ratio > self.max_pe_ratio:
            logger.debug(json.dumps({
                "level": "DEBUG",
                "message": "Strategy not triggered",
                "strategy": self.name,
                "ticker": ticker,
                "reason": "PE ratio too high or missing",
                "pe_ratio": pe_ratio,
                "threshold": self.max_pe_ratio
            }))
            return None

        # Calculate position size based on impact magnitude
        quantity = self._calculate_position_size(impact, drop)
        signal = self._generate_signal(ticker, "BUY", quantity)
        
        logger.info(json.dumps({
            "level": "INFO",
            "message": "Strategy triggered",
            "strategy": self.name,
            "ticker": ticker,
            "trigger_conditions": {
                "price_drop": drop,
                "impact": impact,
                "pe_ratio": pe_ratio
            },
            "signal": signal
        }))
        
        return [signal]

    def _calculate_position_size(self, impact: float, drop: float) -> int:
        """
        Calculate position size based on impact and drop magnitude.
        Returns number of shares to trade.
        """
        # Base position of 100 shares
        base_position = 100
        
        # Adjust based on impact (0.3 to 1.0 scale)
        impact_multiplier = min(max(abs(impact), 0.3), 1.0)
        
        # Adjust based on drop magnitude (larger drops -> larger positions)
        drop_multiplier = min(abs(drop) / 5.0, 2.0)
        
        position_size = int(base_position * impact_multiplier * drop_multiplier)
        
        logger.debug(json.dumps({
            "level": "DEBUG",
            "message": "Calculated position size",
            "strategy": self.name,
            "calculation": {
                "base_position": base_position,
                "impact_multiplier": impact_multiplier,
                "drop_multiplier": drop_multiplier,
                "final_size": position_size
            }
        }))
        
        return max(position_size, 10)  # Minimum 10 shares