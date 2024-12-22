"""
Command Line Interface for MSTO
"""
import argparse
import logging
import sys
from typing import List, Optional
import json

from msto.core.orchestrator import Orchestrator
from msto.strategies.fundamental_event_driven import FundamentalEventDrivenStrategy
from msto.strategies.simple_volatility import SimpleVolatilityStrategy
from msto.utils.config import load_config

logger = logging.getLogger(__name__)

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Market Sentiment Trading Orchestrator")
    
    parser.add_argument(
        "--config",
        type=str,
        default=".env",
        help="Path to configuration file (default: .env)",
    )
    
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        required=True,
        help="List of stock tickers to monitor",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        choices=["fundamental", "volatility", "all"],
        default=["all"],
        help="Strategies to use (default: all)",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["continuous", "once"],
        default="continuous",
        help="Run mode: continuous monitoring or one-time check (default: continuous)",
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds for continuous mode (default: 300)",
    )
    
    parser.add_argument(
        "--max-signals",
        type=int,
        default=3,
        help="Maximum number of signals per ticker (default: 3)",
    )
    
    return parser.parse_args(args)

def setup_logging(log_level: str) -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("msto.log")
        ]
    )

def get_strategies(config: dict, selected_strategies: List[str]) -> List[Strategy]:
    """
    Initialize selected trading strategies.
    
    Args:
        config: Configuration dictionary
        selected_strategies: List of strategy names to initialize
        
    Returns:
        List of initialized strategy objects
    """
    all_strategies = {
        "fundamental": lambda: FundamentalEventDrivenStrategy(
            min_impact_threshold=float(config.get("DEFAULT_MIN_IMPACT_THRESHOLD", 0.3)),
            max_pe_ratio=float(config.get("MAX_PE_RATIO", 30.0)),
            min_drop_threshold=float(config.get("MIN_DROP_THRESHOLD", -5.0))
        ),
        "volatility": lambda: SimpleVolatilityStrategy(
            min_impact_threshold=float(config.get("DEFAULT_MIN_IMPACT_THRESHOLD", 0.3)),
            min_drop_threshold=float(config.get("MIN_DROP_THRESHOLD", -2.0)),
            min_sentiment_threshold=float(config.get("MIN_SENTIMENT_THRESHOLD", -0.5)),
            position_size=int(config.get("BASE_POSITION_SIZE", 10))
        )
    }
    
    strategies = []
    strategy_names = list(all_strategies.keys()) if "all" in selected_strategies else selected_strategies
    
    for name in strategy_names:
        if name in all_strategies:
            strategies.append(all_strategies[name]())
    
    return strategies

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parsed_args = parse_args(args)
    setup_logging(parsed_args.log_level)
    
    try:
        # Update configuration with CLI arguments
        config = load_config(parsed_args.config)
        config["CHECK_INTERVAL_SECONDS"] = parsed_args.interval
        config["MAX_SIGNALS_PER_TICKER"] = parsed_args.max_signals
        
        # Initialize selected strategies
        strategies = get_strategies(config, parsed_args.strategies)
        
        if not strategies:
            logger.error("No valid strategies selected")
            return 1
        
        # Initialize orchestrator
        orchestrator = Orchestrator(strategies)
        
        logger.info(json.dumps({
            "level": "INFO",
            "message": "Starting MSTO",
            "mode": parsed_args.mode,
            "tickers": parsed_args.tickers,
            "strategies": parsed_args.strategies,
            "interval": parsed_args.interval if parsed_args.mode == "continuous" else None
        }))
        
        # Run in selected mode
        if parsed_args.mode == "continuous":
            orchestrator.start_monitoring(parsed_args.tickers)
        else:
            orchestrator.process_all_tickers(parsed_args.tickers)
            
        return 0
        
    except KeyboardInterrupt:
        logger.info("MSTO stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error running MSTO: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 