import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from msto.core.analytics import (
    detect_unusual_drop,
    sentiment_analysis,
    classify_events,
    estimate_impact
)
from msto.core.data_sources import (
    fetch_stock_data,
    fetch_news,
    get_fundamental_metrics
)
from msto.core.execution import TradingViewIntegration
from msto.core.health import HealthCheckHandler
from msto.core.health_server import start_health_server
from msto.strategies.base import Strategy
from msto.utils.config import load_config

logger = logging.getLogger(__name__)

class Orchestrator:
    """Orchestrates the processing of tickers and execution of strategies."""

    def __init__(
        self,
        strategies: List[Strategy],
        config: Optional[Dict[str, Any]] = None,
        health_port: int = 8080
    ):
        """
        Initialize the orchestrator.

        Args:
            strategies: List of trading strategies to use
            config: Optional configuration dictionary
            health_port: Port for health check server
        """
        self.strategies = strategies
        self.config = config or load_config()
        self.executor = ThreadPoolExecutor(
            max_workers=int(self.config.get("PARALLEL_WORKERS", 4))
        )
        self.trading_view = TradingViewIntegration(
            self.config.get("TRADINGVIEW_WEBHOOK_URL")
        )
        self.health_handler = HealthCheckHandler()
        self.health_server = start_health_server(self.health_handler, health_port)
        self.running = False
        self._lock = threading.Lock()

    def start(self, tickers: List[str], interval: int = 300):
        """
        Start processing tickers.

        Args:
            tickers: List of stock tickers to monitor
            interval: Time between checks in seconds
        """
        logger.info(json.dumps({
            "level": "INFO",
            "message": "Starting ticker processing",
            "tickers": tickers,
            "interval": interval
        }))

        self.running = True
        while self.running:
            try:
                futures = []
                for ticker in tickers:
                    future = self.executor.submit(self.process_ticker, ticker)
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(json.dumps({
                            "level": "ERROR",
                            "message": "Error processing ticker",
                            "error": str(e)
                        }))
                        self.health_handler.record_error()

                time.sleep(interval)
            except Exception as e:
                logger.error(json.dumps({
                    "level": "ERROR",
                    "message": "Error in processing loop",
                    "error": str(e)
                }))
                self.health_handler.record_error()

    def stop(self):
        """Stop processing tickers."""
        logger.info(json.dumps({
            "level": "INFO",
            "message": "Stopping ticker processing"
        }))
        self.running = False
        self.executor.shutdown(wait=True)
        self.health_server.shutdown()

    def process_ticker(self, ticker: str) -> None:
        """
        Process a single ticker.

        Args:
            ticker: Stock ticker symbol
        """
        try:
            logger.info(json.dumps({
                "level": "INFO",
                "message": "Processing ticker",
                "ticker": ticker
            }))

            # Fetch data
            stock_data = fetch_stock_data(ticker)
            news_data = fetch_news(ticker)
            fundamental_data = get_fundamental_metrics(ticker)

            # Analyze data
            drop = detect_unusual_drop(stock_data)
            sentiment = sentiment_analysis(news_data)
            events = classify_events(news_data)
            impact = estimate_impact(events, sentiment)

            # Prepare processed data
            processed_data = {
                "ticker": ticker,
                "drop": drop,
                "sentiment": sentiment,
                "events": events,
                "impact": impact,
                "fundamentals": fundamental_data,
                "stock_data": stock_data
            }

            # Process strategies
            signals = self._process_strategies(processed_data)

            # Execute signals
            if signals:
                self._execute_signals(signals)
                self.health_handler.increment_signals()

            logger.info(json.dumps({
                "level": "INFO",
                "message": "Ticker processed successfully",
                "ticker": ticker,
                "signals_generated": len(signals) if signals else 0
            }))

        except Exception as e:
            logger.error(json.dumps({
                "level": "ERROR",
                "message": "Error processing ticker",
                "ticker": ticker,
                "error": str(e)
            }))
            self.health_handler.record_error()
            raise

    def _process_strategies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process all strategies with the given data.

        Args:
            data: Processed market data

        Returns:
            List of generated trading signals
        """
        all_signals = []
        for strategy in self.strategies:
            try:
                signals = strategy.process_data(data)
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                logger.error(json.dumps({
                    "level": "ERROR",
                    "message": "Strategy processing error",
                    "strategy": strategy.__class__.__name__,
                    "error": str(e)
                }))
                self.health_handler.record_error()

        return all_signals

    def _execute_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Execute the generated trading signals.

        Args:
            signals: List of trading signals to execute
        """
        with self._lock:
            for signal in signals:
                try:
                    self.trading_view.execute_signal(signal)
                    logger.info(json.dumps({
                        "level": "INFO",
                        "message": "Signal executed",
                        "signal": signal
                    }))
                except Exception as e:
                    logger.error(json.dumps({
                        "level": "ERROR",
                        "message": "Signal execution error",
                        "signal": signal,
                        "error": str(e)
                    }))
                    self.health_handler.record_error()
