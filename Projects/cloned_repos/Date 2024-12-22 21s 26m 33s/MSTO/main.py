import logging
import json
from orchestrator import Orchestrator
from strategies import SimpleVolatilityStrategy, FundamentalEventDrivenStrategy

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Instantiate multiple strategies like separate microservices
    strategies = [
        SimpleVolatilityStrategy(),
        FundamentalEventDrivenStrategy()
    ]

    orchestrator = Orchestrator(strategies)
    tickers = ["AAPL", "MSFT"]
    for ticker in tickers:
        orchestrator.process_ticker(ticker)

if __name__ == "__main__":
    main()
