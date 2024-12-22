# Market Sentiment Trading Orchestrator (MSTO)

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)

A sophisticated trading system that monitors stock market movements, analyzes news sentiment, and executes trades based on multiple strategies. The system runs continuously, monitoring selected tickers for unusual price drops and correlating them with news sentiment to generate trading signals.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Strategy Development](#strategy-development)
- [Deployment Guide](#deployment-guide)
- [Monitoring & Operations](#monitoring--operations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

MSTO is designed to automate trading decisions based on market sentiment and price movements. It:
- Monitors multiple stock tickers in parallel
- Analyzes price movements for unusual patterns
- Fetches and analyzes news sentiment
- Classifies market events
- Generates trading signals based on configurable strategies
- Executes trades through TradingView integration

### Key Benefits
- **Automated Trading**: Reduce emotional trading with systematic strategies
- **Sentiment Analysis**: Leverage news sentiment for trading decisions
- **Scalable Architecture**: Handle multiple tickers and strategies in parallel
- **Cloud-Ready**: Deploy to AWS with built-in monitoring
- **Extensible**: Easy to add new strategies and data sources

## Features

### Core Functionality

#### Price Monitoring
- Real-time stock price tracking via yfinance
- Configurable lookback periods
- Unusual price movement detection
- Technical indicator support
  - Moving averages
  - Volume analysis
  - Price patterns

#### Sentiment Analysis
- News article fetching and processing
- NLTK-based sentiment scoring
- Event classification:
  - Earnings reports
  - Mergers & acquisitions
  - Management changes
  - Product launches
  - Legal events
  - Market movements

#### Trading Strategies

1. **Fundamental Event-Driven Strategy**
   ```python
   # Example configuration
   {
       "min_impact_threshold": 0.3,
       "max_pe_ratio": 30.0,
       "min_drop_threshold": -5.0,
       "position_sizing": {
           "base_size": 100,
           "impact_multiplier": true,
           "max_position": 1000
       }
   }
   ```
   - Analyzes fundamental metrics:
     - P/E ratio evaluation
     - Price drops analysis
     - News impact assessment
   - Position sizing based on:
     - Impact magnitude
     - Drop significance
     - Market conditions

2. **Simple Volatility Strategy**
   ```python
   # Example configuration
   {
       "min_drop_threshold": -2.0,
       "min_sentiment_threshold": -0.5,
       "position_size": 10,
       "max_positions": 3
   }
   ```
   - Volatility-based triggers:
     - Sudden price drops
     - High volume events
   - Sentiment correlation:
     - News sentiment scoring
     - Event impact analysis

### Technical Features

#### Health Monitoring
- Real-time health checks
- Metric tracking:
  ```json
  {
      "status": "healthy",
      "metrics": {
          "uptime_seconds": 3600,
          "total_checks": 1200,
          "errors": 0,
          "signals_generated": 15
      }
  }
  ```

#### Parallel Processing
- Multi-threaded ticker processing
- Strategy parallelization
- Configurable worker pools:
  ```python
  # Example configuration
  {
      "max_parallel_tickers": 10,
      "strategy_workers": 4,
      "queue_size": 100
  }
  ```

#### Logging System
- Structured JSON logging
- Log levels:
  ```json
  {
      "timestamp": "2023-12-21T10:30:00Z",
      "level": "INFO",
      "component": "strategy",
      "message": "Signal generated",
      "details": {
          "ticker": "AAPL",
          "action": "BUY",
          "quantity": 100
      }
  }
  ```

## Architecture

### System Components
```
msto/
├── core/                   # Core functionality
│   ├── analytics.py       # Market analysis and sentiment
│   │   ├── detect_unusual_drop()
│   │   ├── sentiment_analysis()
│   │   └── estimate_impact()
│   ├── data_sources.py    # Data fetching utilities
│   │   ├── fetch_stock_data()
│   │   ├── fetch_news()
│   │   └── get_fundamental_metrics()
│   ├── execution.py       # Trade execution
│   │   └── TradingViewIntegration
│   ├── health.py         # Health monitoring
│   │   ├── HealthStatus
│   │   └── HealthCheckHandler
│   └── orchestrator.py    # Main coordination logic
│       └── Orchestrator
├── strategies/            # Trading strategies
│   ├── base.py           # Strategy base class
│   │   └── Strategy
│   ├── fundamental_event_driven.py
│   │   └── FundamentalEventDrivenStrategy
│   └── simple_volatility.py
│       └── SimpleVolatilityStrategy
├── utils/                 # Utility modules
│   └── config.py         # Configuration management
└── cli.py                # Command-line interface

deploy/                   # Deployment configuration
├── ecs-task-definition.json
└── deploy.sh

tests/                    # Test suite
├── unit/                 # Unit tests
├── integration/          # Integration tests
└── conftest.py          # Test configuration
```

### Data Flow
1. **Data Collection**
   ```mermaid
   graph LR
       A[Stock Data] --> C[Orchestrator]
       B[News Data] --> C
       C --> D[Analytics]
       D --> E[Strategies]
       E --> F[Execution]
   ```

2. **Signal Generation**
   ```mermaid
   graph TD
       A[Price Drop Detection] --> D[Signal Generation]
       B[Sentiment Analysis] --> D
       C[Strategy Evaluation] --> D
       D --> E[Signal Validation]
       E --> F[Execution]
   ```

## Prerequisites

### Local Development
- Python 3.10+
  ```bash
  python --version  # Should be 3.10 or higher
  ```
- Docker and Docker Compose
  ```bash
  docker --version
  docker-compose --version
  ```
- PostgreSQL 14+
  ```bash
  psql --version
  ```

### AWS Deployment
1. AWS CLI configured with appropriate permissions:
   ```bash
   aws configure
   ```

2. Required IAM Permissions:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "ecs:*",
                   "ecr:*",
                   "efs:*",
                   "logs:*",
                   "ssm:*"
               ],
               "Resource": "*"
           }
       ]
   }
   ```

3. AWS Services Setup:
   - ECS Fargate cluster
   - ECR repository
   - EFS filesystem
   - CloudWatch log groups
   - Systems Manager parameters

## Installation

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/cenab/MSTO.git
   cd MSTO
   ```

2. Create and activate a virtual environment:
   ```bash
   # Linux/macOS
   python -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Docker Setup

1. Build the image:
   ```bash
   docker build -t msto:latest .
   ```

2. Run with docker-compose:
   ```bash
   # Development mode
   docker-compose up --build

   # Production mode
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
   ```

3. Run specific configurations:
   ```bash
   # Custom tickers
   docker-compose run --rm msto --tickers AAPL MSFT --strategies all

   # Override environment
   POSTGRES_PASSWORD=custom docker-compose up
   ```

## Configuration

### Environment Variables

1. Development setup:
   ```bash
   cp .env.local .env
   ```

2. Configuration categories:

#### Core Settings
```env
# Environment configuration
ENV=dev                    # dev/prod
TRADING_MODE=paper         # paper/live
DROP_LOOKBACK_DAYS=60     # Days to analyze

# Performance tuning
PARALLEL_WORKERS=4         # Number of worker threads
BATCH_SIZE=100            # Batch size for processing
```

#### API Keys
```env
# External services
TRADINGVIEW_WEBHOOK_URL=https://your-webhook-url
NEWS_API_KEY=your-api-key

# Optional integrations
SLACK_WEBHOOK_URL=your-slack-webhook    # For notifications
TELEGRAM_BOT_TOKEN=your-bot-token      # For alerts
```

#### Database Configuration
```env
# PostgreSQL settings
POSTGRES_USER=msto_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=msto_dev
DB_CONNECTION_STRING=postgresql://msto_user:secure_password@localhost:5432/msto_dev

# Connection pool
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
```

#### Strategy Parameters
```env
# Common parameters
DEFAULT_MIN_IMPACT_THRESHOLD=0.3
MIN_DROP_THRESHOLD=-5.0

# Fundamental strategy
MAX_PE_RATIO=30.0
MIN_MARKET_CAP=1000000000

# Volatility strategy
MIN_SENTIMENT_THRESHOLD=-0.5
VOLATILITY_WINDOW=20
```

#### Monitoring Configuration
```env
# Health checks
HEALTH_CHECK_PORT=8080
HEALTH_CHECK_INTERVAL=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/app/logs/msto.log

# Market hours
MARKET_HOURS_ONLY=true
MARKET_TIMEZONE=America/New_York
```

## Usage Examples

### Basic Usage

1. Run with default settings:
   ```bash
   msto --tickers AAPL MSFT GOOGL
   ```

2. Run specific strategies:
   ```bash
   msto --tickers AAPL --strategies fundamental volatility
   ```

3. Debug mode with detailed logging:
   ```bash
   msto --tickers AAPL --log-level DEBUG --mode once
   ```

### Advanced Usage

1. Custom configuration file:
   ```bash
   msto --config custom.env --tickers AAPL MSFT
   ```

2. Multiple strategies with parameters:
   ```bash
   msto --tickers AAPL MSFT \
        --strategies fundamental volatility \
        --fundamental-threshold 0.4 \
        --volatility-window 30
   ```

3. Continuous monitoring with custom interval:
   ```bash
   msto --tickers AAPL MSFT GOOGL \
        --mode continuous \
        --interval 600 \
        --max-signals 5
   ```

### Docker Usage

1. Development environment:
   ```bash
   docker-compose up --build
   ```

2. Production deployment:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

3. Custom configuration:
   ```bash
   docker-compose run --rm msto \
       --tickers AAPL MSFT \
       --strategies all \
       --log-level INFO
   ```

## Strategy Development

### Creating a New Strategy

1. Create a new strategy file:
   ```python
   # msto/strategies/my_strategy.py
   from typing import List, Dict, Any, Optional
   from msto.strategies.base import Strategy

   class MyStrategy(Strategy):
       def __init__(self,
                   name: str = "MyStrategy",
                   min_impact_threshold: float = 0.3):
           super().__init__(name, min_impact_threshold)
           # Add custom initialization

       def process_data(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
           if not self._validate_data(data):
               return None

           # Implement strategy logic
           signals = []
           # Generate signals based on data
           return signals
   ```

2. Add configuration parameters:
   ```python
   # msto/utils/config.py
   MY_STRATEGY_THRESHOLD = float(os.getenv("MY_STRATEGY_THRESHOLD", "0.5"))
   ```

3. Register strategy in CLI:
   ```python
   # msto/cli.py
   def get_strategies(config: dict, selected_strategies: List[str]) -> List[Strategy]:
       strategies = {
           "my_strategy": lambda: MyStrategy(
               min_impact_threshold=float(config.get("MY_STRATEGY_THRESHOLD", 0.5))
           )
       }
   ```

### Testing Strategies

1. Unit tests:
   ```python
   # tests/unit/test_my_strategy.py
   def test_my_strategy():
       strategy = MyStrategy()
       data = {
           "ticker": "AAPL",
           "drop": -3.0,
           "avg_sentiment": -0.5
       }
       signals = strategy.process_data(data)
       assert len(signals) == 1
       assert signals[0]["action"] == "BUY"
   ```

2. Integration tests:
   ```python
   # tests/integration/test_strategies.py
   def test_strategy_integration(mock_data):
       strategy = MyStrategy()
       orchestrator = Orchestrator([strategy])
       orchestrator.process_ticker("AAPL")
   ```

## Deployment Guide

### AWS ECS Deployment

1. Initial setup:
   ```bash
   # Configure AWS credentials
   aws configure

   # Create ECR repository
   aws ecr create-repository --repository-name msto

   # Create ECS cluster
   aws ecs create-cluster --cluster-name msto-cluster
   ```

2. Store secrets:
   ```bash
   # Store API keys
   aws ssm put-parameter \
       --name /msto/tradingview_webhook_url \
       --value "your_url" \
       --type SecureString

   aws ssm put-parameter \
       --name /msto/news_api_key \
       --value "your_key" \
       --type SecureString
   ```

3. Deploy:
   ```bash
   # Run deployment script
   ./deploy/deploy.sh
   ```

### Monitoring Setup

1. CloudWatch Logs:
   ```bash
   # Create log group
   aws logs create-log-group --log-group-name /ecs/msto

   # Set retention
   aws logs put-retention-policy \
       --log-group-name /ecs/msto \
       --retention-in-days 30
   ```

2. Alarms:
   ```bash
   # Create CPU utilization alarm
   aws cloudwatch put-metric-alarm \
       --alarm-name msto-cpu-utilization \
       --metric-name CPUUtilization \
       --namespace AWS/ECS \
       --statistic Average \
       --period 300 \
       --threshold 80 \
       --comparison-operator GreaterThanThreshold
   ```

## Monitoring & Operations

### Health Checks

1. Endpoint information:
   ```bash
   curl http://localhost:8080/health
   ```
   Response:
   ```json
   {
       "status": "healthy",
       "last_check_time": "2023-12-21T10:30:00Z",
       "metrics": {
           "uptime_seconds": 3600,
           "total_checks": 1200,
           "errors": 0
       }
   }
   ```

2. Metrics available:
   - System health
   - Processing statistics
   - Strategy performance
   - Error rates

### Logging

1. Log format:
   ```json
   {
       "timestamp": "2023-12-21T10:30:00Z",
       "level": "INFO",
       "component": "strategy",
       "message": "Processing ticker",
       "details": {
           "ticker": "AAPL",
           "strategy": "fundamental",
           "duration_ms": 150
       }
   }
   ```

2. Log levels:
   - DEBUG: Detailed debugging information
   - INFO: General operational information
   - WARNING: Warning messages
   - ERROR: Error conditions
   - CRITICAL: Critical conditions

### Performance Monitoring

1. System metrics:
   - CPU usage
   - Memory utilization
   - Network I/O
   - Disk usage

2. Application metrics:
   - Processing time per ticker
   - Strategy execution time
   - Signal generation rate
   - Error rates

## Troubleshooting

### Common Issues

1. Connection Issues
   ```bash
   # Check database connection
   psql $DB_CONNECTION_STRING -c "\conninfo"

   # Test API endpoints
   curl -v $TRADINGVIEW_WEBHOOK_URL
   ```

2. Performance Issues
   ```bash
   # Check system resources
   docker stats msto

   # Monitor logs
   tail -f logs/msto.log | jq .
   ```

3. Strategy Issues
   ```bash
   # Enable debug logging
   export LOG_LEVEL=DEBUG
   msto --tickers AAPL --strategies fundamental
   ```

### Debug Tools

1. Health check:
   ```bash
   curl http://localhost:8080/health
   ```

2. Log analysis:
   ```bash
   # Search for errors
   grep -i error logs/msto.log | jq .

   # Monitor real-time
   tail -f logs/msto.log | jq 'select(.level=="ERROR")'
   ```

3. Database debugging:
   ```bash
   # Connect to database
   psql $DB_CONNECTION_STRING

   # Check signals table
   SELECT * FROM signals ORDER BY created_at DESC LIMIT 10;
   ```

## Contributing

### Development Workflow

1. Fork and clone:
   ```bash
   git clone https://github.com/cenab/MSTO.git
   cd MSTO
   ```

2. Set up development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   pre-commit install
   ```

3. Create feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

4. Make changes and test:
   ```bash
   # Run tests
   pytest tests/

   # Check code style
   black .
   flake8 .
   mypy .
   ```

5. Submit pull request:
   - Write clear description
   - Include test coverage
   - Update documentation

### Coding Standards

1. Style guide:
   - Follow PEP 8
   - Use type hints
   - Write docstrings
   - Keep functions focused

2. Testing:
   - Write unit tests
   - Include integration tests
   - Maintain test coverage

3. Documentation:
   - Update README
   - Add inline comments
   - Update API documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-party Licenses

- yfinance: Apache 2.0
- NLTK: Apache 2.0
- PostgreSQL: PostgreSQL License
