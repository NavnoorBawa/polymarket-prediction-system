# Polymarket Prediction System

A state-of-the-art machine learning system for predicting Polymarket outcomes using advanced ensemble methods.

## Overview

This project uses sophisticated ML algorithms including XGBoost, LightGBM, CatBoost, and stacking ensembles to analyze prediction markets on Polymarket and generate trading signals with confidence scores.

## Features

- **Advanced ML Models**: Combines XGBoost and LightGBM with stacking ensembles
- **CatBoost Integration**: Ordered boosting model added to ensemble when available
- **Real-time Market Analysis**: Fetches live trading data from Polymarket API
- **Quantitative Metrics**: RSI, volatility, order book imbalance, expected value calculations
- **Risk Management**: Kelly criterion position sizing and terminal risk adjustments
- **Smart Filtering**: Automatically excludes resolved and low-volume markets
- **Cross-Domain Sampling**: Training/predictions now prioritize domain diversity (not sports-only)
- **Persistent Memory Layer**: Stores markets, price history, training samples, model runs, and predictions in a local DB (chDB when available)
- **Predictive Validation**: Reports baseline lift, AUC, high-confidence hit rate, and signal verdict
- **Comprehensive Reporting**: Detailed run reports with models, probabilities, timing, and decision rationale
- **Trend Discovery Mode**: Scores and ranks all analyzed markets (trend signal + trend score) even when trade action is HOLD
- **Class-Imbalance Hardening**: Ensemble now auto-applies class-balance settings across all classifiers

## Components

- `main.py` - Main prediction engine and CLI interface
- `polymarket_fetcher.py` - Polymarket API integration for fetching market data
- `prediction_model.py` - ML models and prediction algorithms
- `test_real_data.py` - Testing with real market data
- `test_real_training.py` - Model training and validation tests

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/polymarket-prediction-system.git
cd polymarket-prediction-system

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run predictions on top markets:

```bash
# Analyze top 10 markets
python main.py

# Analyze custom number of markets
python main.py 20

# Save detailed JSON report to a custom path
python main.py 20 --report-file data\my_run_report.json

# Run report-only entrypoint
python final_report.py --markets 20 --report-file data\final_report.json
```

## Output

The system provides:
- Current vs. predicted prices
- Buy/Sell/Hold recommendations
- Confidence scores
- Trading signals (STRONG BUY, BUY, HOLD)
- Trend signals and ranked watchlist across all analyzed markets
- Key insights (RSI, volatility, order book pressure)
- Kelly-optimized position sizes

## Persistent Storage

The app now keeps a local persistent memory database under `data/`:

- `market_snapshots` - fetched market lists
- `trade_snapshots` - recent trade batches per token
- `price_history_points` - cached historical price points
- `training_samples` - prepared training feature vectors/labels
- `model_training_runs` - training metrics and run metadata
- `prediction_runs` - generated predictions for traceability
- `run_reports` - full run-level report payloads (timing, models, metrics, predictions)

Backend selection:
- **chDB** is used automatically when available
- **SQLite** fallback is used on platforms where chDB wheels are unavailable (e.g., Windows)

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Disclaimer

This software is for educational and research purposes only. It is NOT financial advice. Prediction markets involve substantial risk. Always do your own research and never invest more than you can afford to lose.

## License

MIT License - See LICENSE file for details
