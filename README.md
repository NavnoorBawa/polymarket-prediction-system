# Polymarket Prediction System

A state-of-the-art machine learning system for predicting Polymarket outcomes using advanced ensemble methods.

## Overview

This project uses sophisticated ML algorithms including XGBoost, LightGBM, and stacking ensembles to analyze prediction markets on Polymarket and generate trading signals with confidence scores.

## Features

- **Advanced ML Models**: Combines XGBoost and LightGBM with stacking ensembles
- **Real-time Market Analysis**: Fetches live trading data from Polymarket API
- **Quantitative Metrics**: RSI, volatility, order book imbalance, expected value calculations
- **Risk Management**: Kelly criterion position sizing and terminal risk adjustments
- **Smart Filtering**: Automatically excludes resolved and low-volume markets

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
```

## Output

The system provides:
- Current vs. predicted prices
- Buy/Sell/Hold recommendations
- Confidence scores
- Trading signals (STRONG BUY, BUY, HOLD)
- Key insights (RSI, volatility, order book pressure)
- Kelly-optimized position sizes

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Disclaimer

This software is for educational and research purposes only. It is NOT financial advice. Prediction markets involve substantial risk. Always do your own research and never invest more than you can afford to lose.

## License

MIT License - See LICENSE file for details
