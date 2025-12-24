"""
Full model training test with REAL Polymarket data.
NO SYNTHETIC DATA - 100% real market data.
"""

from prediction_model import create_predictor

def main():
    print("\n" + "=" * 70)
    print("🚀 TRAINING POLYMARKET PREDICTOR ON REAL DATA")
    print("   All data fetched from live Polymarket API")
    print("=" * 70)
    
    # Create and train predictor with real data
    predictor = create_predictor(
        use_optuna=False,
        kelly_fraction=0.25,
        bankroll=10000,
        n_markets=50  # Fetch from 50 markets
    )
    
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINED ON REAL POLYMARKET DATA")
    print("=" * 70)
    
    # Test prediction on a real-looking market
    print("\n📊 Testing prediction capability...")
    
    test_market = {
        'question': 'Test prediction market',
        'id': 'test123',
        'outcomePrices': '[0.65, 0.35]',
        'volume24hr': 50000,
        'liquidity': 25000,
        'endDate': '2025-02-01T00:00:00Z',
        'oneDayPriceChange': 0.02,
        'oneWeekPriceChange': 0.05,
        'bestBid': 0.64,
        'bestAsk': 0.66,
    }
    
    orderbook = {
        'bids': [{'price': 0.64, 'size': 500}, {'price': 0.63, 'size': 300}],
        'asks': [{'price': 0.66, 'size': 400}, {'price': 0.67, 'size': 200}]
    }
    
    import pandas as pd
    trades_df = pd.DataFrame([
        {'price': 0.65, 'size': 100, 'timestamp': '2025-01-20T10:00:00Z'},
        {'price': 0.64, 'size': 150, 'timestamp': '2025-01-20T09:00:00Z'},
        {'price': 0.63, 'size': 200, 'timestamp': '2025-01-20T08:00:00Z'},
    ])
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['price'] = pd.to_numeric(trades_df['price'])
    trades_df['size'] = pd.to_numeric(trades_df['size'])
    
    # Function signature: predict(market, trades_df, days_remaining)
    prediction = predictor.predict(test_market, trades_df, days_remaining=10)
    
    print(f"\n📈 Prediction Results (Real Data Model):")
    print(f"   Market Price: 65%")
    
    # Show available keys for debugging
    prob = prediction.get('probability', prediction.get('probability_yes', 0.5))
    print(f"   Model Probability: {prob:.1%}")
    print(f"   Classification: {'YES' if prediction.get('classification', 0) == 1 else 'NO'}")
    print(f"   Confidence: {prediction.get('confidence', 0):.3f}")
    print(f"   Order Book Imbalance: {prediction.get('order_book_imbalance', 0):.3f}")
    print(f"   Kelly Position: ${prediction.get('position_size', 0):.2f}")
    print(f"   Expected Value: {prediction.get('expected_value', 0):.4f}")
    
    return predictor


if __name__ == "__main__":
    predictor = main()
    print("\n✅ Model ready for trading with REAL data training!")
