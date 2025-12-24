"""
Test script for REAL DATA fetching from Polymarket API.
NO SYNTHETIC DATA - 100% real market data.
"""

from polymarket_fetcher import PolymarketFetcher

def main():
    print("=" * 70)
    print("🌐 TESTING REAL DATA FETCHING FROM POLYMARKET API")
    print("   No synthetic data - only real market data")
    print("=" * 70)
    
    fetcher = PolymarketFetcher(verbose=True)
    
    # Fetch real training data
    training_data = fetcher.fetch_real_training_data(
        n_markets=30,
        min_volume=1000,
        include_closed=True
    )
    
    print(f"\n✅ Fetched {len(training_data)} real training samples")
    
    # Show sample data
    if training_data:
        print("\n📊 Sample Training Data:")
        for i, sample in enumerate(training_data[:5]):
            question = sample.get("question", "Unknown")
            print(f"\n   {i+1}. {question}")
            print(f"      Price: {sample['current_price']:.4f}")
            print(f"      Outcome: {sample['outcome']}")
            print(f"      Resolved: {sample['is_resolved']}")
            vol = sample['volume']
            print(f"      Volume: ${vol:,.0f}")
            
            # Show feature values
            features = sample['features'].flatten()
            print(f"      Features (10 dims): [{', '.join(f'{f:.3f}' for f in features[:5])}...]")
    
    return training_data


if __name__ == "__main__":
    data = main()
    print(f"\n✅ Total real samples ready for training: {len(data)}")
