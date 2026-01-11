#!/usr/bin/env python3
"""
Polymarket Prediction System
State-of-the-Art ML Predictions using XGBoost, LightGBM, and Stacking Ensembles
"""

# Suppress sklearn parallel warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import json
import time
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from polymarket_fetcher import PolymarketFetcher
from prediction_model import PolymarketPredictor, create_predictor

# Configuration
EXCLUDE_POLITICS_MARKETS = True  # Set False to include Politics markets

# Module-level cache for politics tag ID
_POLITICS_TAG_ID: Optional[str] = None


def get_politics_tag_id(fetcher: PolymarketFetcher) -> Optional[str]:
    """
    Get politics tag ID from Polymarket tags API.
    
    Caches the result to avoid repeated API calls.
    
    Args:
        fetcher: PolymarketFetcher instance
    
    Returns: Politics tag ID as string, or None if not found
    """
    global _POLITICS_TAG_ID
    
    # Return cached value if available
    if _POLITICS_TAG_ID:
        return _POLITICS_TAG_ID
    
    try:
        # Use the robust tag discovery function that handles pagination
        tag_id, slug, tag_obj = fetcher.find_tag_by_label("Politics")
        
        if tag_id:
            _POLITICS_TAG_ID = str(tag_id)
            print(f"✅ Found politics tag ID: {_POLITICS_TAG_ID} (slug: {slug})")
            return _POLITICS_TAG_ID
        else:
            print("⚠️  Warning: Politics tag not found. Using keyword fallback.")
            return None
        
    except Exception as e:
        print(f"⚠️  Warning: Error fetching politics tag ID: {e}")
        return None


def print_header():
    print("\n" + "="*65)
    print("   🎯 POLYMARKET LIVE PREDICTIONS")
    print("   State-of-the-Art ML (XGBoost + LightGBM + Stacking)")
    print("="*65)


def is_politics_market(market: Dict, fetcher: PolymarketFetcher, politics_tag_id: Optional[str] = None) -> bool:
    """
    Detect if a market is a Politics market based on Polymarket tags.
    
    Uses official Polymarket tag IDs for accurate categorization instead of keyword matching.
    Falls back to keyword matching if tag-based detection fails.
    
    Args:
        market: Market dictionary from Polymarket API
        fetcher: PolymarketFetcher instance for tag lookup
        politics_tag_id: Optional politics tag ID (will be fetched if not provided)
        
    Returns: True if Politics market detected, False otherwise
    """
    # Get politics tag ID if not provided
    if not politics_tag_id:
        politics_tag_id = get_politics_tag_id(fetcher)
        if not politics_tag_id:
            # Fallback to keyword matching if tag ID not found
            return _is_politics_market_keywords(market)
    
    # Extract tags field from market
    tags = market.get('tags', [])
    
    if not tags:
        # No tags field - fallback to keyword matching
        return _is_politics_market_keywords(market)
    
    # Check if politics tag ID exists in market's tags
    for tag in tags:
        if isinstance(tag, dict):
            # Tag is a dict object: {"id": "123", "label": "...", "slug": "..."}
            tag_id = tag.get('id')
            if tag_id and str(tag_id) == str(politics_tag_id):
                return True
        elif isinstance(tag, (str, int)):
            # Tag is a string/int ID: "123" or 123
            if str(tag) == str(politics_tag_id):
                return True
    
    # Not found in tags - not a politics market
    return False


def _is_politics_market_keywords(market: Dict) -> bool:
    """
    Fallback keyword-based politics detection.
    
    Used when tag-based detection fails (tag not found, no tags field, etc.).
    
    Args:
        market: Market dictionary from Polymarket API
        
    Returns: True if Politics market detected by keywords, False otherwise
    """
    # Politics keywords (based on patterns from other files in codebase)
    POLITICS_KEYWORDS = [
        'trump', 'biden', 'election', 'vote', 'voting', 'president', 
        'congress', 'senate', 'house', 'democrat', 'democratic', 
        'republican', 'gop', 'nominee', 'primary', 'caucus', 
        'cabinet', 'governor', 'mayor', 'impeach', 'impeachment',
        'presidential', 'elections', 'ballot', 'poll', 'polls',
        'democracy', 'political', 'politics', 'senator', 
        'representative', 'congressional', 'electoral', 'campaign'
    ]
    
    # Extract question and description
    question = market.get('question', '').lower()
    description = market.get('description', '').lower()
    combined_text = question + " " + description
    
    # Check if any politics keyword appears in text (case-insensitive)
    return any(keyword in combined_text for keyword in POLITICS_KEYWORDS)


def analyze_market(predictor, fetcher, market, show_details=False):
    """Analyze a single market and return prediction"""
    
    yes_token, _ = fetcher.get_token_ids_for_market(market)
    
    prices_str = market.get('outcomePrices', '[0.5, 0.5]')
    try:
        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        current_price = float(prices[0]) if prices else 0.5
    except:
        current_price = 0.5
    
    # Fetch REAL trade data from Polymarket API
    # Strategy: API requires 'market' parameter (condition ID) to fetch trades
    # We then filter by asset_id (token_id) to get only YES token trades
    trades_df = pd.DataFrame()
    
    # Get condition ID (market ID) - REQUIRED by API
    condition_id = market.get('id') or market.get('conditionId') or market.get('slug')
    
    if condition_id:
        # Fetch trades for the market, filtering by YES token if available
        if yes_token:
            # Fetch all trades for market, then filter to YES token only
            trades = fetcher.get_trades(market=condition_id, asset_id=yes_token, limit=500)
        else:
            # Fetch all trades for market (both YES and NO)
            trades = fetcher.get_trades(market=condition_id, limit=500)
        
        if trades and len(trades) > 0:
            trades_df = fetcher.trades_to_dataframe(trades)
            if show_details:
                outcome_type = "YES token" if yes_token else "all outcomes"
                print(f"✅ Fetched {len(trades)} trades for market ({outcome_type})")
        else:
            if show_details:
                print(f"⚠️ No trades found for market (may be inactive or no recent trades)")
    else:
        if show_details:
            print(f"⚠️ Cannot fetch trades: No market ID found in market data")
    
    # Track if we have real trade data (for reliability warning)
    has_real_trades = not (trades_df is None or trades_df.empty)
    
    # If no trades, create minimal DataFrame with current price for prediction
    if trades_df is None or trades_df.empty:
        # Don't print warning here - will be shown in print_prediction function
        trades_df = pd.DataFrame({
            'price': [current_price],
            'size': [0],
            'timestamp': [datetime.now()]
        })
    
    prediction = predictor.predict(market, trades_df)
    
    question = market.get('question', 'Unknown Market')
    
    signal = prediction['signal']
    action = prediction['action']
    
    # Add trade data availability flag to prediction for display
    prediction['has_real_trades'] = has_real_trades
    
    if signal == 'STRONG' and action == 'BUY_YES':
        recommendation = 'STRONG BUY YES'
    elif signal == 'STRONG' and action == 'BUY_NO':
        recommendation = 'STRONG BUY NO'
    elif action == 'BUY_YES':
        recommendation = 'BUY YES'
    elif action == 'BUY_NO':
        recommendation = 'BUY NO'
    else:
        recommendation = 'HOLD'
    
    insights = []
    rsi = prediction['rsi'] * 100
    if rsi > 70:
        insights.append(f"Overbought (RSI: {rsi:.0f}) - potential reversal")
    elif rsi < 30:
        insights.append(f"Oversold (RSI: {rsi:.0f}) - potential bounce")
    
    if prediction['volatility'] > 0.05:
        insights.append(f"High volatility: {prediction['volatility']:.1%}")
    
    # Order Book Imbalance signal (from microstructure analysis)
    obi = prediction.get('order_book_imbalance', 0)
    if abs(obi) > 0.3:
        if obi > 0:
            insights.append(f"Strong bid pressure (OBI: {obi:.2f})")
        else:
            insights.append(f"Strong ask pressure (OBI: {obi:.2f})")
    
    # Terminal risk warning
    days = prediction.get('days_remaining', 30)
    if days < 7:
        reduction = prediction.get('terminal_risk_reduction', 1.0)
        insights.append(f"⚠️ Terminal risk: {days:.0f} days left (pos reduced {(1-reduction)*100:.0f}%)")
    
    # Expected Value
    ev = prediction.get('expected_value', 0)
    if abs(ev) > 0.02:
        insights.append(f"Expected Value: {ev:+.1%}")
    
    if prediction['kelly_size'] > 50:
        insights.append(f"Kelly position: ${prediction['kelly_size']:.0f}")
    
    if not insights:
        insights.append(f"Edge: {prediction['edge']:.1%} | Direction: {prediction['direction']}")
    
    return {
        'question': question,
        'prediction': {
            'current_price': prediction['current_price'],
            'predicted_price': prediction['predicted_price'],
            'price_change': prediction['predicted_price'] - prediction['current_price'],  # Absolute change
            'confidence': prediction['confidence'],
            'recommendation': recommendation,
            'direction': prediction['direction'],
            'edge': prediction['edge'],
            'kelly_size': prediction['kelly_size'],
            # New quant metrics
            'expected_value': prediction.get('expected_value', 0),
            'order_book_imbalance': prediction.get('order_book_imbalance', 0),
            'gamma_risk': prediction.get('gamma_risk', 0),
            'aggregated_probability': prediction.get('aggregated_probability', 0.5),
            # Trade data availability flag
            'has_real_trades': has_real_trades,
        },
        'insights': insights,
    }


def print_prediction(analysis, rank=None):
    """Print a single market prediction"""
    pred = analysis['prediction']
    question = analysis.get('question', 'Unknown')[:50]
    
    rec = pred['recommendation']
    if 'STRONG BUY YES' in rec:
        signal = "🟢🟢 STRONG YES"
    elif 'BUY YES' in rec:
        signal = "🟢 BUY YES"
    elif 'STRONG BUY NO' in rec:
        signal = "🔴🔴 STRONG NO"
    elif 'BUY NO' in rec:
        signal = "🔴 BUY NO"
    else:
        signal = "⚪ HOLD"
    
    prefix = f"#{rank} " if rank else ""
    
    # Display absolute edge (in cents/percentage points)
    edge_display = pred['price_change'] * 100  # Convert to cents
    
    # Check if we have real trade data (for reliability indicator)
    has_real_trades = pred.get('has_real_trades', True)  # Default to True for backward compatibility
    reliability_indicator = "" if has_real_trades else " ⚠️ (Limited Data)"
    
    print(f"\n{prefix}{question}...")
    print(f"   Current: {pred['current_price']:.1%} → Predicted: {pred['predicted_price']:.1%} ({edge_display:+.1f}¢)")
    print(f"   Signal: {signal} | Confidence: {pred['confidence']:.0%}{reliability_indicator}")
    
    # Add warning if no trade data available
    if not has_real_trades:
        print(f"   ⚠️ WARNING: No trade data available - prediction based on market metadata only")
        print(f"   ⚠️ Prediction reliability may be reduced. Use with caution.")
    
    if analysis.get('insights'):
        print(f"   💡 {analysis['insights'][0]}") 


def run_single_analysis(num_markets=10):
    """Run a single prediction analysis"""
    
    print_header()
    
    fetcher = PolymarketFetcher(verbose=False)
    predictor = create_predictor(use_optuna=False)
    
    print(f"\n📡 Fetching top {num_markets} markets by volume...")
    markets = fetcher.get_markets(limit=num_markets * 5, order='volume24hr')  # Fetch 5x to account for filtering
    
    # Filter for truly active markets:
    # 1. Has trading volume > $1000
    # 2. Price is NOT at extremes (100% or 0%) - these are effectively resolved
    # 3. NOT a Politics market (if EXCLUDE_POLITICS_MARKETS is True)
    active_markets = []
    politics_filtered = 0  # Track how many Politics markets filtered
    
    for m in markets:
        volume = float(m.get('volume24hr', 0) or 0)
        if volume <= 1000:
            continue
            
        # Get price to check if effectively resolved
        prices_str = m.get('outcomePrices', '[0.5, 0.5]')
        try:
            prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
            price = float(prices[0]) if prices else 0.5
        except:
            price = 0.5
        
        # Skip "effectively resolved" markets (price at >=95% or <=5%)
        # These have minimal trading opportunity
        if price >= 0.95 or price <= 0.05:
            continue
        
        # Skip Politics markets if filtering enabled
        if EXCLUDE_POLITICS_MARKETS and is_politics_market(m, fetcher):
            politics_filtered += 1
            continue
            
        active_markets.append(m)
        if len(active_markets) >= num_markets:
            break
    
    # Update logging to show Politics filtering statistics
    if EXCLUDE_POLITICS_MARKETS:
        print(f"✅ Found {len(active_markets)} active markets (excluded: resolved/near-resolved, Politics={politics_filtered})\n")
    else:
        print(f"✅ Found {len(active_markets)} active markets (excluded resolved/near-resolved)\n")
    
    predictions = []
    for i, market in enumerate(active_markets):
        try:
            analysis = analyze_market(predictor, fetcher, market)
            predictions.append(analysis)
            print_prediction(analysis, rank=i+1)
            time.sleep(0.2)
        except Exception as e:
            print(f"⚠️ Error: {e}")
    
    print("\n" + "="*65)
    print("📊 PREDICTION SUMMARY")
    print("="*65)
    
    strong_yes = [p for p in predictions if 'STRONG BUY YES' in p['prediction']['recommendation']]
    strong_no = [p for p in predictions if 'STRONG BUY NO' in p['prediction']['recommendation']]
    buy_yes = [p for p in predictions if p['prediction']['recommendation'] == 'BUY YES']
    buy_no = [p for p in predictions if p['prediction']['recommendation'] == 'BUY NO']
    
    print(f"\n🟢🟢 Strong YES: {len(strong_yes)}")
    print(f"🟢   Buy YES:    {len(buy_yes)}")
    print(f"🔴   Buy NO:     {len(buy_no)}")
    print(f"🔴🔴 Strong NO:  {len(strong_no)}")
    
    # Check if any predictions were made without real trade data
    predictions_without_trades = [p for p in predictions if not p['prediction'].get('has_real_trades', True)]
    if predictions_without_trades:
        print(f"\n⚠️ WARNING: No trade data available - prediction based on market metadata only")
        print(f"   ⚠️ Prediction reliability may be reduced. Use with caution.")
    
    if strong_yes or strong_no:
        print(f"\n📊 TOP OPPORTUNITIES:")
        for p in (strong_yes + strong_no)[:5]:
            q = p.get('question', '')[:45]
            curr = p['prediction']['current_price']
            pred_price = p['prediction']['predicted_price']
            edge = (pred_price - curr) * 100  # Edge in cents
            rec = p['prediction']['recommendation']
            conf = p['prediction']['confidence']
            has_real_trades = p['prediction'].get('has_real_trades', True)
            
            print(f"\n   {q}...")
            print(f"   Current: {curr:.1%} → Predicted: {pred_price:.1%} ({edge:+.1f}¢ edge)")
            print(f"   {rec} | Confidence: {conf:.0%}")
            
            # Show warning if trade data is missing for this opportunity
            if not has_real_trades:
                print(f"   ⚠️ WARNING: No trade data available - prediction based on market metadata only")
                print(f"   ⚠️ Prediction reliability may be reduced. Use with caution.")
    
    print("\n" + "="*65)
    print("⚠️  Not financial advice. Predictions based on ML patterns.")
    print("="*65 + "\n")


if __name__ == "__main__":
    import sys
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_single_analysis(num)
