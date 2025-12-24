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
from typing import List, Dict

import pandas as pd
import numpy as np

from polymarket_fetcher import PolymarketFetcher
from prediction_model import PolymarketPredictor, create_predictor


def print_header():
    print("\n" + "="*65)
    print("   🎯 POLYMARKET LIVE PREDICTIONS")
    print("   State-of-the-Art ML (XGBoost + LightGBM + Stacking)")
    print("="*65)


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
    trades_df = pd.DataFrame()
    if yes_token:
        trades = fetcher.get_trades(yes_token, limit=500)
        if trades:
            trades_df = fetcher.trades_to_dataframe(trades)
    
    # If no trades, create minimal DataFrame with current price for prediction
    if trades_df is None or trades_df.empty:
        trades_df = pd.DataFrame({
            'price': [current_price],
            'size': [0],
            'timestamp': [datetime.now()]
        })
    
    prediction = predictor.predict(market, trades_df)
    
    question = market.get('question', 'Unknown Market')
    
    signal = prediction['signal']
    action = prediction['action']
    
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
    
    print(f"\n{prefix}{question}...")
    print(f"   Current: {pred['current_price']:.1%} → Predicted: {pred['predicted_price']:.1%} ({edge_display:+.1f}¢)")
    print(f"   Signal: {signal} | Confidence: {pred['confidence']:.0%}")
    
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
    active_markets = []
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
            
        active_markets.append(m)
        if len(active_markets) >= num_markets:
            break
    
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
    
    if strong_yes or strong_no:
        print(f"\n📊 TOP OPPORTUNITIES:")
        for p in (strong_yes + strong_no)[:5]:
            q = p.get('question', '')[:45]
            curr = p['prediction']['current_price']
            pred_price = p['prediction']['predicted_price']
            edge = (pred_price - curr) * 100  # Edge in cents
            rec = p['prediction']['recommendation']
            conf = p['prediction']['confidence']
            print(f"\n   {q}...")
            print(f"   Current: {curr:.1%} → Predicted: {pred_price:.1%} ({edge:+.1f}¢ edge)")
            print(f"   {rec} | Confidence: {conf:.0%}")
    
    print("\n" + "="*65)
    print("⚠️  Not financial advice. Predictions based on ML patterns.")
    print("="*65 + "\n")


if __name__ == "__main__":
    import sys
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_single_analysis(num)
