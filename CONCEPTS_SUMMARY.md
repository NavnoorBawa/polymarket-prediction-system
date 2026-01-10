# Key Concepts & Logic Summary - Prediction Model

## Quick Reference Guide

### 🎯 Core Philosophy

**This is a probability-based trading system**, not a binary classifier. Everything is about:
- Estimating probabilities accurately
- Sizing positions based on edge and confidence
- Managing risk as contracts approach expiration

---

## 🔑 7 Core Components Explained Simply

### 1. OrderBookMicrostructure
**What**: Analyzes buy vs sell pressure to predict short-term moves  
**Formula**: `OBI = (bids - asks) / (bids + asks)`  
**Key Insight**: More bids than asks → Price likely to increase  
**Research**: OBI explains 65% of short-term price variance (Cont et al. 2014)

### 2. ArbitrageDetector
**What**: Finds guaranteed profit opportunities  
**Key Rule**: For mutually exclusive outcomes, prices must sum to $1.00  
**Example**: Prices [0.38, 0.33, 0.27] = $0.98 → Buy all for guaranteed $0.02 profit  
**Reality Check**: Must account for 1.5% transaction fees!

### 3. TerminalRiskManager
**What**: Reduces position size as contract nears expiration  
**Why**: Binary options get very volatile near expiration (gamma risk)  
**Formula**: `Position(t) = Initial × √(time_remaining / initial_time)`  
**Example**: 30 days → 7 days = 52% position reduction

### 4. BayesianAggregator
**What**: Combines multiple probability estimates into one optimal estimate  
**Formula**: `P_final = (w₁P₁ + w₂P₂ + w₃P₃ + w₄P₄) / Σw`  
**Weights**: Model (40%), Market (35%), Momentum (15%), Sentiment (10%)  
**Optimization**: Weights learned by minimizing Brier score on historical data

### 5. TechnicalIndicators
**What**: Standard technical analysis (RSI, MACD, Bollinger Bands, etc.)  
**Purpose**: Capture momentum, volatility, and mean reversion signals  
**Why**: ML models benefit from these engineered features

### 6. KellyCriterion
**What**: Mathematical formula for optimal position sizing  
**Full Kelly**: `f* = (P_true - P_market) / (1 - P_market)`  
**Fractional Kelly**: Use 25% of full Kelly (industry standard)  
**Example**: Model says 55%, market says 48% → Full Kelly = 13.4% of bankroll  
**Real World**: Use Quarter Kelly (0.25) = 3.35% of bankroll (safer)

### 7. Machine Learning Pipeline
**What**: Stacking ensemble of multiple models  
**Models**: XGBoost, LightGBM, HistGradient, ExtraTrees, RandomForest  
**Stacking**: Meta-learner combines base models optimally  
**Calibration**: Probabilities adjusted to be statistically reliable  
**Output**: Two models trained:
- **Direction Model**: P(price goes UP) [binary classifier]
- **Price Model**: Predicted future price [regressor]

---

## 📊 Prediction Flow (Step-by-Step)

```
1. FEATURE EXTRACTION
   ├─ Trade features (RSI, momentum, volume, etc.)
   ├─ Market features (price, liquidity, days remaining)
   └─ Combine into 10-dimensional vector

2. MODEL PREDICTIONS
   ├─ Direction Model → P(price UP) = 0.68
   ├─ Price Model → Raw predicted price = 0.52
   └─ Scale features using trained scaler

3. PRICE REFINEMENT
   ├─ Calculate move: 0.52 - 0.48 = +0.04 (4% increase)
   ├─ Check model agreement (both agree on UP)
   ├─ Apply confidence scaling (models agree → boost confidence)
   └─ Cap move (max 20% per prediction)

4. ORDER BOOK ANALYSIS
   ├─ Calculate OBI: +0.35 (more bids than asks)
   ├─ Imbalance ratio: 0.67 (67% bids)
   └─ Microstructure signal: UP with 75% confidence

5. BAYESIAN AGGREGATION
   ├─ Model: 0.68 (68% probability)
   ├─ Market: 0.48 (48% current price)
   ├─ Momentum: 0.55 (slight upward momentum)
   ├─ Sentiment: 0.75 (order flow bullish)
   └─ Weighted average → Aggregated probability = 0.61

6. CONFIDENCE CALCULATION
   ├─ Base confidence from model: 65%
   ├─ Boost if models agree: +5%
   ├─ Boost for large moves: +3%
   └─ Blend with calibrated confidence → Final: 72%

7. POSITION SIZING (KELLY)
   ├─ Edge: |0.52 - 0.48| = 0.04 (4%)
   ├─ Full Kelly: (0.52 - 0.48) / (1 - 0.48) = 0.077 (7.7%)
   ├─ Quarter Kelly: 0.077 × 0.25 = 0.019 (1.9%)
   ├─ With confidence: 0.019 × 0.72 = 0.014 (1.4%)
   ├─ Position: $10,000 × 0.014 = $140
   └─ Cap at 25%: min($140, $2,500) = $140 ✓

8. TERMINAL RISK ADJUSTMENT
   ├─ Days remaining: 5 (less than 7, need reduction)
   ├─ Reduction factor: √(5/7) = 0.85
   ├─ Volatility adjustment: ×0.9 (high volatility)
   └─ Final position: $140 × 0.85 × 0.9 = $107

9. SIGNAL CLASSIFICATION
   ├─ Price change: 0.04 (4%)
   ├─ Confidence: 72%
   ├─ Check thresholds (normal price range)
   └─ Signal: "MODERATE" (edge > 5%, conf > 55%)

10. OUTPUT
    └─ Action: BUY_YES, Position: $107, Confidence: 72%, Signal: MODERATE
```

---

## 🧮 Key Formulas Cheat Sheet

### Order Book Imbalance (OBI)
```
OBI = (Q_bid - Q_ask) / (Q_bid + Q_ask)
```
- Range: [-1, +1]
- Positive → Bullish, Negative → Bearish

### Volume-Adjusted Mid Price (VAMP)
```
VAMP = (P_bid × Q_ask + P_ask × Q_bid) / (Q_bid + Q_ask)
```
- Better than simple mid-price
- Weights by opposite-side liquidity

### Full Kelly Criterion
```
f* = (P_true - P_market) / (1 - P_market)
```
- For binary contracts (prediction markets)
- P_true = Your estimated probability
- P_market = Current market price

### Fractional Kelly
```
Position = f* × fraction × confidence × bankroll
```
- Industry standard: fraction = 0.25 (Quarter Kelly)
- Max position: 25% of bankroll per trade

### Terminal Risk Position Scaling
```
Position(t) = Initial × √(T_remaining / T_initial)
```
- Reduces position as time remaining decreases
- Square-root scaling (more conservative than linear)

### Brier Score (Calibration Metric)
```
Brier = (1/N) × Σ(P_predicted - Y_actual)²
```
- Range: [0, 1], Lower is better
- Measures probability calibration quality
- Perfect predictions = 0

### Expected Value
```
EV(YES) = P_true - P_market
EV(NO) = (1 - P_true) - (1 - P_market)
```
- Positive EV → Profitable trade in expectation
- Used to determine BUY_YES vs BUY_NO

---

## 🎓 Learning Path

### Beginner Level
1. Understand what prediction markets are (YES/NO contracts)
2. Learn basic probability concepts (0-1 range)
3. Understand expected value (EV)
4. Read about Kelly Criterion basics

### Intermediate Level
1. Learn order book microstructure (bids vs asks)
2. Understand arbitrage opportunities
3. Study binary options Greeks (gamma risk)
4. Learn about probability calibration

### Advanced Level
1. Master stacking ensembles in ML
2. Understand Bayesian model averaging
3. Study options pricing theory (for terminal risk)
4. Learn about Brier score and proper scoring rules

---

## 🔍 Common Questions Answered

### Q: Why two models (direction + price)?
**A**: Direction model predicts probability of price increase (binary). Price model predicts actual future price (continuous). Using both provides redundancy and better confidence estimates.

### Q: Why calibrate probabilities?
**A**: Raw ML outputs are often poorly calibrated (e.g., predicts 70% but only occurs 60% of the time). Calibration adjusts these to match actual frequencies, essential for proper Kelly sizing.

### Q: Why fractional Kelly (25%)?
**A**: Full Kelly is too aggressive (33% chance of halving bankroll before doubling). Quarter Kelly is safer (<3% chance of ruin) and is industry standard. Better to survive long-term than maximize short-term gains.

### Q: Why reduce positions near expiration?
**A**: Binary options have increasing gamma risk (price sensitivity to underlying changes) as expiration approaches. Small movements cause large P&L swings. Risk management requires position reduction.

### Q: Why 10 specific features?
**A**: These 10 features capture:
- Current state (price, volume, liquidity)
- Momentum (short, medium, long-term)
- Volatility and mean reversion (RSI, Bollinger)
- Order flow (imbalance)
- Market structure (spread)

More features risk overfitting on limited real data.

### Q: What if models disagree?
**A**: Reduce confidence significantly (0.3 vs 0.6-1.0). Disagreement indicates uncertainty. Better to pass on uncertain trades than overbet.

### Q: How is confidence calculated?
**A**: Base confidence (55-90%) from model certainty, boosted by:
- Model agreement (+10% if direction + price models agree)
- Momentum alignment (+10% if model agrees with momentum)
- Move magnitude (larger moves → higher confidence)

Then blended with calibrated confidence model.

---

## ⚠️ Critical Gotchas

### 1. Feature Order Mismatch
**Problem**: Training uses features in order [A, B, C], prediction uses [B, A, C]  
**Result**: Garbage predictions (model sees wrong features)  
**Solution**: Always use explicit feature list, verify order matches

### 2. Uncalibrated Probabilities
**Problem**: Model outputs 80% but actual is 60% → Overbetting → Ruin  
**Solution**: Always use CalibratedClassifierCV with sigmoid/isotonic scaling

### 3. Ignoring Terminal Risk
**Problem**: Large positions near expiration → Gamma explosion → Catastrophic loss  
**Solution**: Always apply terminal risk reduction for contracts < 7 days

### 4. Class Imbalance
**Problem**: Training data has 90% YES, 10% NO → Model always predicts YES  
**Solution**: Check class balance, use stratified splits, adjust labels based on prices

### 5. Data Leakage
**Problem**: Using future information (e.g., resolution price) in predictions  
**Solution**: Only use historical trades, market price at prediction time

### 6. Overconfidence
**Problem**: Confidence > 90% → Overbetting on uncertain predictions  
**Solution**: Cap confidence at 88%, reduce when models disagree

---

## 📈 Performance Metrics Explained

### Accuracy
- % of correct direction predictions
- >55% is good for prediction markets
- >60% is excellent

### Brier Score
- Probability calibration quality
- Range: [0, 1], Lower is better
- <0.20 is good, <0.15 is excellent
- Perfect = 0.00

### Log Loss
- Penalizes confident wrong predictions heavily
- Lower is better
- Random baseline = 0.693
- <0.60 is good

### Calibration Error
- Average difference between predicted and actual frequencies
- <0.05 is well-calibrated
- <0.03 is excellent

### Price RMSE
- Root mean squared error for price predictions
- <0.10 is good
- <0.05 is excellent

---

## 🛠️ Debugging Checklist

When predictions seem wrong, check:

- [ ] Features match training exactly (order and values)
- [ ] Scaler was fitted on training data (not refit)
- [ ] Model is actually trained (is_trained = True)
- [ ] No NaN or Inf values in features
- [ ] Confidence is reasonable (52-88% range)
- [ ] Position size is capped (max 25% of bankroll)
- [ ] Terminal risk applied if days_remaining < 7
- [ ] Class balance handled during training
- [ ] Probabilities are calibrated (use calibration=True)

---

## 💡 Pro Tips

1. **Start Conservative**: Use kelly_fraction=0.20 (20% of Kelly) until confident in model
2. **Monitor Calibration**: Check Brier score regularly, recalibrate if it increases
3. **Feature Engineering**: Experiment with different feature combinations (but always maintain consistency!)
4. **Backtest Properly**: Use walk-forward validation, not simple train/test split
5. **Risk First**: Better to pass on trades than overbet. Preservation > Growth
6. **Real Data Only**: Synthetic data doesn't capture real market dynamics
7. **Watch Terminal Risk**: Always reduce positions in final week
8. **Model Agreement**: Higher confidence when models agree is crucial
9. **Arbitrage First**: Always check for arbitrage before model predictions
10. **Continuous Learning**: Update model weights (Bayesian) as more data arrives

---

## 🚀 Quick Start Example

```python
from prediction_model import create_predictor
import pandas as pd

# 1. Create and train predictor
predictor = create_predictor(
    kelly_fraction=0.25,  # Quarter Kelly (industry standard)
    bankroll=10000,       # $10,000 starting capital
    n_markets=100         # Fetch 100 markets for training
)

# 2. Prepare market data
market = {
    'outcomePrices': '[0.48, 0.52]',  # YES price = 0.48, NO = 0.52
    'volume24hr': 50000,
    'liquidity': 100000,
    'endDate': '2024-12-31',
    # ... other market fields
}

trades_df = pd.DataFrame([
    {'price': 0.47, 'size': 100, 'side': 'buy'},
    {'price': 0.48, 'size': 200, 'side': 'sell'},
    # ... more trades
])

# 3. Make prediction
prediction = predictor.predict(market, trades_df)

# 4. Interpret results
print(f"Action: {prediction['action']}")              # BUY_YES or BUY_NO or HOLD
print(f"Position Size: ${prediction['kelly_size']:.2f}")  # How much to bet
print(f"Confidence: {prediction['confidence']:.1%}")  # Model confidence
print(f"Predicted Price: {prediction['predicted_price']:.2f}")  # Expected price
print(f"Current Price: {prediction['current_price']:.2f}")  # Market price
print(f"Edge: {prediction['edge']:.4f}")              # Expected advantage
print(f"Signal: {prediction['signal']}")              # STRONG/MODERATE/WEAK/HOLD
```

---

## 📚 Further Reading

1. **Kelly Criterion**: Essential for position sizing
2. **Options Greeks**: Understand gamma risk (terminal risk)
3. **Order Book Microstructure**: Understand OBI and VAMP
4. **Probability Calibration**: Platt scaling, isotonic regression
5. **Stacking Ensembles**: Advanced ML technique
6. **Brier Score**: Proper scoring rules for probabilities
7. **Bayesian Model Averaging**: Combining multiple models optimally

---

*This summary complements the detailed tutorial. For code-level details, see `TUTORIAL_prediction_model.md`.*
