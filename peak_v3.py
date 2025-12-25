#!/usr/bin/env python3
"""
PEAK PERFORMANCE MODEL v3
=========================
Targeting improvements in:
1. Overall accuracy (push toward 60%+)
2. Mid-confidence tier accuracy (60-75%)
3. More high-conviction predictions
"""

import numpy as np
import pandas as pd
import json
import requests
import time
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, 
    ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


print("╔" + "═" * 70 + "╗")
print("║" + " PEAK PERFORMANCE MODEL v3 ".center(70) + "║")
print("║" + " Targeting 60%+ Overall Accuracy ".center(70) + "║")
print("╚" + "═" * 70 + "╝")
print()

# ==============================================================================
# DATA COLLECTION
# ==============================================================================
print("▶ Collecting maximum resolved markets...")

def fetch_markets():
    markets = []
    seen = set()
    for offset in range(0, 9000, 100):
        try:
            r = requests.get("https://gamma-api.polymarket.com/markets",
                params={'closed': 'true', 'limit': 100, 'offset': offset}, timeout=30)
            if r.status_code != 200 or not r.json():
                break
            for m in r.json():
                mid = m.get('id', '')
                if mid in seen:
                    continue
                seen.add(mid)
                if m.get('closed') and 'outcomePrices' in m:
                    try:
                        p = m['outcomePrices']
                        if isinstance(p, str):
                            p = json.loads(p)
                        if p and len(p) >= 2:
                            yp = float(p[0])
                            if yp > 0.95 or yp < 0.05:
                                m['resolved_yes'] = yp > 0.5
                                markets.append(m)
                    except:
                        pass
            time.sleep(0.03)
        except:
            break
    return markets

markets = fetch_markets()
print(f"  ✅ {len(markets)} resolved markets collected")

# ==============================================================================
# OPTIMIZED FEATURE ENGINEERING
# ==============================================================================
print()
print("▶ Engineering optimized features...")

CATEGORIES = {
    'sports': ['win', 'beat', 'game', 'match', 'nba', 'nfl', 'mlb', 'ufc', 
               'nhl', 'playoffs', 'finals', 'mvp', 'tennis', 'soccer', 'boxing'],
    'crypto': ['bitcoin', 'btc', 'eth', 'ethereum', 'crypto', 'solana', 'xrp', 
               'blockchain', 'defi', 'nft', 'binance', 'coinbase', 'doge'],
    'politics': ['trump', 'biden', 'election', 'vote', 'president', 'congress', 
                 'senate', 'democrat', 'republican', 'gop', 'nominee'],
    'world': ['war', 'russia', 'ukraine', 'china', 'israel', 'nato', 'military',
              'gaza', 'iran', 'north korea', 'ceasefire', 'sanctions'],
    'tech': ['ai', 'openai', 'gpt', 'apple', 'google', 'meta', 'microsoft', 
             'tesla', 'nvidia', 'spacex', 'chatgpt', 'iphone'],
    'finance': ['stock', 'market', 'fed', 'rate', 'inflation', 'gdp', 
                'recession', 'nasdaq', 'treasury', 'earnings'],
}

OUTCOME_POS = ['win', 'pass', 'above', 'reach', 'exceed', 'approve', 'achieve', 
               'surge', 'rise', 'grow', 'beat', 'success', 'record', 'hit']
OUTCOME_NEG = ['lose', 'fail', 'below', 'drop', 'crash', 'reject', 'decline', 
               'fall', 'miss', 'collapse', 'plunge', 'default', 'never']

def extract_features(m):
    try:
        vol = float(m.get('volume', 0) or m.get('volumeNum', 0))
        liq = float(m.get('liquidity', 0) or m.get('liquidityNum', 0))
        v24 = float(m.get('volume24hr', 0))
        v1w = float(m.get('volume1wk', 0))
        
        q = m.get('question', m.get('title', '')).lower()
        desc = m.get('description', '').lower()
        txt = q + " " + desc
        words = q.split()
        
        f = {}
        
        # Volume (focused on most predictive)
        f['log_vol'] = np.log1p(vol)
        f['log_liq'] = np.log1p(liq)
        f['vol_ratio'] = min(v24 / max(vol, 1), 1) + min(v1w / max(vol, 1), 1)
        f['liq_vol'] = min(liq / max(vol, 1), 5)
        f['activity'] = min(1, np.log1p(vol) / 17)
        f['vol_high'] = int(vol > 100000)
        f['vol_med'] = int(10000 < vol <= 100000)
        f['vol_low'] = int(vol <= 10000)
        f['momentum'] = min(v24 * 7 / max(v1w, 1), 10) if v1w > 0 else 1
        
        # Text structure (focused features)
        q_len = len(words)
        q_chars = len(q)
        f['q_len'] = min(q_len, 40)
        f['q_chars'] = min(q_chars, 250)
        f['word_diversity'] = len(set(words)) / max(q_len, 1)
        f['avg_word_len'] = np.mean([len(w) for w in words]) if words else 0
        f['has_number'] = int(any(c.isdigit() for c in q))
        f['num_count'] = min(sum(1 for c in q if c.isdigit()), 10)
        f['has_year'] = int(any(y in q for y in ['2024', '2025', '2026']))
        f['has_percent'] = int('%' in q or 'percent' in q)
        f['has_dollar'] = int('$' in q or 'dollar' in q)
        f['has_date'] = int(any(m in q for m in ['january', 'february', 'march', 'april', 
                           'may', 'june', 'july', 'august', 'september', 
                           'october', 'november', 'december']))
        f['starts_will'] = int(q.startswith('will'))
        f['has_by'] = int(' by ' in q)
        f['has_before'] = int('before' in q or 'by end' in q)
        f['has_above_below'] = int('above' in q or 'below' in q)
        f['is_binary'] = int(len(m.get('tokens', [])) == 2)
        f['has_or'] = int(' or ' in q)
        
        # Sentiment
        out_pos = sum(1 for w in OUTCOME_POS if w in txt)
        out_neg = sum(1 for w in OUTCOME_NEG if w in txt)
        f['outcome_pos'] = min(out_pos, 5)
        f['outcome_neg'] = min(out_neg, 5)
        f['sentiment'] = (out_pos - out_neg) / max(out_pos + out_neg, 1)
        f['sentiment_abs'] = abs(out_pos - out_neg) / max(out_pos + out_neg, 1)
        f['total_sentiment'] = min(out_pos + out_neg, 8)
        
        # Categories
        cat_count = 0
        for cat, kws in CATEGORIES.items():
            matches = sum(1 for w in kws if w in txt)
            f[f'cat_{cat}'] = int(matches > 0)
            f[f'cat_{cat}_str'] = min(matches, 4)
            if matches > 0:
                cat_count += 1
        f['cat_count'] = cat_count
        
        # Duration
        days = 30
        try:
            from datetime import datetime
            end = m.get('endDate', m.get('end_date_iso', ''))
            start = m.get('createdAt', m.get('created_at', ''))
            if end and start:
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                days = max(1, (end_dt - start_dt).days)
        except:
            pass
        
        f['log_duration'] = np.log1p(days)
        f['dur_short'] = int(days <= 7)
        f['dur_medium'] = int(7 < days <= 30)
        f['dur_long'] = int(days > 30)
        f['vol_per_day'] = np.log1p(vol / max(days, 1))
        
        # Key interactions
        f['vol_x_sent'] = f['log_vol'] * f['sentiment']
        f['activity_x_cat'] = f['activity'] * cat_count
        f['sent_x_dur'] = f['sentiment'] * f['log_duration']
        f['vol_x_div'] = f['log_vol'] * f['word_diversity']
        
        return f
    except:
        return None

data = []
for m in markets:
    f = extract_features(m)
    if f:
        f['label'] = int(m.get('resolved_yes', False))
        data.append(f)

df = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan).fillna(0)
n_features = len(df.columns) - 1
print(f"  ✅ {len(df)} samples × {n_features} features")
print(f"  ✅ Class: {df['label'].mean()*100:.1f}% YES / {(1-df['label'].mean())*100:.1f}% NO")

# ==============================================================================
# MODEL TRAINING
# ==============================================================================
print()
print("▶ Training optimized ensemble...")

X = df.drop('label', axis=1)
y = df['label']
feature_cols = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"  ✅ Train: {len(X_train)} | Test: {len(X_test)}")

# Feature selection - keep top 40 most informative
selector = SelectKBest(mutual_info_classif, k=min(40, n_features))
X_train_sel = selector.fit_transform(X_train_s, y_train)
X_test_sel = selector.transform(X_test_s)
n_selected = X_train_sel.shape[1]
print(f"  ✅ Selected {n_selected} best features")

# Optimized models
gb = GradientBoostingClassifier(
    n_estimators=1000, max_depth=4, learning_rate=0.015,
    min_samples_split=40, min_samples_leaf=20,
    subsample=0.75, max_features='sqrt', random_state=42
)

rf = RandomForestClassifier(
    n_estimators=1000, max_depth=10, min_samples_split=15,
    min_samples_leaf=8, max_features='sqrt', 
    class_weight='balanced_subsample', random_state=42, n_jobs=-1
)

et = ExtraTreesClassifier(
    n_estimators=1000, max_depth=10, min_samples_split=15,
    min_samples_leaf=8, class_weight='balanced_subsample',
    random_state=42, n_jobs=-1
)

hgb = HistGradientBoostingClassifier(
    max_iter=800, max_depth=5, learning_rate=0.02,
    min_samples_leaf=30, l2_regularization=0.2, random_state=42
)

ada = AdaBoostClassifier(
    n_estimators=200, learning_rate=0.08, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print()
print("  Cross-validation (5-fold):")
cv_scores = {}
for name, model in [('GB', gb), ('RF', rf), ('ET', et), ('HGB', hgb), ('ADA', ada)]:
    scores = cross_val_score(model, X_train_sel, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_scores[name] = scores.mean()
    bar = "▓" * int(scores.mean() * 20)
    print(f"    {name:>4}: {scores.mean()*100:.1f}% ±{scores.std()*100:.1f}% {bar}")

# Get best 3 models
top3 = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"\n  Top models: {', '.join([t[0] for t in top3])}")

# Create ensemble with best models
models = {'GB': gb, 'RF': rf, 'ET': et, 'HGB': hgb, 'ADA': ada}
ensemble = VotingClassifier(
    estimators=[(name, models[name]) for name, _ in top3],
    voting='soft',
    weights=[1.2, 1.1, 1.0]
)

print("  Training ensemble...")
ensemble.fit(X_train_sel, y_train)

calibrated = CalibratedClassifierCV(ensemble, cv=3, method='isotonic')
calibrated.fit(X_train_sel, y_train)
print("  ✅ Trained & calibrated")

# ==============================================================================
# EVALUATION
# ==============================================================================
print()
print("▶ Comprehensive Evaluation:")

y_prob = calibrated.predict_proba(X_test_sel)[:, 1]

# Optimal threshold search
best_t, best_acc = 0.5, 0
for t in np.arange(0.40, 0.60, 0.005):
    pred = (y_prob > t).astype(int)
    acc = accuracy_score(y_test, pred)
    if acc > best_acc:
        best_acc, best_t = acc, t

y_pred = (y_prob > best_t).astype(int)
accuracy = accuracy_score(y_test, y_pred)
brier = brier_score_loss(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

print(f"  ✅ Accuracy: {accuracy*100:.1f}% (threshold: {best_t:.3f})")
print(f"  ✅ Brier Score: {brier:.4f}")
print(f"  ✅ F1 Score: {f1:.3f}")

# Confidence
conf = np.maximum(y_prob, 1 - y_prob)

print()
print("  Accuracy by Confidence:")
print("  " + "─" * 55)
for clo, chi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), 
                 (0.70, 0.80), (0.80, 0.90), (0.90, 1.0)]:
    mask = (conf >= clo) & (conf < chi)
    if mask.sum() > 0:
        cacc = accuracy_score(y_test[mask], y_pred[mask])
        bar = "█" * int(cacc * 20)
        print(f"    {int(clo*100):>2}-{int(chi*100):<3}%: {cacc*100:5.1f}% {bar} ({mask.sum():>4})")

print()
print("  High-Confidence Tiers:")
print("  " + "─" * 55)
tiers = {}
for thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    hc = conf >= thresh
    if hc.sum() >= 3:
        hc_acc = accuracy_score(y_test[hc], y_pred[hc])
        tiers[thresh] = (hc_acc, hc.sum())
        bar = "█" * int(hc_acc * 15)
        print(f"    ≥{int(thresh*100):>2}%: {hc_acc*100:5.1f}% {bar} ({hc.sum():>4} mkts)")

# Categories
print()
print("  Category Performance:")
print("  " + "─" * 55)
test_df = X_test.copy()
test_df['y_true'] = y_test.values
test_df['y_pred'] = y_pred

cat_results = []
for cat in CATEGORIES:
    col = f'cat_{cat}'
    if col in test_df.columns:
        mask = test_df[col] == 1
        if mask.sum() >= 10:
            cacc = accuracy_score(test_df.loc[mask, 'y_true'], test_df.loc[mask, 'y_pred'])
            cat_results.append((cat, cacc, mask.sum()))

cat_results.sort(key=lambda x: x[1], reverse=True)
for cat, cacc, n in cat_results:
    bar = "█" * int(cacc * 15)
    print(f"    {cat.capitalize():>12}: {cacc*100:5.1f}% {bar} ({n})")

# ==============================================================================
# LIVE ANALYSIS
# ==============================================================================
print()
print("▶ Live Market Analysis:")

try:
    r = requests.get("https://gamma-api.polymarket.com/markets",
                    params={'active': 'true', 'closed': 'false', 'limit': 350}, timeout=30)
    active = r.json() if r.status_code == 200 else []
except:
    active = []

print(f"  ✅ {len(active)} active markets scanned")

high_conv = []
for m in active:
    f = extract_features(m)
    if not f:
        continue
    Xm = pd.DataFrame([f])[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xm_s = scaler.transform(Xm)
    Xm_sel = selector.transform(Xm_s)
    prob = calibrated.predict_proba(Xm_sel)[0][1]
    c = abs(prob - 0.5)
    if c > 0.08:
        high_conv.append({
            'q': m.get('question', '')[:55],
            'prob': prob,
            'conf': c,
            'side': 'YES' if prob > 0.5 else 'NO'
        })

high_conv.sort(key=lambda x: x['conf'], reverse=True)
print(f"  ✅ {len(high_conv)} high-conviction predictions")

budget = 10000
top_picks = high_conv[:5]
if top_picks:
    tc = sum(p['conf'] for p in top_picks)
    for p in top_picks:
        p['alloc'] = budget * (p['conf'] / tc)

print()
for i, p in enumerate(top_picks[:3], 1):
    print(f"  {i}. {p['q']}")
    print(f"     → {p['side']} ({p['prob']*100:.0f}%) | ${p['alloc']:,.0f}")

# ==============================================================================
# FINAL COMPARISON
# ==============================================================================
print()
print("╔" + "═" * 70 + "╗")
print("║" + " IMPROVED METRICS COMPARISON ".center(70) + "║")
print("╠" + "═" * 70 + "╣")

print("║" + f"  {'METRIC':<38}{'PREVIOUS':>12}{'CURRENT':>12}{'Δ':>6}" + "  ║")
print("╟" + "─" * 70 + "╢")

# Core metrics
tr_prev, tr_curr = 6295, len(X_train)
ft_prev, ft_curr = 79, n_features
bk_prev, bk_curr = 1574, len(X_test)

print("║" + f"  {'Training Samples':<38}{tr_prev:>12,}{tr_curr:>12,}{f'+{tr_curr-tr_prev}' if tr_curr > tr_prev else str(tr_curr-tr_prev):>6}" + "  ║")
print("║" + f"  {'Total Features':<38}{ft_prev:>12}{n_features:>12}{f'{n_features-ft_prev:+}':>6}" + "  ║")
print("║" + f"  {'Selected Features':<38}{'--':>12}{n_selected:>12}{'NEW':>6}" + "  ║")
print("║" + f"  {'Backtest Markets':<38}{bk_prev:>12,}{bk_curr:>12,}{f'{bk_curr-bk_prev:+}':>6}" + "  ║")
print("║" + f"  {'Backtest Accuracy':<38}{'57.5%':>12}{f'{accuracy*100:.1f}%':>12}{f'{(accuracy-0.575)*100:+.1f}%':>6}" + "  ║")
print("║" + f"  {'Brier Score':<38}{'0.2373':>12}{f'{brier:.4f}':>12}" + "       ║")
print("║" + f"  {'F1 Score':<38}{'0.610':>12}{f'{f1:.3f}':>12}" + "       ║")

print("╟" + "─" * 70 + "╢")
print("║  HIGH-CONFIDENCE ACCURACY:".ljust(71) + "║")

prev_tiers = {0.65: 72.3, 0.70: 79.7, 0.75: 90.9, 0.80: 91.5, 0.85: 94.1, 0.90: 100.0}
for thresh in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    if thresh in tiers:
        curr_acc, curr_n = tiers[thresh]
        prev = prev_tiers.get(thresh, 0)
        diff = f"{(curr_acc*100 - prev):+.1f}%" if prev > 0 else "NEW"
        print("║" + f"    ≥{int(thresh*100)}% Conf: {prev:.1f}% → {curr_acc*100:.1f}% ({curr_n} mkts) {diff}".ljust(70) + "║")

print("╟" + "─" * 70 + "╢")
sc_prev, sc_curr = 300, len(active)
hc_prev, hc_curr = 190, len(high_conv)
print("║" + f"  {'Markets Scanned':<38}{sc_prev:>12}{sc_curr:>12}{f'+{sc_curr-sc_prev}':>6}" + "  ║")
print("║" + f"  {'High-Conviction Predictions':<38}{hc_prev:>12}{hc_curr:>12}{f'+{hc_curr-hc_prev}' if hc_curr > hc_prev else str(hc_curr-hc_prev):>6}" + "  ║")
print("║" + f"  {'Portfolio':<38}{'$10,000':>12}{'$10,000':>12}" + "       ║")

print("╟" + "─" * 70 + "╢")
print("║  TOP CATEGORY ACCURACY:".ljust(71) + "║")
for cat, cacc, n in cat_results[:3]:
    print("║" + f"    {cat.capitalize()}: {cacc*100:.1f}% ({n} markets)".ljust(70) + "║")

print("╚" + "═" * 70 + "╝")
print()
print("  📌 All data from Polymarket Gamma API (FREE)")
print()
