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
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from polymarket_fetcher import PolymarketFetcher
from prediction_model import PolymarketPredictor, create_predictor

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass


def print_header():
    print("\n" + "="*65)
    print("   🎯 POLYMARKET LIVE PREDICTIONS")
    print("   State-of-the-Art ML (XGBoost + LightGBM + CatBoost + Stacking)")
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
    domain = fetcher.infer_market_domain(market)
    
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
        'market_id': market.get('id') or market.get('conditionId') or 'unknown',
        'question': question,
        'domain': domain,
        'prediction': {
            'current_price': prediction['current_price'],
            'predicted_price': prediction['predicted_price'],
            'price_change': prediction['predicted_price'] - prediction['current_price'],  # Absolute change
            'confidence': prediction['confidence'],
            'direction_probability': prediction.get('direction_probability', 0.5),
            'aggregated_probability': prediction.get('aggregated_probability', 0.5),
            'recommendation': recommendation,
            'direction': prediction['direction'],
            'edge': prediction['edge'],
            'kelly_size': prediction['kelly_size'],
            'expected_growth_rate': prediction.get('expected_growth_rate', 0),
            # New quant metrics
            'expected_value': prediction.get('expected_value', 0),
            'order_book_imbalance': prediction.get('order_book_imbalance', 0),
            'gamma_risk': prediction.get('gamma_risk', 0),
            'signal': prediction.get('signal', 'HOLD'),
            'action': prediction.get('action', 'HOLD'),
            'predictive_signal_verified': prediction.get('predictive_signal_verified', False),
            'trend_signal': prediction.get('trend_signal', 'FLAT'),
            'trend_label': prediction.get('trend_label', 'FLAT'),
            'trend_direction': prediction.get('trend_direction', prediction['direction']),
            'trend_score': prediction.get('trend_score', 0.0),
            'trend_interesting': prediction.get('trend_interesting', False),
        },
        'insights': insights,
    }


def print_prediction(analysis, rank=None):
    """Print a single market prediction"""
    pred = analysis['prediction']
    question = analysis.get('question', 'Unknown')[:50]
    domain = analysis.get('domain', 'other')
    
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
    print(f"   Domain: {domain}")
    print(f"   Current: {pred['current_price']:.1%} → Predicted: {pred['predicted_price']:.1%} ({edge_display:+.1f}¢)")
    print(
        f"   Signal: {signal} | Confidence: {pred['confidence']:.0%} | "
        f"P_up: {pred['direction_probability']:.1%} | P_agg: {pred['aggregated_probability']:.1%} | "
        f"ModelEdge: {'YES' if pred.get('predictive_signal_verified') else 'NO'}"
    )
    print(
        f"   Trend: {pred.get('trend_signal', 'FLAT')} "
        f"(score {pred.get('trend_score', 0.0):.1f}/100)"
    )
    
    if analysis.get('insights'):
        print(f"   💡 {analysis['insights'][0]}") 


def build_run_report(
    predictor: PolymarketPredictor,
    predictions: List[Dict[str, Any]],
    domain_counts: Dict[str, int],
    run_started_at: datetime,
    run_completed_at: datetime,
    requested_markets: int,
) -> Dict[str, Any]:
    training_metrics = predictor.training_metrics or {}
    training_domain_counts = training_metrics.get(
        'training_domain_counts',
        getattr(predictor, 'latest_training_domain_counts', {}),
    )
    training_class_counts = training_metrics.get(
        'training_class_counts',
        getattr(predictor, 'latest_training_class_counts', {}),
    )
    model_manifest = getattr(predictor, 'model_manifest', {})

    prediction_rows: List[Dict[str, Any]] = []
    for analysis in predictions:
        pred = analysis['prediction']
        prediction_rows.append(
            {
                'market_id': analysis.get('market_id', 'unknown'),
                'question': analysis.get('question', 'Unknown'),
                'domain': analysis.get('domain', 'other'),
                'current_price': pred.get('current_price', 0.5),
                'predicted_price': pred.get('predicted_price', 0.5),
                'price_change': pred.get('price_change', 0.0),
                'recommendation': pred.get('recommendation', 'HOLD'),
                'signal': pred.get('signal', 'HOLD'),
                'action': pred.get('action', 'HOLD'),
                'confidence': pred.get('confidence', 0.0),
                'direction_probability': pred.get('direction_probability', 0.5),
                'aggregated_probability': pred.get('aggregated_probability', 0.5),
                'edge': pred.get('edge', 0.0),
                'expected_value': pred.get('expected_value', 0.0),
                'expected_growth_rate': pred.get('expected_growth_rate', 0.0),
                'predictive_signal_verified': pred.get('predictive_signal_verified', False),
                'trend_signal': pred.get('trend_signal', 'FLAT'),
                'trend_label': pred.get('trend_label', 'FLAT'),
                'trend_direction': pred.get('trend_direction', pred.get('direction', 'DOWN')),
                'trend_score': pred.get('trend_score', 0.0),
                'trend_interesting': pred.get('trend_interesting', False),
                'insight': analysis.get('insights', [''])[0] if analysis.get('insights') else '',
            }
        )

    avg_confidence = float(np.mean([row['confidence'] for row in prediction_rows])) if prediction_rows else 0.0
    avg_abs_edge = float(np.mean([abs(row['price_change']) for row in prediction_rows])) if prediction_rows else 0.0
    trend_ranked = sorted(prediction_rows, key=lambda row: row.get('trend_score', 0.0), reverse=True)
    buy_yes_count = sum(1 for row in prediction_rows if 'BUY YES' in row['recommendation'])
    buy_no_count = sum(1 for row in prediction_rows if 'BUY NO' in row['recommendation'])
    strong_buy_yes_count = sum(1 for row in prediction_rows if row['recommendation'] == 'STRONG BUY YES')
    strong_buy_no_count = sum(1 for row in prediction_rows if row['recommendation'] == 'STRONG BUY NO')

    report = {
        'run': {
            'started_at': run_started_at.isoformat(),
            'completed_at': run_completed_at.isoformat(),
            'duration_seconds': round((run_completed_at - run_started_at).total_seconds(), 3),
            'requested_markets': requested_markets,
            'analyzed_markets': len(prediction_rows),
        },
        'models': model_manifest,
        'training': {
            'metrics': training_metrics,
            'domain_counts': training_domain_counts,
            'class_counts': training_class_counts,
            'predictive_signal_verified': bool(training_metrics.get('has_predictive_signal', False)),
        },
        'prediction_summary': {
            'domain_mix': domain_counts,
            'average_confidence': avg_confidence,
            'average_absolute_edge': avg_abs_edge,
            'buy_yes': buy_yes_count,
            'buy_no': buy_no_count,
            'strong_buy_yes': strong_buy_yes_count,
            'strong_buy_no': strong_buy_no_count,
            'hold': sum(1 for row in prediction_rows if row['recommendation'] == 'HOLD'),
            'trend_interesting': sum(1 for row in prediction_rows if row.get('trend_interesting')),
            'trend_up': sum(1 for row in prediction_rows if row.get('trend_direction') == 'UP'),
            'trend_down': sum(1 for row in prediction_rows if row.get('trend_direction') == 'DOWN'),
            'trend_flat': sum(1 for row in prediction_rows if row.get('trend_label') == 'FLAT'),
            'max_trend_score': max((row.get('trend_score', 0.0) for row in prediction_rows), default=0.0),
        },
        'top_trends': [
            {
                'market_id': row.get('market_id', 'unknown'),
                'question': row.get('question', 'Unknown'),
                'domain': row.get('domain', 'other'),
                'trend_signal': row.get('trend_signal', 'FLAT'),
                'trend_score': row.get('trend_score', 0.0),
                'confidence': row.get('confidence', 0.0),
                'price_change': row.get('price_change', 0.0),
                'recommendation': row.get('recommendation', 'HOLD'),
            }
            for row in trend_ranked[: min(len(trend_ranked), 8)]
        ],
        'predictions': prediction_rows,
    }
    return report


def print_detailed_run_report(report: Dict[str, Any]):
    print("\n" + "=" * 65)
    print("🧾 DETAILED RUN REPORT")
    print("=" * 65)

    run = report.get('run', {})
    models = report.get('models', {})
    training = report.get('training', {})
    summary = report.get('prediction_summary', {})

    print(
        f"Run window: {run.get('started_at', 'n/a')} -> {run.get('completed_at', 'n/a')} "
        f"({run.get('duration_seconds', 0)}s)"
    )
    print(
        f"Markets: requested {run.get('requested_markets', 0)} | "
        f"analyzed {run.get('analyzed_markets', 0)}"
    )

    base_models = models.get('base_models', [])
    print(f"Models used: {', '.join(base_models) if base_models else 'n/a'}")
    if models:
        print(
            f"Calibration: {'ON' if models.get('use_calibration', False) else 'OFF'} | "
            f"CatBoost: {'ON' if models.get('has_catboost', False) else 'OFF'}"
        )

    metrics = training.get('metrics', {})
    if metrics:
        print(
            f"Validation: split={metrics.get('split_strategy', 'n/a')} | "
            f"acc={metrics.get('direction_accuracy', 0):.1%} | "
            f"baseline={metrics.get('baseline_accuracy', 0):.1%} | "
            f"lift={metrics.get('lift_vs_baseline', 0):+.1%} | "
            f"auc={metrics.get('auc', 0):.3f} | "
            f"brier={metrics.get('brier_score', 0):.4f}"
        )
        print(
            f"Validation (balanced): acc={metrics.get('direction_balanced_accuracy', 0):.1%} | "
            f"baseline={metrics.get('baseline_balanced_accuracy', 0):.1%} | "
            f"lift={metrics.get('balanced_lift_vs_baseline', 0):+.1%}"
        )
        print(
            f"Predictive signal: "
            f"{'YES' if training.get('predictive_signal_verified', False) else 'NO'} | "
            f"high-conf={metrics.get('high_conf_accuracy', 0):.1%} "
            f"(coverage {metrics.get('high_conf_coverage', 0):.1%}) | "
            f"high-conf balanced={metrics.get('high_conf_balanced_accuracy', 0):.1%}"
        )

    if training.get('domain_counts'):
        domains = ", ".join(
            f"{domain}:{count}"
            for domain, count in sorted(training['domain_counts'].items())
        )
        print(f"Training domains: {domains}")
    if training.get('class_counts'):
        classes = ", ".join(
            f"{label}:{count}"
            for label, count in sorted(training['class_counts'].items())
        )
        print(f"Training classes: {classes}")

    domain_mix = summary.get('domain_mix', {})
    if domain_mix:
        live_domains = ", ".join(f"{domain}:{count}" for domain, count in sorted(domain_mix.items()))
        print(f"Live domain mix: {live_domains}")

    print(
        f"Prediction summary: BUY_YES={summary.get('buy_yes', 0)} | "
        f"BUY_NO={summary.get('buy_no', 0)} | HOLD={summary.get('hold', 0)} | "
        f"avg_conf={summary.get('average_confidence', 0):.1%} | "
        f"avg_|edge|={summary.get('average_absolute_edge', 0):.2%}"
    )
    print(
        f"Trend summary: interesting={summary.get('trend_interesting', 0)} | "
        f"UP={summary.get('trend_up', 0)} | DOWN={summary.get('trend_down', 0)} | "
        f"FLAT={summary.get('trend_flat', 0)} | max_score={summary.get('max_trend_score', 0.0):.1f}"
    )

    top_trends = report.get('top_trends', [])
    if top_trends:
        print("\nTrend watchlist (ranked):")
        for index, row in enumerate(top_trends, start=1):
            print(
                f"{index}. [{row['domain']}] {row['question'][:56]} | "
                f"{row['trend_signal']} ({row['trend_score']:.1f}) | "
                f"Conf={row['confidence']:.1%} | Edge={row['price_change']:+.2%} | "
                f"{row['recommendation']}"
            )

    predictions = report.get('predictions', [])
    if predictions:
        print("\nPer-market details (probability + decision):")
        for index, row in enumerate(predictions, start=1):
            print(
                f"{index}. [{row['domain']}] {row['question'][:56]} | "
                f"P_up={row['direction_probability']:.1%} | P_agg={row['aggregated_probability']:.1%} | "
                f"Conf={row['confidence']:.1%} | {row['recommendation']} | "
                f"Edge={row['price_change']:+.2%} | "
                f"Trend={row.get('trend_signal', 'FLAT')}({row.get('trend_score', 0.0):.1f}) | "
                f"ModelEdge={'YES' if row.get('predictive_signal_verified') else 'NO'}"
            )


def save_report_json(report: Dict[str, Any], report_file: str) -> Path:
    def _json_default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return str(value)

    report_path = Path(report_file)
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding='utf-8',
    )
    return report_path


def run_single_analysis(
    num_markets: int = 10,
    detailed_report: bool = True,
    report_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single prediction analysis"""
    
    run_started_at = datetime.now()
    print_header()
    
    predictor = create_predictor(use_optuna=False)
    fetcher = PolymarketFetcher(verbose=False, storage=predictor.storage)
    
    print(f"\n📡 Fetching top markets across all domains...")
    markets = fetcher.get_markets(limit=max(num_markets * 25, 300), order='volume24hr')
    active_markets = fetcher.select_diverse_active_markets(
        markets=markets,
        limit=num_markets,
        min_volume=1000,
    )

    domain_counts: Dict[str, int] = {}
    for market in active_markets:
        domain = fetcher.infer_market_domain(market)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print(f"✅ Found {len(active_markets)} active markets (excluded resolved/near-resolved)")
    if domain_counts:
        distribution = ", ".join(f"{domain}:{count}" for domain, count in sorted(domain_counts.items()))
        print(f"🌐 Domain mix: {distribution}\n")
    else:
        print("")
    
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

    trend_ranked = sorted(
        predictions,
        key=lambda row: row['prediction'].get('trend_score', 0.0),
        reverse=True,
    )
    if trend_ranked:
        print("\n📈 TREND WATCHLIST (all analyzed markets):")
        for idx, p in enumerate(trend_ranked, start=1):
            q = p.get('question', '')[:45]
            pred = p.get('prediction', {})
            print(
                f"   {idx}. {q}... | "
                f"{pred.get('trend_signal', 'FLAT')} ({pred.get('trend_score', 0.0):.1f}) | "
                f"Conf {pred.get('confidence', 0.0):.0%} | "
                f"Edge {pred.get('price_change', 0.0):+.2%}"
            )
    
    print("\n" + "="*65)
    print("⚠️  Not financial advice. Predictions based on ML patterns.")
    print("="*65 + "\n")

    run_completed_at = datetime.now()
    run_report = build_run_report(
        predictor=predictor,
        predictions=predictions,
        domain_counts=domain_counts,
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
        requested_markets=num_markets,
    )

    if predictor.storage:
        predictor.storage.save_run_report(run_report, run_type="live_prediction")

    if detailed_report:
        print_detailed_run_report(run_report)

    target_report_file = report_file or "data\\last_run_report.json"
    report_path = save_report_json(run_report, target_report_file)
    print(f"\n🗂️ Report saved: {report_path}")

    return run_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket live prediction runner")
    parser.add_argument("num", nargs="?", type=int, default=10, help="Number of markets to analyze")
    parser.add_argument(
        "--report-file",
        default="data\\last_run_report.json",
        help="Path for JSON report output",
    )
    parser.add_argument(
        "--no-detailed-report",
        action="store_true",
        help="Disable detailed report section in console output",
    )
    args = parser.parse_args()

    run_single_analysis(
        num_markets=args.num,
        detailed_report=not args.no_detailed_report,
        report_file=args.report_file,
    )
