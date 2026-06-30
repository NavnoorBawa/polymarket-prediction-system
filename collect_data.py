"""
Historical data collector for the Polymarket prediction system.

Repeatedly harvests FRESH market samples from the Polymarket API and accumulates
them (deduped) into the persistent DB, WITHOUT training any model. This is the
"foundations" tool: run it periodically to grow the training pool so the models
have enough non-skewed history to learn from.

Usage:
    python collect_data.py --iterations 10 --markets 40 --sleep 5

Each iteration:
  1. Fetches fresh diverse markets from the API.
  2. Saves new samples (duplicates are skipped via sample_id dedup).
  3. Reports pool growth and class/domain balance.
"""

import argparse
import os
import sys
import time
from collections import Counter

# Force UTF-8 console on Windows so emojis/logging don't crash.
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from chdb_memory import PolymarketCHDBStore
from polymarket_fetcher import PolymarketFetcher


def _class_balance(storage) -> str:
    """Return a short string describing positive/negative class balance in the pool."""
    try:
        rows = storage._query(
            "SELECT outcome, count(DISTINCT sample_id) FROM training_samples GROUP BY outcome"
        )
    except Exception:
        return "n/a"
    counts = {int(r[0]): int(r[1]) for r in rows if r and r[0] is not None}
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    total = pos + neg
    if total == 0:
        return "empty"
    return f"{pos} up / {neg} down ({pos / total:.0%} positive)"


def _domain_balance(storage, fetcher, sample_limit: int = 2000) -> str:
    """Approximate domain distribution across the most recent accumulated samples."""
    samples = storage.load_training_samples(limit=sample_limit)
    if not samples:
        return "empty"
    domains = Counter(
        sample.get("domain")
        or fetcher.infer_market_domain({"question": sample.get("question", "")})
        for sample in samples
    )
    top = ", ".join(f"{name}:{count}" for name, count in domains.most_common(6))
    return f"{len(domains)} domains -> {top}"


def collect(iterations: int, markets: int, sleep_seconds: float,
            min_volume: float, storage_dir: str) -> None:
    storage = PolymarketCHDBStore(db_dir=storage_dir, prefer_chdb=True, verbose=True)
    fetcher = PolymarketFetcher(verbose=True, storage=storage)

    print("=" * 70)
    print("📥 POLYMARKET DATA COLLECTOR (no training, accumulation only)")
    print(f"   backend: {storage.backend} @ {storage.db_path}")
    print(f"   iterations={iterations} markets/iter={markets} sleep={sleep_seconds}s")
    print("=" * 70)

    start_pool = storage.count_training_samples()
    print(f"\n📊 Starting accumulated pool: {start_pool} distinct samples")

    total_new = 0
    for i in range(1, iterations + 1):
        print("\n" + "-" * 70)
        print(f"Iteration {i}/{iterations}")
        print("-" * 70)

        before = storage.count_training_samples()
        try:
            # force_refresh=True guarantees a fresh API harvest each iteration.
            fetcher.fetch_real_training_data(
                n_markets=markets,
                min_volume=min_volume,
                include_closed=True,
                train_pool_size=before + markets,
                force_refresh=True,
            )
        except Exception as exc:  # network/API hiccups should not abort the loop
            print(f"⚠️  Iteration {i} failed: {exc}")
            time.sleep(sleep_seconds)
            continue

        after = storage.count_training_samples()
        new_this_iter = after - before
        total_new += max(new_this_iter, 0)

        print(f"\n✅ Pool: {before} -> {after} (+{new_this_iter} new)")
        print(f"   Class balance: {_class_balance(storage)}")
        print(f"   Domain spread: {_domain_balance(storage, fetcher)}")

        if i < iterations:
            time.sleep(sleep_seconds)

    end_pool = storage.count_training_samples()
    print("\n" + "=" * 70)
    print("📈 COLLECTION SUMMARY")
    print(f"   Pool grew: {start_pool} -> {end_pool} (+{end_pool - start_pool})")
    print(f"   New samples added this session: {total_new}")
    print(f"   Final class balance: {_class_balance(storage)}")
    print(f"   Final domain spread: {_domain_balance(storage, fetcher)}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Accumulate Polymarket training data in the DB")
    parser.add_argument("--iterations", type=int, default=10, help="Number of harvest iterations")
    parser.add_argument("--markets", type=int, default=40, help="Fresh markets to harvest per iteration")
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds to sleep between iterations")
    parser.add_argument("--min-volume", type=float, default=1000.0, help="Minimum 24h volume filter")
    parser.add_argument("--storage-dir", type=str, default="data", help="DB storage directory")
    args = parser.parse_args()

    collect(
        iterations=args.iterations,
        markets=args.markets,
        sleep_seconds=args.sleep,
        min_volume=args.min_volume,
        storage_dir=args.storage_dir,
    )


if __name__ == "__main__":
    main()
