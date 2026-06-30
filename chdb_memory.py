"""
Persistent storage layer for Polymarket data and model memory.

Primary backend: chDB (ClickHouse in-process)
Fallback backend: SQLite (used when chDB is not available for the platform)
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import chdb

    HAS_CHDB = True
except ImportError:
    HAS_CHDB = False
    chdb = None


class PolymarketCHDBStore:
    """Persistent data store for market data, training samples and predictions."""

    def __init__(
        self,
        db_dir: str = "data",
        database_name: str = "polymarket_memory",
        prefer_chdb: bool = True,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.backend = "chdb" if prefer_chdb and HAS_CHDB else "sqlite"
        if self.backend == "chdb":
            self.db_path = self.db_dir / f"{database_name}.chdb"
            self.connection = chdb.connect(str(self.db_path))
        else:
            self.db_path = self.db_dir / f"{database_name}.sqlite3"
            self.connection = sqlite3.connect(str(self.db_path))

        self._init_schema()

    def _log(self, message: str):
        if self.verbose:
            print(f"[Store:{self.backend}] {message}")

    @staticmethod
    def _sql_literal(value) -> str:
        if value is None:
            return "NULL"

        if isinstance(value, bool):
            return "1" if value else "0"

        if isinstance(value, (int, np.integer)):
            return str(int(value))

        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                return "NULL"
            return repr(numeric)

        text = str(value).replace("\\", "\\\\").replace("'", "''")
        return f"'{text}'"

    @staticmethod
    def _to_epoch_seconds(timestamp_value) -> Optional[int]:
        if timestamp_value is None:
            return None

        if isinstance(timestamp_value, (int, float, np.integer, np.floating)):
            numeric = float(timestamp_value)
            if numeric > 1_000_000_000_000:
                numeric = numeric / 1000.0
            return int(numeric)

        parsed = pd.to_datetime(timestamp_value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return int(parsed.timestamp())

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                return default
            return numeric
        except (TypeError, ValueError):
            return default

    def _execute(self, sql: str):
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            commit = getattr(self.connection, "commit", None)
            if callable(commit):
                commit()
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()

    def _query(self, sql: str) -> List[Tuple]:
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            return rows
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()

    def _insert_rows(self, table: str, columns: Sequence[str], rows: Sequence[Tuple]):
        if not rows:
            return

        cols = ", ".join(columns)
        values_sql = []
        for row in rows:
            values = ", ".join(self._sql_literal(value) for value in row)
            values_sql.append(f"({values})")

        sql = f"INSERT INTO {table} ({cols}) VALUES {', '.join(values_sql)}"
        self._execute(sql)

    def _init_schema(self):
        if self.backend == "chdb":
            self._init_chdb_schema()
        else:
            self._init_sqlite_schema()

    def _init_chdb_schema(self):
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                cache_key String,
                market_id String,
                fetched_epoch Int64,
                raw_json String
            ) ENGINE = MergeTree
            ORDER BY (cache_key, fetched_epoch, market_id)
            """
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS trade_snapshots (
                token_id String,
                fetched_epoch Int64,
                trade_ts Int64,
                raw_json String
            ) ENGINE = MergeTree
            ORDER BY (token_id, fetched_epoch, trade_ts)
            """
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS price_history_points (
                token_id String,
                interval_name String,
                fidelity Int32,
                fetched_epoch Int64,
                point_ts Int64,
                price Float64
            ) ENGINE = MergeTree
            ORDER BY (token_id, interval_name, fidelity, fetched_epoch, point_ts)
            """
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS training_samples (
                sample_id String,
                source String,
                created_epoch Int64,
                market_id String,
                question String,
                current_price Float64,
                future_price Float64,
                outcome UInt8,
                volume Float64,
                features_json String,
                raw_json String
            ) ENGINE = MergeTree
            ORDER BY (created_epoch, sample_id)
            """
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS model_training_runs (
                run_id String,
                created_epoch Int64,
                sample_count Int32,
                metrics_json String,
                params_json String
            ) ENGINE = MergeTree
            ORDER BY (created_epoch, run_id)
            """
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_runs (
                prediction_id String,
                created_epoch Int64,
                market_id String,
                question String,
                current_price Float64,
                predicted_price Float64,
                confidence Float64,
                action String,
                signal String,
                edge Float64,
                payload_json String
            ) ENGINE = MergeTree
            ORDER BY (created_epoch, prediction_id)
            """
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS run_reports (
                report_id String,
                created_epoch Int64,
                run_type String,
                report_json String
            ) ENGINE = MergeTree
            ORDER BY (created_epoch, report_id)
            """
        )

    def _init_sqlite_schema(self):
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                cache_key TEXT NOT NULL,
                market_id TEXT NOT NULL,
                fetched_epoch INTEGER NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_market_snapshots_cache_epoch "
            "ON market_snapshots (cache_key, fetched_epoch)"
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS trade_snapshots (
                token_id TEXT NOT NULL,
                fetched_epoch INTEGER NOT NULL,
                trade_ts INTEGER,
                raw_json TEXT NOT NULL
            )
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_snapshots_token_epoch "
            "ON trade_snapshots (token_id, fetched_epoch)"
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS price_history_points (
                token_id TEXT NOT NULL,
                interval_name TEXT NOT NULL,
                fidelity INTEGER NOT NULL,
                fetched_epoch INTEGER NOT NULL,
                point_ts INTEGER NOT NULL,
                price REAL NOT NULL
            )
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_price_history_lookup "
            "ON price_history_points (token_id, interval_name, fidelity, fetched_epoch)"
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS training_samples (
                sample_id TEXT NOT NULL,
                source TEXT NOT NULL,
                created_epoch INTEGER NOT NULL,
                market_id TEXT,
                question TEXT,
                current_price REAL NOT NULL,
                future_price REAL NOT NULL,
                outcome INTEGER NOT NULL,
                volume REAL NOT NULL,
                features_json TEXT NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_training_samples_created "
            "ON training_samples (created_epoch)"
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS model_training_runs (
                run_id TEXT NOT NULL,
                created_epoch INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                params_json TEXT NOT NULL
            )
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_model_training_runs_epoch "
            "ON model_training_runs (created_epoch)"
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_runs (
                prediction_id TEXT NOT NULL,
                created_epoch INTEGER NOT NULL,
                market_id TEXT,
                question TEXT,
                current_price REAL NOT NULL,
                predicted_price REAL NOT NULL,
                confidence REAL NOT NULL,
                action TEXT NOT NULL,
                signal TEXT NOT NULL,
                edge REAL NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_runs_epoch "
            "ON prediction_runs (created_epoch)"
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS run_reports (
                report_id TEXT NOT NULL,
                created_epoch INTEGER NOT NULL,
                run_type TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_run_reports_epoch "
            "ON run_reports (created_epoch)"
        )

    def _latest_epoch(
        self,
        table: str,
        filters: Dict[str, object],
        epoch_column: str = "fetched_epoch",
        max_age_seconds: Optional[int] = None,
    ) -> Optional[int]:
        where_parts = [f"{key} = {self._sql_literal(value)}" for key, value in filters.items()]
        if max_age_seconds is not None:
            threshold = int(time.time()) - int(max_age_seconds)
            where_parts.append(f"{epoch_column} >= {threshold}")

        where_sql = " AND ".join(where_parts) if where_parts else "1 = 1"
        sql = (
            f"SELECT {epoch_column} FROM {table} "
            f"WHERE {where_sql} ORDER BY {epoch_column} DESC LIMIT 1"
        )
        rows = self._query(sql)
        if not rows:
            return None
        return int(rows[0][0])

    def save_markets(self, cache_key: str, markets: List[Dict]):
        if not markets:
            return

        fetched_epoch = int(time.time())
        rows = []
        for market in markets:
            market_id = str(market.get("id") or market.get("conditionId") or "")
            raw_json = json.dumps(market, separators=(",", ":"), default=str)
            rows.append((cache_key, market_id, fetched_epoch, raw_json))

        self._insert_rows(
            "market_snapshots",
            ("cache_key", "market_id", "fetched_epoch", "raw_json"),
            rows,
        )

    def load_markets(
        self,
        cache_key: str,
        limit: int = 100,
        max_age_seconds: Optional[int] = None,
    ) -> List[Dict]:
        latest_epoch = self._latest_epoch(
            table="market_snapshots",
            filters={"cache_key": cache_key},
            max_age_seconds=max_age_seconds,
        )
        if latest_epoch is None:
            return []

        safe_limit = max(int(limit), 1)
        sql = (
            "SELECT raw_json FROM market_snapshots "
            f"WHERE cache_key = {self._sql_literal(cache_key)} "
            f"AND fetched_epoch = {latest_epoch} LIMIT {safe_limit}"
        )
        rows = self._query(sql)
        markets = []
        for row in rows:
            try:
                markets.append(json.loads(row[0]))
            except json.JSONDecodeError:
                continue
        return markets

    def save_trades(self, token_id: str, trades: List[Dict]):
        if not token_id or not trades:
            return

        fetched_epoch = int(time.time())
        rows = []
        for trade in trades:
            trade_ts = self._to_epoch_seconds(trade.get("timestamp"))
            if trade_ts is None:
                trade_ts = fetched_epoch
            raw_json = json.dumps(trade, separators=(",", ":"), default=str)
            rows.append((token_id, fetched_epoch, trade_ts, raw_json))

        self._insert_rows(
            "trade_snapshots",
            ("token_id", "fetched_epoch", "trade_ts", "raw_json"),
            rows,
        )

    def load_trades(
        self,
        token_id: str,
        limit: int = 500,
        max_age_seconds: Optional[int] = None,
    ) -> List[Dict]:
        latest_epoch = self._latest_epoch(
            table="trade_snapshots",
            filters={"token_id": token_id},
            max_age_seconds=max_age_seconds,
        )
        if latest_epoch is None:
            return []

        safe_limit = max(int(limit), 1)
        sql = (
            "SELECT raw_json FROM trade_snapshots "
            f"WHERE token_id = {self._sql_literal(token_id)} "
            f"AND fetched_epoch = {latest_epoch} "
            "ORDER BY trade_ts DESC "
            f"LIMIT {safe_limit}"
        )
        rows = self._query(sql)
        trades = []
        for row in rows:
            try:
                trades.append(json.loads(row[0]))
            except json.JSONDecodeError:
                continue
        return trades

    def save_price_history(
        self,
        token_id: str,
        interval_name: str,
        fidelity: int,
        history_df: pd.DataFrame,
    ):
        if history_df is None or history_df.empty:
            return

        fetched_epoch = int(time.time())
        rows = []
        for _, point in history_df.iterrows():
            point_ts = self._to_epoch_seconds(point.get("timestamp"))
            if point_ts is None:
                continue
            price = self._to_float(point.get("price"), default=0.0)
            rows.append((token_id, interval_name, int(fidelity), fetched_epoch, point_ts, price))

        self._insert_rows(
            "price_history_points",
            ("token_id", "interval_name", "fidelity", "fetched_epoch", "point_ts", "price"),
            rows,
        )

    def load_price_history(
        self,
        token_id: str,
        interval_name: str,
        fidelity: int,
        max_age_seconds: Optional[int] = None,
    ) -> pd.DataFrame:
        latest_epoch = self._latest_epoch(
            table="price_history_points",
            filters={"token_id": token_id, "interval_name": interval_name, "fidelity": int(fidelity)},
            max_age_seconds=max_age_seconds,
        )
        if latest_epoch is None:
            return pd.DataFrame(columns=["timestamp", "price"])

        sql = (
            "SELECT point_ts, price FROM price_history_points "
            f"WHERE token_id = {self._sql_literal(token_id)} "
            f"AND interval_name = {self._sql_literal(interval_name)} "
            f"AND fidelity = {int(fidelity)} "
            f"AND fetched_epoch = {latest_epoch} "
            "ORDER BY point_ts"
        )
        rows = self._query(sql)
        if not rows:
            return pd.DataFrame(columns=["timestamp", "price"])

        df = pd.DataFrame(rows, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df.dropna(subset=["price"]).reset_index(drop=True)

    def save_training_samples(self, samples: List[Dict], source: str = "api"):
        if not samples:
            return

        created_epoch = int(time.time())
        rows = []
        for sample in samples:
            features = sample.get("features")
            if isinstance(features, np.ndarray):
                feature_list = features.flatten().astype(float).tolist()
            elif isinstance(features, list):
                feature_list = [self._to_float(x) for x in features]
            else:
                continue

            market_id = str(sample.get("market_id", "unknown"))
            current_price = self._to_float(sample.get("current_price"), default=0.5)
            future_price = self._to_float(sample.get("future_price"), default=current_price)
            outcome = int(sample.get("outcome", 0))
            volume = self._to_float(sample.get("volume"), default=0.0)
            question = str(sample.get("question", "Unknown"))[:200]

            sample_key = (
                f"{market_id}|{current_price:.6f}|{future_price:.6f}|{outcome}|"
                f"{','.join(f'{x:.6f}' for x in feature_list)}"
            )
            sample_id = hashlib.sha1(sample_key.encode("utf-8")).hexdigest()
            raw_json = json.dumps(sample, default=str, separators=(",", ":"))
            features_json = json.dumps(feature_list, separators=(",", ":"))

            rows.append(
                (
                    sample_id,
                    source,
                    created_epoch,
                    market_id,
                    question,
                    current_price,
                    future_price,
                    outcome,
                    volume,
                    features_json,
                    raw_json,
                )
            )

        self._insert_rows(
            "training_samples",
            (
                "sample_id",
                "source",
                "created_epoch",
                "market_id",
                "question",
                "current_price",
                "future_price",
                "outcome",
                "volume",
                "features_json",
                "raw_json",
            ),
            rows,
        )

    def load_training_samples(
        self,
        limit: int = 100,
        max_age_seconds: Optional[int] = None,
    ) -> List[Dict]:
        where_sql = "1 = 1"
        if max_age_seconds is not None:
            threshold = int(time.time()) - int(max_age_seconds)
            where_sql = f"created_epoch >= {threshold}"

        safe_limit = max(int(limit), 1)
        sql = (
            "SELECT market_id, question, current_price, future_price, outcome, volume, features_json, raw_json "
            "FROM training_samples "
            f"WHERE {where_sql} ORDER BY created_epoch DESC LIMIT {safe_limit}"
        )
        rows = self._query(sql)

        samples = []
        for row in rows:
            try:
                feature_values = json.loads(row[6])
            except json.JSONDecodeError:
                continue

            try:
                raw_payload = json.loads(row[7]) if row[7] else {}
            except json.JSONDecodeError:
                raw_payload = {}

            if not isinstance(feature_values, list) or not feature_values:
                continue

            features_array = np.array(feature_values, dtype=float).reshape(1, -1)
            sample = {
                "features": features_array,
                "current_price": self._to_float(row[2], 0.5),
                "future_price": self._to_float(row[3], 0.5),
                "outcome": int(row[4]),
                "market_id": row[0] or "unknown",
                "question": row[1] or "Unknown",
                "is_resolved": False,
                "volume": self._to_float(row[5], 0.0),
            }
            if "sample_timestamp" in raw_payload:
                sample["sample_timestamp"] = raw_payload.get("sample_timestamp")
            if "domain" in raw_payload:
                sample["domain"] = raw_payload.get("domain")

            samples.append(sample)

        return samples

    def save_model_training_run(self, metrics: Dict, sample_count: int, params: Optional[Dict] = None):
        if not metrics:
            return

        created_epoch = int(time.time())
        run_seed = f"{created_epoch}|{sample_count}|{json.dumps(metrics, sort_keys=True, default=str)}"
        run_id = hashlib.sha1(run_seed.encode("utf-8")).hexdigest()

        self._insert_rows(
            "model_training_runs",
            ("run_id", "created_epoch", "sample_count", "metrics_json", "params_json"),
            [
                (
                    run_id,
                    created_epoch,
                    int(sample_count),
                    json.dumps(metrics, default=str, separators=(",", ":")),
                    json.dumps(params or {}, default=str, separators=(",", ":")),
                )
            ],
        )

    def save_prediction(self, market: Dict, prediction: Dict):
        if not prediction:
            return

        created_epoch = int(time.time())
        market_id = str(market.get("id") or market.get("conditionId") or "unknown")
        question = str(market.get("question", "Unknown"))[:200]

        current_price = self._to_float(prediction.get("current_price"), default=0.5)
        predicted_price = self._to_float(prediction.get("predicted_price"), default=current_price)
        confidence = self._to_float(prediction.get("confidence"), default=0.5)
        edge = self._to_float(prediction.get("edge"), default=0.0)
        action = str(prediction.get("action", "HOLD"))
        signal = str(prediction.get("signal", "HOLD"))

        prediction_seed = (
            f"{created_epoch}|{market_id}|{current_price:.6f}|{predicted_price:.6f}|{confidence:.6f}|{action}"
        )
        prediction_id = hashlib.sha1(prediction_seed.encode("utf-8")).hexdigest()

        self._insert_rows(
            "prediction_runs",
            (
                "prediction_id",
                "created_epoch",
                "market_id",
                "question",
                "current_price",
                "predicted_price",
                "confidence",
                "action",
                "signal",
                "edge",
                "payload_json",
            ),
            [
                (
                    prediction_id,
                    created_epoch,
                    market_id,
                    question,
                    current_price,
                    predicted_price,
                    confidence,
                    action,
                    signal,
                    edge,
                    json.dumps(prediction, default=str, separators=(",", ":")),
                )
            ],
        )

    def save_run_report(self, report: Dict, run_type: str = "live_prediction"):
        if not report:
            return

        created_epoch = int(time.time())
        report_seed = f"{created_epoch}|{run_type}|{json.dumps(report, sort_keys=True, default=str)}"
        report_id = hashlib.sha1(report_seed.encode("utf-8")).hexdigest()

        self._insert_rows(
            "run_reports",
            ("report_id", "created_epoch", "run_type", "report_json"),
            [
                (
                    report_id,
                    created_epoch,
                    run_type,
                    json.dumps(report, default=str, separators=(",", ":")),
                )
            ],
        )

    def load_latest_run_report(self, run_type: Optional[str] = None) -> Optional[Dict]:
        where_sql = "1 = 1"
        if run_type:
            where_sql = f"run_type = {self._sql_literal(run_type)}"

        sql = (
            "SELECT report_json FROM run_reports "
            f"WHERE {where_sql} ORDER BY created_epoch DESC LIMIT 1"
        )
        rows = self._query(sql)
        if not rows:
            return None

        try:
            return json.loads(rows[0][0])
        except json.JSONDecodeError:
            return None
