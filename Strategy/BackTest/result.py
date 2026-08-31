"""The shared output format both engines produce, and its on-disk layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Common.config import settings
from Common.io import read_json, read_parquet, write_json, write_parquet
from Common.provenance import stamp

EQUITY_FILE = "equity.parquet"
POSITIONS_FILE = "positions.parquet"
TRADES_FILE = "trades.parquet"
METRICS_FILE = "metrics.json"
SPEC_FILE = "spec.json"

TRADE_COLUMNS = (
    "date",
    "symbol",
    "side",
    "qty",
    "price",
    "notional",
    "commission",
    "spread",
    "slippage",
)


@dataclass
class BacktestResult:
    """Equity, positions, trades and headline metrics for one run."""

    name: str
    run_id: str
    equity: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    spec: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    engine: str = "vectorised"
    adjustments: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))

    @property
    def returns(self) -> pd.Series:
        return self.equity["equity"].pct_change().fillna(0.0).rename("ret")

    @property
    def final_equity(self) -> float:
        return float(self.equity["equity"].iloc[-1])

    def cost_summary(self) -> dict[str, float]:
        if self.trades.empty:
            return {"commission": 0.0, "spread": 0.0, "slippage": 0.0, "total": 0.0}
        totals = {
            name: float(self.trades[name].sum())
            for name in ("commission", "spread", "slippage")
        }
        totals["total"] = float(sum(totals.values()))
        return totals

    def directory(self, root: Path | str | None = None) -> Path:
        base = Path(root) if root else settings().backtests_root
        return base / self.name / self.run_id

    def save(self, root: Path | str | None = None) -> Path:
        target = self.directory(root)
        target.mkdir(parents=True, exist_ok=True)

        write_parquet(self.equity.reset_index(), target / EQUITY_FILE)
        write_parquet(self.positions, target / POSITIONS_FILE)
        write_parquet(
            self.trades if not self.trades.empty else _empty_trades(),
            target / TRADES_FILE,
        )
        write_json(stamp(engine=self.engine, **self.metrics), target / METRICS_FILE)
        write_json(stamp(**self.spec), target / SPEC_FILE)
        return target

    @classmethod
    def load(cls, path: Path | str) -> BacktestResult:
        directory = Path(path)
        equity = read_parquet(directory / EQUITY_FILE).set_index("date")
        equity.index = pd.DatetimeIndex(pd.to_datetime(equity.index, utc=True), name="date")

        spec = read_json(directory / SPEC_FILE) if (directory / SPEC_FILE).exists() else {}
        metrics = read_json(directory / METRICS_FILE) if (directory / METRICS_FILE).exists() else {}

        return cls(
            name=str(spec.get("name", directory.parent.name)),
            run_id=directory.name,
            equity=equity,
            positions=read_parquet(directory / POSITIONS_FILE),
            trades=read_parquet(directory / TRADES_FILE),
            spec=spec,
            metrics=metrics,
            engine=str(metrics.get("engine", "vectorised")),
        )

    def check_invariants(self, tolerance: float = 1e-6) -> list[str]:
        """Accounting identities that must hold on every bar. Returns the failures."""
        problems: list[str] = []

        identity = self.equity["cash"] + self.equity["position_value"] - self.equity["equity"]
        worst = float(identity.abs().max()) if len(identity) else 0.0
        scale = max(float(self.equity["equity"].abs().max()), 1.0)
        if worst > tolerance * scale:
            problems.append(f"equity != cash + positions (max error {worst:.6g})")

        if not self.trades.empty:
            traded = self.trades.groupby("symbol")["qty"].sum()
            # Splits change the share count without a trade, so they belong on this side
            # of the identity too.
            traded = traded.add(
                self.adjustments.reindex(traded.index).fillna(0.0), fill_value=0.0
            )
            final = (
                self.positions[self.positions["date"] == self.positions["date"].max()]
                .set_index("symbol")["shares"]
                .reindex(traded.index)
                .fillna(0.0)
            )
            drift = float((traded - final).abs().max())
            if drift > 1e-6 * max(float(traded.abs().max()), 1.0):
                problems.append(f"trades do not reconcile positions (max drift {drift:.6g})")

        if self.equity["equity"].isna().any():
            problems.append("equity contains NaN")
        return problems


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object" if name in {"symbol", "side"} else "float64") for name in TRADE_COLUMNS})


def make_run_id(spec_hash: str, timestamp: pd.Timestamp | None = None) -> str:
    stamp_at = timestamp or pd.Timestamp.now(tz="UTC")
    return f"{stamp_at.strftime('%Y%m%dT%H%M%S')}-{spec_hash[:8]}"


def latest_run(name: str, root: Path | str | None = None) -> Path | None:
    base = Path(root) if root else settings().backtests_root
    directory = base / name
    if not directory.is_dir():
        return None
    runs = sorted(path for path in directory.iterdir() if path.is_dir())
    return runs[-1] if runs else None


def build_equity_frame(
    dates: pd.DatetimeIndex,
    equity: np.ndarray,
    cash: np.ndarray,
    long_value: np.ndarray,
    short_value: np.ndarray,
) -> pd.DataFrame:
    position_value = long_value - short_value
    gross = long_value + short_value
    frame = pd.DataFrame(
        {
            "equity": equity,
            "cash": cash,
            "position_value": position_value,
            "long_value": long_value,
            "short_value": short_value,
            "gross": gross,
            "net": position_value,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    frame["leverage"] = frame["gross"] / frame["equity"].replace(0.0, np.nan)
    frame["ret"] = frame["equity"].pct_change().fillna(0.0)
    return frame
