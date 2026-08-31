"""US listing universe from the NASDAQ symbol directory, covering stocks and ETFs."""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

from . import config

# The HTTPS mirror of this directory is frequently unavailable; the FTP one is stable.
NASDAQ_TRADED_URL = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"


def universe_path() -> Path:
    return config.REFERENCE_ROOT / "universe.parquet"


ASSET_CLASSES = ("stock", "etf")

# Security names of ordinary equity issues (including ADRs) look like these.
_COMMON_EQUITY_PATTERN = re.compile(
    r"\b(?:common stock|common shares?|ordinary shares?|capital stock|"
    r"american depositary shares?|shares of beneficial interest)\b",
    re.IGNORECASE,
)

# Warrants, rights, units, preferred issues, debt and listed funds are not common equity.
_NON_COMMON_PATTERN = re.compile(
    r"\b(?:warrants?|rights|units|preferred|preference|depositary units|"
    r"debentures?|notes?|bonds?|subordinated|when[- ]issued|contingent value|"
    r"fund|closed[- ]end|index[- ]linked)\b",
    re.IGNORECASE,
)


def _read_symbol_directory(url: str = NASDAQ_TRADED_URL) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed, trusted URL
        payload = response.read().decode("utf-8", "replace")

    frame = pd.read_csv(io.StringIO(payload), sep="|", dtype=str)
    # The last line is a "File Creation Time" footer rather than a listing.
    return frame[frame["Symbol"].notna() & frame["Security Name"].notna()]


def _to_yahoo_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def build_universe(url: str = NASDAQ_TRADED_URL) -> pd.DataFrame:
    """Return every tradable US listing tagged as ``stock``, ``etf`` or ``other``."""
    frame = _read_symbol_directory(url)
    frame = frame[frame["Test Issue"].str.upper() == "N"]

    is_etf = frame["ETF"].str.upper() == "Y"
    names = frame["Security Name"]
    is_common = names.str.contains(_COMMON_EQUITY_PATTERN) & ~names.str.contains(
        _NON_COMMON_PATTERN
    )

    asset_class = pd.Series("other", index=frame.index, dtype="object")
    asset_class[is_etf] = "etf"
    asset_class[~is_etf & is_common] = "stock"

    exchange = (
        frame["Listing Exchange"]
        if "Listing Exchange" in frame.columns
        else pd.Series("", index=frame.index)
    )
    universe = pd.DataFrame(
        {
            "symbol": frame["Symbol"].map(_to_yahoo_symbol),
            "name": names.str.strip(),
            "exchange": exchange,
            "asset_class": asset_class,
            "region": "US",
            "snapshot_date": pd.Timestamp.now(tz="UTC").normalize(),
        }
    )
    universe = universe[universe["symbol"].str.len() > 0]
    return universe.drop_duplicates(subset="symbol").sort_values("symbol").reset_index(drop=True)


def save_universe(universe: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or universe_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def load_universe(path: Path | None = None) -> pd.DataFrame:
    path = path or universe_path()
    if not path.exists():
        raise FileNotFoundError(
            f"no universe snapshot at {path}; run 'python -m DataAcquisition universe' first"
        )
    return pd.read_parquet(path, engine="pyarrow")


def symbols(
    asset_class: str = "stock",
    region: str = "US",
    refresh: bool = False,
    path: Path | None = None,
) -> list[str]:
    """Symbols for one asset class, downloading a fresh snapshot when needed."""
    path = path or universe_path()
    if refresh or not path.exists():
        save_universe(build_universe(), path)
    universe = load_universe(path)
    selected = universe[
        (universe["asset_class"] == asset_class) & (universe["region"] == region)
    ]
    return sorted(selected["symbol"].tolist())
