"""Validation and cleaning applied to canonical bars before they enter the lake."""

from __future__ import annotations

import pandas as pd

from .schema import SYMBOL, TIMESTAMP, empty_bars

PRICE_COLUMNS = ("open", "high", "low", "close")


def clean_bars(frame: pd.DataFrame, drop_zero_volume: bool = False) -> pd.DataFrame:
    """Drop impossible rows, sort ascending and keep the last row per timestamp."""
    if frame.empty:
        return empty_bars()

    cleaned = frame.dropna(subset=list(PRICE_COLUMNS))
    for column in PRICE_COLUMNS:
        cleaned = cleaned[cleaned[column] > 0]

    consistent = (
        (cleaned["high"] >= cleaned["low"])
        & (cleaned["high"] >= cleaned["open"])
        & (cleaned["high"] >= cleaned["close"])
        & (cleaned["low"] <= cleaned["open"])
        & (cleaned["low"] <= cleaned["close"])
    )
    cleaned = cleaned[consistent]

    cleaned = cleaned[cleaned["volume"].fillna(0.0) >= 0]
    if drop_zero_volume:
        cleaned = cleaned[cleaned["volume"].fillna(0.0) > 0]

    cleaned = cleaned.sort_values(TIMESTAMP)
    cleaned = cleaned[~cleaned[TIMESTAMP].duplicated(keep="last")]
    return cleaned.reset_index(drop=True)


def merge_bars(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Combine stored and freshly downloaded bars, letting the new data win."""
    if existing.empty:
        return incoming.reset_index(drop=True)
    if incoming.empty:
        return existing.reset_index(drop=True)

    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.sort_values([TIMESTAMP, SYMBOL])
    combined = combined[~combined[TIMESTAMP].duplicated(keep="last")]
    return combined.reset_index(drop=True)
