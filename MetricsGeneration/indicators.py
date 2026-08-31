"""Point-in-time metrics computed per symbol from a daily OHLCV frame.

Every function here is pure: it takes a single symbol's OHLCV frame indexed by date
and returns values aligned to that same index. No I/O, no cross-symbol dependencies.

Return, momentum, volatility and shape families are computed on the **adjusted**
price series, because a split or a dividend is not a return. Liquidity features
use the **raw** traded price, because that is the money that actually changed hands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
ANNUALIZE = float(np.sqrt(TRADING_DAYS))

BASE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "adj_factor",
    "dividend",
    "split_ratio",
    "repaired",
)

OHLC_COLUMNS = ("open", "high", "low", "close")

LABEL_COLUMNS = ("fwd_ret_1d", "fwd_ret_5d", "fwd_ret_21d")

# Prices keep full precision; derived features are fine at single precision.
FLOAT64_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "dividend",
        "split_ratio",
        "obv",
        "dollar_vol",
    }
)


def _wilder(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing, the recursive average used by RSI/ATR/ADX."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0.0, np.nan))


def _rolling_mean_abs_dev(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size < window:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    means = windows.mean(axis=1)
    result[window - 1 :] = np.abs(windows - means[:, None]).mean(axis=1)
    return result


def with_adjustments(frame: pd.DataFrame) -> pd.DataFrame:
    """Guarantee ``adj_close`` and ``adj_factor`` exist, deriving them when absent."""
    result = frame.copy()
    close = result["close"]

    if "adj_close" not in result.columns:
        if "adj_factor" in result.columns:
            result["adj_close"] = close * result["adj_factor"]
        else:
            result["adj_close"] = close

    adjusted = result["adj_close"].where(result["adj_close"] > 0.0)
    result["adj_close"] = adjusted.ffill().fillna(close)
    result["adj_factor"] = _safe_div(result["adj_close"], close).fillna(1.0)

    if "repaired" not in result.columns:
        result["repaired"] = 0.0
    return result


def adjusted_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Split- and dividend-adjusted OHLC, scaled by the same factor as the close."""
    factor = frame["adj_factor"]
    adjusted = pd.DataFrame(
        {name: frame[name] * factor for name in OHLC_COLUMNS}, index=frame.index
    )
    adjusted["close"] = frame["adj_close"]
    return adjusted


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    return pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff()
    average_gain = _wilder(change.clip(lower=0.0), window)
    average_loss = _wilder((-change).clip(lower=0.0), window)
    result = 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    # A window with no down days has an undefined ratio but a defined RSI of 100.
    return result.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    smoothed_range = _wilder(true_range(high, low, close), window)
    plus_di = 100.0 * _safe_div(_wilder(plus_dm, window), smoothed_range)
    minus_di = 100.0 * _safe_div(_wilder(minus_dm, window), smoothed_range)

    directional_index = 100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    return _wilder(directional_index, window)


def yang_zhang_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    overnight = np.log(open_ / close.shift(1))
    open_to_close = np.log(close / open_)
    high_to_open = np.log(high / open_)
    low_to_open = np.log(low / open_)

    rogers_satchell = (
        high_to_open * (high_to_open - open_to_close)
        + low_to_open * (low_to_open - open_to_close)
    )
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    variance = (
        overnight.rolling(window).var()
        + k * open_to_close.rolling(window).var()
        + (1.0 - k) * rogers_satchell.rolling(window).mean()
    )
    return np.sqrt(variance.clip(lower=0.0)) * ANNUALIZE


def _returns_block(frame: pd.DataFrame, metrics: dict[str, pd.Series]) -> None:
    open_, close = frame["open"], frame["close"]
    simple_return = close.pct_change()

    metrics["ret_1d"] = simple_return
    metrics["logret_1d"] = np.log(close / close.shift(1))
    for window in (5, 21, 63, 126, 252):
        metrics[f"ret_{window}d"] = close / close.shift(window) - 1.0
    metrics["ret_overnight"] = open_ / close.shift(1) - 1.0
    metrics["ret_intraday"] = close / open_ - 1.0


def _momentum_block(frame: pd.DataFrame, metrics: dict[str, pd.Series]) -> None:
    high, low, close = frame["high"], frame["low"], frame["close"]

    # Classic 12-1 momentum: a year of returns excluding the most recent month.
    metrics["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0

    for window in (10, 20, 50, 200):
        metrics[f"px_to_sma_{window}"] = close / close.rolling(window).mean() - 1.0
    metrics["sma_50_to_200"] = (
        close.rolling(50).mean() / close.rolling(200).mean() - 1.0
    )

    metrics["dist_52w_high"] = close / high.rolling(TRADING_DAYS).max() - 1.0
    metrics["dist_52w_low"] = close / low.rolling(TRADING_DAYS).min() - 1.0

    metrics["rsi_14"] = rsi(close, 14)

    macd_line = (
        close.ewm(span=12, adjust=False, min_periods=12).mean()
        - close.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    metrics["macd_line"] = macd_line
    metrics["macd_signal"] = signal_line
    metrics["macd_hist"] = macd_line - signal_line

    metrics["adx_14"] = adx(high, low, close, 14)


def _volatility_block(frame: pd.DataFrame, metrics: dict[str, pd.Series]) -> None:
    open_, high, low, close = frame["open"], frame["high"], frame["low"], frame["close"]
    log_return = metrics["logret_1d"]

    for window in (20, 60, 252):
        metrics[f"vol_{window}d"] = log_return.rolling(window).std() * ANNUALIZE

    log_hl = np.log(high / low)
    metrics["parkinson_20d"] = (
        np.sqrt((log_hl**2).rolling(20).mean() / (4.0 * np.log(2.0))) * ANNUALIZE
    )

    garman_klass = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    metrics["garman_klass_20d"] = (
        np.sqrt(garman_klass.rolling(20).mean().clip(lower=0.0)) * ANNUALIZE
    )

    metrics["yang_zhang_20d"] = yang_zhang_volatility(open_, high, low, close, 20)

    average_true_range = _wilder(true_range(high, low, close), 14)
    metrics["atr_14"] = average_true_range
    metrics["natr_14"] = average_true_range / close

    metrics["downside_dev_60d"] = (
        log_return.clip(upper=0.0).rolling(60).std() * ANNUALIZE
    )


def _shape_block(frame: pd.DataFrame, metrics: dict[str, pd.Series]) -> None:
    close = frame["close"]
    simple_return = metrics["ret_1d"]

    rolling_mean = simple_return.rolling(TRADING_DAYS).mean()
    rolling_std = simple_return.rolling(TRADING_DAYS).std()
    downside_std = simple_return.clip(upper=0.0).rolling(TRADING_DAYS).std()

    metrics["sharpe_252d"] = _safe_div(rolling_mean, rolling_std) * ANNUALIZE
    metrics["sortino_252d"] = _safe_div(rolling_mean, downside_std) * ANNUALIZE
    metrics["skew_252d"] = simple_return.rolling(TRADING_DAYS).skew()
    metrics["kurt_252d"] = simple_return.rolling(TRADING_DAYS).kurt()

    drawdown = close / close.rolling(TRADING_DAYS, min_periods=1).max() - 1.0
    metrics["dd_from_252d_high"] = drawdown
    metrics["max_dd_252d"] = drawdown.rolling(TRADING_DAYS).min()

    metrics["hit_rate_252d"] = (
        (simple_return > 0.0).where(simple_return.notna()).rolling(TRADING_DAYS).mean()
    )


def _liquidity_block(frame: pd.DataFrame, metrics: dict[str, pd.Series]) -> None:
    high, low, close, volume = (
        frame["high"],
        frame["low"],
        frame["close"],
        frame["volume"],
    )
    dollar_volume = close * volume

    metrics["dollar_vol"] = dollar_volume
    metrics["advd_20"] = dollar_volume.rolling(20).mean()
    metrics["advd_60"] = dollar_volume.rolling(60).mean()

    # Amihud illiquidity: price impact per million dollars traded.
    metrics["amihud_60d"] = (
        _safe_div(metrics["ret_1d"].abs(), dollar_volume).rolling(60).mean() * 1e6
    )

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    metrics["vol_zscore_20"] = _safe_div(volume - volume_mean, volume_std)

    metrics["obv"] = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()

    typical_price = (high + low + close) / 3.0
    rolling_vwap = _safe_div(
        (typical_price * volume).rolling(20).sum(), volume.rolling(20).sum()
    )
    metrics["dist_vwap_20"] = close / rolling_vwap - 1.0

    metrics["zero_vol_frac_20"] = (volume == 0.0).rolling(20).mean()
    metrics["stale_px_frac_20"] = (close.diff() == 0.0).rolling(20).mean()


def _reversion_block(frame: pd.DataFrame, metrics: dict[str, pd.Series]) -> None:
    high, low, close = frame["high"], frame["low"], frame["close"]

    middle_band = close.rolling(20).mean()
    band_std = close.rolling(20).std()
    metrics["bb_pctb_20"] = _safe_div(close - (middle_band - 2.0 * band_std), 4.0 * band_std)
    metrics["bb_width_20"] = _safe_div(4.0 * band_std, middle_band)
    metrics["zscore_20"] = _safe_div(close - middle_band, band_std)

    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    stochastic_k = 100.0 * _safe_div(close - lowest_low, highest_high - lowest_low)
    metrics["stoch_k_14"] = stochastic_k
    metrics["stoch_d_14"] = stochastic_k.rolling(3).mean()
    metrics["willr_14"] = stochastic_k - 100.0

    typical_price = (high + low + close) / 3.0
    mean_absolute_deviation = pd.Series(
        _rolling_mean_abs_dev(typical_price.to_numpy(dtype=float), 20),
        index=typical_price.index,
    )
    metrics["cci_20"] = _safe_div(
        typical_price - typical_price.rolling(20).mean(), 0.015 * mean_absolute_deviation
    )


def compute_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute every point-in-time metric for one symbol's OHLCV frame."""
    frame = with_adjustments(frame)
    adjusted = adjusted_ohlc(frame)

    metrics: dict[str, pd.Series] = {
        column: frame[column] for column in BASE_COLUMNS if column in frame.columns
    }

    _returns_block(adjusted, metrics)
    _momentum_block(adjusted, metrics)
    _volatility_block(adjusted, metrics)
    _shape_block(adjusted, metrics)
    _liquidity_block(frame, metrics)
    _reversion_block(adjusted, metrics)

    result = pd.DataFrame(metrics, index=frame.index)
    return result.replace([np.inf, -np.inf], np.nan)


def compute_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward returns. Lookahead by construction, kept in a separate dataset."""
    close = with_adjustments(frame)["adj_close"]
    labels = {
        f"fwd_ret_{horizon}d": close.shift(-horizon) / close - 1.0
        for horizon in (1, 5, 21)
    }
    return pd.DataFrame(labels, index=frame.index).replace([np.inf, -np.inf], np.nan)


def metric_dtype(name: str) -> str:
    return "float64" if name in FLOAT64_COLUMNS else "float32"


def metric_names() -> tuple[str, ...]:
    """Every metric ``compute_metrics`` produces, in output order."""
    index = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    probe = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
            "adj_close": 1.0,
            "dividend": 0.0,
            "split_ratio": 0.0,
            "repaired": 0.0,
        },
        index=index,
    )
    return tuple(compute_metrics(probe).columns)


def min_periods(metric: str) -> int:
    """Bars of history a metric needs before it stops being NaN."""
    return MIN_PERIODS.get(metric, 1)


# Declared warm-up per feature. Leading values stay NaN: never forward-filled,
# never zero-filled, because a zero is a claim and a NaN is an absence.
MIN_PERIODS: dict[str, int] = {
    "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
    "adj_close": 1, "adj_factor": 1, "dividend": 1, "split_ratio": 1, "repaired": 1,
    "ret_1d": 2, "logret_1d": 2, "ret_5d": 6, "ret_21d": 22, "ret_63d": 64,
    "ret_126d": 127, "ret_252d": 253, "ret_overnight": 2, "ret_intraday": 1,
    "mom_12_1": 253, "px_to_sma_10": 10, "px_to_sma_20": 20, "px_to_sma_50": 50,
    "px_to_sma_200": 200, "sma_50_to_200": 200, "dist_52w_high": 252,
    "dist_52w_low": 252, "rsi_14": 15, "macd_line": 26, "macd_signal": 34,
    "macd_hist": 34, "adx_14": 28,
    "vol_20d": 21, "vol_60d": 61, "vol_252d": 253, "parkinson_20d": 20,
    "garman_klass_20d": 20, "yang_zhang_20d": 21, "atr_14": 15, "natr_14": 15,
    "downside_dev_60d": 61,
    "sharpe_252d": 253, "sortino_252d": 253, "skew_252d": 253, "kurt_252d": 253,
    "dd_from_252d_high": 1, "max_dd_252d": 252, "hit_rate_252d": 253,
    "dollar_vol": 1, "advd_20": 20, "advd_60": 60, "amihud_60d": 61,
    "vol_zscore_20": 20, "obv": 1, "dist_vwap_20": 20, "zero_vol_frac_20": 20,
    "stale_px_frac_20": 21,
    "bb_pctb_20": 20, "bb_width_20": 20, "zscore_20": 20, "stoch_k_14": 14,
    "stoch_d_14": 16, "willr_14": 14, "cci_20": 20,
    "beta_252d": 126, "corr_mkt_252d": 126, "idio_vol_252d": 126,
    "rel_ret_21d": 22, "mkt_ret_1d": 2,
}
