"""Performance, risk and trading metrics.

Everything here takes plain pandas objects, so this module never imports the
backtest package and the dependency arrow points one way only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# --- return ---------------------------------------------------------------
def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    # n observations span n - 1 periods.
    years = (len(equity) - 1) / periods_per_year
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (1.0 / years) - 1.0) if growth > 0 and years > 0 else -1.0


# --- risk -----------------------------------------------------------------
def annual_volatility(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series, threshold: float = 0.0, periods_per_year: float = TRADING_DAYS
) -> float:
    shortfall = (returns - threshold).clip(upper=0.0)
    return float(np.sqrt((shortfall**2).mean()) * np.sqrt(periods_per_year))


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def drawdown_duration(equity: pd.Series) -> int:
    """Longest run of consecutive bars spent below a previous high-water mark."""
    underwater = drawdown(equity) < 0.0
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    return float(np.nanpercentile(returns.dropna(), (1.0 - level) * 100.0))


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    threshold = value_at_risk(returns, level)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) else threshold


def monthly_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).resample("ME").prod() - 1.0


def worst_month(returns: pd.Series) -> float:
    monthly = monthly_returns(returns)
    return float(monthly.min()) if len(monthly) else 0.0


# --- risk-adjusted --------------------------------------------------------
def sharpe(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: float = TRADING_DAYS
) -> float:
    excess = returns - risk_free / periods_per_year
    deviation = excess.std(ddof=1)
    if not deviation:
        return 0.0
    return float(excess.mean() / deviation * np.sqrt(periods_per_year))


def sortino(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: float = TRADING_DAYS
) -> float:
    excess = returns - risk_free / periods_per_year
    deviation = downside_deviation(excess, 0.0, periods_per_year)
    if not deviation:
        return 0.0
    return float(excess.mean() * periods_per_year / deviation)


def calmar(equity: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    worst = abs(max_drawdown(equity))
    return float(cagr(equity, periods_per_year) / worst) if worst else 0.0


def deflated_sharpe(
    returns: pd.Series, trials: int = 1, periods_per_year: float = TRADING_DAYS
) -> float:
    """Probability the Sharpe is real once multiple testing is accounted for.

    Bailey and Lopez de Prado: the more configurations you tried, the higher the
    Sharpe you should expect from luck alone.
    """
    from scipy.stats import norm

    observations = int(returns.notna().sum())
    if observations < 3 or trials < 1:
        return float("nan")

    observed = sharpe(returns, 0.0, periods_per_year) / np.sqrt(periods_per_year)
    skewness = float(returns.skew())
    kurtosis = float(returns.kurt()) + 3.0

    euler = 0.5772156649
    if trials > 1:
        expected_max = np.sqrt(1.0 / (observations - 1)) * (
            (1.0 - euler) * norm.ppf(1.0 - 1.0 / trials)
            + euler * norm.ppf(1.0 - 1.0 / (trials * np.e))
        )
    else:
        expected_max = 0.0

    variance = 1.0 - skewness * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if variance <= 0.0:
        return float("nan")
    statistic = (observed - expected_max) * np.sqrt(observations - 1) / np.sqrt(variance)
    return float(norm.cdf(statistic))


# --- attribution ----------------------------------------------------------
def alpha_beta(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: float = TRADING_DAYS
) -> tuple[float, float]:
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 3:
        return float("nan"), float("nan")
    strategy, market = aligned.iloc[:, 0], aligned.iloc[:, 1]
    variance = market.var(ddof=1)
    if not variance:
        return float("nan"), float("nan")
    beta = float(strategy.cov(market) / variance)
    alpha = float((strategy.mean() - beta * market.mean()) * periods_per_year)
    return alpha, beta


def information_ratio(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: float = TRADING_DAYS
) -> float:
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 3:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    deviation = active.std(ddof=1)
    return float(active.mean() / deviation * np.sqrt(periods_per_year)) if deviation else 0.0


def contribution_by_symbol(positions: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Approximate P&L contribution per symbol from held value and daily return."""
    if positions.empty:
        return pd.DataFrame(columns=["symbol", "mean_weight", "days_held"])
    grouped = positions.groupby("symbol").agg(
        mean_weight=("weight", "mean"), days_held=("date", "count")
    )
    return grouped.sort_values("days_held", ascending=False).head(top).reset_index()


# --- trading --------------------------------------------------------------
def holding_period(positions: pd.DataFrame, dates: pd.DatetimeIndex) -> float:
    """Mean length in bars of an uninterrupted run of holding one symbol."""
    if positions.empty:
        return 0.0

    order = pd.DatetimeIndex(dates)
    located = order.get_indexer(pd.DatetimeIndex(positions["date"]))
    frame = pd.DataFrame(
        {"symbol": positions["symbol"].to_numpy(), "bar": located}
    ).sort_values(["symbol", "bar"])

    starts = frame.groupby("symbol")["bar"].diff().ne(1.0).sum()
    return float(len(frame) / starts) if starts else 0.0


def trading_stats(
    trades: pd.DataFrame, equity: pd.DataFrame, periods_per_year: float = TRADING_DAYS
) -> dict[str, float]:
    if trades.empty or equity.empty:
        return {
            "annual_turnover": 0.0,
            "trade_count": 0.0,
            "avg_holding_days": 0.0,
            "total_cost": 0.0,
            "cost_ratio": 0.0,
        }

    notional = trades.groupby("date")["notional"].sum()
    average_equity = equity["equity"].reindex(notional.index).ffill().replace(0.0, np.nan)
    turnover = (notional / average_equity).fillna(0.0)
    years = max(len(equity) / periods_per_year, 1e-9)

    costs = float(trades[["commission", "spread", "slippage"]].to_numpy().sum())
    gross_pnl = float(equity["equity"].iloc[-1] - equity["equity"].iloc[0]) + costs

    return {
        "annual_turnover": float(turnover.sum() / years),
        "trade_count": float(len(trades)),
        "avg_trade_notional": float(trades["notional"].mean()),
        "total_cost": costs,
        "cost_ratio": float(costs / gross_pnl) if gross_pnl else float("nan"),
        "commission": float(trades["commission"].sum()),
        "spread_cost": float(trades["spread"].sum()),
        "slippage": float(trades["slippage"].sum()),
    }


def return_stats(returns: pd.Series) -> dict[str, float]:
    positive = returns > 0.0
    wins = returns[positive]
    losses = returns[returns < 0.0]
    return {
        "hit_rate": float(positive.mean()) if len(returns) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else float("inf"),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurt()),
    }


# --- headline -------------------------------------------------------------
def performance(
    equity: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    trials: int = 1,
    periods_per_year: float = TRADING_DAYS,
) -> dict[str, float]:
    """Every headline number, in one dictionary, for one run."""
    curve = equity["equity"]
    returns = curve.pct_change().dropna()

    summary: dict[str, float] = {
        "start": str(equity.index.min().date()),
        "end": str(equity.index.max().date()),
        "bars": float(len(equity)),
        "total_return": total_return(curve),
        "cagr": cagr(curve, periods_per_year),
        "volatility": annual_volatility(returns, periods_per_year),
        "downside_deviation": downside_deviation(returns, 0.0, periods_per_year),
        "sharpe": sharpe(returns, 0.0, periods_per_year),
        "sortino": sortino(returns, 0.0, periods_per_year),
        "calmar": calmar(curve, periods_per_year),
        "max_drawdown": max_drawdown(curve),
        "drawdown_duration": float(drawdown_duration(curve)),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "worst_month": worst_month(returns),
        "deflated_sharpe": deflated_sharpe(returns, trials, periods_per_year),
        "trials": float(trials),
        "avg_leverage": float(equity["leverage"].replace([np.inf, -np.inf], np.nan).mean())
        if "leverage" in equity
        else float("nan"),
        "avg_net_exposure": float(equity["net"].div(curve.replace(0.0, np.nan)).mean())
        if "net" in equity
        else float("nan"),
    }
    summary.update(return_stats(returns))

    if trades is not None:
        summary.update(trading_stats(trades, equity, periods_per_year))

    if positions is not None:
        summary["avg_holding_days"] = holding_period(positions, equity.index)
        summary["avg_positions"] = float(len(positions) / max(len(equity), 1))

    if benchmark is not None and len(benchmark):
        alpha, beta = alpha_beta(returns, benchmark, periods_per_year)
        summary["alpha"] = alpha
        summary["beta"] = beta
        summary["information_ratio"] = information_ratio(returns, benchmark, periods_per_year)

    return summary
