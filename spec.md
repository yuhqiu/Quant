# Quant Research Platform — Specification

End-to-end pipeline from market data acquisition to strategy backtesting.
Live order execution is **out of scope** for this revision, but every interface is
designed so that an `Execution` module can be added later without reworking the
upstream stages.

---

## 1. Scope

```
Acquisition -> Analysis/Cleaning -> Metrics/Features -> Signals -> Portfolio -> BackTest -> Evaluation
```

| Stage | Module | Status |
|---|---|---|
| Download & store raw market data | `DataAcquisition/` | Implemented |
| Clean, validate, quality report | `DataAcquisition/` (`cleaning.py`, `quality.py`) | Implemented |
| Derived indicators & panel features | `MetricsGeneration/` | Implemented, needs alignment (§5) |
| Alpha signals from features | `Signals/` | To build |
| Positions from signals | `Portfolio/` | To build |
| Simulate P&L with costs | `Strategy/BackTest/` | To build |
| Metrics, tearsheet, attribution | `Evaluation/` | To build |
| Shared config, calendar, logging, IO | `Common/` | To build |

---

## 2. General Requirements

1. Every module lives in its own top-level folder (PascalCase, e.g. `DataAcquisition`).
2. Every module folder contains a `test/` folder for unit tests and other test code.
3. Every module has an `__init__.py` exporting its public API, plus `__main__.py`
   and `cli.py` if it is runnable from the command line.
4. Modules communicate **only through files on disk and public Python APIs** —
   never by reaching into another module's internals.
5. Python 3.12+, `from __future__ import annotations`, PEP 604 unions
   (`str | None`), `pathlib.Path` (never `os.path`), `@dataclass` for value
   objects. No Pydantic.
6. No hidden state: every stage is idempotent and re-runnable. Re-running a stage
   on the same inputs produces the same outputs.
7. Determinism: any randomness (sampling, CV splits, tie-breaking) takes an
   explicit seed.
8. Storage is Parquet with `zstd` compression throughout. DuckDB is the query
   layer over Parquet; it is never the system of record.
9. All timestamps are timezone-aware UTC (`pa.timestamp("us", tz="UTC")`).
10. Structured logging via the `logging` module in library code; `print()` only
    in CLI presentation layers.

### 2.1 Repository Layout (target)

```
customized/
  Common/                 shared config, calendar, logging, parquet IO, types
    test/
  DataAcquisition/        download + clean + quality
    providers/            pluggable data vendors
    test/
  MetricsGeneration/      per-symbol indicators + cross-sectional features
    test/
  Signals/                feature -> alpha score
    test/
  Portfolio/              alpha -> target weights
    test/
  Strategy/
    BackTest/             vectorised + event-driven simulation engines
    Library/              concrete strategy definitions
    test/
  Evaluation/             performance metrics + reports
    test/
  DataSource/lake/        raw bar lake + DuckDB catalog (git-ignored)
  Metrics/                wide feature matrices (git-ignored)
  Results/                backtest artifacts (git-ignored)
  scripts/                PowerShell helpers (check-env.ps1, run-pipeline.ps1)
  pyproject.toml
  spec.md
```

---

## 3. Cross-Cutting Contracts

### 3.1 Canonical Bar Schema

Single source of truth: `DataAcquisition/schema.py`.

| Column | Type | Meaning |
|---|---|---|
| `ts` | `timestamp[us, UTC]` | Bar open timestamp |
| `symbol` | `string` | Normalised ticker |
| `open`, `high`, `low`, `close` | `float64` | **Unadjusted** prices |
| `volume` | `float64` | Share volume |
| `adj_close` | `float64` | Split + dividend adjusted close |
| `adj_factor` | `float64` | `adj_close / close` |
| `dividend` | `float64` | Cash dividend on this bar |
| `split_ratio` | `float64` | Split ratio on this bar |
| `repaired` | `bool` | A repair was applied to this bar |

Rule: **raw prices are stored unadjusted; adjustments are stored alongside as
derived columns.** This keeps the lake stable under future corporate actions — a
new dividend rewrites `adj_factor`, never `close`.

### 3.2 Partition Key

Every dataset is addressed by a `Partition(region, asset_class, interval)`, using
Hive-style directories so DuckDB can prune:

```
DataSource/lake/bars/region=US/asset_class=stock/interval=1d/{SYMBOL}.parquet
```

`region ∈ {US}` (extensible), `asset_class ∈ {stock, etf, other}`,
`interval ∈ {1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo}`.

### 3.3 Point-in-Time Discipline (non-negotiable)

Lookahead is the largest source of fake backtest returns. Therefore:

1. A feature stamped `ts = T` may only use information observable at or before
   the close of bar `T`.
2. Labels (`fwd_ret_*`) are the only forward-looking columns, and they live in a
   **separate directory** so they cannot be accidentally globbed into a feature
   set.
3. Signals computed on bar `T` execute no earlier than bar `T+1`; the backtest
   engine enforces a configurable `execution_lag` (default 1 bar).
4. The universe must be point-in-time: a symbol enters when listed and leaves
   when delisted. Universe snapshots carry `snapshot_date` so historical
   membership is reconstructable.
5. Delisted symbols stay in the lake. Dropping them creates survivorship bias
   and inflates returns.

### 3.4 Wide Matrix Format (features)

One Parquet file per metric, wide layout: index = date (UTC), columns = symbol,
values `float32` (`float64` for prices, volume, `obv`, `dollar_vol`).
Byte-stream-split encoding + zstd.

```
Metrics/{region}/{asset_class}/{interval}/{metric}.parquet
Metrics/{region}/{asset_class}/labels_{interval}/{label}.parquet
Metrics/{region}/{asset_class}/{interval}/_manifest.json
```

`_manifest.json` records `created_utc`, `source`, `layout`, `rows`, `symbols`,
`start`, `end`, `build_seconds`, library versions, `metrics[]`, `symbol_list[]`.

Rationale: strategies are cross-sectional and ask for "one feature, all symbols,
all dates". Wide layout makes that one file read instead of thousands.

---

## 4. Module: DataAcquisition — *implemented*

**Purpose.** Download bars from pluggable vendors, normalise, clean, persist to
the Parquet lake, and track ingestion watermarks in DuckDB.

### 4.1 Requirements

1. Download all US Stocks and ETFs. The universe is built from the NASDAQ traded
   symbol directory and classified into `stock` / `etf` / `other`.
2. Selectable interval and time range.
3. Extensible provider interface for future vendors (currently Yahoo only).
4. Incremental download: fetch only bars newer than the stored watermark, so a
   daily top-up does not re-download twenty years of history.
5. Derived values (`adj_close`, `adj_factor`, `dividend`, `split_ratio`) are
   stored explicitly, not recomputed ad hoc by consumers.
6. One bad symbol never fails the run; per-symbol status is recorded.

### 4.2 Provider Interface

```python
class MarketDataProvider(ABC):
    name: ClassVar[str]
    intervals: ClassVar[frozenset[str]]
    max_batch_size: ClassVar[int]
    request_pause: ClassVar[float]
    max_lookback: ClassVar[Mapping[str, pd.Timedelta]]

    @abstractmethod
    def fetch(self, request: FetchRequest) -> FetchResult: ...
```

`FetchResult` carries `frames: dict[str, DataFrame]` **and**
`errors: dict[str, str]`. A provider must never raise for a single bad symbol.

Adding a vendor: implement the ABC in `providers/`, register it. No other module
changes.

### 4.3 Ingestion Modes

| Mode | Behaviour |
|---|---|
| `full` | Fetch the whole requested range, ignoring catalog state |
| `incremental` | Fetch from `last_ts` onward, re-fetching the final stored bar to correct a partial candle |
| `auto` | `incremental` if a watermark exists, else `full` |

### 4.4 Catalog

DuckDB at `DataSource/lake/catalog.duckdb`.

- Table `ingest_state(provider, region, asset_class, interval, symbol, first_ts,
  last_ts, row_count, status, message, updated_at)`, primary key on the first
  five columns.
- View `bars` over the Hive-partitioned Parquet tree.
- View `universe` over `reference/universe.parquet`.
- `refresh_from_lake()` rebuilds watermarks by scanning files, so the catalog is
  always reconstructible and never authoritative.

### 4.5 Cleaning Rules

Applied by `clean_bars()` before write:

- Drop rows with NaN or non-positive `open/high/low/close`.
- Enforce `high >= max(open, close, low)` and `low <= min(open, close)`.
- Drop negative volume; optionally drop zero-volume bars.
- Sort by `ts`, keep the last row per timestamp.

`merge_bars(existing, incoming)` concatenates and keeps the newest row per `ts`,
so re-fetched bars supersede stale ones.

### 4.6 Quality Report

Per symbol, written to
`DataSource/lake/reports/quality_{region}_{asset_class}_{interval}.parquet`:
`rows`, `first_ts`, `last_ts`, `stale_days`, `null_close`, `zero_volume`,
`repaired_rows`, `bad_adj_factor`, `price_jumps` (close ratio outside
`[0.25, 4]`), `max_gap_days`, `repaired_ratio`, `zero_volume_ratio`.

**To add:** a `--fail-on` threshold set so scheduled runs exit non-zero when data
health degrades, and a calendar-aware `missing_sessions` count (needs the trading
calendar from `Common/`).

### 4.7 CLI

```
python -m DataAcquisition <command>
```

| Command | Purpose |
|---|---|
| `universe` | Refresh the US listing snapshot |
| `download` | Download bars (`--mode auto\|full\|incremental`) |
| `update` | Incremental top-up of everything already stored |
| `status` | Stored coverage per partition |
| `quality` | Build the quality report |
| `migrate` | Import legacy per-symbol CSV into the lake |
| `refresh-state` | Rebuild catalog watermarks from the lake |
| `query` | Run SQL against the lake |
| `export` | Export a partition to Parquet or CSV |

### 4.8 Configuration

`config.py` constants, with `DATA_ROOT` overridable via `QUANT_DATA_ROOT`.
Defaults: provider `yahoo`, region `US`, asset class `stock`, interval `1d`,
compression `zstd`.

---

## 5. Module: MetricsGeneration — *implemented, requires alignment*

**Purpose.** Turn per-symbol OHLCV into a wide feature panel plus forward-return
labels. Covers the original "data analysis" requirement: cleaning lives in
`DataAcquisition`, enrichment lives here.

### 5.1 Known Gaps (must be fixed)

| Gap | Fix |
|---|---|
| Reads CSV from `DataSource/US/Stock/day`, bypassing the Parquet lake | Read via `DataAcquisition.read_bars(partition, symbols)` |
| No `__init__.py` | Add, exporting `build()`, `assemble_metric()`, `read_matrix()` |
| No `test/` folder | Add, per §2 rule 2 |
| Uses raw `close` instead of `adj_close` | Return/momentum/volatility features must use the **adjusted** series; only liquidity features use raw price |
| Full rebuild every run (~190 s over 5.4k symbols) | Incremental mode keyed on the manifest `end` date |

### 5.2 Feature Blocks

Per-symbol, all causal:

- **Returns** — `ret_1d`, `logret_1d`, `ret_{5,21,63,126,252}d`, `ret_overnight`, `ret_intraday`
- **Momentum** — `mom_12_1`, `px_to_sma_{10,20,50,200}`, `sma_50_to_200`, `dist_52w_high`, `dist_52w_low`, `rsi_14`, `macd_line/signal/hist`, `adx_14`
- **Volatility** — `vol_{20,60,252}d`, `parkinson_20d`, `garman_klass_20d`, `yang_zhang_20d`, `atr_14`, `natr_14`, `downside_dev_60d`
- **Shape** — `sharpe_252d`, `sortino_252d`, `skew_252d`, `kurt_252d`, `dd_from_252d_high`, `max_dd_252d`, `hit_rate_252d`
- **Liquidity** — `dollar_vol`, `advd_20`, `advd_60`, `amihud_60d`, `stale_px_frac_20`, `zero_vol_frac_20`
- **Oscillators** — `cci_20`, `willr_14`, `stoch_k_14`, `stoch_d_14`, `bb_pctb_20`, `bb_width_20`, `obv`, `zscore_20`, `vol_zscore_20`, `dist_vwap_20`, `rel_ret_21d`

Cross-sectional, computed after the per-symbol pass:

- `cs_rank_{ret_21d, mom_12_1, vol_20d, advd_20, rsi_14}` — percentile rank per date
- `cs_z_{ret_21d, mom_12_1}` — z-score per date
- `beta_252d`, `corr_mkt_252d`, `idio_vol_252d` against an equal-weight market proxy
- `mkt_ret_1d`, stored as `_market.parquet`
- Dates with fewer than `MIN_CROSS_SECTION = 20` valid symbols are blanked

Labels, in a separate directory: `fwd_ret_1d`, `fwd_ret_5d`, `fwd_ret_21d`.

### 5.3 Build Pipeline

1. Resolve the symbol list from the lake / universe.
2. `ProcessPoolExecutor` over symbol batches, writing per-metric shards to staging.
3. Assemble each metric by concatenating shards on the symbol axis, reindex
   columns to the full symbol list, write the final Parquet.
4. Cross-sectional pass over the assembled matrices.
5. Write `_manifest.json`.

### 5.4 Feature Contract

- Naming convention `{family}_{window}{unit}`: `vol_20d`, `rsi_14`.
- Adding a feature is one pure function `f(frame: DataFrame) -> Series` plus a
  registry entry. The build driver does not change.
- Every feature declares its `min_periods`. Leading values stay `NaN` — never
  forward-filled, never zero-filled.

---

## 6. Module: Common — *to build*

Shared primitives, so the same logic is not reimplemented three times.

1. **`config.py`** — layered settings: defaults → `config.toml` → environment
   variables → CLI flags, resolved into one `Settings` dataclass.
2. **`calendar.py`** — NYSE/Nasdaq sessions, holidays, early closes. Powers
   `missing_sessions`, resampling, and backtest clocking.
3. **`logging.py`** — `get_logger(name)`; JSON lines to `DataSource/lake/logs/`,
   human-readable to console.
4. **`io.py`** — `write_parquet` / `read_parquet` with project compression and
   encoding defaults, plus atomic write (temp file + rename).
5. **`types.py`** — `Partition`, `Interval`, `AssetClass`, `Region`, so
   `DataAcquisition` need not be a dependency of every module.

---

## 7. Module: Signals — *to build*

**Purpose.** Map the feature panel to a per-symbol, per-date alpha score. A
signal is an opinion, not a position.

### 7.1 Requirements

1. A signal reads named metrics and returns a wide `DataFrame`
   (index = date, columns = symbol). It writes no files.
2. Interface:
   ```python
   class Signal(Protocol):
       name: str
       required_metrics: tuple[str, ...]
       def compute(self, panel: FeaturePanel) -> pd.DataFrame: ...
   ```
   `FeaturePanel` lazily loads and caches requested metric matrices, so a signal
   declaring three metrics reads exactly three files.
3. Composition primitives: `WeightedSignal`, `RankCombine`, `ZScoreCombine`.
4. Neutralisation helpers: cross-sectional demean, sector-neutral, beta-neutral,
   winsorise at configurable quantiles.
5. Every signal is masked by a **tradability filter** before use: price ≥
   `min_price`, `advd_20` ≥ `min_dollar_volume`, present in the point-in-time
   universe, not halted.

### 7.2 Signal Quality Report

A signal must pass this before it reaches a backtest:

- **IC** — Spearman rank correlation of score vs `fwd_ret_{1,5,21}d`, per date.
- **IC mean / std / IR / t-stat**, plus IC decay across horizons.
- **Quantile spread** — mean forward return of the top decile minus the bottom.
- **Turnover** — mean absolute rank change per period. A signal with 200 % daily
  turnover is dead on arrival after costs.
- **Autocorrelation** of the score at lags 1, 5, 21.
- **Coverage** — non-NaN symbol count per date.

Written to `Results/signals/{signal_name}/report.parquet` plus a plot bundle.

---

## 8. Module: Portfolio — *to build*

**Purpose.** Convert alpha scores into target weights, subject to constraints.

### 8.1 Requirements

1. Interface:
   ```python
   class PortfolioConstructor(Protocol):
       def target_weights(
           self, scores: pd.DataFrame, context: MarketContext
       ) -> pd.DataFrame: ...
   ```
   Returns weights indexed date × symbol, summing to the configured gross.
2. Built-in constructors:
   - `TopNEqualWeight(n, long_only=True)`
   - `QuantileLongShort(quantiles, gross, net=0.0)`
   - `ScoreProportional(cap_per_name)`
   - `InverseVolWeighted(vol_metric="vol_20d")`
   - `MeanVariance(cov_estimator, risk_aversion)` with Ledoit-Wolf shrinkage
3. Constraints as a composable chain: `max_weight_per_name`, `max_sector_weight`,
   `max_gross`, `max_net`, `max_beta`, `min_positions`,
   `max_turnover_per_rebalance`.
4. Rebalance schedule `daily | weekly | monthly | on_signal_change`, aligned to
   the trading calendar, with an optional no-trade band (skip names whose target
   moves less than `epsilon`) to suppress churn.
5. Weights derive from information available at bar `T` and apply at
   `T + execution_lag`.

---

## 9. Module: Strategy / BackTest — *to build*

**Purpose.** Simulate the P&L of `target_weights` against historical bars with
realistic frictions.

### 9.1 Design Decision

Two engines, one shared result format:

| Engine | Use |
|---|---|
| **Vectorised** (primary) | Daily cross-sectional strategies over thousands of symbols. NumPy over the wide matrices. Fast enough for parameter sweeps. |
| **Event-driven** (secondary) | Path-dependent logic — stops, trailing exits, intraday, order-level detail. Slower; validates the vectorised result. |

For any strategy expressible in both, they must produce an identical
`BacktestResult`. That equivalence is itself a test.

### 9.2 Requirements

1. A strategy binds the pieces declaratively:
   ```python
   @dataclass(frozen=True)
   class StrategySpec:
       name: str
       universe: UniverseSpec
       signal: Signal
       constructor: PortfolioConstructor
       costs: CostModel
       start: pd.Timestamp
       end: pd.Timestamp
       initial_capital: float = 1_000_000.0
       execution_lag: int = 1
       rebalance: str = "weekly"
       seed: int = 0
   ```
2. **Execution assumptions** are explicit and configurable:
   - Fill price: `next_open` (default), `vwap_proxy`, or `close`.
   - Partial fills capped at `participation_rate * volume` of the fill bar;
     unfilled quantity carried or cancelled per config.
   - No trading on a bar in which the symbol did not trade.
3. **Cost model**, pluggable, defaults:
   - Commission: per-share or bps of notional, with a per-order minimum.
   - Spread: `half_spread_bps`, or estimated from high/low when unavailable.
   - Slippage: square-root impact, `k * sigma * sqrt(notional / advd_20)`.
   - Borrow cost on shorts, accrued daily.
   - Interest on idle cash.

   Costs are always on. A zero-cost run requires an explicit opt-in flag, because
   a cost-free backtest is a marketing document, not a result.
4. **Corporate actions** — positions adjust for splits; dividends credit cash on
   the ex-date. This is why the lake stores `dividend` and `split_ratio`.
5. **Accounting** — daily mark-to-market, separate cash / position / equity
   ledgers, no negative cash unless margin is explicitly enabled.
6. **Output** to `Results/backtests/{strategy}/{run_id}/`:
   - `equity.parquet` — date, equity, cash, gross, net, leverage
   - `positions.parquet` — date × symbol shares and weights
   - `trades.parquet` — date, symbol, side, qty, price, commission, slippage
   - `metrics.json` — headline performance numbers
   - `spec.json` — resolved `StrategySpec`, git commit, input manifest hashes
7. **Determinism** — same spec plus same data manifest ⇒ byte-identical outputs.

### 9.3 Anti-Overfitting Protocol

1. **Split the data.** In-sample for development; out-of-sample held back and run
   at most once per strategy version. The engine records how many times a given
   `spec_hash` has touched the OOS window.
2. **Walk-forward** — rolling train/test windows with a purge and embargo gap
   sized to the label horizon, so overlapping labels cannot leak.
3. **Parameter sweeps** report the full surface, not the maximum. A strategy that
   works at exactly one parameter value is noise.
4. **Deflated Sharpe** or an equivalent multiple-testing adjustment whenever more
   than one configuration was evaluated.
5. **Baselines** — buy-and-hold SPY, equal-weight universe, and a random-signal
   null distribution matched on turnover and constraints.

---

## 10. Module: Evaluation — *to build*

**Purpose.** Turn a `BacktestResult` into a judgement.

### 10.1 Metrics

- **Return** — total, CAGR, annualised mean.
- **Risk** — annualised vol, downside deviation, max drawdown, drawdown
  duration, VaR/CVaR at 95 %, worst month.
- **Risk-adjusted** — Sharpe, Sortino, Calmar, deflated Sharpe.
- **Attribution** — alpha and beta vs benchmark, sector/factor exposure over
  time, contribution by symbol and by holding period.
- **Trading** — annual turnover, average holding period, hit rate, profit factor,
  average win/loss, total cost as a fraction of gross P&L.
- **Capacity** — P&L decay as AUM scales, from the impact model.

### 10.2 Outputs

- `report.html` — equity curve (log scale), underwater drawdown plot, rolling
  252-day Sharpe, monthly return heatmap, exposure over time, cost breakdown.
- `metrics.json` — machine-readable, for cross-run comparison.
- `compare(run_ids)` — side-by-side table across runs.

---

## 11. Testing Strategy

Each module's `test/` folder contains:

| Layer | Content |
|---|---|
| `test_offline.py` | Pure-function unit tests on synthetic fixtures. No network, no lake. Runs in seconds. |
| `test_live.py` | Integration tests against real vendors / the real lake. Skipped unless `QUANT_LIVE_TESTS=1`. |
| `fixtures/` | Small deterministic Parquet/CSV samples committed to the repo. |

Required correctness tests:

1. **Indicator ground truth** — each indicator checked against a hand-computed
   fixture, not against another implementation of itself.
2. **Lookahead detection** — recompute every feature on a truncated history; the
   value at date `T` must be unchanged. Any feature that moves when future data
   is removed is leaking. Runs across the whole feature registry.
3. **Backtest invariants** — equity equals cash plus marked positions on every
   bar; trades reconcile position deltas exactly; a zero-signal strategy returns
   exactly the cash rate.
4. **Known-answer backtest** — a two-symbol, ten-bar scenario whose expected
   equity path is written out by hand in the test.
5. **Engine equivalence** — vectorised and event-driven agree within tolerance on
   a shared strategy.
6. **Idempotence** — ingesting twice adds zero rows; building metrics twice
   produces identical files.

Standardise on `pytest`, replacing the bespoke harness in
`DataAcquisition/test/harness.py`.

---

## 12. Operations

### 12.1 Daily Pipeline

```powershell
python -m DataAcquisition universe
python -m DataAcquisition update   --interval 1d
python -m DataAcquisition quality  --interval 1d --fail-on stale_days=5
python -m MetricsGeneration build  --incremental
python -m Strategy backtest --spec Strategy/Library/momentum.toml
python -m Evaluation report --run latest
```

Wrapped in `scripts/run-pipeline.ps1`, exiting non-zero on any stage failure.

### 12.2 Dependencies

Consolidate the three `requirements.txt` files into one `pyproject.toml` with
optional groups, resolving the current conflicts (`pyarrow>=17` vs
`pyarrow==25.0.1`; two different `yfinance` pins).

```
core:      pandas, pyarrow, duckdb, numpy
acquire:   yfinance, requests
metrics:   scipy
backtest:  own engine; vectorbt/backtrader only if a dependency is justified
report:    matplotlib, jinja2
dev:       pytest, ruff, mypy
```

The VS Code tasks reference `scripts/check-env.ps1` and `scripts/lean.ps1`
(QuantConnect LEAN), neither of which exists. Either create `scripts/` or delete
those tasks — a broken task is worse than no task.

### 12.3 Reproducibility

Every artifact records the git commit, library versions, input manifest hashes,
the resolved config, and the run timestamp. A result you cannot reproduce is not
a result.

---

## 13. Build Order

1. `Common/` — config, calendar, logging, IO. Everything else depends on it.
2. Align `MetricsGeneration/` to the lake and to `adj_close`; add tests.
3. The lookahead test harness — build it *before* the strategy code, so the
   strategy is validated from day one.
4. `Signals/` with two reference signals (12-1 momentum, short-term reversal)
   plus the IC/turnover report.
5. `Portfolio/` with `QuantileLongShort` and the constraint chain.
6. `Strategy/BackTest/` vectorised engine plus the cost model.
7. `Evaluation/` tearsheet.
8. Event-driven engine and the equivalence test.
9. Walk-forward and multiple-testing tooling.

---

## 14. Explicitly Out of Scope

Live order execution, broker connectivity, real-time streaming, order management,
risk kill-switches, and anything touching real capital. `BacktestResult` is the
seam where an `Execution/` module attaches later.





