"""Common: calendar rules, settings layering, atomic IO, partitions."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from Common.calendar import TradingCalendar, early_close_days, easter, holidays
from Common.config import Settings, load_settings, reset_settings
from Common.io import atomic_path, read_matrix, read_parquet, write_matrix, write_parquet
from Common.provenance import hash_payload
from Common.types import Partition


class TestCalendar:
    def test_easter_matches_known_dates(self):
        assert easter(2024) == date(2024, 3, 31)
        assert easter(2025) == date(2025, 4, 20)
        assert easter(2000) == date(2000, 4, 23)

    def test_known_holidays(self):
        days = holidays(2024)
        assert date(2024, 1, 1) in days       # New Year's Day
        assert date(2024, 1, 15) in days      # MLK
        assert date(2024, 3, 29) in days      # Good Friday
        assert date(2024, 5, 27) in days      # Memorial Day
        assert date(2024, 6, 19) in days      # Juneteenth
        assert date(2024, 7, 4) in days
        assert date(2024, 11, 28) in days     # Thanksgiving
        assert date(2024, 12, 25) in days

    def test_saturday_new_year_is_not_observed(self):
        # 2022-01-01 was a Saturday; the NYSE did not close on 2021-12-31.
        assert date(2021, 12, 31) not in holidays(2021)

    def test_sunday_holiday_rolls_to_monday(self):
        # 2021-07-04 was a Sunday, observed on Monday the 5th.
        assert date(2021, 7, 5) in holidays(2021)

    def test_juneteenth_only_from_2022(self):
        assert date(2021, 6, 18) not in holidays(2021)
        assert date(2022, 6, 20) in holidays(2022)

    def test_ad_hoc_closure(self):
        assert date(2012, 10, 29) in holidays(2012)

    def test_sessions_exclude_weekends_and_holidays(self):
        calendar = TradingCalendar()
        sessions = calendar.sessions("2024-07-01", "2024-07-08")
        observed = [stamp.date() for stamp in sessions]
        assert date(2024, 7, 4) not in observed
        assert date(2024, 7, 6) not in observed
        assert len(observed) == 5

    def test_2024_has_252_sessions(self):
        assert len(TradingCalendar().sessions("2024-01-01", "2024-12-31")) == 252

    def test_next_and_previous_session(self):
        calendar = TradingCalendar()
        assert calendar.next_session("2024-07-03").date() == date(2024, 7, 5)
        assert calendar.previous_session("2024-07-05").date() == date(2024, 7, 3)

    def test_early_closes(self):
        assert date(2024, 11, 29) in early_close_days(2024)  # day after Thanksgiving
        assert date(2024, 7, 3) in early_close_days(2024)

    def test_missing_sessions_finds_the_gap(self):
        calendar = TradingCalendar()
        full = calendar.sessions("2024-03-01", "2024-03-15")
        with_gap = full.delete([3, 4])
        missing = calendar.missing_sessions(with_gap, "2024-03-01", "2024-03-15")
        assert list(missing) == [full[3], full[4]]

    def test_rebalance_dates_are_a_subset(self):
        calendar = TradingCalendar()
        sessions = calendar.sessions("2024-01-01", "2024-06-30")
        weekly = calendar.rebalance_dates(sessions, "weekly")
        assert set(weekly).issubset(set(sessions))
        assert 20 < len(weekly) < 30
        assert len(calendar.rebalance_dates(sessions, "monthly")) == 6


class TestSettings:
    def test_environment_overrides_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUANT_DATA_ROOT", str(tmp_path / "lake_root"))
        reset_settings()
        resolved = load_settings()
        assert resolved.data_root == (tmp_path / "lake_root").resolve()
        assert resolved.bars_root == resolved.lake_root / "bars"
        reset_settings()

    def test_file_is_overridden_by_environment(self, monkeypatch, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text('[quant]\nregion = "US"\ninterval = "1wk"\n', encoding="utf-8")
        monkeypatch.setenv("QUANT_INTERVAL", "1d")
        resolved = load_settings(config)
        assert resolved.interval == "1d"

    def test_unknown_key_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown settings"):
            load_settings(tmp_path / "missing.toml", nonsense=1)

    def test_derived_paths_hang_off_data_root(self):
        settings = Settings()
        assert settings.catalog_path.parent == settings.lake_root
        assert settings.backtests_root == settings.results_root / "backtests"


class TestPartition:
    def test_key_and_hive_path(self):
        partition = Partition("US", "stock", "1d")
        assert partition.key == "US/stock/1d"
        assert partition.hive_path.as_posix() == "region=US/asset_class=stock/interval=1d"

    def test_parse_round_trip(self):
        assert Partition.parse("US/etf/1wk") == Partition("US", "etf", "1wk")

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError):
            Partition("US", "crypto", "1d")
        with pytest.raises(ValueError):
            Partition("US", "stock", "1sec")

    def test_is_hashable_and_frozen(self):
        assert len({Partition(), Partition()}) == 1


class TestIO:
    def test_matrix_round_trip(self, tmp_path):
        index = pd.date_range("2024-01-01", periods=5, tz="UTC")
        frame = pd.DataFrame({"AAA": [1.0, 2, 3, 4, 5], "BBB": [5.0, 4, 3, 2, 1]}, index=index)
        path = tmp_path / "metric.parquet"
        write_matrix(frame, path)
        restored = read_matrix(path)
        pd.testing.assert_frame_equal(frame, restored, check_names=False, check_freq=False)

    def test_matrix_read_tolerates_missing_columns(self, tmp_path):
        index = pd.date_range("2024-01-01", periods=3, tz="UTC")
        write_matrix(pd.DataFrame({"AAA": [1.0, 2, 3]}, index=index), tmp_path / "m.parquet")
        restored = read_matrix(tmp_path / "m.parquet", columns=["AAA", "ZZZ"])
        assert list(restored.columns) == ["AAA", "ZZZ"]
        assert restored["ZZZ"].isna().all()

    def test_parquet_round_trip(self, tmp_path):
        frame = pd.DataFrame({"symbol": ["A", "B"], "value": [1.5, 2.5]})
        write_parquet(frame, tmp_path / "flat.parquet")
        pd.testing.assert_frame_equal(frame, read_parquet(tmp_path / "flat.parquet"))

    def test_atomic_write_leaves_nothing_behind_on_failure(self, tmp_path):
        target = tmp_path / "artifact.bin"
        with pytest.raises(RuntimeError), atomic_path(target) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("interrupted")
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_hash_payload_is_order_independent(self):
        assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})
        assert hash_payload({"a": 1}) != hash_payload({"a": 2})
