import pandas as pd
import pytest

from src.common import paths
from src.data import lake_io, query
from src.data.etl.futures import stitch


def test_continuous_dir_matches_get_continuous_path(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    out = stitch.continuous_dir("cme", "es", 1, "1d")
    assert out == tmp_path / "processed" / "futures" / "cme" / "es" / "continuous" / "es1" / "1d"


def test_get_continuous_empty_when_unstitched(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    df = query.get_continuous("cme", "es", strip=1, timeframe="1d")
    assert df.empty and df.index.name == "ts"


# --- pure roll-logic helpers ---

def _daily(dates, close, volume):
    idx = pd.DatetimeIndex(dates, name="ts")
    return pd.DataFrame(
        {"open": close, "high": [c + 1 for c in close], "low": [c - 1 for c in close],
         "close": close, "volume": volume},
        index=idx,
    )


def _contracts():
    return pd.DataFrame({
        "contract": ["ESH4", "ESM4"],
        "fnd": [pd.NaT, pd.NaT], "ltd": [pd.NaT, pd.NaT],
        "last_trade": [pd.Timestamp("2024-03-15"), pd.Timestamp("2024-06-21")],
    }).set_index("contract")


def test_anchor_and_expiry_fall_back_to_last_trade():
    row = _contracts().loc["ESH4"]
    assert stitch._anchor(row) == pd.Timestamp("2024-03-15")   # fnd/ltd null -> last_trade
    assert stitch._expiry(row) == pd.Timestamp("2024-03-15")


def test_stitch_daily_force_roll_path():
    h = pd.bdate_range("2024-02-01", "2024-03-15")
    m = pd.bdate_range("2024-02-01", "2024-06-21")
    daily = {
        "ESH4": _daily(h, [100.0] * len(h), [1000.0] * len(h)),  # vol always high => no crossover
        "ESM4": _daily(m, [110.0] * len(m), [10.0] * len(m)),
    }
    results, schedule = stitch._stitch_daily(daily, _contracts(), [1],
                                             ma_start=20, ma_window=3, days=5, max_ratio=2.0)
    assert len(schedule) == 1
    assert pd.Timestamp(schedule[0]["roll_date"]) == pd.Timestamp("2024-03-08")  # 5 bdays before LTD
    assert schedule[0]["ratio"] == pytest.approx(1.1)   # 110/100
    assert schedule[0]["spread"] == pytest.approx(10.0)
    df = results[1]
    assert set(df["contract"].unique()) == {"ESH4", "ESM4"}
    # spread back-adjustment makes the front segment line up with the back (continuous):
    front = df[df["contract"] == "ESH4"]
    assert front["spread_adjusted_close"].iloc[-1] == pytest.approx(110.0)


def test_stitch_daily_volume_crossover_rolls_early():
    h = pd.bdate_range("2024-02-01", "2024-03-15")
    m = pd.bdate_range("2024-02-01", "2024-06-21")
    cut = pd.Timestamp("2024-02-21")
    daily = {
        "ESH4": _daily(h, [100.0] * len(h), [1000.0 if d < cut else 10.0 for d in h]),
        "ESM4": _daily(m, [110.0] * len(m), [10.0 if d < cut else 1000.0 for d in m]),
    }
    _, schedule = stitch._stitch_daily(daily, _contracts(), [1],
                                       ma_start=20, ma_window=3, days=5, max_ratio=2.0)
    roll = pd.Timestamp(schedule[0]["roll_date"])
    assert pd.Timestamp("2024-02-21") <= roll < pd.Timestamp("2024-03-08")  # earlier than force-roll


def test_snap_returns_none_past_data():
    front = _daily(pd.bdate_range("2024-02-01", "2024-02-10"), [1.0] * 7, [1.0] * 7)
    back = front.copy()
    assert stitch._snap(pd.Timestamp("2024-03-01"), front, back) is None  # target past the data


def test_stitch_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    days = pd.bdate_range("2024-03-01", "2024-03-15")

    def write_contract(contract, close, vol):
        for d in days:
            ts = pd.to_datetime([d + pd.Timedelta(minutes=mn) for mn in (0, 1)])
            frame = pd.DataFrame({"ts": ts, "open": [close] * 2, "high": [close + 1] * 2,
                                  "low": [close - 1] * 2, "close": [close] * 2, "volume": [vol] * 2})
            part = (paths.data_path("processed", "futures", "cme", "es", "bars", "1m")
                    / f"contract={contract}" / f"date={d.strftime('%Y-%m-%d')}" / "part.parquet")
            lake_io.write_table(frame, part)

    write_contract("ESH4", 100.0, 1000.0)
    write_contract("ESM4", 110.0, 10.0)
    contracts = pd.DataFrame({
        "contract": ["ESH4", "ESM4"], "symbol": ["es", "es"],
        "expiry": [pd.Timestamp("2024-03-15"), pd.Timestamp("2024-04-19")],
        "first_trade": [pd.NaT, pd.NaT],
        "last_trade": [pd.Timestamp("2024-03-15"), pd.Timestamp("2024-04-19")],
        "fnd": [pd.NaT, pd.NaT], "ltd": [pd.NaT, pd.NaT],
        "tick_size": [0.25, 0.25], "multiplier": [50.0, 50.0], "currency": ["USD", "USD"],
    })
    lake_io.write_csv(contracts, paths.data_path("processed", "futures", "cme", "es", "contracts") / "es.csv")

    summary = stitch.stitch("cme", "es", strips=(1,), bars_timeframe="1m", intraday_timeframes=("1m",))
    assert summary.get(1, 0) > 0

    cont = query.get_continuous("cme", "es", strip=1, timeframe="1d")
    assert cont.index.name == "ts"
    assert set(cont["contract"].unique()) == {"ESH4", "ESM4"}
    assert "ratio_adjusted_close" in cont.columns
    # roll schedule written alongside the strip
    sched = stitch.continuous_dir("cme", "es", 1, "1d").parent / "roll_schedule.csv"
    assert sched.is_file()
