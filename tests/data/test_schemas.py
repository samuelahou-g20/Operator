import pandas as pd
import pytest

from src.data import schemas


def _bars_df():
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [10.0, 20.0],
        }
    )


def test_validate_accepts_conforming():
    df = _bars_df()
    assert schemas.validate(df, "bars") is df


def test_validate_allows_extra_contract_column():
    df = _bars_df()
    df["contract"] = pd.array(["ESM6", "ESM6"], dtype="string")
    schemas.validate(df, "bars")  # must not raise


def test_validate_missing_column():
    df = _bars_df().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        schemas.validate(df, "bars")


def test_validate_wrong_dtype():
    df = _bars_df()
    df["close"] = df["close"].astype("int64")
    with pytest.raises(ValueError, match="dtype"):
        schemas.validate(df, "bars")


def test_validate_string_accepts_object():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01"]),
            "price": [1.0],
            "size": [1.0],
            "side": ["buy"],  # object dtype, expected 'string'
        }
    )
    schemas.validate(df, "trades")  # must not raise


def test_validate_unknown_data_type():
    with pytest.raises(ValueError, match="unknown data_type"):
        schemas.validate(_bars_df(), "nope")
