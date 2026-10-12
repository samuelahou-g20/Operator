"""Canonical schema for every data_type, defined exactly once (ARCHITECTURE §5).

``SCHEMAS`` maps each data_type to its ``{column: dtype}``. Fetchers normalize
vendor data into these on write; ``query`` guarantees output conforms.
``validate`` checks column presence and dtypes (extra columns such as
``contract`` on raw futures are allowed). Timeframe lives in the path, not here.

MUST NOT: grow into pandera/pydantic models, coerce data, or define schemas
anywhere else in the codebase.
"""

from __future__ import annotations

import pandas as pd

SCHEMAS: dict[str, dict[str, str]] = {
    "bars": {
        "ts": "datetime64[ns]",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    },
    "trades": {
        "ts": "datetime64[ns]",
        "price": "float64",
        "size": "float64",
        "side": "string",
    },
    "l1": {
        "ts": "datetime64[ns]",
        "bid_px": "float64",
        "bid_sz": "float64",
        "ask_px": "float64",
        "ask_sz": "float64",
    },
    "l2": {
        "ts": "datetime64[ns]",
        "level": "int64",
        "side": "string",
        "px": "float64",
        "sz": "float64",
        "action": "string",
    },
    "funding_rate": {
        "ts": "datetime64[ns]",
        "rate": "float64",
        "next_funding_ts": "datetime64[ns]",
    },
    "open_interest": {
        "ts": "datetime64[ns]",
        "oi": "float64",
    },
    "rates": {
        "ts": "datetime64[ns]",       # observation date
        "value": "float64",           # annualized rate as a DECIMAL (0.0525 = 5.25%)
    },
    "contracts": {
        "contract": "string",
        "symbol": "string",
        "expiry": "datetime64[ns]",
        "first_trade": "datetime64[ns]",
        "last_trade": "datetime64[ns]",
        "fnd": "datetime64[ns]",
        "ltd": "datetime64[ns]",
        "tick_size": "float64",
        "multiplier": "float64",
        "currency": "string",
    },
}


def _dtype_ok(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    # String columns may arrive as object, the pandas 3.0 default 'str' dtype, or
    # a StringDtype variant ('string', 'string[python]', 'string[pyarrow]').
    if expected == "string" and (
        actual in ("string", "object", "str") or actual.startswith("string")
    ):
        return True
    # Accept any tz-naive datetime64 spelling (e.g. read back from parquet).
    if expected.startswith("datetime64") and actual.startswith("datetime64"):
        return True
    return False


def validate(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Check that ``df`` has the required columns and dtypes for ``data_type``.

    Extra columns are allowed. Raises ``ValueError`` on a missing column or an
    incompatible dtype. Returns ``df`` unchanged.
    """
    if data_type not in SCHEMAS:
        raise ValueError(f"unknown data_type {data_type!r}; known: {sorted(SCHEMAS)}")
    schema = SCHEMAS[data_type]

    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(
            f"{data_type}: missing columns {missing} (have {list(df.columns)})"
        )
    for col, expected in schema.items():
        actual = str(df[col].dtype)
        if not _dtype_ok(actual, expected):
            raise ValueError(
                f"{data_type}.{col}: expected dtype {expected!r}, got {actual!r}"
            )
    return df
