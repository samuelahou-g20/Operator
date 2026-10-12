"""Dashboard-parity JASON trend scoring (vendored from JasonAnalytics/jason_engine).

Faithful copy of ``jason_engine/trend_analysis.py`` — the multi-factor, cross-sectionally
normalized trend engine behind the Trend Table dashboard. The six default factors are each a
half-life-weighted EWMA of a JASON ``low_z``/``high_z`` (+OHLC) signal; ``run_analysis_from_data``
normalizes every factor by its max-abs across the supplied group and averages them into
``Average Score`` (per ticker, in [-1, 1]). Run with a 7d half-life for the short score and 28d
for the long score to reproduce the dashboard's regime axes.

Only the module's imports/helpers are localized: ``get_logger`` and the trading-day/hour constants
come from ``src.signals.jason`` (already vendored); the sigmoid + annualization helpers are copied
here verbatim. The scoring logic below is UNCHANGED from the source.

MUST NOT: fetch data, place orders, or touch the lake — callers supply OHLC + JASON frames.
"""

from __future__ import annotations

import datetime as dt  # noqa: F401 - kept for parity with the source module
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.signals.jason import TRADING_DAYS_PER_YEAR, TRADING_HOURS_PER_DAY, get_logger


# --- helpers copied verbatim from jason_engine/utils.py --------------------- #
def sigmoid_z(x: pd.Series, threshold: float = 2.0, k: float = 10.0) -> pd.Series:
    numeric_x = x.astype(float).to_numpy()
    result = 1 / (1 + np.exp(-k * (numeric_x - threshold)))
    return pd.Series(result, index=x.index)


def sigmoid_diff(x: pd.Series, threshold: float = 1.0, k: float = 10.0) -> pd.Series:
    numeric_x = x.astype(float).to_numpy()
    result = 1 / (1 + np.exp(-k * (numeric_x - threshold))) - 1 / (1 + np.exp(-k * (-numeric_x - threshold)))
    return pd.Series(result, index=x.index)


def sigmoid_vol(x: pd.Series, hist_vol: float = 50.0, k: float = 0.2) -> pd.Series:
    numeric_x = x.astype(float).to_numpy()
    result = 1 / (1 + np.exp(-k * (numeric_x - hist_vol)))
    return pd.Series(result, index=x.index)


def get_annualization_factor(freq: str, asset_class: str = "crypto") -> float:
    trading_days = TRADING_DAYS_PER_YEAR.get(asset_class, 365)
    hours_per_day = TRADING_HOURS_PER_DAY.get(asset_class, 24.0)
    freq_to_periods = {
        "1min": hours_per_day * 60, "5min": hours_per_day * 12,
        "15min": hours_per_day * 4, "30min": hours_per_day * 2,
        "1h": hours_per_day, "4h": hours_per_day / 4,
        "12h": hours_per_day / 12, "1d": 1.0,
    }
    periods_per_day = freq_to_periods.get(freq, 1.0)
    return trading_days * periods_per_day


logger = get_logger("jason_trend_analysis")



# ============================================================================
# TREND FUNCTION DEFINITIONS
# ============================================================================

# Default list of trend functions to use
DEFAULT_TREND_FUNCTIONS = [
    "Z-Score > Threshold",
    "Z-Score Low Vol",
    "Avg Z-Diff",
    "Z-Diff > Threshold",
    "Time Since Pivots",
    "Trend Efficiency",
]


# ============================================================================
# TREND SCORE CALCULATION FUNCTIONS
# ============================================================================

def _calculate_weighted_average_score(
    data_series: pd.Series,
    half_life_weights: Dict[pd.Timedelta, float]
) -> float:
    """
    Helper to calculate the weighted average score for a given data series.

    Args:
        data_series: Time series of values
        half_life_weights: Dict mapping half-life timedeltas to weights

    Returns:
        Weighted average of EWM scores across half-lives
    """
    weighted_scores = 0.0
    total_weight = 0.0

    for hl, weight in half_life_weights.items():
        score = data_series.ewm(halflife=hl, times=data_series.index).mean().iloc[-1] # .ewm.mean(): list exp weight means of all lengths within the window; iloc[-1]: EWMA of the whole series (last one)
        weighted_scores += (score * weight) # different EWMA with different half lifes has got different weights
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_scores / total_weight


def _log_returns(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    **kwargs
) -> float:
    """Calculates trend score based on time-averaged log returns."""
    log_returns = np.log(ohlc['close']).diff()

    weighted_scores = 0.0
    total_weight = 0.0
    for hl, weight in half_life_weights.items():
        ewm_mean_return = log_returns.ewm(halflife=hl, times=log_returns.index).mean().iloc[-1] # at the end of series only
        score = np.clip(ewm_mean_return * 100, -1, 1)
        weighted_scores += (score * weight)
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_scores / total_weight


def _z_score_above_threshold(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    threshold: float = 2.0,
    **kwargs
) -> float:
    """Computes trend based on time-averaged sigmoid-weighted Z-scores."""
    z_signal = sigmoid_z(jason['low_z'], threshold) - sigmoid_z(jason['high_z'], threshold) # [-1,1]
    return _calculate_weighted_average_score(z_signal, half_life_weights)


def _z_score_low_vol(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    freq: str = '4h',
    threshold: float = 2.0,
    asset_class: str = 'crypto',
    **kwargs
) -> float:
    """Computes trend based on time-averaged Z-scores during low vol."""
    # Get the correct annualization factor based on asset class
    annualization_factor = get_annualization_factor(freq, asset_class)

    historical_vol = 100 * np.log(ohlc['close']).diff().std() * np.sqrt(annualization_factor) # historical annualized realized vol

    vol_weight_low = sigmoid_vol(jason['low_vol'], hist_vol=historical_vol) # normal sigmoid with threhold being hist_vol -> higher low_vol -> higher vol_weight_low
    vol_weight_high = sigmoid_vol(jason['high_vol'], hist_vol=historical_vol)

    z_signal = (vol_weight_low * sigmoid_z(jason['low_z'], threshold) -
                vol_weight_high * sigmoid_z(jason['high_z'], threshold)) # uptrend (if z_low is large) is amplified if low_vol is large -> is this the intention????

    return _calculate_weighted_average_score(z_signal, half_life_weights)


def _avg_z_diff(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    **kwargs
) -> float:
    """Computes trend based on the time-averaged EWM of (low_z - high_z)."""
    z_diff = jason['low_z'] - jason['high_z']
    return _calculate_weighted_average_score(z_diff, half_life_weights)


def _avg_z_diff_above_threshold(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    threshold: float = 1.0,
    **kwargs
) -> float:
    """Computes trend based on the time-averaged EWM of sigmoid-transformed Z-diffs."""
    z_diff = jason['low_z'] - jason['high_z']
    z_signal = sigmoid_diff(z_diff, threshold=threshold)
    return _calculate_weighted_average_score(z_signal, half_life_weights)


def _pivots_difference(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    **kwargs
) -> float:
    """Computes trend based on the time-averaged EWM of pivot difference changes."""
    price_range = abs(ohlc['close'].iloc[-1] - ohlc['close'].iloc[0]) # forward looking normalization
    if price_range == 0:
        return 0.0

    pivots_diff = (jason['high_pivot'].diff() - jason['low_pivot'].diff()) / price_range # not a clear trend, I guess? higher high + lower low -> close to zero ????
    return _calculate_weighted_average_score(pivots_diff, half_life_weights)


def _pivots_time_difference(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    **kwargs
) -> float:
    """Computes trend based on the time-averaged EWM of time since pivots."""
    time_pivot_diff = jason['low_days'] - jason['high_days']
    return _calculate_weighted_average_score(time_pivot_diff, half_life_weights)


def _obv_momentum(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    **kwargs
) -> float:
    """Computes trend based on the time-averaged EWM of On-Balance Volume changes."""
    if 'Volume' not in ohlc.columns and 'volume' not in ohlc.columns:
        return 0.0

    vol_col = 'Volume' if 'Volume' in ohlc.columns else 'volume'
    price_change = ohlc['close'].diff()
    signed_volume = np.sign(price_change).fillna(0) * ohlc[vol_col]
    obv = signed_volume.cumsum()
    obv_diff = obv.diff() ####################### I think obv_diff is the same thing as signed volume ???

    weighted_scores = 0.0
    total_weight = 0.0
    for hl, weight in half_life_weights.items():
        obv_momentum = obv_diff.ewm(halflife=hl, times=obv_diff.index).mean().iloc[-1]
        ewm_volume = ohlc[vol_col].ewm(halflife=hl, times=ohlc[vol_col].index).mean().iloc[-1]

        if ewm_volume == 0:
            score = 0.0
        else:
            score = np.clip(obv_momentum / ewm_volume, -1, 1)

        weighted_scores += (score * weight)
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_scores / total_weight


def _trend_efficiency(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    **kwargs
) -> float:
    """Measures trend quality/directness, from -1 to +1, averaged over time."""
    price_diffs = ohlc['close'].diff()

    weighted_scores = 0.0
    total_weight = 0.0
    for hl, weight in half_life_weights.items():
        ewm_direction = price_diffs.ewm(halflife=hl, times=price_diffs.index).mean().iloc[-1]
        ewm_volatility = price_diffs.abs().ewm(halflife=hl, times=price_diffs.index).mean().iloc[-1]

        if ewm_volatility == 0:
            score = 0.0
        else:
            efficiency_score = ewm_direction / ewm_volatility
            score = np.clip(efficiency_score, -1, 1)

        weighted_scores += (score * weight)
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_scores / total_weight


def _volatility_return_correlation(
    ohlc: pd.DataFrame,
    jason: pd.DataFrame,
    half_life_weights: Dict[pd.Timedelta, float],
    **kwargs
) -> float:
    """Measures the time-averaged EWM correlation between returns and volatility."""
    returns = np.log(ohlc['close']).diff()
    # Use a short rolling window for instantaneous volatility
    volatility = returns.rolling(window=5).std()

    weighted_scores = 0.0
    total_weight = 0.0
    for hl, weight in half_life_weights.items():
        correlation = returns.ewm(halflife=hl, times=returns.index).corr(volatility).fillna(0) # the correlation uses the same r for r and for vol -> mechanical correlation, maybe shift by 1????
        score = correlation.iloc[-1]

        weighted_scores += (score * weight)
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_scores / total_weight # positive: large upmoves associated with high vol?


# Mapping of function names to implementations
TREND_FUNCTION_MAP: Dict[str, Callable] = {
    "Log Returns": _log_returns,
    "Z-Score > Threshold": _z_score_above_threshold,
    "Z-Score Low Vol": _z_score_low_vol,
    "Avg Z-Diff": _avg_z_diff,
    "Z-Diff > Threshold": _avg_z_diff_above_threshold,
    "Pivots Difference": _pivots_difference,
    "Time Since Pivots": _pivots_time_difference,
    "OBV Momentum": _obv_momentum,
    "Trend Efficiency": _trend_efficiency,
    "Vol-Return Correlation": _volatility_return_correlation,
}

# Export as TREND_FUNCTIONS for backward compatibility
TREND_FUNCTIONS = list(TREND_FUNCTION_MAP.keys())


# ============================================================================
# JASON TREND ANALYZER CLASS
# ============================================================================

class JasonTrendAnalyzer:
    """
    A class to perform trend analysis on assets using JASON and OHLC data.

    This analyzer computes various trend scores by averaging them across
    multiple time horizons (half-lives) using a weighted average
    that prioritizes shorter time frames.
    """

    def __init__(self, config: Dict = None, half_life_weights: Dict = {pd.Timedelta('1d'): 3.0, pd.Timedelta('3d'): 2.0, pd.Timedelta('7d'): 1.0}):
        """
        Initializes the analyzer with a given configuration.

        Args:
            config: Optional dictionary containing configuration parameters like
                    coins, frequency, and trend functions. If None, uses defaults.
        """
        config = config or {}
        self.config = config
        self.coins = config.get("coins", [])
        self.freq = config.get("freq", "4h")
        self.asset_class = config.get("asset_class", "crypto")
        self.trend_functions = config.get("trend_functions", DEFAULT_TREND_FUNCTIONS)

        # Set a fixed list of half-lives with weights
        # More weight is given to shorter half-lives
        
        self.half_life_weights = half_life_weights

        self.ohlc_data = None
        self.jason_data = None
        self.results = None

    def run_analysis_from_data(
        self,
        ohlc_data: Dict[str, pd.DataFrame],
        jason_data: Dict[str, pd.DataFrame],
        trend_functions: List[str] = None,
        asset_class_by_symbol: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Runs trend analysis on pre-fetched data.

        This is an efficient method for use when data is already available.

        Args:
            ohlc_data: Dictionary of OHLC DataFrames keyed by ticker
            jason_data: Dictionary of JASON DataFrames keyed by ticker
            trend_functions: Optional list of trend function names to use
            asset_class_by_symbol: Optional mapping from ticker to asset class.

        Returns:
            DataFrame with trend scores, sorted by Average Score
        """
        if trend_functions is None:
            trend_functions = self.trend_functions

        all_scores = {}
        active_coins = list(jason_data.keys())

        for coin in active_coins:
            coin_ohlc = ohlc_data.get(coin)
            coin_jason = jason_data.get(coin)

            if coin_ohlc is None or coin_jason is None:
                continue
            if coin_ohlc.empty or coin_jason.empty:
                continue

            coin_scores = {}
            for func_name in trend_functions:
                try:
                    score_method = TREND_FUNCTION_MAP.get(func_name)
                    if score_method:
                        coin_asset_class = (asset_class_by_symbol or {}).get(coin, self.asset_class)
                        score = score_method(
                            coin_ohlc,
                            coin_jason,
                            self.half_life_weights,
                            freq=self.freq,
                            asset_class=coin_asset_class,
                        )
                        coin_scores[func_name] = score
                    else:
                        coin_scores[func_name] = 0.0
                except Exception as e:
                    logger.warning(f"Error computing {func_name} for {coin}: {e}")
                    coin_scores[func_name] = 0.0

            all_scores[coin] = coin_scores

        if not all_scores:
            return pd.DataFrame()

        df = pd.DataFrame(all_scores).T

        # Ensure column order matches requested functions
        available_funcs = [f for f in trend_functions if f in df.columns]
        if available_funcs:
            df = df[available_funcs]

        # Normalize each column by its max absolute value
        df_normalized = df.div(df.abs().max()).fillna(0)
        df_normalized['Average Score'] = df_normalized.mean(axis=1)

        return df_normalized.sort_values(by='Average Score', ascending=False).round(4)

    def get_summary(self) -> str:
        """Generates a string summary of the analysis results."""
        if self.results is None:
            return "Analysis has not been run yet."

        summary = [
            "\n" + "=" * 100,
            "TREND ANALYSIS SUMMARY",
            "=" * 100,
            f"\nDataset: {len(self.results)} coins, {len(self.results.columns)-1} trend functions",
            f"Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Half-lives: {[hl.days for hl in self.half_life_weights.keys()]} days (Weighted)",
            "\nTop Performers by Function:"
        ]

        for func in self.results.columns:
            if func == 'Average Score':
                continue
            top_coin = self.results[func].idxmax()
            top_score = self.results[func].max()
            bottom_coin = self.results[func].idxmin()
            bottom_score = self.results[func].min()
            summary.append(f"  - {func}: {top_coin} ({top_score:.4f}) | {bottom_coin} ({bottom_score:.4f})")

        summary.append("\nOverall Ranking (by Average Score):")
        for i, (coin, row) in enumerate(self.results.iterrows(), 1):
            summary.append(f"  {i:2d}. {coin}: {row['Average Score']:.4f}")

        summary.append("=" * 100)
        return "\n".join(summary)


# ============================================================================
# STANDALONE FUNCTION FOR TREND TABLE CREATION
# ============================================================================

def create_trend_analysis_table(
    ohlc_data: Dict[str, pd.DataFrame],
    jason_data: Dict[str, pd.DataFrame],
    coins: List[str],
    freq: str = "4h",
    lookback: int = 7,
    start_date: Optional[dt.datetime] = None,
    end_date: Optional[dt.datetime] = None,
    trend_functions: List[str] = None,
    asset_class_by_symbol: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Creates a trend analysis ranking table from OHLC and JASON data.

    This is a convenience function that wraps JasonTrendAnalyzer for
    one-off analyses.

    Args:
        ohlc_data: Dictionary of OHLC DataFrames keyed by ticker
        jason_data: Dictionary of JASON DataFrames keyed by ticker
        coins: List of coin tickers to analyze
        freq: Data frequency string
        lookback: Lookback period in days
        start_date: Optional start date for filtering data
        end_date: Optional end date for filtering data
        trend_functions: Optional list of trend function names
        asset_class_by_symbol: Optional mapping from ticker to asset class.

    Returns:
        DataFrame with trend scores and Average Score, sorted by rank
    """
    if trend_functions is None:
        trend_functions = DEFAULT_TREND_FUNCTIONS

    if end_date is None:
        end_date = dt.datetime.now(dt.timezone.utc)
    if start_date is None:
        start_date = end_date - pd.Timedelta(days=lookback)

    all_scores = {}

    half_life_weights = {
        pd.Timedelta('1d'): 3.0,
        pd.Timedelta('3d'): 2.0,
        pd.Timedelta('7d'): 1.0
    }

    for coin in coins:
        coin_ohlc = ohlc_data.get(coin)
        coin_jason = jason_data.get(coin)

        if coin_ohlc is None or coin_jason is None:
            logger.warning(f"Skipping {coin} due to missing data.")
            continue

        # Filter to date range
        coin_ohlc_filtered = coin_ohlc.loc[start_date:end_date] if not coin_ohlc.empty else coin_ohlc
        coin_jason_filtered = coin_jason.loc[start_date:end_date] if not coin_jason.empty else coin_jason

        if coin_ohlc_filtered.empty or coin_jason_filtered.empty:
            continue

        coin_asset_class = (asset_class_by_symbol or {}).get(coin, 'crypto')
        try:
            periods_per_trading_day = int(round(
                get_annualization_factor(freq, coin_asset_class) /
                TRADING_DAYS_PER_YEAR.get(coin_asset_class, 365)
            ))
        except ZeroDivisionError:
            periods_per_trading_day = 0
        periods_per_trading_day = max(periods_per_trading_day, 1)

        scores = {}

        # Calculate price change metrics. For traditional assets, one "day"
        # means one trading session (e.g. 4 x 4h equity bars, no weekend bars).
        if not coin_ohlc_filtered.empty and len(coin_ohlc_filtered) > 2 * periods_per_trading_day:
            last_price = coin_ohlc_filtered['close'].iloc[-1]
            price_24h_ago = coin_ohlc_filtered['close'].iloc[-1 - periods_per_trading_day]
            price_48h_ago = coin_ohlc_filtered['close'].iloc[-1 - 2 * periods_per_trading_day]

            scores['Last Spot Price'] = last_price
            scores['24h % Change'] = (last_price / price_24h_ago) - 1 if price_24h_ago else 0
            scores['Previous 24h % Change'] = (price_24h_ago / price_48h_ago) - 1 if price_48h_ago else 0
        else:
            scores['Last Spot Price'] = coin_ohlc_filtered['close'].iloc[-1] if not coin_ohlc_filtered.empty else np.nan
            scores['24h % Change'] = np.nan
            scores['Previous 24h % Change'] = np.nan

        # Calculate trend function scores
        for func_name in trend_functions:
            score_method = TREND_FUNCTION_MAP.get(func_name)
            if score_method:
                try:
                    score = score_method(
                        coin_ohlc_filtered,
                        coin_jason_filtered,
                        half_life_weights,
                        freq=freq,
                        asset_class=coin_asset_class,
                    )
                    scores[func_name] = score
                except Exception as e:
                    logger.error(f"Error computing {func_name} for {coin}: {e}")
                    scores[func_name] = 0.0
            else:
                scores[func_name] = 0.0

        all_scores[coin] = scores

    df = pd.DataFrame(all_scores).T
    if df.empty:
        return pd.DataFrame()

    # Reorder columns
    col_order = ['Last Spot Price', '24h % Change', 'Previous 24h % Change'] + trend_functions
    df = df.reindex(columns=[c for c in col_order if c in df.columns]).fillna(0)

    # Normalize only the trend function columns
    df_normalized = df.copy()
    for col in trend_functions:
        if col in df.columns:
            max_abs_val = df[col].abs().max()
            if max_abs_val > 0:
                df_normalized[col] = df[col] / max_abs_val

    # Calculate average score
    available_funcs = [f for f in trend_functions if f in df_normalized.columns]
    if available_funcs:
        df_normalized['Average Score'] = df_normalized[available_funcs].mean(axis=1)
    else:
        df_normalized['Average Score'] = 0

    return df_normalized.sort_values(by='Average Score', ascending=False)


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def compute_trend_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute basic trend features for the given DataFrame.

    This is a simple legacy function for basic trend detection.

    Args:
        df: Input DataFrame with OHLCV data.
        window: Window size for moving average.

    Returns:
        DataFrame with trend features added.
    """
    if df.empty:
        return df
    df = df.copy()
    df['moving_avg'] = df['close'].rolling(window=window).mean()
    df['trend'] = df['close'] > df['moving_avg']
    return df
