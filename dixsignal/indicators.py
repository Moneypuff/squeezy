"""Signal ingredients: moving averages and the high-decile DIX gate."""

import numpy as np
import pandas as pd


def sma(series_or_frame, window):
    """Simple moving average over `window` bars. min_periods=window, so the first
    `window-1` bars are NaN (no lookahead, no short-window artifacts)."""
    return series_or_frame.rolling(window, min_periods=window).mean()


def top_decile_gate_trailing(d_panel, window=252, q=0.90, min_periods=60):
    """Boolean panel: True where each name's D is in the TOP decile of its OWN trailing
    `window`-day distribution (rolling `q` quantile). This is a per-name, self-relative
    'is this name unusually accumulated for itself' gate -- robust to names having
    structurally different average dark ratios.

    A NaN D (name not yet trading, or a missing day) yields False.
    """
    if d_panel is None or d_panel.empty:
        return pd.DataFrame()
    thresh = d_panel.rolling(window, min_periods=min_periods).quantile(q)
    gate = d_panel >= thresh
    return gate.where(d_panel.notna() & thresh.notna(), False)


def top_decile_gate_cross_sectional(d_panel, q=0.90, min_names=10):
    """Boolean panel: True where each name's D is in the top decile ACROSS names on that
    day (cross-sectional rank). Answers 'which names are most accumulated today'. Days
    with fewer than `min_names` observations are all-False."""
    if d_panel is None or d_panel.empty:
        return pd.DataFrame()
    counts = d_panel.notna().sum(axis=1)
    thresh = d_panel.quantile(q, axis=1)
    gate = d_panel.ge(thresh, axis=0)
    gate = gate.where(d_panel.notna(), False)
    gate.loc[counts < min_names] = False
    return gate


def decile_gate(d_panel, mode="trailing", **kw):
    """Dispatch to the trailing (default) or cross-sectional decile gate."""
    if mode == "cross_sectional":
        return top_decile_gate_cross_sectional(d_panel, **kw)
    return top_decile_gate_trailing(d_panel, **kw)


def _consec_true_series(s):
    """Run-length of consecutive True values ending at each row, for one boolean Series."""
    b = s.astype(bool).astype("int64")
    grp = (b == 0).cumsum()                 # new group id at every False (reset boundary)
    return b.groupby(grp).cumsum()          # cumulative 1s within each all-True run


def consecutive_true(bool_obj):
    """Run-length of consecutive True values ending at each row (0 where False). Works on
    a Series or (column-wise) a DataFrame."""
    if isinstance(bool_obj, pd.Series):
        return _consec_true_series(bool_obj)
    return bool_obj.apply(_consec_true_series)


def held_for(bool_frame, n):
    """True where a condition has been continuously True for at least `n` rows (inclusive
    of the current row) — e.g. 'DIX above the 7th decile for 5 days'."""
    return consecutive_true(bool_frame) >= n


def cross_above(price, level):
    """True on the bar where `price` closes above `level` having been at/below it the prior
    bar — a fresh upward cross ('momentum inflecting up'), not merely being above."""
    above = price > level
    return above & ~above.shift(1, fill_value=False)


def two_closes_below(price, level):
    """True where `price` has closed below `level` on this bar AND the previous one — the
    'chop/void' condition (two consecutive closes back below the line)."""
    below = price < level
    return below & below.shift(1, fill_value=False)


def align_daily_to_hourly(daily_frame, hourly_index):
    """Forward-fill a DAILY panel (date index) onto an HOURLY tz-aware index. Each hourly
    bar takes the most recent COMPLETED daily value strictly before its session, i.e. the
    prior day's close/D -- no lookahead into the same day's not-yet-known daily figure.

    Returns a frame indexed by `hourly_index` with the daily columns.
    """
    if daily_frame is None or daily_frame.empty or len(hourly_index) == 0:
        return pd.DataFrame(index=hourly_index)
    tz = hourly_index.tz
    # Daily values become available for use on the NEXT session; shift the effective
    # timestamp to the day AFTER the close and localize to the hourly index tz so a
    # merge_asof/reindex picks up only already-known data.
    daily = daily_frame.copy()
    idx = pd.to_datetime(daily.index).normalize()
    if tz is not None:
        # place each daily stamp at midnight of the following day, in exchange tz
        idx = (idx + pd.Timedelta(days=1)).tz_localize(tz)
    daily.index = idx
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    return daily.reindex(daily.index.union(hourly_index)).ffill().reindex(hourly_index)
