"""Synthetic dataset generator -- lets the whole pipeline be exercised end-to-end with
NO network (FINRA/Yahoo unreachable, offline dev, CI). It fabricates internally
consistent hourly closes, daily closes, and per-name D such that D carries a mild,
checkable edge (high-D + uptrend names drift up a little more), so a run can confirm
that (a) the plumbing works and (b) turning the DIX gate on visibly changes the result.

None of these numbers say anything about the real market -- that is what the live
FINRA/Yahoo data determines. This is a test fixture, not a signal.
"""

import numpy as np
import pandas as pd

from .data import EXCHANGE_TZ

BARS_PER_DAY = 6  # hours retained per session after dropping the partial close bar


def _hourly_index(dates):
    """tz-aware ET hourly index: BARS_PER_DAY bars (10:00..15:00) per business day."""
    stamps = []
    for d in dates:
        for h in range(10, 10 + BARS_PER_DAY):
            stamps.append(pd.Timestamp(d.year, d.month, d.day, h, 0))
    idx = pd.DatetimeIndex(stamps).tz_localize(EXCHANGE_TZ)
    return idx


def make_dataset(tickers, start="2023-07-01", end=None, seed=7, bench="SPY"):
    """Return a dataset dict shaped exactly like data.build_dataset()."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    hidx = _hourly_index(dates)
    n_bars = len(hidx)
    day_of_bar = np.repeat(np.arange(n_days), BARS_PER_DAY)

    syms = list(dict.fromkeys([*tickers, bench]))

    def ar1(mu, phi, sig, n, x0=None):
        x = np.empty(n); x[0] = mu if x0 is None else x0
        for t in range(1, n):
            x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0, sig)
        return x

    # common market hourly drift (shared factor)
    market_h = rng.normal(0.0002, 0.006, n_bars)

    hourly_cols, daily_close_cols, d_cols = {}, {}, {}
    for sym in syms:
        # daily D as AR(1) in a plausible band; benchmark ETF gets a flat-ish D
        mu_d = 0.45 if sym != bench else 0.42
        d_daily = np.clip(ar1(mu_d, 0.94, 0.02, n_days), 0.28, 0.66)
        d_z = (d_daily - d_daily.mean()) / (d_daily.std() + 1e-9)

        beta = 1.0 if sym == bench else rng.uniform(0.6, 1.3)
        # idiosyncratic hourly noise + a name trend + a small edge: yesterday's high D
        # nudges today's drift up (the relationship the gate is meant to harvest)
        idio = rng.normal(0, 0.008, n_bars)
        edge_daily = np.zeros(n_days) if sym == bench else 0.0009 * d_z  # per bar, lagged 1 day
        edge_h = np.concatenate([[0.0] * BARS_PER_DAY,
                                 np.repeat(edge_daily[:-1], BARS_PER_DAY)])[:n_bars]
        ret_h = beta * market_h + idio + edge_h
        price = 100.0 * np.cumprod(1.0 + ret_h)
        s = pd.Series(price, index=hidx)
        hourly_cols[sym] = s
        # daily close = last hourly bar of each day
        last_of_day = pd.Series(price, index=day_of_bar).groupby(level=0).last().values
        daily_close_cols[sym] = pd.Series(last_of_day, index=dates)
        d_cols[sym] = pd.Series(d_daily, index=dates)

    hourly_close = pd.DataFrame(hourly_cols)
    daily_close = pd.DataFrame(daily_close_cols)
    d_panel = pd.DataFrame(d_cols)
    # keep only requested tickers as tradable columns; bench stays for benchmarking
    return {
        "hourly_close": hourly_close,
        "daily_close": daily_close,
        "daily_close_raw": daily_close,
        "volume": pd.DataFrame(1e6, index=daily_close.index, columns=daily_close.columns),
        "d": d_panel,
        "dpi": d_panel,
    }
