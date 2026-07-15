"""Combine the ingredients into a per-name long/flat signal on the hourly timeline.

Entry (long) when ALL hold on an hourly bar:
  * hourly close > 100-hour SMA          (fast intraday trend up)
  * hourly close > 20-day SMA            (medium trend up; prior day's SMA)
  * name's D in the top decile of its trailing history  (high dark accumulation)

Exit (flat) when any condition breaks. The daily inputs (20-day SMA, D gate) are
forward-filled onto the hourly index using only already-completed daily values, so
there is no intraday lookahead.
"""

from dataclasses import dataclass, field

import pandas as pd

from . import indicators as ind


@dataclass
class ConvictionParams:
    """Daily-only 'conviction' setup: go long on a fresh close back above the 20-day SMA
    when DIX has held above the 7th decile for >= `dix_days`; void on two consecutive
    closes back below the SMA."""
    sma_daily: int = 20
    dix_decile: float = 0.70          # "above the 7th decile" = >= 70th trailing percentile
    dix_days: int = 5                 # DIX must clear the decile this many consecutive days
    dix_window: int = 252             # trailing window for the decile threshold
    dix_min_periods: int = 60
    void_closes: int = 2              # consecutive closes below the SMA that void the trade
    require_cross: bool = True        # True: enter only on a fresh cross up; False: any close above
    dix_hold: bool = False            # True: also exit if DIX drops back below the decile
    strategy: str = "conviction"


@dataclass
class Params:
    sma_hourly: int = 100
    sma_daily: int = 20
    decile_mode: str = "trailing"        # "trailing" | "cross_sectional"
    decile_window: int = 252             # trailing-mode lookback (trading days)
    decile_q: float = 0.90               # top decile
    decile_min_periods: int = 60
    require_gate: bool = True             # if False, ignore DIX (SMA-only baseline)
    extra: dict = field(default_factory=dict)


def build_signals(dataset, params: Params):
    """Return {ticker: DataFrame(close, sma_h, sma_d, gate, long)} on the hourly index.

    Tickers absent from the hourly panel (no intraday data) are skipped. The DIX gate is
    computed once across all names, then aligned per ticker.
    """
    hourly = dataset["hourly_close"]
    daily = dataset["daily_close"]
    d_panel = dataset.get("d")
    if hourly is None or hourly.empty:
        return {}

    sma_d_daily = ind.sma(daily, params.sma_daily) if daily is not None and not daily.empty \
        else pd.DataFrame()

    if params.require_gate and d_panel is not None and not d_panel.empty:
        gate_daily = ind.decile_gate(
            d_panel, mode=params.decile_mode,
            **({"window": params.decile_window, "q": params.decile_q,
                "min_periods": params.decile_min_periods}
               if params.decile_mode == "trailing" else {"q": params.decile_q}))
    else:
        gate_daily = pd.DataFrame()

    out = {}
    for tkr in hourly.columns:
        close = hourly[tkr].dropna()
        if close.empty:
            continue
        sma_h = ind.sma(close, params.sma_hourly)

        # daily 20-day SMA aligned to this name's hourly bars (prior-day value)
        if not sma_d_daily.empty and tkr in sma_d_daily.columns:
            sma_d = ind.align_daily_to_hourly(sma_d_daily[[tkr]], close.index)[tkr]
        else:
            sma_d = pd.Series(index=close.index, dtype=float)

        # DIX decile gate aligned to hourly (prior-day value)
        if params.require_gate:
            if not gate_daily.empty and tkr in gate_daily.columns:
                gate = ind.align_daily_to_hourly(
                    gate_daily[[tkr]].astype(float), close.index)[tkr].fillna(0) > 0.5
            else:
                gate = pd.Series(False, index=close.index)
        else:
            gate = pd.Series(True, index=close.index)

        long = (close > sma_h) & (close > sma_d) & gate
        long = long.where(sma_h.notna() & sma_d.notna(), False)  # no signal before SMAs warm up

        out[tkr] = pd.DataFrame({
            "close": close, "sma_h": sma_h, "sma_d": sma_d,
            "gate": gate.astype(bool), "long": long.astype(bool),
        })
    return out


def build_conviction_signals(dataset, params: ConvictionParams):
    """Daily conviction setup. Returns {ticker: DataFrame(close, sma, dix_ok, entry, long)}.

    Per name, on DAILY bars:
      * entry when a fresh close crosses above the 20-day SMA (or, if require_cross is
        False, any close above) AND DIX has been >= its `dix_decile` trailing threshold
        for at least `dix_days` consecutive days;
      * once long, stay long until `void_closes` consecutive closes fall below the SMA
        (the chop/void), optionally also exiting if DIX drops back below the decile.

    `long[t]` means 'in the trade as decided at the close of day t'; the backtest enters
    the next bar (position = long.shift(1)), so no bar uses its own unknown close.
    """
    daily = dataset["daily_close"]
    d_panel = dataset.get("d")
    if daily is None or daily.empty:
        return {}

    sma = ind.sma(daily, params.sma_daily)
    # DIX >= 7th-decile threshold, held for >= dix_days consecutive sessions
    if d_panel is not None and not d_panel.empty:
        dix_gate = ind.top_decile_gate_trailing(
            d_panel, window=params.dix_window, q=params.dix_decile,
            min_periods=params.dix_min_periods)
        dix_ok_panel = ind.held_for(dix_gate, params.dix_days)
    else:
        dix_ok_panel = pd.DataFrame()

    out = {}
    for tkr in daily.columns:
        close = daily[tkr].dropna()
        if close.empty or tkr not in sma.columns:
            continue
        s = sma[tkr].reindex(close.index)
        above = close > s
        entry_trigger = ind.cross_above(close, s) if params.require_cross else above
        entry_trigger = entry_trigger & s.notna()
        below_twice = ind.two_closes_below(close, s) if params.void_closes == 2 \
            else (ind.consecutive_true(close < s) >= params.void_closes)

        if tkr in dix_ok_panel.columns:
            dix_ok = dix_ok_panel[tkr].reindex(close.index).fillna(False).astype(bool)
            dix_live = ind.top_decile_gate_trailing(
                d_panel[[tkr]], window=params.dix_window, q=params.dix_decile,
                min_periods=params.dix_min_periods)[tkr].reindex(close.index).fillna(False)
        else:
            dix_ok = pd.Series(False, index=close.index)
            dix_live = pd.Series(False, index=close.index)

        # state machine over daily bars
        et = entry_trigger.values
        dk = dix_ok.values
        dv = below_twice.values
        dl = dix_live.values
        long = [False] * len(close)
        in_trade = False
        for i in range(len(close)):
            if in_trade:
                exit_now = dv[i] or (params.dix_hold and not dl[i])
                if exit_now:
                    in_trade = False
                    long[i] = False
                else:
                    long[i] = True
            if not in_trade and et[i] and dk[i]:
                in_trade = True
                long[i] = True
        out[tkr] = pd.DataFrame({
            "close": close, "sma_d": s, "dix_ok": dix_ok,
            "entry": pd.Series(entry_trigger.values & dk, index=close.index),
            "long": pd.Series(long, index=close.index),
        })
    return out
