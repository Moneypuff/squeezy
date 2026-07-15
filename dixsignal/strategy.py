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
