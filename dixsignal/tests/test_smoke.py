"""Offline smoke tests -- exercise the full pipeline on synthetic data (no network).

Run:  python -m dixsignal.tests.test_smoke      (plain asserts, no pytest needed)
  or: pytest dixsignal/tests/
"""

import numpy as np

from dixsignal import backtest as bt
from dixsignal import synth
from dixsignal.strategy import Params

UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "IWM"]


def _dataset():
    return synth.make_dataset(UNIVERSE, start="2023-07-01", end="2025-07-01", seed=7)


def test_dataset_shapes():
    ds = _dataset()
    assert not ds["hourly_close"].empty
    assert not ds["daily_close"].empty
    assert not ds["d"].empty
    # hourly index is tz-aware and intraday; daily is not
    assert ds["hourly_close"].index.tz is not None
    assert len(ds["hourly_close"]) > len(ds["daily_close"])


def test_no_lookahead_gate_alignment():
    """Signal on day D must use D-1's daily D, never the same-day value."""
    from dixsignal import indicators as ind
    ds = _dataset()
    daily = ds["d"][["AAPL"]]
    hourly_idx = ds["hourly_close"]["AAPL"].dropna().index
    aligned = ind.align_daily_to_hourly(daily, hourly_idx)["AAPL"].dropna()
    # The earliest daily value (date 0) only becomes usable on the NEXT session, so the
    # first non-NaN hourly bar must equal daily.iloc[0] -- never a same-day value.
    assert np.isclose(aligned.iloc[0], float(daily["AAPL"].iloc[0]))
    # And every hourly bar's value must come from a STRICTLY earlier daily date: the set
    # of values seen intraday excludes the final daily value (never available in-sample).
    seen = set(np.round(aligned.values, 10))
    assert np.round(float(daily["AAPL"].iloc[-1]), 10) not in seen


def test_backtest_runs_and_gate_changes_result():
    ds = _dataset()
    on = bt.backtest(ds, Params(require_gate=True))
    off = bt.backtest(ds, Params(require_gate=False))
    assert on["portfolio"]["n_names"] == len(UNIVERSE)
    # the gate must change exposure (fewer bars invested when gated)
    assert on["portfolio"]["avg_exposure"] < off["portfolio"]["avg_exposure"]
    # metrics are finite
    assert np.isfinite(on["portfolio"]["total_return"])
    assert np.isfinite(on["portfolio"]["sharpe"])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} smoke tests passed.")


if __name__ == "__main__":
    _run_all()
