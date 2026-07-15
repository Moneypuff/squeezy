"""Command-line entry point.

    # real 2-year backtest (needs cdn.finra.org + query*.finance.yahoo.com reachable)
    python -m dixsignal.cli run --years 2

    # offline smoke test on synthetic data (no network)
    python -m dixsignal.cli demo

    # SMA-only baseline (DIX gate off) to isolate the gate's contribution
    python -m dixsignal.cli demo --no-gate
"""

import argparse
import sys

import pandas as pd

from . import backtest as bt
from . import synth
from .data import DEFAULT_CACHE_DIR, build_dataset
from .strategy import Params

# The 28-name dashboard universe (from squeezy/index.html).
DEFAULT_UNIVERSE = [
    "AAPL", "AMZN", "DJT", "FSLR", "GLD", "GOOG", "IBIT", "IWM", "JPM", "KWEB",
    "META", "MSFT", "NVDA", "QQQ", "SLV", "SMH", "SPY", "TLT", "TSLA", "XLE",
    "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XOM",
]


def _params_from_args(a):
    return Params(
        sma_hourly=a.sma_hourly,
        sma_daily=a.sma_daily,
        decile_mode=a.decile_mode,
        decile_window=a.decile_window,
        decile_q=a.decile_q,
        require_gate=not a.no_gate,
    )


def _add_common(sub):
    sub.add_argument("--sma-hourly", type=int, default=100)
    sub.add_argument("--sma-daily", type=int, default=20)
    sub.add_argument("--decile-mode", choices=["trailing", "cross_sectional"], default="trailing")
    sub.add_argument("--decile-window", type=int, default=252)
    sub.add_argument("--decile-q", type=float, default=0.90)
    sub.add_argument("--no-gate", action="store_true", help="disable DIX gate (SMA-only baseline)")
    sub.add_argument("--cost-bps", type=float, default=0.0, help="per-side cost in bps")
    sub.add_argument("--benchmark", default="SPY")
    sub.add_argument("--csv", default=None, help="write per-name metrics to this CSV path")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dixsignal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = ap.add_subparsers(dest="cmd", required=True)

    r = subs.add_parser("run", help="live backtest from FINRA + Yahoo")
    _add_common(r)
    r.add_argument("--tickers", nargs="*", default=DEFAULT_UNIVERSE)
    r.add_argument("--years", type=float, default=2.0, help="lookback (Yahoo 60m caps ~2y)")
    r.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--keep-partial", action="store_true",
                   help="keep the partial 15:30 ET bar in the hourly series")

    d = subs.add_parser("demo", help="offline synthetic smoke test (no network)")
    _add_common(d)
    d.add_argument("--tickers", nargs="*", default=DEFAULT_UNIVERSE)
    d.add_argument("--seed", type=int, default=7)

    a = ap.parse_args(argv)
    params = _params_from_args(a)

    if a.cmd == "demo":
        dataset = synth.make_dataset(a.tickers, seed=a.seed, bench=a.benchmark)
        print("DEMO MODE: synthetic data, no network. Numbers are a plumbing check only.\n",
              file=sys.stderr)
    else:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=int(a.years * 365.25))
        dataset = build_dataset(a.tickers, start, end, workers=a.workers,
                                cache_dir=a.cache_dir, drop_partial=not a.keep_partial)
        if dataset["hourly_close"] is None or dataset["hourly_close"].empty:
            print("No hourly data returned -- is query*.finance.yahoo.com reachable from "
                  "this environment? (Egress policy may block it.)", file=sys.stderr)
            return 2
        if dataset["d"] is None or dataset["d"].empty:
            print("WARNING: no FINRA DIX data (is cdn.finra.org reachable?). "
                  "Running SMA-only.", file=sys.stderr)
            params.require_gate = False

    result = bt.backtest(dataset, params, cost_bps=a.cost_bps, benchmark=a.benchmark)
    print(bt.format_report(result, params))

    if a.csv and not result["per_name"].empty:
        result["per_name"].to_csv(a.csv)
        print(f"\nper-name metrics -> {a.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
