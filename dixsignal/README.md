# dixsignal — DIX + dual-SMA signal engine

Backtests a long-only single-stock signal that combines, per name:

- **100-hour SMA** on Yahoo hourly closes — fast intraday trend
- **20-day SMA** on Yahoo daily closes — medium-term trend
- **high-decile DIX** — the name's `D` (5-day MA of FINRA off-exchange
  ShortVolume ÷ TotalVolume, SqueezeMetrics' construction) in the top decile of
  its own trailing history

**Entry (long):** hourly close is above *both* SMAs **and** the name is in its top DIX
decile. **Exit (flat):** any condition breaks. Next-bar execution — a bar's signal is
acted on the following bar, so no bar uses its own not-yet-known close.

Everything is built from **free, key-less public data**: FINRA's consolidated daily
short-sale files (`cdn.finra.org`, history from 2018-08-01) and Yahoo's chart API
(`query*.finance.yahoo.com`). The FINRA/Yahoo fetch + DIX construction are vendored from
the earlier `ndx_dark_residual.py` project; the hourly fetch, SMAs, decile gate, and
backtest are new here.

## Why "2 years" first

Yahoo caps **hourly** (`60m`) history at ~**730 days**. FINRA daily DIX goes back to 2018,
so the hourly feed is the binding constraint. The plan is to backtest on this 2-year
window first and decide whether a paid intraday feed (for deeper history) is worth it.

## Install

```bash
pip install -r dixsignal/requirements.txt
```

## Run

```bash
# Real 2-year backtest over the 28-name dashboard universe.
# Requires cdn.finra.org and query*.finance.yahoo.com to be reachable.
python -m dixsignal.cli run --years 2

# Isolate the DIX gate's contribution (SMA-only baseline):
python -m dixsignal.cli run --years 2 --no-gate

# Offline plumbing check on synthetic data (no network):
python -m dixsignal.cli demo

# Write per-name metrics to CSV:
python -m dixsignal.cli run --years 2 --csv out.csv
```

Useful flags: `--sma-hourly`, `--sma-daily`, `--decile-q`, `--decile-mode
{trailing,cross_sectional}`, `--cost-bps`, `--keep-partial`, `--tickers ...`.

## ⚠️ Network / egress note

This engine fetches directly from FINRA and Yahoo. In a **Claude Code on the web** session
whose environment egress policy does **not** allowlist `cdn.finra.org` and
`query1/2.finance.yahoo.com`, every fetch returns empty and `run` exits with a clear
message (exit code 2). To get real numbers, either:

1. run it on a machine with open network access, or
2. use a web environment whose egress policy allows those two hosts.

The `demo` command needs no network and always works.

## Layout

| file | role |
|------|------|
| `data.py` | FINRA + Yahoo (daily **and** hourly) fetch/cache; per-name `D` |
| `indicators.py` | SMAs, top-decile gate (trailing / cross-sectional), daily→hourly alignment |
| `strategy.py` | combines ingredients into the per-name long/flat signal |
| `backtest.py` | next-bar backtest, per-name + pooled metrics, text report |
| `synth.py` | synthetic dataset for offline/CI runs |
| `cli.py` | `run` (live) / `demo` (offline) entry point |
| `tests/` | offline smoke tests (`python -m dixsignal.tests.test_smoke`) |

## Caveats

- Yahoo hourly bars are unofficial, exchange-local, and not corporate-action adjusted;
  fine for trend SMAs, not for precise far-back intraday returns. The final ~30-min 15:30
  ET bar is dropped by default (`--keep-partial` to keep it).
- DIX is an **end-of-day daily** series — it gates entries; it is never resampled to hourly.
- Synthetic (`demo`) numbers are a plumbing check only and say nothing about the market.
