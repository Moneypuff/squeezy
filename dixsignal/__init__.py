"""dixsignal -- DIX + dual-SMA signal engine.

Combines three ingredients into a long-only equity signal, per single stock:

  * 100-hour SMA on Yahoo hourly closes   (fast intraday trend)
  * 20-day   SMA on Yahoo daily  closes   (medium-term trend)
  * high-decile per-name DIX / dark ratio (D = 5-day MA of FINRA off-exchange
    ShortVolume / TotalVolume -- SqueezeMetrics' construction, computed free
    from FINRA's consolidated daily short-sale files)

Entry when price is above BOTH SMAs and the name's D sits in the top decile of
its own trailing history; exit when any of those breaks. See backtest.py.

The FINRA/Yahoo fetch + DIX construction are vendored from the prior
`ndx_dark_residual.py` project (attribution in data.py).
"""

__all__ = ["data", "indicators", "strategy", "backtest", "synth"]
