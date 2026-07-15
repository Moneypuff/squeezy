"""Long-only, next-bar backtest of the DIX + dual-SMA signal, per name and pooled.

Execution model: a bar's signal is acted on at the NEXT bar's open-to-close (we use
close-to-close returns with position = signal.shift(1)), so no bar uses its own
not-yet-observed close. Flat = cash = 0 return. Costs are optional (bps per side).
"""

import numpy as np
import pandas as pd

from .strategy import Params, build_signals


def _years(index):
    if len(index) < 2:
        return np.nan
    span = index[-1] - index[0]
    return span.days / 365.25 if hasattr(span, "days") else np.nan


def _bars_per_year(index):
    yrs = _years(index)
    if not yrs or np.isnan(yrs) or yrs <= 0:
        return np.nan
    return len(index) / yrs


def _max_drawdown(equity):
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def _extract_trades(position, bar_ret):
    """List of per-trade compounded returns from a 0/1 position series."""
    pos = position.fillna(0).astype(int).values
    ret = bar_ret.fillna(0).values
    trades, in_trade, cum = [], False, 1.0
    for i in range(len(pos)):
        if pos[i] == 1:
            cum *= (1.0 + ret[i])
            in_trade = True
        if in_trade and (i == len(pos) - 1 or pos[i] == 0 or (i + 1 < len(pos) and pos[i + 1] == 0)):
            # close the trade when the position is (or is about to go) flat, or at the end
            if pos[i] == 1 and (i == len(pos) - 1 or pos[i + 1] == 0):
                trades.append(cum - 1.0)
                cum, in_trade = 1.0, False
            elif pos[i] == 0:
                in_trade = False
    return trades


def backtest_series(close, long_signal, cost_bps=0.0):
    """Backtest one name. Returns (metrics dict, strat bar-return series, equity series)."""
    close = close.dropna()
    long_signal = long_signal.reindex(close.index).fillna(False)
    bar_ret = close.pct_change().fillna(0.0)
    position = long_signal.shift(1).fillna(False).astype(float)  # next-bar execution
    strat_ret = position * bar_ret
    if cost_bps:
        turns = position.diff().abs().fillna(position.abs())
        strat_ret = strat_ret - turns * (cost_bps / 1e4)
    equity = (1.0 + strat_ret).cumprod()
    trades = _extract_trades(position, bar_ret)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    bpy = _bars_per_year(close.index)
    sd = strat_ret.std()
    metrics = {
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else np.nan,
        "cagr": float(equity.iloc[-1] ** (1.0 / _years(close.index)) - 1.0)
                if len(equity) and _years(close.index) and _years(close.index) > 0 else np.nan,
        "sharpe": float(strat_ret.mean() / sd * np.sqrt(bpy)) if sd and bpy and not np.isnan(bpy) else np.nan,
        "max_drawdown": _max_drawdown(equity),
        "exposure": float(position.mean()),
        "n_trades": len(trades),
        "hit_rate": float(len(wins) / len(trades)) if trades else np.nan,
        "avg_win": float(np.mean(wins)) if wins else np.nan,
        "avg_loss": float(np.mean(losses)) if losses else np.nan,
    }
    return metrics, strat_ret, equity


def backtest(dataset, params: Params = None, cost_bps=0.0, benchmark="SPY"):
    """Full backtest across the universe.

    Returns a dict with:
      per_name   : DataFrame of per-ticker metrics (index = ticker)
      portfolio  : dict of pooled equal-weight metrics
      equity     : DataFrame(portfolio, benchmark) hourly equity curves
      signals    : the raw per-ticker signal frames (for inspection/plots)
    """
    params = params or Params()
    sig = build_signals(dataset, params)
    if not sig:
        return {"per_name": pd.DataFrame(), "portfolio": {}, "equity": pd.DataFrame(), "signals": {}}

    rows, strat_rets = {}, {}
    for tkr, frame in sig.items():
        m, sr, _eq = backtest_series(frame["close"], frame["long"], cost_bps=cost_bps)
        rows[tkr] = m
        strat_rets[tkr] = sr
    per_name = pd.DataFrame(rows).T.sort_values("total_return", ascending=False)

    # equal-weight pooled portfolio: 1/N capital per name, cash when a name is flat
    ret_panel = pd.DataFrame(strat_rets).sort_index()
    port_ret = ret_panel.mean(axis=1).fillna(0.0)
    port_eq = (1.0 + port_ret).cumprod()

    equity = pd.DataFrame({"portfolio": port_eq})
    hourly = dataset["hourly_close"]
    if benchmark and hourly is not None and benchmark in hourly.columns:
        bench_close = hourly[benchmark].reindex(port_eq.index).ffill()
        bench_eq = (1.0 + bench_close.pct_change().fillna(0.0)).cumprod()
        equity[benchmark] = bench_eq

    sd = port_ret.std()
    bpy = _bars_per_year(port_ret.index)
    portfolio = {
        "total_return": float(port_eq.iloc[-1] - 1.0),
        "cagr": float(port_eq.iloc[-1] ** (1.0 / _years(port_ret.index)) - 1.0)
                if _years(port_ret.index) and _years(port_ret.index) > 0 else np.nan,
        "sharpe": float(port_ret.mean() / sd * np.sqrt(bpy)) if sd and bpy and not np.isnan(bpy) else np.nan,
        "max_drawdown": _max_drawdown(port_eq),
        "avg_exposure": float((ret_panel != 0).mean(axis=1).mean()),
        "n_names": int(ret_panel.shape[1]),
        "avg_hit_rate": float(per_name["hit_rate"].mean(skipna=True)),
        "total_trades": int(per_name["n_trades"].sum()),
    }
    if benchmark in equity.columns:
        portfolio["benchmark_total_return"] = float(equity[benchmark].iloc[-1] - 1.0)

    return {"per_name": per_name, "portfolio": portfolio, "equity": equity, "signals": sig}


def to_payload(result, params: Params = None, benchmark="SPY", max_points=1500):
    """JSON-serialisable dict for the HTML panel: equity curves + metrics + per-name
    table. The hourly equity curve is decimated to ~`max_points` for a light payload."""
    import numpy as _np
    if result["per_name"].empty:
        return {"empty": True}
    eq = result["equity"]
    step = max(1, len(eq) // max_points)
    eqd = eq.iloc[::step]
    dates = [t.strftime("%Y-%m-%d %H:%M") for t in eqd.index]
    curves = {"portfolio": [round(float(x), 5) for x in eqd["portfolio"].values]}
    if benchmark in eqd.columns:
        curves[benchmark] = [round(float(x), 5) for x in eqd[benchmark].values]

    def _clean(d):
        return {k: (None if (isinstance(v, float) and _np.isnan(v)) else
                    (round(float(v), 6) if isinstance(v, (int, float, _np.floating)) else v))
                for k, v in d.items()}

    per_name = [{"ticker": t, **_clean(row.to_dict())}
                for t, row in result["per_name"].iterrows()]
    meta = {}
    if params:
        meta = {"sma_hourly": params.sma_hourly, "sma_daily": params.sma_daily,
                "decile_q": params.decile_q, "decile_mode": params.decile_mode,
                "require_gate": params.require_gate}
    return {"empty": False, "dates": dates, "curves": curves,
            "portfolio": _clean(result["portfolio"]), "per_name": per_name, "params": meta}


def format_report(result, params: Params = None):
    """Plain-text summary suitable for stdout."""
    if result["per_name"].empty:
        return "No signals produced (empty dataset -- were the data hosts reachable?)."
    p = result["portfolio"]
    lines = []
    lines.append("=" * 64)
    lines.append("DIX + dual-SMA signal -- backtest summary")
    if params:
        lines.append(f"  {params.sma_hourly}h SMA  x  {params.sma_daily}d SMA  x  "
                     f"top-{int((1-params.decile_q)*100)}% DIX ({params.decile_mode})"
                     + ("" if params.require_gate else "  [GATE OFF: SMA-only baseline]"))
    lines.append("=" * 64)
    lines.append(f"  names traded      : {p['n_names']}")
    lines.append(f"  portfolio return  : {p['total_return']*100:8.2f}%")
    if "benchmark_total_return" in p:
        lines.append(f"  SPY buy & hold    : {p['benchmark_total_return']*100:8.2f}%")
    lines.append(f"  CAGR              : {p['cagr']*100:8.2f}%")
    lines.append(f"  Sharpe (hourly)   : {p['sharpe']:8.2f}")
    lines.append(f"  max drawdown      : {p['max_drawdown']*100:8.2f}%")
    lines.append(f"  avg exposure      : {p['avg_exposure']*100:8.2f}%")
    lines.append(f"  avg hit-rate      : {p['avg_hit_rate']*100:8.2f}%")
    lines.append(f"  total trades      : {p['total_trades']}")
    lines.append("-" * 64)
    lines.append("  top 8 names by total return:")
    cols = ["total_return", "sharpe", "max_drawdown", "n_trades", "hit_rate", "exposure"]
    head = result["per_name"][cols].head(8)
    for tkr, r in head.iterrows():
        lines.append(f"    {tkr:6s}  ret {r['total_return']*100:7.2f}%  "
                     f"sharpe {r['sharpe']:5.2f}  dd {r['max_drawdown']*100:6.1f}%  "
                     f"trades {int(r['n_trades']):3d}  hit {r['hit_rate']*100:4.0f}%  "
                     f"exp {r['exposure']*100:4.0f}%")
    lines.append("=" * 64)
    return "\n".join(lines)
