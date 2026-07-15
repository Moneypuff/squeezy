"""Data acquisition: FINRA off-exchange volumes, Yahoo daily + hourly prices.

The FINRA fetch/cache stack, the Yahoo daily fetcher, and the per-name D
construction (`finra_dpi_to_d`) are vendored verbatim (lightly trimmed) from the
prior `ndx_dark_residual.py` project -- the same free, key-less public sources:

  * FINRA consolidated daily short-sale files (CNMSshvol*): per-symbol
    ShortVolume + off-exchange TotalVolume, history from 2018-08-01.
  * Yahoo chart API: daily bars (this module) AND hourly bars (new here --
    interval=60m, ~730-day lookback cap).

Nothing in here needs an API key. When run behind an egress proxy that blocks
`cdn.finra.org` or `query*.finance.yahoo.com`, every fetch degrades to empty
panels rather than raising -- run it where those hosts are reachable.
"""

import concurrent.futures
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # requests is only needed for live fetching
    requests = None

# --------------------------------------------------------------------------
# Endpoints / constants
# --------------------------------------------------------------------------
FINRA_TMPL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
FINRA_MIN_DATE = pd.Timestamp("2018-08-01")  # earliest consolidated NMS short-volume file

YAHOO_CHART_DAILY = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                     "?period1={p1}&period2={p2}&interval=1d")
YAHOO_CHART_HOURLY = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                      "?period1={p1}&period2={p2}&interval=60m")
_YF_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
EXCHANGE_TZ = "America/New_York"

DEFAULT_CACHE_DIR = str(Path.home() / ".dixsignal_cache")
FINRA_DOC_NAMES = {"offexch": "finra_offexch_volume.csv", "short": "finra_short_volume.csv"}
YAHOO_DAILY_CACHE = "yahoo_daily.pkl"
YAHOO_HOURLY_CACHE = "yahoo_hourly.pkl"


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def make_session(pool_size=8):
    """A requests.Session whose connection pool is sized for concurrent use."""
    if requests is None:
        raise RuntimeError("The 'requests' package is required for live fetching.")
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def parallel_map(fn, items, workers):
    """Map `fn` over `items` concurrently, preserving order. `fn` must swallow its
    own errors -- an exception here would abort the whole batch."""
    items = list(items)
    if workers and workers > 1 and len(items) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            yield from ex.map(fn, items)
    else:
        yield from map(fn, items)


def to_yahoo_symbol(sym):
    """FINRA/holdings ticker -> Yahoo convention (class shares use '-' not '.')."""
    return sym.strip().upper().replace(".", "-")


def _unix(ts):
    return int(pd.Timestamp(ts).timestamp())


# --------------------------------------------------------------------------
# FINRA daily short-sale volume files  (vendored)
# --------------------------------------------------------------------------
def _finra_missing(status, text):
    """True if this response means 'no file for this date' (weekend/holiday)."""
    if status == 404:
        return True
    if status == 403 and ("AccessDenied" in text or "NoSuchKey" in text):
        return True
    return False


def _parse_finra_volumes(text):
    """Parse a FINRA short-volume body -> {symbol: (ShortVolume, TotalVolume)}."""
    out = {}
    for line in text.splitlines()[1:]:  # skip header
        parts = line.split("|")
        if len(parts) < 5:
            continue
        try:
            out[parts[1]] = (float(parts[2]), float(parts[4]))
        except ValueError:  # trailing summary/footer line, or a blank field
            continue
    return out


def fetch_finra_offexchange_volume(date_str, session=None, retries=2, pause=0.3):
    """{symbol: (ShortVolume, TotalVolume)} for one date (YYYYMMDD), plus a `resolved`
    flag. resolved=False means a transient failure (leave the date for a later run);
    resolved=True with an empty dict means a confirmed holiday (no file)."""
    if requests is None:
        raise RuntimeError("The 'requests' package is required for live fetching.")
    get = (session or requests).get
    url = FINRA_TMPL.format(date=date_str)
    for attempt in range(retries):
        try:
            r = get(url, timeout=20)
            if _finra_missing(r.status_code, r.text):
                return {}, True
            if r.status_code != 200:
                time.sleep(pause * (attempt + 1))
                continue
            return _parse_finra_volumes(r.text), True
        except Exception:  # noqa: BLE001
            time.sleep(pause * (attempt + 1))
    return {}, False


def finra_doc_path(cache_dir, kind="offexch", ns=""):
    name = FINRA_DOC_NAMES[kind]
    if ns:
        name = name.replace(".csv", f"_{ns}.csv")
    return Path(cache_dir) / name


def load_finra_document(cache_dir, kind="offexch", ns=""):
    if not cache_dir:
        return pd.DataFrame()
    p = finra_doc_path(cache_dir, kind, ns)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not read cache {p.name} ({e}); rebuilding", file=sys.stderr)
        return pd.DataFrame()


def save_finra_document(df, cache_dir, kind="offexch", ns=""):
    if not cache_dir or df is None or df.empty:
        return
    p = finra_doc_path(cache_dir, kind, ns)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".csv.tmp")
        df.sort_index().to_csv(tmp)
        tmp.replace(p)
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not write cache {p.name} ({e})", file=sys.stderr)


def fetch_finra_dark_volume_panel(dates, symbols, workers=8, cache_dir=None, ns=""):
    """(ShortVolume panel, TotalVolume panel) for `symbols` over `dates`, incrementally
    cached: one HTTP request per genuinely-new trading day; holidays recorded as NaN
    rows so they are never re-requested."""
    wanted = list(dict.fromkeys(symbols))
    dates = [d for d in dates if d >= FINRA_MIN_DATE]
    if not dates:
        return pd.DataFrame(), pd.DataFrame()

    doc_t = load_finra_document(cache_dir, "offexch", ns)
    doc_s = load_finra_document(cache_dir, "short", ns)
    have = (set(doc_t.index) if not doc_t.empty else set()) & \
           (set(doc_s.index) if not doc_s.empty else set())
    missing = [d for d in dates if d not in have]
    print(f"FINRA cache: {len(dates) - len(missing)}/{len(dates)} day(s) cached; "
          f"fetching {len(missing)} new day(s)...", file=sys.stderr)

    if missing:
        session = make_session(workers) if requests else None
        total = len(missing)
        counter = {"n": 0}
        lock = threading.Lock()

        def _one(d):
            vols, resolved = fetch_finra_offexchange_volume(d.strftime("%Y%m%d"), session=session)
            with lock:
                counter["n"] += 1
                if counter["n"] % 50 == 0 or counter["n"] == total:
                    print(f"[{counter['n']:>4}/{total}] FINRA fetched", file=sys.stderr)
            return d, (vols if resolved else None)

        rows_s, rows_t, recorded = {}, {}, []
        for d, vols in parallel_map(_one, missing, workers):
            if vols is None:            # transient failure -> leave for a future run
                continue
            recorded.append(d)          # holidays (empty vols) become NaN rows below
            if vols:
                rows_s[d] = {s: vols[s][0] for s in wanted if s in vols}
                rows_t[d] = {s: vols[s][1] for s in wanted if s in vols}
        if recorded:
            idx = pd.DatetimeIndex(sorted(recorded))
            for rows, doc, kind in ((rows_s, doc_s, "short"), (rows_t, doc_t, "offexch")):
                new_df = pd.DataFrame.from_dict(rows, orient="index").reindex(idx)
                doc2 = pd.concat([doc, new_df]) if not doc.empty else new_df
                doc2 = doc2[~doc2.index.duplicated(keep="last")].sort_index()
                save_finra_document(doc2, cache_dir, kind, ns)
                if kind == "short":
                    doc_s = doc2
                else:
                    doc_t = doc2

    def _slice(doc):
        if doc.empty:
            return pd.DataFrame()
        cols = [s for s in wanted if s in doc.columns]
        return doc.reindex(pd.DatetimeIndex(dates))[cols]
    return _slice(doc_s), _slice(doc_t)


def finra_dpi_to_d(short_panel, total_panel, smooth=5):
    """Per-name D = `smooth`-day MA of the off-exchange DPI (short / total), clipped
    0..1. SqueezeMetrics' D construction, computed from FINRA directly. Returns
    (raw 1-day dpi, smoothed D)."""
    if short_panel.empty or total_panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    dpi = (short_panel / total_panel.replace(0, np.nan)).clip(lower=0, upper=1)
    return dpi, dpi.rolling(smooth, min_periods=1).mean()


# --------------------------------------------------------------------------
# Yahoo daily prices  (vendored)
# --------------------------------------------------------------------------
def fetch_yahoo_daily_one(sym, start, end, session=None, retries=3, pause=0.5):
    """(close, adjclose, volume) DataFrame indexed by date for one symbol, or empty."""
    if requests is None:
        raise RuntimeError("The 'requests' package is required for live fetching.")
    get = (session or requests).get
    url = YAHOO_CHART_DAILY.format(sym=to_yahoo_symbol(sym), p1=_unix(start), p2=_unix(end))
    last = None
    for a in range(retries):
        try:
            r = get(url, timeout=30, headers={"User-Agent": _YF_UA})
            if r.status_code == 429:
                last = "429"; time.sleep(pause * (a + 2) + 0.5); continue
            if r.status_code not in (200, 404):
                last = f"HTTP {r.status_code}"; time.sleep(pause * (a + 1)); continue
            j = r.json()
            res = j.get("chart", {}).get("result")
            if not res or j.get("chart", {}).get("error"):
                return pd.DataFrame(columns=["close", "adjclose", "volume"])
            res = res[0]
            ts = res.get("timestamp")
            if not ts:
                return pd.DataFrame(columns=["close", "adjclose", "volume"])
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
            idx = pd.to_datetime(ts, unit="s").normalize()
            df = pd.DataFrame({"close": q.get("close"),
                               "adjclose": adj if adj is not None else q.get("close"),
                               "volume": q.get("volume")}, index=idx)
            return df[~df.index.duplicated(keep="last")].dropna(how="all")
        except Exception as e:  # noqa: BLE001
            last = str(e); time.sleep(pause * (a + 1))
    print(f"  ! yahoo daily {sym}: failed ({last})", file=sys.stderr)
    return pd.DataFrame(columns=["close", "adjclose", "volume"])


def load_yahoo_daily_panels(symbols, start, end, workers=8, cache_dir=None, label="symbol"):
    """{'close','adjclose','volume'} wide daily panels over [start, end]. Same-day pkl
    cache; only names absent/stale are refetched."""
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols))
    cache = (Path(cache_dir) / YAHOO_DAILY_CACHE) if cache_dir else None
    cached = {}
    if cache is not None and cache.exists():
        try:
            cached = pd.read_pickle(cache)
        except Exception:  # noqa: BLE001
            cached = {}
    fields = ("close", "adjclose", "volume")
    base = {f: cached.get(f, pd.DataFrame()) for f in fields}
    end_n = pd.Timestamp(end).normalize()

    def _fetch_start(sym):
        c = base["close"]
        if sym not in c.columns:
            return pd.Timestamp(start)
        s = c[sym].dropna()
        if s.empty:
            return pd.Timestamp(start)
        if s.index.max() >= end_n:
            return None
        return s.index.max()

    todo = [(s, st) for s in symbols for st in [_fetch_start(s)] if st is not None]
    print(f"Yahoo daily: {len(symbols) - len(todo)}/{len(symbols)} {label}s cached; "
          f"fetching {len(todo)}...", file=sys.stderr)

    fetched = {}
    if todo:
        session = make_session(workers) if requests else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for sym, df in ex.map(lambda it: (it[0], fetch_yahoo_daily_one(it[0], it[1], end,
                                                                           session=session)), todo):
                if len(df):
                    fetched[sym] = df

    out = {}
    for f in fields:
        new = pd.DataFrame({s: d[f] for s, d in fetched.items() if f in d})
        out[f] = (new.combine_first(base[f]) if not base[f].empty else new).sort_index()
    if cache is not None and fetched:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(out, cache)
        except Exception:  # noqa: BLE001
            pass
    win = [d for d in out["close"].index if pd.Timestamp(start) <= d <= end_n]
    return {f: out[f].reindex(index=win, columns=symbols) for f in fields}


# --------------------------------------------------------------------------
# Yahoo hourly prices  (NEW -- interval=60m, ~730-day lookback cap)
# --------------------------------------------------------------------------
def fetch_yahoo_hourly_one(sym, start, end, session=None, retries=3, pause=0.5,
                           drop_partial=True):
    """Hourly (60m) close+volume for one symbol as a tz-aware (exchange time) DataFrame,
    or empty. Yahoo caps 60m history at ~730 days; `start` is clamped by the caller.

    `drop_partial` removes the final short bar of each session (Yahoo emits a ~30-min
    15:30 ET bar at the close) so a fixed-length hourly SMA counts uniform bars."""
    if requests is None:
        raise RuntimeError("The 'requests' package is required for live fetching.")
    get = (session or requests).get
    url = YAHOO_CHART_HOURLY.format(sym=to_yahoo_symbol(sym), p1=_unix(start), p2=_unix(end))
    last = None
    for a in range(retries):
        try:
            r = get(url, timeout=40, headers={"User-Agent": _YF_UA})
            if r.status_code == 429:
                last = "429"; time.sleep(pause * (a + 2) + 0.5); continue
            if r.status_code not in (200, 404):
                last = f"HTTP {r.status_code}"; time.sleep(pause * (a + 1)); continue
            j = r.json()
            res = j.get("chart", {}).get("result")
            if not res or j.get("chart", {}).get("error"):
                return pd.DataFrame(columns=["close", "volume"])
            res = res[0]
            ts = res.get("timestamp")
            if not ts:
                return pd.DataFrame(columns=["close", "volume"])
            q = res["indicators"]["quote"][0]
            idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(EXCHANGE_TZ)
            df = pd.DataFrame({"close": q.get("close"), "volume": q.get("volume")}, index=idx)
            df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
            if drop_partial and not df.empty:
                df = _drop_partial_bars(df)
            return df
        except Exception as e:  # noqa: BLE001
            last = str(e); time.sleep(pause * (a + 1))
    print(f"  ! yahoo hourly {sym}: failed ({last})", file=sys.stderr)
    return pd.DataFrame(columns=["close", "volume"])


def _drop_partial_bars(df):
    """Drop the last bar of each trading day (the partial 15:30 ET half-hour bar) so
    every retained bar spans a full hour."""
    day = df.index.tz_convert(EXCHANGE_TZ).normalize()
    is_last = day != np.roll(day.values, -1)
    is_last[-1] = True  # final bar overall is always a session's last
    return df[~is_last]


def hourly_lookback_start(start, end, max_days=730):
    """Clamp `start` so the 60m request stays within Yahoo's ~730-day window."""
    end = pd.Timestamp(end)
    floor = end - pd.Timedelta(days=max_days)
    start = pd.Timestamp(start)
    return max(start, floor)


def load_yahoo_hourly_panel(symbols, start, end, workers=8, cache_dir=None,
                            drop_partial=True, label="symbol"):
    """Wide hourly CLOSE panel (tz-aware ET index x symbols) over the clamped 730-day
    window. Simpler cache than the daily loader: a same-run pkl of the full pull."""
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols))
    start = hourly_lookback_start(start, end)
    cache = (Path(cache_dir) / YAHOO_HOURLY_CACHE) if cache_dir else None
    if cache is not None and cache.exists():
        try:
            cached = pd.read_pickle(cache)
            if (set(symbols) <= set(cached.columns)
                    and cached.index.max() is not pd.NaT
                    and cached.index.max().normalize() >= pd.Timestamp(end).normalize()
                    .tz_localize(EXCHANGE_TZ)):
                print(f"Yahoo hourly: reusing cache ({len(cached)} bars)", file=sys.stderr)
                return cached.reindex(columns=symbols)
        except Exception:  # noqa: BLE001
            pass

    print(f"Yahoo hourly: fetching {len(symbols)} {label}(s) over "
          f"[{start.date()}..{pd.Timestamp(end).date()}] (60m)...", file=sys.stderr)
    session = make_session(workers) if requests else None
    cols = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, df in ex.map(lambda s: (s, fetch_yahoo_hourly_one(
                s, start, end, session=session, drop_partial=drop_partial)), symbols):
            if len(df):
                cols[sym] = df["close"]
    panel = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
    if cache is not None and not panel.empty:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(panel, cache)
        except Exception:  # noqa: BLE001
            pass
    return panel.reindex(columns=symbols) if not panel.empty else panel


# --------------------------------------------------------------------------
# One-call bundle
# --------------------------------------------------------------------------
def build_dataset(symbols, start, end, workers=8, cache_dir=DEFAULT_CACHE_DIR,
                  drop_partial=True, d_smooth=5):
    """Fetch everything the signal needs and return a dict of panels:

        hourly_close : tz-aware ET index x symbols   (Yahoo 60m, clamped 730d)
        daily_close  : date index x symbols          (Yahoo adjusted daily close)
        d            : date index x symbols          (per-name D, 5-day dark ratio)
        dpi          : date index x symbols          (raw 1-day dark ratio)

    Any source that is unreachable (blocked host, offline) yields an empty panel;
    callers should check `daily_close`/`d` emptiness before backtesting.
    """
    daily = load_yahoo_daily_panels(symbols, start, end, workers=workers, cache_dir=cache_dir)
    hourly_close = load_yahoo_hourly_panel(symbols, start, end, workers=workers,
                                           cache_dir=cache_dir, drop_partial=drop_partial)
    dates = [pd.Timestamp(d) for d in daily["close"].index]
    short, total = fetch_finra_dark_volume_panel(dates, symbols, workers=workers,
                                                 cache_dir=cache_dir)
    dpi, d = finra_dpi_to_d(short, total, smooth=d_smooth)
    return {
        "hourly_close": hourly_close,
        "daily_close": daily["adjclose"],
        "daily_close_raw": daily["close"],
        "volume": daily["volume"],
        "d": d,
        "dpi": dpi,
    }
