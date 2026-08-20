"""Data loading: public Binance REST, no API key, small on-disk cache.

A single-venue illustration of the ingestion layer. The lab this comes from
reads a multi-venue store (Binance, Hyperliquid, Kraken, Lighter, Paradex)
with the same shape: page the REST endpoint, normalise into one schema, cache
incrementally so only missing ranges are fetched.

One implementation detail worth keeping: never cache the bar that is still
forming. An in-progress daily candle looks exactly like a complete one and
quietly corrupts every close-based calculation until the day rolls over, a bug
that cost real debugging time in the original codebase.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.binance.com/api/v3/klines"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
MS_PER_DAY = 86_400_000


def _interval_ms(interval):
    unit = interval[-1]
    n = int(interval[:-1])
    return n * {"m": 60_000, "h": 3_600_000, "d": MS_PER_DAY}[unit]


def fetch_klines(symbol, interval="1d", start="2021-01-01", end=None, pause=0.15):
    """Fetch OHLCV from Binance public REST, paging until `end`.

    Returns a DataFrame indexed by open time (UTC) with the in-progress bar
    dropped -- see the module docstring.
    """
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end \
        else int(time.time() * 1000)
    step = _interval_ms(interval)
    rows = []
    while start_ms < end_ms:
        r = requests.get(BASE, params=dict(symbol=symbol, interval=interval,
                                           startTime=start_ms, limit=1000),
                         timeout=20)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        rows += batch
        start_ms = batch[-1][0] + step
        if len(batch) < 1000:
            break
        time.sleep(pause)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df = df.astype({c: float for c in
                    ["open", "high", "low", "close", "volume", "quote_volume",
                     "taker_buy_quote"]})
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["open", "high", "low", "close", "volume", "quote_volume",
             "taker_buy_quote", "close_time"]]
    # Drop the still-forming bar: its close is not a close.
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] <= now_ms]
    return df.drop(columns="close_time")


def _cache_path(symbol, interval):
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{symbol}_{interval}.parquet"


def load_symbol(symbol, interval="1d", start="2021-01-01", use_cache=True):
    """Single symbol OHLCV, cached to parquet under .cache/."""
    path = _cache_path(symbol, interval)
    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        if len(cached) and cached.index[0] <= pd.Timestamp(start):
            return cached.loc[start:]
    df = fetch_klines(symbol, interval=interval, start=start)
    if use_cache and len(df):
        df.to_parquet(path)
    return df


def load_panel(symbols, interval="1d", start="2021-01-01", quote="USDT",
               use_cache=True, verbose=True):
    """Load a panel of assets -> (px_mat, dv_mat, missing).

    px_mat : date x asset close prices
    dv_mat : date x asset dollar volume (quote volume)
    missing: symbols with no data on this venue -- always report these. A
             silently dropped asset is a silently biased universe.
    """
    px, dv, missing = {}, {}, []
    for i, base in enumerate(symbols):
        sym = f"{base}{quote}"
        try:
            df = load_symbol(sym, interval=interval, start=start, use_cache=use_cache)
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            missing.append(base)
            continue
        px[base], dv[base] = df["close"], df["quote_volume"]
        if verbose and (i + 1) % 20 == 0:
            print(f"  loaded {i+1}/{len(symbols)}")
    px_mat = pd.DataFrame(px).sort_index()
    dv_mat = pd.DataFrame(dv).reindex_like(px_mat)
    return px_mat, dv_mat, missing


def top_symbols_by_volume(n=60, quote="USDT", exclude=("USDC", "FDUSD", "TUSD",
                                                       "BUSD", "EUR", "DAI")):
    """Current top-n Binance spot bases by 24h quote volume.

    Convenience for the demo only. Selecting today's most liquid names and then
    backtesting them over history is itself a survivorship bias -- the demo
    notebook says so explicitly, and the universe module exists to do this
    properly with life windows.
    """
    r = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = df[df["symbol"].str.endswith(quote)].copy()
    df["quoteVolume"] = df["quoteVolume"].astype(float)
    df["base"] = df["symbol"].str[: -len(quote)]
    df = df[~df["base"].isin(exclude)]
    df = df[~df["base"].str.contains(r"UP$|DOWN$|BULL$|BEAR$", regex=True)]
    return df.nlargest(n, "quoteVolume")["base"].tolist()
