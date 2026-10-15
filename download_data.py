"""
Download daily crypto prices from Binance (public API, no account needed)
and save them as prices_cache.csv, which the evolve_bots scripts read.

Usage:
  python download_data.py                 # last 5 years
  python download_data.py --years 3
  python download_data.py --start 2023-01-01 --end 2025-10-14
"""
import argparse, time
import pandas as pd
import requests

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA"]
HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]


def fetch(symbol, start_ms, end_ms):
    rows, cur = [], start_ms
    while cur < end_ms:
        data, err = None, None
        for host in HOSTS:
            try:
                r = requests.get(f"{host}/api/v3/klines", timeout=20, params={
                    "symbol": symbol + "USDT", "interval": "1d",
                    "startTime": cur, "endTime": end_ms, "limit": 1000})
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                err = e
        if data is None:
            raise RuntimeError(f"{symbol}: {err}")
        if not data:
            break
        rows += data
        cur = data[-1][0] + 24 * 3600 * 1000      # continue after the last candle
        if len(data) < 1000:
            break
        time.sleep(0.2)                            # be polite to the API
    idx = pd.to_datetime([x[0] for x in rows], unit="ms").normalize()
    return pd.Series([float(x[4]) for x in rows], index=idx, name=symbol).loc[lambda s: ~s.index.duplicated()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    end = pd.Timestamp(a.end) if a.end else pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = pd.Timestamp(a.start) if a.start else end - pd.Timedelta(days=int(a.years * 365))
    s_ms, e_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    series = []
    for c in COINS:
        s = fetch(c, s_ms, e_ms)
        print(f"{c}: {len(s)} days  ({s.index[0].date()} -> {s.index[-1].date()})")
        series.append(s)
    df = pd.concat(series, axis=1).dropna()        # keep only days where every coin has a price
    df.to_csv("prices_cache.csv")
    print(f"\nSaved prices_cache.csv: {len(df)} days, {df.index[0].date()} -> {df.index[-1].date()}")


if __name__ == "__main__":
    main()
