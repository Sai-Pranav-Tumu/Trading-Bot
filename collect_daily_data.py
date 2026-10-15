"""
Point-in-time data collector for Phase 2 research.

Backfill allowed:
  - Binance funding rates
  - Fear & Greed history

Forward append-only:
  - news/headlines

Safety rule for later simulations:
  A bot at decision time T may only use rows where available_at_utc <= T - 1 hour.
  The collector stores available_at_utc; feature builders must enforce the 1-hour lag.

Each row records the code version and the time the collector saw the data. Raw API
responses are saved under data/raw/. CSV writes are idempotent by source keys.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests


COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA"]
FUNDING_SYMBOLS = [f"{coin}USDT" for coin in COINS]
RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
}
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
RUN_LOG = os.path.join(DATA_DIR, "collector_runs.jsonl")
SAFETY_LAG_HOURS = 1
CODE_VERSION = "collector-v2"


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ms_to_iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).replace(microsecond=0).isoformat()


def sec_to_iso(sec):
    return datetime.fromtimestamp(int(sec), tz=timezone.utc).replace(microsecond=0).isoformat()


def fear_greed_available_iso(sec):
    value_day = datetime.fromtimestamp(int(sec), tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return (value_day + timedelta(days=1)).isoformat()


def date_to_ms(value):
    if not value:
        return None
    dt = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_rss_time(value):
    if not value:
        return ""
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    except Exception:
        return value


def parse_gdelt_time(value):
    if not value:
        return ""
    try:
        return datetime.strptime(value[:14], "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc).isoformat()
    except Exception:
        return value


def git_version():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        return out or CODE_VERSION
    except Exception:
        return CODE_VERSION


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)


def raw_path(source, collected_at, suffix):
    stamp = collected_at.replace(":", "").replace("-", "")
    safe_suffix = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in suffix)
    return os.path.join(RAW_DIR, f"{stamp}_{source}_{safe_suffix}.json")


def save_raw(source, suffix, payload, collected_at):
    path = raw_path(source, collected_at, suffix)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def existing_keys(path, key_fields):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {tuple(row.get(k, "") for k in key_fields) for row in csv.DictReader(f)}


def append_rows_unique(path, fieldnames, rows, key_fields):
    keys = existing_keys(path, key_fields)
    fresh = []
    for row in rows:
        key = tuple(row.get(k, "") for k in key_fields)
        if key not in keys:
            fresh.append(row)
            keys.add(key)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(fresh)
    return len(fresh), len(rows) - len(fresh)


def request_with_retry(url, params=None, response_type="json", attempts=4, min_delay=1.0):
    last_exc = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=30,
                                    headers={"User-Agent": "AiBotResearchCollector/1.0"})
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else max(5.0, min_delay)
                time.sleep(delay * (attempt + 1))
                continue
            response.raise_for_status()
            if response_type == "json":
                return response.json(), response.url
            return response.text, response.url
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(min_delay * (2 ** attempt))
    raise last_exc


def log_run(started_at, code_version, counts, failures):
    row = {
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "code_version": code_version,
        "counts": counts,
        "failures": failures,
    }
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def collect_fear_greed(limit, code_version):
    collected_at = now_utc()
    payload, url = request_with_retry("https://api.alternative.me/fng/", {"limit": limit})
    raw = save_raw("alternative_me", "fear_greed", payload, collected_at)
    rows = []
    for item in payload.get("data", []):
        value_time = sec_to_iso(item["timestamp"])
        rows.append({
            "source": "alternative.me",
            "value_date_utc": value_time,
            "available_at_utc": fear_greed_available_iso(item["timestamp"]),
            "collected_at_utc": collected_at,
            "value": item.get("value"),
            "classification": item.get("value_classification"),
            "api_url": url,
            "raw_path": raw,
            "code_version": code_version,
        })
    return append_rows_unique(os.path.join(DATA_DIR, "fear_greed.csv"), [
        "source", "value_date_utc", "available_at_utc", "collected_at_utc",
        "value", "classification", "api_url", "raw_path", "code_version",
    ], rows, ["source", "value_date_utc"])


def collect_funding_rates(symbols, start_ms, end_ms, limit, code_version):
    appended = skipped = 0
    for symbol in symbols:
        cursor = start_ms
        while True:
            collected_at = now_utc()
            params = {"symbol": symbol, "limit": limit}
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms
            payload, url = request_with_retry("https://fapi.binance.com/fapi/v1/fundingRate", params)
            raw = save_raw("binance", f"funding_{symbol}", payload, collected_at)
            rows = []
            for item in payload:
                funding_ms = int(item["fundingTime"])
                funding_time = ms_to_iso(funding_ms)
                rows.append({
                    "exchange": "binance",
                    "symbol": item.get("symbol", symbol),
                    "funding_time_utc": funding_time,
                    "available_at_utc": funding_time,
                    "collected_at_utc": collected_at,
                    "funding_rate": item.get("fundingRate"),
                    "mark_price": item.get("markPrice"),
                    "rate_type": item.get("rateType"),
                    "api_url": url,
                    "raw_path": raw,
                    "code_version": code_version,
                })
            fresh, dupes = append_rows_unique(os.path.join(DATA_DIR, "funding_rates.csv"), [
                "exchange", "symbol", "funding_time_utc", "available_at_utc",
                "collected_at_utc", "funding_rate", "mark_price", "rate_type",
                "api_url", "raw_path", "code_version",
            ], rows, ["exchange", "symbol", "funding_time_utc"])
            appended += fresh
            skipped += dupes
            if not payload or len(payload) < limit or start_ms is None:
                break
            next_cursor = int(payload[-1]["fundingTime"]) + 1
            if cursor is not None and next_cursor <= cursor:
                break
            cursor = next_cursor
            time.sleep(0.2)
    return appended, skipped


def news_coin_hint(text):
    text = text.lower()
    hits = []
    pairs = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "ether", "eth"],
        "SOL": ["solana", "sol"],
        "BNB": ["bnb", "binance coin"],
        "XRP": ["xrp", "ripple"],
        "ADA": ["cardano", "ada"],
    }
    for coin, terms in pairs.items():
        if any(term in text for term in terms):
            hits.append(coin)
    return ",".join(hits)


def collect_gdelt_headlines(query, code_version, max_records=50):
    time.sleep(5.0)
    collected_at = now_utc()
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "hybridrel",
    }
    payload, url = request_with_retry("https://api.gdeltproject.org/api/v2/doc/doc",
                                      params, attempts=5, min_delay=5.0)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    raw = save_raw("gdelt", digest[:12], payload, collected_at)
    rows = []
    for article in payload.get("articles", []):
        title = article.get("title", "")
        article_url = article.get("url", "")
        published_at = parse_gdelt_time(article.get("seendate", ""))
        rows.append({
            "source": article.get("domain", "gdelt"),
            "coin": news_coin_hint(f"{title} {article_url}"),
            "headline": title,
            "url": article_url,
            "published_at_utc": published_at,
            "collected_at_utc": collected_at,
            "raw_hash": digest,
            "raw_path": raw,
            "api_url": url,
            "code_version": code_version,
        })
    return append_news_rows(rows)


def collect_rss_feed(name, url, code_version):
    collected_at = now_utc()
    text, final_url = request_with_retry(url, response_type="text")
    payload = {"source": name, "url": final_url, "xml": text}
    raw = save_raw("rss", name, payload, collected_at)
    root = ET.fromstring(text)
    rows = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        published = parse_rss_time(item.findtext("pubDate") or item.findtext("published"))
        rows.append({
            "source": name,
            "coin": news_coin_hint(f"{title} {link}"),
            "headline": title,
            "url": link,
            "published_at_utc": published,
            "collected_at_utc": collected_at,
            "raw_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "raw_path": raw,
            "api_url": final_url,
            "code_version": code_version,
        })
    return append_news_rows(rows)


def collect_rss_headlines(code_version):
    appended = skipped = 0
    failures = []
    for name, url in RSS_FEEDS.items():
        try:
            fresh, dupes = collect_rss_feed(name, url, code_version)
            appended += fresh
            skipped += dupes
        except Exception as exc:
            failures.append({"source": f"rss:{name}", "error": str(exc)})
    return appended, skipped, failures


def append_news_rows(rows):
    return append_rows_unique(os.path.join(DATA_DIR, "news_headlines.csv"), [
        "source", "coin", "headline", "url", "published_at_utc",
        "collected_at_utc", "raw_hash", "raw_path", "api_url", "code_version",
    ], rows, ["source", "url", "published_at_utc"])


def report_data():
    reports = []
    funding_path = os.path.join(DATA_DIR, "funding_rates.csv")
    if os.path.exists(funding_path):
        by_symbol = {}
        with open(funding_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_symbol.setdefault(row["symbol"], []).append(row["funding_time_utc"])
        for symbol, values in sorted(by_symbol.items()):
            times = sorted(datetime.fromisoformat(v) for v in values if v)
            gaps = []
            for prev, cur in zip(times, times[1:]):
                hours = (cur - prev).total_seconds() / 3600
                if hours > 12:
                    gaps.append(hours)
            reports.append({
                "dataset": "funding",
                "symbol": symbol,
                "rows": len(times),
                "first": times[0].isoformat() if times else "",
                "last": times[-1].isoformat() if times else "",
                "gaps_over_12h": len(gaps),
                "max_gap_hours": round(max(gaps), 2) if gaps else 0,
            })
    fg_path = os.path.join(DATA_DIR, "fear_greed.csv")
    if os.path.exists(fg_path):
        with open(fg_path, newline="", encoding="utf-8") as f:
            times = sorted(datetime.fromisoformat(r["value_date_utc"]) for r in csv.DictReader(f))
        gaps = []
        for prev, cur in zip(times, times[1:]):
            days = (cur - prev).total_seconds() / 86400
            if days > 1.5:
                gaps.append(days)
        reports.append({
            "dataset": "fear_greed",
            "symbol": "ALL",
            "rows": len(times),
            "first": times[0].isoformat() if times else "",
            "last": times[-1].isoformat() if times else "",
            "gaps_over_1_5d": len(gaps),
            "max_gap_days": round(max(gaps), 2) if gaps else 0,
        })
    return reports


def print_report():
    reports = report_data()
    if not reports:
        print("No data files found yet.")
        return
    for row in reports:
        print(json.dumps(row, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fear-greed", action="store_true")
    parser.add_argument("--fear-greed-limit", type=int, default=0,
                        help="0 asks Alternative.me for all available history.")
    parser.add_argument("--funding", action="store_true")
    parser.add_argument("--symbols", default=",".join(FUNDING_SYMBOLS))
    parser.add_argument("--start-ms", type=int)
    parser.add_argument("--end-ms", type=int)
    parser.add_argument("--start-date", help="UTC date like 2021-10-15 for funding backfill.")
    parser.add_argument("--end-date", help="UTC date like 2026-10-14 for funding backfill.")
    parser.add_argument("--funding-limit", type=int, default=1000)
    parser.add_argument("--news", action="store_true")
    parser.add_argument("--news-query", default="bitcoin OR ethereum OR solana OR crypto")
    parser.add_argument("--news-max-records", type=int, default=50)
    parser.add_argument("--rss", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    started_at = now_utc()
    code_version = git_version()
    counts = {}
    failures = []

    if args.all or args.fear_greed:
        try:
            fresh, dupes = collect_fear_greed(args.fear_greed_limit, code_version)
            counts["fear_greed_appended"] = fresh
            counts["fear_greed_duplicates"] = dupes
            print(f"fear_greed rows appended: {fresh} (duplicates skipped: {dupes})")
        except Exception as exc:
            failures.append({"source": "fear_greed", "error": str(exc)})
            print(f"fear_greed collection failed: {exc}")
    if args.all or args.funding:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        start_ms = args.start_ms if args.start_ms is not None else date_to_ms(args.start_date)
        end_ms = args.end_ms if args.end_ms is not None else date_to_ms(args.end_date)
        try:
            fresh, dupes = collect_funding_rates(symbols, start_ms, end_ms,
                                                 args.funding_limit, code_version)
            counts["funding_appended"] = fresh
            counts["funding_duplicates"] = dupes
            print(f"funding rows appended: {fresh} (duplicates skipped: {dupes})")
        except Exception as exc:
            failures.append({"source": "funding", "error": str(exc)})
            print(f"funding collection failed: {exc}")
    if args.all or args.news:
        try:
            fresh, dupes = collect_gdelt_headlines(args.news_query, code_version,
                                                   args.news_max_records)
            counts["gdelt_news_appended"] = fresh
            counts["gdelt_news_duplicates"] = dupes
            print(f"gdelt news rows appended: {fresh} (duplicates skipped: {dupes})")
        except Exception as exc:
            failures.append({"source": "gdelt_news", "error": str(exc)})
            print(f"gdelt news collection failed: {exc}")
    if args.all or args.rss:
        fresh, dupes, rss_failures = collect_rss_headlines(code_version)
        counts["rss_news_appended"] = fresh
        counts["rss_news_duplicates"] = dupes
        failures.extend(rss_failures)
        print(f"rss news rows appended: {fresh} (duplicates skipped: {dupes})")
        for failure in rss_failures:
            print(f"{failure['source']} failed: {failure['error']}")

    log_run(started_at, code_version, counts, failures)
    if args.report:
        print_report()


if __name__ == "__main__":
    main()
