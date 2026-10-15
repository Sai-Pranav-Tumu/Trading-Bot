"""
Evolutionary crypto bot simulator (SIMULATION ONLY - no real orders, no keys).

Each bot = a small genome: feature weights, a confidence threshold, a bet size.
Every simulated day a bot scores each coin, bets on the best one (if confident
enough) for one day, otherwise stays in cash.

Data split (never mixed):
  historical pool -> evolution generations 0-8 of every 12-generation cycle
  recent pool     -> generations 9-11 of every cycle ("today's conditions")
  holdout (last 60 days) -> never touched until the final report

Usage:
  python evolve_bots.py --synthetic          # offline smoke test (fake data)
  python evolve_bots.py                      # real Binance daily data
  python evolve_bots.py --x 3 --gens 120     # survive if >3% return per window
"""
import argparse, os
import numpy as np
import pandas as pd

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA"]
FEE, SLIP = 0.001, 0.0005
COST = 2 * (FEE + SLIP)            # round trip cost per trade
FEATS = ["ret1", "ret3", "ret7", "ret30", "vol7", "sentiment"]
WINDOW, HOLDOUT, RECENT = 30, 60, 60
N_BOTS, ELITE = 1000, 20


# ---------------------------------------------------------------- data
def fetch_binance(sym, days=730):
    import requests
    err = None
    for host in ("https://api.binance.com", "https://data-api.binance.vision"):
        try:
            r = requests.get(f"{host}/api/v3/klines", timeout=20,
                             params={"symbol": sym + "USDT", "interval": "1d", "limit": days})
            r.raise_for_status()
            rows = r.json()
            idx = pd.to_datetime([x[0] for x in rows], unit="ms").normalize()
            return pd.Series([float(x[4]) for x in rows], index=idx, name=sym)
        except Exception as e:
            err = e
    raise RuntimeError(f"Could not fetch {sym}: {err}")


def load_prices(synthetic, days=730, cache="prices_cache.csv"):
    if synthetic:
        rng = np.random.default_rng(0)
        common = rng.standard_t(4, days) * 0.02
        cols = {c: 100 * np.exp(np.cumsum(0.6 * common + rng.standard_t(4, days) * 0.02))
                for c in COINS}
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
        return pd.DataFrame(cols, index=idx)
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    df = pd.concat([fetch_binance(c, days) for c in COINS], axis=1).dropna()
    df.to_csv(cache)
    return df


def load_sentiment(index, coins):
    """PLUG-IN POINT for news. Must return [days x coins] scores in -1..+1 that
    were computed ONLY from articles published before each day's decision time.
    Zeros for now = price-only bots."""
    return pd.DataFrame(0.0, index=index, columns=coins)


def build(px):
    r1 = px.pct_change()
    f = {"ret1": r1, "ret3": px.pct_change(3), "ret7": px.pct_change(7),
         "ret30": px.pct_change(30), "vol7": r1.rolling(7).std(),
         "sentiment": load_sentiment(px.index, px.columns)}
    F = np.stack([f[k].values for k in FEATS], axis=2)        # [T, C, K]
    nxt = (px.shift(-1) / px - 1).values                      # return from close t to close t+1
    ok = ~(np.isnan(F[:, :, :-1]).any((1, 2)) | np.isnan(nxt).any(1))
    return F[ok], nxt[ok], px.index[ok]


# ---------------------------------------------------------------- bots
def evaluate(pop, F, nxt):
    W, thr, frac = pop
    D = F.shape[0]
    scores = np.einsum("dck,nk->ndc", F, W)
    best, bscore = scores.argmax(2), scores.max(2)
    trade = bscore > thr[:, None]
    ret = nxt[np.arange(D)[None, :], best]
    daily = np.where(trade, frac[:, None] * (ret - COST), 0.0)
    eq = np.cumprod(1 + daily, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = ((peak - eq) / peak).max(1)
    total = eq[:, -1] - 1
    return total, dd, trade.sum(1)


def random_pop(rng, n=N_BOTS):
    return [rng.normal(0, 1, (n, len(FEATS))), rng.uniform(0, 1, n), rng.uniform(0.1, 1, n)]


def breed(pop, surv_idx, rng, n=N_BOTS):
    W, thr, frac = pop
    nw, nt, nf = W[surv_idx[:ELITE]], thr[surv_idx[:ELITE]], frac[surv_idx[:ELITE]]
    need = n - len(nw)
    a = rng.choice(surv_idx, need); b = rng.choice(surv_idx, need)
    mask = rng.random((need, W.shape[1])) < 0.5
    cw = np.where(mask, W[a], W[b]) + rng.normal(0, 0.15, (need, W.shape[1]))
    ct = np.where(rng.random(need) < 0.5, thr[a], thr[b]) + rng.normal(0, 0.05, need)
    cf = np.clip(np.where(rng.random(need) < 0.5, frac[a], frac[b]) + rng.normal(0, 0.05, need), 0.05, 1.0)
    return [np.vstack([nw, cw]), np.concatenate([nt, ct]), np.concatenate([nf, cf])]


def pick(pop, idx):
    return [p[idx] for p in pop]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--x", type=float, default=2.0, help="survive if window return > x percent")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    px = load_prices(a.synthetic)
    F, nxt, dates = build(px)
    hold_F, hold_n = F[-HOLDOUT:], nxt[-HOLDOUT:]
    rec_F, rec_n = F[-(HOLDOUT + RECENT):-HOLDOUT], nxt[-(HOLDOUT + RECENT):-HOLDOUT]
    his_F, his_n = F[:-(HOLDOUT + RECENT)], nxt[:-(HOLDOUT + RECENT)]
    # standardise with historical stats only (no peeking at recent/holdout)
    mu, sd = his_F[:, :, :-1].mean((0, 1)), his_F[:, :, :-1].std((0, 1)) + 1e-9
    for arr in (his_F, rec_F, hold_F):
        arr[:, :, :-1] = np.clip((arr[:, :, :-1] - mu) / sd, -4, 4)
    print(f"days: historical={len(his_F)} recent={len(rec_F)} holdout={len(hold_F)}  "
          f"({dates[0].date()} -> {dates[-1].date()})")

    random_baseline = random_pop(rng)
    pop = random_pop(rng)
    rows = []
    for g in range(a.gens):
        use_recent = (g % 12) >= 9
        PF, PN = (rec_F, rec_n) if use_recent else (his_F, his_n)
        s = rng.integers(0, len(PF) - WINDOW + 1)
        total, dd, ntr = evaluate(pop, PF[s:s + WINDOW], PN[s:s + WINDOW])
        fit = total - 0.5 * dd
        surv = np.where((total * 100 > a.x) & (ntr >= 3))[0]
        if len(surv) < 50:                       # fallback: keep top 10%
            surv = np.argsort(-fit)[:100]
        surv = surv[np.argsort(-fit[surv])]
        rows.append(dict(gen=g, pool="recent" if use_recent else "historical",
                         survivors=int(((total * 100 > a.x) & (ntr >= 3)).sum()),
                         best_ret_pct=100 * total[surv[0]], mean_ret_pct=100 * total.mean()))
        if g % 6 == 0 or g == a.gens - 1:
            r = rows[-1]
            print(f"gen {g:3d} [{r['pool']:10s}] survivors={r['survivors']:4d} "
                  f"best={r['best_ret_pct']:+6.1f}%  pop mean={r['mean_ret_pct']:+6.2f}%")
        pop = breed(pop, surv, rng)

    pd.DataFrame(rows).to_csv("evolution_log.csv", index=False)

    # pick champion on ALL training data (historical + recent), then judge on holdout
    tr_F, tr_n = np.concatenate([his_F, rec_F]), np.concatenate([his_n, rec_n])
    t, d, _ = evaluate(pop, tr_F, tr_n)
    champ = int(np.argmax(t - 0.5 * d))
    h_t, h_d, h_n = evaluate(pop, hold_F, hold_n)
    r_t, _, _ = evaluate(random_baseline, hold_F, hold_n)
    btc_hold = np.prod(1 + hold_n[:, 0]) - 1

    print("\n=== HOLDOUT (last 60 days, never seen during evolution) ===")
    print(f"champion bot        : {100*h_t[champ]:+.1f}%  (max drawdown {100*h_d[champ]:.1f}%, {h_n[champ]} trades)")
    print(f"evolved pop median  : {100*np.median(h_t):+.1f}%   top-10% cutoff {100*np.quantile(h_t,0.9):+.1f}%")
    print(f"random bots median  : {100*np.median(r_t):+.1f}%   (luck baseline)")
    print(f"BTC buy & hold      : {100*btc_hold:+.1f}%")
    print("Evolution only matters if the evolved bots beat BOTH random bots and buy&hold on holdout.")
    print("Log saved to evolution_log.csv")


if __name__ == "__main__":
    main()
