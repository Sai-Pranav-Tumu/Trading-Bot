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
def multi_eval(pop, PF, PN, rng, k):
    """Judge every bot on k random windows -> much less luck-driven."""
    tot, dds, trs = [], [], []
    for _ in range(k):
        s = rng.integers(0, len(PF) - WINDOW + 1)
        t, d, n = evaluate(pop, PF[s:s + WINDOW], PN[s:s + WINDOW])
        tot.append(t); dds.append(d); trs.append(n)
    return np.mean(tot, 0), np.mean(dds, 0), np.mean(trs, 0), np.min(tot, 0)


def run_once(seed, a, his, rec, hold):
    rng = np.random.default_rng(seed)
    (his_F, his_n), (rec_F, rec_n), (hold_F, hold_n) = his, rec, hold
    baseline, pop = random_pop(rng), random_pop(rng)
    for g in range(a.gens):
        use_recent = (g % 12) >= 9                    # 9 historical, then 3 recent
        PF, PN = (rec_F, rec_n) if use_recent else (his_F, his_n)
        mean_t, mean_d, mean_n, worst = multi_eval(pop, PF, PN, rng, a.windows)
        fit = mean_t - 0.5 * mean_d
        ok = (mean_t * 100 > a.x) & (mean_n >= 3) & (worst > -0.15)
        surv = np.where(ok)[0]
        if len(surv) < 50:                            # fallback: keep top 10%
            surv = np.argsort(-fit)[:100]
        surv = surv[np.argsort(-fit[surv])]
        if g % 12 == 0 or g == a.gens - 1:
            print(f"  seed {seed} gen {g:3d} [{'recent' if use_recent else 'historical':10s}] "
                  f"survivors={int(ok.sum()):4d}  mean window return={100*mean_t.mean():+6.2f}%")
        pop = breed(pop, surv, rng)

    tr_F, tr_n = np.concatenate([his_F, rec_F]), np.concatenate([his_n, rec_n])
    t, d, _ = evaluate(pop, tr_F, tr_n)
    champ = int(np.argmax(t - 0.5 * d))
    h_t, h_d, h_n = evaluate(pop, hold_F, hold_n)
    r_t, _, _ = evaluate(baseline, hold_F, hold_n)
    return dict(seed=seed, champ=h_t[champ], champ_dd=h_d[champ],
                pop_median=np.median(h_t), top10=np.quantile(h_t, 0.9),
                random_median=np.median(r_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--x", type=float, default=2.0, help="survive if mean window return > x percent")
    ap.add_argument("--windows", type=int, default=8, help="windows per generation")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    px = load_prices(a.synthetic)
    F, nxt, dates = build(px)
    his = [F[:-(HOLDOUT + RECENT)], nxt[:-(HOLDOUT + RECENT)]]
    rec = [F[-(HOLDOUT + RECENT):-HOLDOUT], nxt[-(HOLDOUT + RECENT):-HOLDOUT]]
    hold = [F[-HOLDOUT:], nxt[-HOLDOUT:]]
    mu, sd = his[0][:, :, :-1].mean((0, 1)), his[0][:, :, :-1].std((0, 1)) + 1e-9
    for arr in (his[0], rec[0], hold[0]):             # standardise on historical stats only
        arr[:, :, :-1] = np.clip((arr[:, :, :-1] - mu) / sd, -4, 4)
    print(f"days: historical={len(his[0])} recent={len(rec[0])} holdout={len(hold[0])} "
          f"({dates[0].date()} -> {dates[-1].date()})")

    res = [run_once(s, a, his, rec, hold) for s in range(1, a.seeds + 1)]
    df = pd.DataFrame(res)
    df.to_csv("holdout_results.csv", index=False)
    btc = np.prod(1 + hold[1][:, 0]) - 1
    ew = np.prod(1 + hold[1].mean(1)) - 1

    print("\n=== HOLDOUT per seed (last 60 days, never seen in evolution) ===")
    print((df.drop(columns="seed") * 100).round(1).rename(
        columns=dict(champ="champion%", champ_dd="champ_dd%", pop_median="evolved_med%",
                     top10="top10%", random_median="random_med%")).to_string())
    print(f"\nMEDIAN over seeds: champion {100*df.champ.median():+.1f}% | "
          f"evolved bots {100*df.pop_median.median():+.1f}% | random bots {100*df.random_median.median():+.1f}%")
    print(f"BENCHMARKS: BTC buy&hold {100*btc:+.1f}% | equal-weight 6 coins buy&hold {100*ew:+.1f}%")
    beats_rand = (df.pop_median > df.random_median).mean()
    print(f"Evolved beat random in {100*beats_rand:.0f}% of seeds. "
          f"Real edge needs: beat random consistently AND beat buy&hold.")


if __name__ == "__main__":
    main()