#!/usr/bin/env python3
"""Mehrmarkt-Labor: testet die Gold-Regel V1 UNVERÄNDERT auf weiteren Märkten (1h-Rücksetzer im Tagestrend, nur 07-17 UTC).
Dazu Zufallsvergleich je Markt und gesamt sowie Drawdown-/Risiko-Analyse. Keine neuen Einstellungen.
Start: GitHub -> Actions -> multi -> Run workflow (am besten früh am Tag, braucht ca. 45 Twelve-Data-Abrufe)."""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
import scanner as S
import lab as L
import gold2 as G

MARKETS = {"XAUUSD": "XAU/USD", "XAGUSD": "XAG/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
           "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "QQQ": "QQQ", "SPY": "SPY"}
PAGES = 5
TPS = (1.5, 2.0, 3.0)
MAIN_TP = 1.5                 # Ziel, das der Bot nutzt (vorab festgelegt)
SIMS = 100
MAX_HOLD = 36


def cost_price(sym, price):
    """Angenommene Kosten pro Trade in Preis-Einheiten."""
    fixed = {"XAUUSD": 0.20, "XAGUSD": 0.03}
    if sym in fixed: return fixed[sym]
    if sym in S.COST_ABS: return S.COST_ABS[sym]
    if sym == "BTCUSD": return price * 0.001
    return price * 0.0003


def fetch_1h(symbol, pages=PAGES):
    frames, end, last = [], None, {}
    for p in range(pages):
        params = {"symbol": symbol, "interval": "1h", "outputsize": 5000, "timezone": "UTC", "apikey": S.env("TWELVEDATA_KEY", required=True)}
        if end: params["end_date"] = end
        for attempt in range(3):
            last = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=60).json()
            if last.get("status") == "error" and "minute" in str(last.get("message", "")).lower() and attempt < 2:
                time.sleep(65); continue
            break
        if last.get("status") == "error" or not last.get("values"):
            break
        d = pd.DataFrame(last["values"]); d["time"] = pd.to_datetime(d["datetime"], utc=True)
        for c in ("open", "high", "low", "close"): d[c] = d[c].astype(float)
        frames.append(d[["time", "open", "high", "low", "close"]]); end = d["datetime"].min()
        if len(d) < 4000: break
        time.sleep(12)
    if not frames:
        raise RuntimeError(str(last.get("message", "keine Daten"))[:150])
    df = pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)


def signals_v1(h):
    return G.cooldown([s for s in G.base_signals(h) if G.f_session(h, *s)])


def evaluate(sym, h, sig):
    """Je Trade: Zeit, Seite, Ergebnis nach Kosten je Ziel (einfacher Modus, Stop vor Ziel in derselben Kerze)."""
    hi, lo, cl = h.high.values, h.low.values, h.close.values; n = len(cl); rows = []
    for i, side, stop in sig:
        sg = 1 if side == "long" else -1
        entry = cl[i]; risk = abs(entry - stop)
        if risk <= 0 or i + 5 >= n: continue
        c = cost_price(sym, entry) / risk
        row = {"time": h.time.iloc[i], "symbol": sym, "side": side}
        for k in TPS:
            tp = entry + sg * k * risk; g = 0.0
            for j in range(i + 1, min(i + 1 + MAX_HOLD, n)):
                if (lo[j] <= stop) if sg == 1 else (hi[j] >= stop): g = -1.0; break
                if (hi[j] >= tp) if sg == 1 else (lo[j] <= tp): g = k; break
            row[f"r_{k}"] = g - c
        rows.append(row)
    return rows


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


def dd_stats(d, col):
    d = d.sort_values("time"); eq = d[col].cumsum()
    dd = (eq.cummax() - eq).max()
    streak = cur = 0
    for v in d[col]:
        cur = cur + 1 if v < 0 else 0; streak = max(streak, cur)
    m = d.set_index("time")[col].resample("ME").sum()
    return dd, streak, (m.min() if len(m) else 0.0), ((m > 0).mean() if len(m) else 0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    frames, notes = {}, []
    for sym, ts in MARKETS.items():
        try:
            frames[sym] = fetch_1h(ts); print(f"{sym}: {len(frames[sym])} Kerzen ab {frames[sym].time.min():%Y-%m-%d}", flush=True)
        except Exception as e:
            notes.append(f"{sym}: {S.scrub(e)[:100]}"); print(notes[-1], flush=True)
            if "for the day" in str(e): break
        time.sleep(12)
    try:
        frames["BTCUSD"] = L.hist_bitstamp("btcusd", 3600, 1400)
    except Exception as e:
        notes.append(f"BTCUSD: {S.scrub(e)[:100]}")
    run(frames, notes, a.dry)


def run(frames, notes, dry):
    preps, sigs, rows = {}, {}, []
    for sym, raw in frames.items():
        if len(raw) < 1500: notes.append(f"{sym}: zu wenig Daten"); continue
        h = G.prepare(raw); s = signals_v1(h)
        preps[sym], sigs[sym] = h, s
        rows += evaluate(sym, h, s)
    if not rows:
        S.telegram("Mehrmarkt-Labor: keine Daten/Trades. " + " | ".join(notes), dry); return
    d = pd.DataFrame(rows); d["q"] = d.time.dt.to_period("Q")
    col = f"r_{MAIN_TP}"
    lines = [f"Mehrmarkt-Labor: Regel V1 unverändert (1h-Rücksetzer im Tagestrend, nur 07-17 UTC), Kosten je Markt, Ziel {MAIN_TP}R\n"]

    # Zufallsvergleich je Markt und gesamt
    rng = np.random.default_rng(7)
    sim_means = {sym: [] for sym in preps}; sim_pool = []
    for _ in range(SIMS):
        pool = []
        for sym, h in preps.items():
            A = {"cl": h.close.values, "atr": h.atr.values, "td": h.trend_d.values}
            tw = G.random_twins(h, A, sigs[sym], [G.f_session], rng)
            r = [x[col] for x in evaluate(sym, h, tw)]
            sim_means[sym].append(np.mean(r) if r else np.nan); pool += r
        sim_pool.append(np.mean(pool) if pool else np.nan)

    lines.append("Je Markt: n | Quote | Ø nach Kosten | t | Quartale im Plus | 1. Hälfte / 2. Hälfte | long / short | Zufall Ø, p")
    for sym, g in d.groupby("symbol"):
        g = g.sort_values("time"); half = g.time.iloc[len(g) // 2]
        qs = g.groupby("q")[col].mean(); s = np.array([x for x in sim_means[sym] if not np.isnan(x)])
        p = float((s >= g[col].mean()).mean()) if len(s) else np.nan
        lines.append(f"  {sym}: n={len(g)} | {round((g[col] > 0).mean() * 100)}% | {g[col].mean():+.2f}R | t={tstat(g[col]):.1f} | {(qs > 0).sum()}/{len(qs)} | "
                     f"{g[g.time < half][col].mean():+.2f} / {g[g.time >= half][col].mean():+.2f} | "
                     f"{g[g.side == 'long'][col].mean():+.2f} / {g[g.side == 'short'][col].mean():+.2f} | Zufall {np.nanmean(s):+.2f}, p={p:.2f}")
    sp = np.array([x for x in sim_pool if not np.isnan(x)])
    allp = float((sp >= d[col].mean()).mean())
    nog = d[d.symbol != "XAUUSD"]
    lines.append(f"\nGESAMT: n={len(d)}, Ø {d[col].mean():+.2f}R, t={tstat(d[col]):.1f}, Zufall Ø {sp.mean():+.2f} (95%: {np.percentile(sp, 95):+.2f}), p={allp:.2f}")
    if len(nog):
        lines.append(f"OHNE GOLD (echter Gegentest): n={len(nog)}, Ø {nog[col].mean():+.2f}R, t={tstat(nog[col]):.1f}, "
                     f"Märkte im Plus: {int((nog.groupby('symbol')[col].mean() > 0).sum())}/{nog.symbol.nunique()}")
    lines.append("Andere Ziele (gesamt): " + " | ".join(f"{k}R: {d[f'r_{k}'].mean():+.2f}" for k in TPS))

    # Risiko-Analyse
    lines.append("\nRISIKO (Ziel %gR, in R; bei x %% Risiko je Trade = x * R in %% vom Konto):" % MAIN_TP)
    for name, sub in (("Nur Gold", d[d.symbol == "XAUUSD"]), ("Alle Märkte", d)):
        if not len(sub): continue
        dd, streak, worst_m, pos_m = dd_stats(sub, col)
        yrs = max((sub.time.max() - sub.time.min()).days / 365.25, 0.1); per_y = sub[col].sum() / yrs
        lines.append(f"  {name}: {per_y:+.1f}R pro Jahr, größter Rückgang {dd:.1f}R, längste Verlustserie {streak}, schlechtester Monat {worst_m:+.1f}R, Monate im Plus {pos_m * 100:.0f}%")
        lines.append("    " + " | ".join(f"{r_}% Risiko: {per_y * r_:+.0f}%/Jahr, Rückgang ca. {dd * r_:.0f}%" for r_ in (0.5, 1.0, 1.5, 2.0)))
    lines.append("\nLesart: Hält die Regel auf Märkten, an denen sie nicht entwickelt wurde (vor allem OHNE GOLD), spricht das für einen echten Effekt. "
                 "Nur dann lohnt es, Märkte im Bot zu ergänzen. Rückgänge sind historisch, künftige können größer sein (eher x1,5 einplanen). "
                 "Einstieg zum Schlusskurs, Kosten pauschal, ETFs handeln nur in US-Börsenzeiten.")
    if notes: lines.append("Hinweise: " + " | ".join(notes))
    text = "\n".join(lines); print(text)
    d.drop(columns="q").to_csv("multi_trades.csv", index=False)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    for i in range(0, len(text), 3800): S.telegram(text[i:i + 3800], dry)


if __name__ == "__main__":
    main()
