#!/usr/bin/env python3
"""Gold-Labor: testet Daytrading-Ideen nur auf XAU/USD, mit 1-2 Jahren 15-Minuten-Daten (nachgeladen in Seiten).
Kandidat nur, wenn: n>=100, t>=3, in >=70% der Quartale positiv und bei doppelten Kosten noch positiv.
Start: GitHub -> Actions -> gold -> Run workflow."""
import os, time, argparse, warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
import scanner as S
import lab as L

SYMBOL = "XAU/USD"
PAGES = 8                       # je 5000 Kerzen (ca. 10 Wochen) -> bis ca. 1,5 Jahre
GOLD_COST = 0.20                # $ pro Trade (Spread 0,13 + etwas Slippage), Stresstest mit dem Doppelten
TPS = (1.0, 1.5, 2.0, 3.0)
MIN_N, MIN_T, MIN_QPOS = 100, 3.0, 0.7
MAX_HOLD_BARS = 144             # 36 Stunden


def fetch_pages(pages=PAGES):
    frames, end = [], None
    for p in range(pages):
        params = {"symbol": SYMBOL, "interval": "15min", "outputsize": 5000, "timezone": "UTC", "apikey": S.env("TWELVEDATA_KEY", required=True)}
        if end: params["end_date"] = end
        for attempt in range(3):
            j = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=60).json()
            if j.get("status") == "error" and "credits" in str(j.get("message", "")).lower() and attempt < 2:
                time.sleep(65); continue
            break
        if j.get("status") == "error" or not j.get("values"):
            print(f"Seite {p + 1}: {str(j.get('message', 'leer'))[:100]}"); break
        d = pd.DataFrame(j["values"])
        d["time"] = pd.to_datetime(d["datetime"], utc=True)
        for c in ("open", "high", "low", "close"): d[c] = d[c].astype(float)
        frames.append(d[["time", "open", "high", "low", "close"]])
        end = d["datetime"].min()
        print(f"Seite {p + 1}: {len(d)} Kerzen bis {end}", flush=True)
        if len(d) < 4000: break
        time.sleep(12)
    df = pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)


def to_1h(df):
    h = df.set_index("time").resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    return h


# ---------------- Strategien: (Index, Seite, Stop) ----------------
def strat_asia(df, stop_mode):
    out = []
    for day, g in df.groupby(df.time.dt.date):
        rng = g[g.time.dt.hour < 7]
        if len(rng) < 20: continue
        hi, lo = rng.high.max(), rng.low.min()
        for idx, r in g[(g.time.dt.hour >= 7) & (g.time.dt.hour < 10)].iterrows():
            a = r.atr
            if np.isnan(a) or not (0.8 * a <= hi - lo <= 8 * a): continue
            mid = (hi + lo) / 2
            if r.close > hi:
                stop = lo if stop_mode == "far" else mid
                if 0.5 * a <= r.close - stop <= 5 * a: out.append((idx, "long", stop))
                break
            if r.close < lo:
                stop = hi if stop_mode == "far" else mid
                if 0.5 * a <= stop - r.close <= 5 * a: out.append((idx, "short", stop))
                break
    return out


def strat_orb(df, start_hour, span=4):
    out = []
    for day, g in df.groupby(df.time.dt.date):
        rng = g[g.time.dt.hour == start_hour]
        if len(rng) < 4: continue
        hi, lo = rng.high.max(), rng.low.min()
        for idx, r in g[(g.time.dt.hour > start_hour) & (g.time.dt.hour <= start_hour + span - 1)].iterrows():
            a = r.atr
            if np.isnan(a) or not (0.5 * a <= hi - lo <= 4 * a): continue
            if r.close > hi: out.append((idx, "long", lo)); break
            if r.close < lo: out.append((idx, "short", hi)); break
    return out


def strat_prevday(df, mode):
    """Vortageshoch/-tief (UTC-Tag). mode 'break': erster Schluss darüber/darunter; 'fade': Docht darüber, Schluss zurück (Fehlausbruch)."""
    d = df.set_index("time").resample("1D").agg({"high": "max", "low": "min"}).dropna()
    pdh = d.high.shift(1); pdl = d.low.shift(1)
    out, done = [], set()
    dates = df.time.dt.normalize()
    for i in range(1, len(df)):
        r = df.iloc[i]
        if not (7 <= r.time.hour < 17) or np.isnan(r.atr): continue
        day = dates.iloc[i]
        h, l = pdh.get(day, np.nan), pdl.get(day, np.nan)
        if np.isnan(h): continue
        prev = df.close.iloc[i - 1]
        if mode == "break":
            if r.close > h and prev <= h and (day, "L") not in done: done.add((day, "L")); out.append((i, "long", h - r.atr))
            if r.close < l and prev >= l and (day, "S") not in done: done.add((day, "S")); out.append((i, "short", l + r.atr))
        else:
            if r.high > h and r.close < h and (day, "S") not in done: done.add((day, "S")); out.append((i, "short", r.high + 0.2 * r.atr))
            if r.low < l and r.close > l and (day, "L") not in done: done.add((day, "L")); out.append((i, "long", r.low - 0.2 * r.atr))
    return out


# ---------------- Auswertung ----------------
def evaluate(df, tf_min, sig, name):
    rows, last = [], {}
    hi, lo = df.high.values, df.low.values
    maxc = int(MAX_HOLD_BARS * 15 / tf_min)
    for i, side, stop in sig:
        t = df.time.iloc[i]
        if not (t.weekday() < 5 and 6 <= t.hour < 20): continue
        if i - last.get(side, -999) < 16 * 15 // tf_min: continue
        last[side] = i
        entry = float(df.close.iloc[i]); risk = abs(entry - stop)
        if risk <= 0: continue
        sgn = 1 if side == "long" else -1
        fut = list(zip(hi[i + 1:i + 1 + maxc], lo[i + 1:i + 1 + maxc]))
        if len(fut) < 4: continue
        row = {"time": t, "strategie": name, "side": side, "stop_dist": risk}
        for k in TPS:
            st, _ = S.simulate_path(side, entry, stop, entry + sgn * k * risk, entry + sgn * k * risk, fut, "simple")
            g = k if st == "tp1" else -1.0 if st == "stop" else 0.0
            row[f"n_{k}"] = g - GOLD_COST / risk
            row[f"d_{k}"] = g - 2 * GOLD_COST / risk
        rows.append(row)
    return rows


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


def report(rows):
    d = pd.DataFrame(rows)
    d["q"] = d.time.dt.to_period("Q")
    lines, good = [], []
    for name, g in d.groupby("strategie"):
        lines.append(f"{name}: n={len(g)}, Ø Stop {g.stop_dist.mean():.1f} $")
        for k in TPS:
            x, x2 = g[f"n_{k}"], g[f"d_{k}"]
            qs = g.groupby("q")[f"n_{k}"].mean()
            qpos = (qs > 0).mean() if len(qs) else 0
            t = tstat(x)
            lines.append(f"  Ziel {k}R: Quote {round((x > 0).mean() * 100)}%, Ø {x.mean():+.2f}R, t={t:.1f}, Quartale im Plus {(qs > 0).sum()}/{len(qs)}, doppelte Kosten Ø {x2.mean():+.2f}R")
            if len(g) >= MIN_N and t >= MIN_T and qpos >= MIN_QPOS and len(qs) >= 5 and x2.mean() > 0:
                good.append(f"{name}, Ziel {k}R: Ø {x.mean():+.2f}R, t={t:.1f}, n={len(g)}, Quartale {(qs > 0).sum()}/{len(qs)}, doppelte Kosten {x2.mean():+.2f}R")
    return "\n".join(lines), good


def seasonality(h):
    d = h.copy(); d["ret"] = (d.close - d.open) / d.atr.shift(1)
    d = d.dropna(subset=["ret"]); d = d[(d.time.dt.weekday < 5) & ~((d.time.dt.weekday == 4) & (d.time.dt.hour >= 21))]
    d["q"] = d.time.dt.to_period("Q"); cost_atr = GOLD_COST / float(d.atr.mean())
    out = []
    for key, nm in ((d.time.dt.hour, "Stunde"), (d.time.dt.weekday, "Wochentag")):
        for v, g in d.groupby(key):
            if len(g) < 100 or g.ret.std() == 0: continue
            m, t = g.ret.mean(), tstat(g.ret)
            qs = g.groupby("q").ret.mean(); cons = (np.sign(qs) == np.sign(m)).mean()
            out.append((nm, int(v), len(g), m, t, cons))
    strong = [o for o in out if abs(o[4]) >= 3.5 and o[5] >= 0.7]
    lines = [f"Zeit-Muster Gold (1h, Wochenenden bereinigt): {len(out)} Zellen, {len(strong)} mit |t|>=3,5 und gleichem Vorzeichen in >=70% der Quartale (Zufall: ca. {len(out) * 0.0005:.1f}); Kosten ca. {cost_atr:.3f} ATR/Kerze"]
    for nm, v, n, m, t, c in sorted(strong, key=lambda o: -abs(o[4]))[:8]:
        lines.append(f"  {nm} {v}{' UTC' if nm == 'Stunde' else ''}: {m:+.3f} ATR, t={t:.1f}, Quartale gleichgerichtet {c * 100:.0f}% -> {'ueber' if abs(m) > cost_atr else 'unter'} den Kosten")
    # NFP-Näherung: erster Freitag im Monat
    day = h.set_index("time").resample("1D").agg({"high": "max", "low": "min"}).dropna()
    day["rng"] = (day.high - day.low); day = day[day.index.weekday < 5]
    nfp = day[(day.index.weekday == 4) & (day.index.day <= 7)]
    lines.append(f"Erster Freitag im Monat (NFP-Näherung): Tagesspanne Ø {nfp.rng.mean():.1f} $ gegenüber {day.rng.mean():.1f} $ an normalen Tagen (n={len(nfp)})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    raw = fetch_pages()
    print(f"Gold: {len(raw)} Kerzen ({raw.time.min():%Y-%m-%d} bis {raw.time.max():%Y-%m-%d})")
    run(raw, a.dry)


def run(raw, dry):
    df = L.indicators(raw); h = L.indicators(to_1h(raw))
    rows = []
    for nm, mode in (("Asia-Ausbruch, Stop Gegenseite", "far"), ("Asia-Ausbruch, Stop Mitte", "mid")):
        rows += evaluate(df, 15, strat_asia(df, mode), nm)
    rows += evaluate(df, 15, strat_orb(df, 7), "ORB London (07-08 UTC)")
    rows += evaluate(df, 15, strat_orb(df, 14), "ORB New York (14-15 UTC)")
    rows += evaluate(df, 15, strat_prevday(df, "break"), "Vortages-Ausbruch")
    rows += evaluate(df, 15, strat_prevday(df, "fade"), "Vortages-Fehlausbruch")
    tr = L.trend_filter(h, "1D")
    rows += evaluate(h, 60, L.strat_pullback(h, 60, tr), "Trend-Rücksetzer 1h (Tagestrend)")
    txt, good = report(rows) if rows else ("Keine Trades.", [])
    out = (f"Gold-Labor (XAU/USD, {len(raw)} 15m-Kerzen, {raw.time.min():%Y-%m-%d} bis {raw.time.max():%Y-%m-%d}, Kosten {GOLD_COST} $ pro Trade)\n\n"
           + txt + "\n\n" + seasonality(h) + "\n\n"
           + ("KANDIDATEN (n>=100, t>=3, >=70% Quartale im Plus, bei doppelten Kosten positiv):\n" + "\n".join(good) if good else "Kein Kandidat erfüllt alle Bedingungen (n>=100, t>=3, >=70% Quartale im Plus, doppelte Kosten positiv).")
           + "\n\nViele getestete Varianten: Zufallstreffer sind möglich. Auch Kandidaten brauchen eine Gegenprobe auf neuen Daten. Einstieg zum Schlusskurs, Sommer-/Winterzeit verschiebt Eröffnungszeiten um eine Stunde.")
    print(out)
    if rows: pd.DataFrame(rows).to_csv("gold_trades.csv", index=False)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + out + "\n```\n")
    for i in range(0, len(out), 3800): S.telegram(out[i:i + 3800], dry)


if __name__ == "__main__":
    main()
