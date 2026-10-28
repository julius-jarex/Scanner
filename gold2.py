#!/usr/bin/env python3
"""Gold-Labor 2: vorab festgelegte Varianten des 1h-Rücksetzers im Tagestrend (XAU/USD), mit mehreren Jahren 1h-Daten.
Kandidat nur, wenn: n>=100, t>=3, >=70% der Quartale im Plus (mind. 6 Quartale) und bei doppelten Kosten positiv.
Start: GitHub -> Actions -> gold2 -> Run workflow."""
import os, time, argparse, warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
import scanner as S
import lab as L

SYMBOL = "XAU/USD"
PAGES = 5                     # je 5000 1h-Kerzen (ca. 10 Monate) -> bis ca. 4 Jahre
COST = 0.20                   # $ pro Trade, Stresstest mit dem Doppelten
TPS = (1.5, 2.0, 3.0)
MAX_HOLD = 36                 # Stunden
ROUND = 50.0                  # runde Marken in $
MIN_N, MIN_T, MIN_QPOS, MIN_Q = 100, 3.0, 0.7, 6


def fetch_1h(pages=PAGES):
    frames, end = [], None
    for p in range(pages):
        params = {"symbol": SYMBOL, "interval": "1h", "outputsize": 5000, "timezone": "UTC", "apikey": S.env("TWELVEDATA_KEY", required=True)}
        if end: params["end_date"] = end
        for attempt in range(3):
            j = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=60).json()
            if j.get("status") == "error" and "credits" in str(j.get("message", "")).lower() and attempt < 2:
                time.sleep(65); continue
            break
        if j.get("status") == "error" or not j.get("values"):
            print(f"Seite {p + 1}: {str(j.get('message', 'leer'))[:100]}"); break
        d = pd.DataFrame(j["values"]); d["time"] = pd.to_datetime(d["datetime"], utc=True)
        for c in ("open", "high", "low", "close"): d[c] = d[c].astype(float)
        frames.append(d[["time", "open", "high", "low", "close"]]); end = d["datetime"].min()
        print(f"Seite {p + 1}: {len(d)} Kerzen bis {end}", flush=True)
        if len(d) < 4000: break
        time.sleep(12)
    df = pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)


def prepare(raw):
    h = L.indicators(raw)
    h["trend_d"] = L.trend_filter(h, "1D")
    h["trend_4h"] = L.trend_filter(h, "4h")
    h["atr_ratio"] = h.atr / h.atr.rolling(200).median()
    d = h.set_index("time").resample("1D").agg({"high": "max", "low": "min"}).dropna()
    day = h.time.dt.normalize()
    h["pdh"] = day.map(d.high.shift(1)); h["pdl"] = day.map(d.low.shift(1))
    return h.reset_index(drop=True)


def base_signals(h, deep=False):
    """(i, side, stop) für jede Kerze, die das Grundmuster erfüllt."""
    out = []
    lo, hi, e20, e50 = h.low.values, h.high.values, h.ema20.values, h.ema50.values
    for i in range(210, len(h)):
        r = h.iloc[i]; a = r.atr
        if np.isnan(a): continue
        w = slice(i - 5, i + 1)
        ref = e50 if deep else e20
        if r.trend_d == 1 and (lo[w] <= ref[w]).any() and r.close > r.ema20 > r.ema50 and r.close > r.open:
            stop = lo[w].min() - 0.2 * a
            if 0.5 * a <= r.close - stop <= 3 * a: out.append((i, "long", stop))
        if r.trend_d == -1 and (hi[w] >= ref[w]).any() and r.close < r.ema20 < r.ema50 and r.close < r.open:
            stop = hi[w].max() + 0.2 * a
            if 0.5 * a <= stop - r.close <= 3 * a: out.append((i, "short", stop))
    return out


# ---------------- Filter (vorab festgelegt) ----------------
def f_session(h, i, side, stop):
    return 7 <= h.time.iloc[i].hour <= 16


def f_level(h, i, side, stop):
    r = h.iloc[i]; risk = abs(r.close - stop)
    lv = [r.pdh, r.pdl, np.floor(r.close / ROUND) * ROUND, np.ceil(r.close / ROUND) * ROUND]
    ahead = [x for x in lv if not np.isnan(x) and ((x > r.close) if side == "long" else (x < r.close)) and abs(x - r.close) > 1e-9]
    return not ahead or min(abs(x - r.close) for x in ahead) >= 1.0 * risk


def f_4h(h, i, side, stop):
    return h.trend_4h.iloc[i] == (1 if side == "long" else -1)


def f_vol(h, i, side, stop):
    x = h.atr_ratio.iloc[i]
    return not np.isnan(x) and 0.8 <= x <= 1.5


# ---------------- Auswertung ----------------
def cooldown(sig, bars=4):
    out, last = [], {}
    for i, side, stop in sig:
        if i - last.get(side, -999) >= bars:
            out.append((i, side, stop)); last[side] = i
    return out


def fixed_exit(h, sig, confirm=False):
    rows = []
    hi, lo, op = h.high.values, h.low.values, h.open.values
    for i, side, stop in sig:
        sg = 1 if side == "long" else -1
        if confirm:          # Einstieg erst, wenn eine der 2 Folgekerzen das Extrem der Signalkerze überschreitet
            trig = hi[i] if sg == 1 else lo[i]
            j = next((j for j in range(i + 1, min(i + 3, len(h))) if (hi[j] >= trig if sg == 1 else lo[j] <= trig)), None)
            if j is None: continue
            entry = max(trig, op[j]) if sg == 1 else min(trig, op[j])
            first = [(entry, lo[j])] if sg == 1 else [(hi[j], entry)]      # Füllkerze: nur Stop-Risiko, kein Zielgewinn
            fut = first + list(zip(hi[j + 1:j + MAX_HOLD], lo[j + 1:j + MAX_HOLD]))
        else:
            entry = h.close.iloc[i]
            fut = list(zip(hi[i + 1:i + 1 + MAX_HOLD], lo[i + 1:i + 1 + MAX_HOLD]))
        risk = abs(entry - stop)
        if risk <= 0 or len(fut) < 4 or (sg == 1 and entry <= stop) or (sg == -1 and entry >= stop): continue
        row = {"time": h.time.iloc[i], "side": side}
        for k in TPS:
            st, _ = S.simulate_path(side, entry, stop, entry + sg * k * risk, entry + sg * k * risk, fut, "simple")
            g = k if st == "tp1" else -1.0 if st == "stop" else 0.0
            row[f"n_{k}"] = g - COST / risk; row[f"d_{k}"] = g - 2 * COST / risk
        rows.append(row)
    return rows


def trail_exit(h, sig, trail_atr=2.0, hold=72):
    """Hälfte bei 1R, Stop auf Einstieg, Rest mit Abstand 2 ATR zum höchsten/tiefsten Schluss nachziehen."""
    rows = []
    hi, lo, cl = h.high.values, h.low.values, h.close.values
    for i, side, stop in sig:
        sg = 1 if side == "long" else -1
        entry, a = cl[i], h.atr.iloc[i]; risk = abs(entry - stop)
        if risk <= 0 or i + 4 >= len(h): continue
        part, st, best, res = False, stop, entry, None
        for j in range(i + 1, min(i + 1 + hold, len(h))):
            if (lo[j] <= st) if sg == 1 else (hi[j] >= st):
                rest = (st - entry) / risk * sg
                res = (0.5 + 0.5 * rest) if part else -1.0; break
            if not part and ((hi[j] >= entry + risk) if sg == 1 else (lo[j] <= entry - risk)):
                part, st = True, entry
            if part:
                best = max(best, cl[j]) if sg == 1 else min(best, cl[j])
                ns = best - sg * trail_atr * a
                st = max(st, ns) if sg == 1 else min(st, ns)
        if res is None:
            j = min(i + hold, len(h) - 1); rest = (cl[j] - entry) / risk * sg
            res = (0.5 + 0.5 * rest) if part else rest
        rows.append({"time": h.time.iloc[i], "side": side, "n_T": res - COST / risk, "d_T": res - 2 * COST / risk})
    return rows


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


def judge(name, rows, keys):
    if not rows: return [f"{name}: keine Trades"], []
    d = pd.DataFrame(rows); d["q"] = d.time.dt.to_period("Q")
    lines, good = [f"{name}: n={len(d)} (long {int((d.side == 'long').sum())}, short {int((d.side == 'short').sum())})"], []
    for k in keys:
        x, x2 = d[f"n_{k}"], d[f"d_{k}"]
        qs = d.groupby("q")[f"n_{k}"].mean(); t = tstat(x)
        lo_, sh_ = d[d.side == "long"][f"n_{k}"].mean(), d[d.side == "short"][f"n_{k}"].mean()
        lab = "Trailing" if k == "T" else f"Ziel {k}R"
        lines.append(f"  {lab}: Quote {round((x > 0).mean() * 100)}%, Ø {x.mean():+.2f}R, t={t:.1f}, Quartale {(qs > 0).sum()}/{len(qs)}, "
                     f"Kosten x2 {x2.mean():+.2f}R | long {lo_:+.2f}, short {sh_:+.2f}")
        if len(d) >= MIN_N and t >= MIN_T and len(qs) >= MIN_Q and (qs > 0).mean() >= MIN_QPOS and x2.mean() > 0:
            good.append(f"{name}, {lab}: Ø {x.mean():+.2f}R, t={t:.1f}, n={len(d)}, Quartale {(qs > 0).sum()}/{len(qs)}")
    return lines, good


def run(raw, dry):
    h = prepare(raw)
    base = base_signals(h); deep = base_signals(h, deep=True)
    flt = lambda sig, *fs: cooldown([s for s in sig if all(f(h, *s) for f in fs)])
    variants = [
        ("V0 Basis", flt(base), "fixed"),
        ("V1 nur 07-17 UTC", flt(base, f_session), "fixed"),
        ("V2 mit Bestätigungskerze", flt(base), "confirm"),
        ("V3 Abstand zum Level >= 1R", flt(base, f_level), "fixed"),
        ("V4 tiefer Rücksetzer (EMA50)", flt(deep), "fixed"),
        ("V5 auch 4h-Trend passend", flt(base, f_4h), "fixed"),
        ("V6 normale Volatilität", flt(base, f_vol), "fixed"),
        ("V7 Teilausstieg 1R + Trailing", flt(base), "trail"),
        ("V8 Kombi V1+V3+V5", flt(base, f_session, f_level, f_4h), "fixed"),
        ("V9 Kombi V8 + Trailing", flt(base, f_session, f_level, f_4h), "trail"),
    ]
    out, good = [], []
    for name, sig, mode in variants:
        if mode == "trail":
            l, g = judge(name, trail_exit(h, sig), ["T"])
        else:
            l, g = judge(name, fixed_exit(h, sig, confirm=(mode == "confirm")), list(TPS))
        out += l; good += g
    yrs = (raw.time.max() - raw.time.min()).days / 365.25
    text = (f"Gold-Labor 2 (XAU/USD 1h, {len(raw)} Kerzen, {raw.time.min():%Y-%m-%d} bis {raw.time.max():%Y-%m-%d}, ca. {yrs:.1f} Jahre, Kosten {COST} $)\n"
            f"Grundmuster: Rücksetzer im Tagestrend, Einstieg zum Schlusskurs, Stop hinter dem 6-Kerzen-Extrem.\n\n" + "\n".join(out) + "\n\n"
            + ("KANDIDATEN (n>=100, t>=3, >=70% von mind. 6 Quartalen im Plus, Kosten x2 positiv):\n" + "\n".join(good) if good
               else "Kein Kandidat erfüllt alle Bedingungen (n>=100, t>=3, >=70% von mind. 6 Quartalen im Plus, Kosten x2 positiv).")
            + "\n\n10 Varianten vorab festgelegt, keine nachträgliche Anpassung. Sommer-/Winterzeit verschiebt Sessions um eine Stunde.")
    print(text)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    for i in range(0, len(text), 3800): S.telegram(text[i:i + 3800], dry)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    raw = fetch_1h()
    print(f"Gold 1h: {len(raw)} Kerzen ({raw.time.min():%Y-%m-%d} bis {raw.time.max():%Y-%m-%d})")
    run(raw, a.dry)


if __name__ == "__main__":
    main()
