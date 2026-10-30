#!/usr/bin/env python3
"""Gold 5-Minuten-Labor: 50-%-Rücksetzer nach starkem Impuls.
Regel (vorab festgelegt, Beispiel Aufwärtstrend 100 -> 200):
  Impuls:   Tief A -> Hoch B innerhalb von höchstens 24 Kerzen (2 h), Größe mindestens 8 x ATR(14, 5 min)
  Einstieg: Limit-Long bei 50 % des Impulses (150), gültig bis 48 Kerzen (4 h) nach B; neues Hoch über B = Impuls neu bewerten
  Ziel:     50 % des Rücksetzers wieder hoch (175)
  Stop:     drei Varianten  S1 = 61,8 % (138,2) | S2 = 78,6 % (121,4) | S3 = unter dem Impuls-Start A (100)
  Spätestens nach 8 h raus. Abwärtstrend spiegelbildlich.
Bestehen: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv, besser als Zufall (p<0,05).
Start: GitHub -> Actions -> gold5 -> Run workflow. Ca. 12 Twelve-Data-Abrufe."""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
import scanner as S
import lab as L

IMP_BARS, IMP_ATR, VALID, HOLD = 24, 8.0, 48, 96
STOPS = {"S1 Stop 61,8 %": 0.618, "S2 Stop 78,6 %": 0.786, "S3 Stop unter Impuls-Start": 1.0}
COST, PAGES, SIMS = 0.20, 12, 100


def fetch(interval="5min", pages=PAGES):
    frames, end, last = [], None, {}
    for p in range(pages):
        params = {"symbol": "XAU/USD", "interval": interval, "outputsize": 5000, "timezone": "UTC", "apikey": S.env("TWELVEDATA_KEY", required=True)}
        if end: params["end_date"] = end
        for k in range(5):
            try:
                last = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=60).json()
            except Exception:
                time.sleep(20 * (k + 1)); continue
            if last.get("status") == "error" and "minute" in str(last.get("message", "")).lower():
                time.sleep(65); continue
            break
        if last.get("status") == "error" or not last.get("values"): break
        d = pd.DataFrame(last["values"]); d["time"] = pd.to_datetime(d["datetime"], utc=True)
        for c in ("open", "high", "low", "close"): d[c] = d[c].astype(float)
        frames.append(d[["time", "open", "high", "low", "close"]]); end = d["datetime"].min()
        print(f"Seite {p + 1}: bis {end}", flush=True)
        if len(d) < 4000: break
        time.sleep(12)
    if not frames: raise RuntimeError(str(last.get("message", "keine Daten"))[:150])
    return pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True).iloc[:-1]


def find_setups(h):
    """Liefert je Impuls ein Setup: (Index B, Seite, A, B). Kausal: nur Daten bis zur aktuellen Kerze."""
    hi, lo, atr = h.high.values, h.low.values, h.atr.values
    out, last_b = [], -1
    for t in range(IMP_BARS + 15, len(h)):
        w0 = t - IMP_BARS
        for side in ("long", "short"):
            if side == "long":
                ia = w0 + int(np.argmin(lo[w0:t + 1])); ib = ia + int(np.argmax(hi[ia:t + 1])); A, B = lo[ia], hi[ib]
            else:
                ia = w0 + int(np.argmax(hi[w0:t + 1])); ib = ia + int(np.argmin(lo[ia:t + 1])); A, B = hi[ia], lo[ib]
            if ib != t - 1 or ib <= ia or np.isnan(atr[ib]) or abs(B - A) < IMP_ATR * atr[ib] or ib == last_b:
                continue                                  # B muss gerade bestätigt sein (Kerze danach kein neues Extrem)
            if (side == "long" and hi[t] > B) or (side == "short" and lo[t] < B):
                continue
            out.append((ib, side, A, B)); last_b = ib
    return out


def trade(h, ib, side, A, B, frac):
    """Simuliert Limit-Einstieg bei 50 %, Ziel 50 % des Rücksetzers, Stop bei frac des Impulses. Ergebnis in R oder None."""
    hi, lo, op, cl = h.high.values, h.low.values, h.open.values, h.close.values
    sg = 1 if side == "long" else -1
    rng = abs(B - A); entry_lvl = B - sg * 0.5 * rng; stop = B - sg * frac * rng - sg * (0.1 * h.atr.values[ib] if frac >= 1 else 0)
    target = entry_lvl + sg * 0.5 * abs(B - entry_lvl)
    for j in range(ib + 1, min(ib + 1 + VALID, len(h))):
        if (hi[j] > B) if sg == 1 else (lo[j] < B): return None          # neues Extrem vor Einstieg -> verworfen
        if (lo[j] <= entry_lvl) if sg == 1 else (hi[j] >= entry_lvl):
            entry = min(entry_lvl, op[j]) if sg == 1 else max(entry_lvl, op[j])
            risk = abs(entry - stop)
            if risk <= 0 or (sg == 1 and entry <= stop) or (sg == -1 and entry >= stop): return None
            first = [(entry, lo[j])] if sg == 1 else [(hi[j], entry)]
            fut = first + list(zip(hi[j + 1:j + HOLD], lo[j + 1:j + HOLD]))
            st, _ = S.simulate_path(side, entry, stop, target, target, fut, "simple")
            k = abs(target - entry) / risk
            g = k if st == "tp1" else -1.0 if st == "stop" else (cl[min(j + HOLD - 1, len(h) - 1)] - entry) * sg / risk
            return {"time": h.time.iloc[ib], "side": side, "r": g - COST / risk, "r2": g - 2 * COST / risk, "k": k, "st": st}
    return None


def random_trade(h, j0, side, d_entry, d_stop, d_target, rng):
    """Zufalls-Gegenstück: gleiche Abstände (in ATR) für Limit, Stop und Ziel, aber zufälliger Zeitpunkt."""
    hi, lo, op, cl, atr = h.high.values, h.low.values, h.open.values, h.close.values, h.atr.values
    j0 = int(rng.integers(100, len(h) - HOLD - VALID)); a = atr[j0]; sg = 1 if side == "long" else -1
    lvl = cl[j0] - sg * d_entry * a; stop = lvl - sg * d_stop * a; target = lvl + sg * d_target * a
    for j in range(j0 + 1, j0 + 1 + VALID):
        if (lo[j] <= lvl) if sg == 1 else (hi[j] >= lvl):
            entry = min(lvl, op[j]) if sg == 1 else max(lvl, op[j]); risk = abs(entry - stop)
            if risk <= 0 or (entry - stop) * sg <= 0: return None
            fut = ([(entry, lo[j])] if sg == 1 else [(hi[j], entry)]) + list(zip(hi[j + 1:j + HOLD], lo[j + 1:j + HOLD]))
            st, _ = S.simulate_path(side, entry, stop, target, target, fut, "simple")
            k = abs(target - entry) / risk
            g = k if st == "tp1" else -1.0 if st == "stop" else (cl[min(j + HOLD - 1, len(h) - 1)] - entry) * sg / risk
            return g - COST / risk
    return None


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


TF = "5min"                    # per --tf änderbar; Kerzenanzahlen bleiben gleich (bei 15 min: Impuls <= 6 h, Limit 12 h gültig, max. 24 h Haltedauer)
P_MAX = 0.05


def run(raw, dry):
    h = L.indicators(raw).reset_index(drop=True)
    setups = find_setups(h)
    mins = int(TF.replace("min", ""))
    lines = [f"Gold {mins}-Minuten-Labor: 50-%-Rücksetzer nach Impuls (>= {IMP_ATR:g} ATR in <= {IMP_BARS * mins / 60:g} h), Ziel 50 % des Rücksetzers",
             f"Daten {raw.time.min():%Y-%m-%d} bis {raw.time.max():%Y-%m-%d}, {len(h)} Kerzen, {len(setups)} Impulse gefunden, Kosten {COST} $\n"]
    rng = np.random.default_rng(5); passed = []
    for name, frac in STOPS.items():
        rows = [x for x in (trade(h, ib, s, A, B, frac) for ib, s, A, B in setups) if x]
        if len(rows) < 20:
            lines.append(f"{name}: zu wenige Trades ({len(rows)})"); continue
        d = pd.DataFrame(rows).sort_values("time"); half = d.time.iloc[len(d) // 2]
        a_, b_ = d[d.time < half].r.mean(), d[d.time >= half].r.mean(); t = tstat(d.r)
        # Zufallsvergleich mit gleichen Abständen in ATR
        geo = []
        for ib, s, A, B in setups:
            at = h.atr.values[ib]; rng_ = abs(B - A); sg = 1 if s == "long" else -1
            lvl = B - sg * 0.5 * rng_; stp = B - sg * frac * rng_; tgt = lvl + sg * 0.25 * rng_
            geo.append((s, abs(h.close.values[ib] - lvl) / at, abs(lvl - stp) / at, abs(tgt - lvl) / at))
        sims = []
        for _ in range(SIMS):
            r = [random_trade(h, 0, s, de, ds, dt, rng) for s, de, ds, dt in geo]
            r = [x for x in r if x is not None]; sims.append(np.mean(r) if r else np.nan)
        sims = np.array([x for x in sims if not np.isnan(x)]); p = float((sims >= d.r.mean()).mean())
        ok = len(d) >= 100 and t >= 2.5 and a_ > 0 and b_ > 0 and d.r2.mean() > 0 and p < P_MAX
        if ok: passed.append(name)
        lines.append(f"{name}: n={len(d)}, Quote {round((d.r > 0).mean() * 100)}%, Ø Chance/Risiko 1:{d.k.mean():.2f}, Ø {d.r.mean():+.2f}R, t={t:.1f}, "
                     f"1. Hälfte {a_:+.2f} / 2. Hälfte {b_:+.2f}, Kosten x2 {d.r2.mean():+.2f}R | long {d[d.side == 'long'].r.mean():+.2f} / short {d[d.side == 'short'].r.mean():+.2f} | "
                     f"Zufall {sims.mean():+.2f}, p={p:.2f}")
        ses = d.groupby(pd.cut(d.time.dt.hour, [-1, 6, 12, 17, 24], labels=["00-07", "07-13", "13-18", "18-24"]))["r"].agg(["count", "mean"])
        lines.append("   je Tageszeit UTC: " + " | ".join(f"{k}: n={int(v['count'])} {v['mean']:+.2f}R" for k, v in ses.iterrows() if v["count"]))
    lines.append("\nERGEBNIS: " + ("Bestanden: " + ", ".join(passed) + ". Kann als Signal in den Bot." if passed else
                                    "Keine Stop-Variante besteht alle Bedingungen. Nicht in den Bot."))
    lines.append(f"Bedingungen: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv, besser als Zufall (p<{str(P_MAX).replace('.', ',')}). Einstieg per Limit, "
                 f"in der Füllkerze zählt nur der Stop. Datenzeitraum hängt davon ab, wie weit der Gratis-Key zurückreicht.")
    text = "\n".join(lines); print(text)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    S.telegram(text[:3900], dry)


def main():
    global TF, P_MAX
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--tf", default="5min"); a = ap.parse_args()
    TF = a.tf
    if TF != "5min": P_MAX = 0.01            # zweiter Versuch derselben Idee -> strengere Hürde
    S.load_env()
    try:
        raw = fetch(TF, 8 if TF == "15min" else PAGES)
    except Exception as e:
        S.telegram(f"⚠️ Gold-{TF}-Labor abgebrochen: {S.scrub(e)[:200]}", a.dry); sys.exit(1)
    run(raw, a.dry)


if __name__ == "__main__":
    main()
