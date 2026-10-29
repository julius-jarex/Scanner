#!/usr/bin/env python3
"""Makro-Test: verbessern US-Dollar, Realzinsen, VIX oder Inflationserwartung (kostenlos von FRED) die Gold-Long-Signale (V1)?
Vier Hypothesen, vorab festgelegt (Long nur, wenn ...):
 M1 Dollar schwach:          Dollar-Index (DTWEXBGS) unter seinem 50-Tage-Durchschnitt
 M2 Realzins fällt:          10-j. Realzins (DFII10) niedriger als vor 20 Handelstagen
 M3 Angst hoch:              VIX (VIXCLS) über seinem 1-Jahres-Median
 M4 Inflationserwartung:     10-j. Breakeven (T10YIE) höher als vor 20 Handelstagen
Übernahme nur, wenn in BEIDEN Zeiträumen (vor/ab 2023) besser als alle Trades (mind. 30) und Unterschied t>=2.
FRED-Werte gelten erst 2 Tage nach ihrem Datum (kein Blick in die Zukunft).
Start: GitHub -> Actions -> macro -> Run workflow. Braucht ca. 12 Twelve-Data-Abrufe."""
import os, sys, io, argparse, warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
import scanner as S
import gold2 as G
import multi as M
from cot import welch_t, DEV_START, COL

SERIES = ["DTWEXBGS", "DFII10", "VIXCLS", "T10YIE"]


def fetch_fred(sid):
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    dcol = "observation_date" if "observation_date" in d else d.columns[0]
    d["date"] = pd.to_datetime(d[dcol], utc=True)
    d["v"] = pd.to_numeric(d[sid] if sid in d else d.iloc[:, 1], errors="coerce")
    return d.dropna(subset=["v"])[["date", "v"]].sort_values("date").reset_index(drop=True)


def _flag(cond, ref):
    """Bedingung als 1/0, NaN wo der Vergleichswert noch nicht berechnet werden kann."""
    f = cond.astype(float)
    f[ref.isna()] = np.nan
    return f


def features(fred):
    out = {}
    dx = fred["DTWEXBGS"].dropna().copy(); ma = dx.v.rolling(50).mean(); dx["f"] = _flag(dx.v < ma, ma)
    out["M1 Dollar schwach (unter 50-Tage-Schnitt)"] = dx
    ry = fred["DFII10"].dropna().copy(); p = ry.v.shift(20); ry["f"] = _flag(ry.v < p, p)
    out["M2 Realzins fällt (20 Tage)"] = ry
    vx = fred["VIXCLS"].dropna().copy(); md = vx.v.rolling(252).median(); vx["f"] = _flag(vx.v > md, md)
    out["M3 Angst hoch (VIX über 1-Jahres-Median)"] = vx
    be = fred["T10YIE"].dropna().copy(); p2 = be.v.shift(20); be["f"] = _flag(be.v > p2, p2)
    out["M4 Inflationserwartung steigt (20 Tage)"] = be
    for d in out.values():
        d["valid_from"] = d.date + pd.Timedelta(days=2)
    return out


def run(raw, fred, dry, notes=None):
    h = G.prepare(raw)
    sig = [s for s in M.signals_v1(h) if s[1] == "long"]
    base = pd.DataFrame(M.evaluate("XAUUSD", h, sig)).sort_values("time")
    lines = [f"Makro-Test Gold (FRED), Basis: V1 nur Long, Ziel {M.MAIN_TP}R, {len(base)} Trades {base.time.min():%Y-%m-%d} bis {base.time.max():%Y-%m-%d}",
             f"Alle Trades: Ø {base[COL].mean():+.2f}R | vor 2023 {base[base.time < DEV_START][COL].mean():+.2f}R | ab 2023 {base[base.time >= DEV_START][COL].mean():+.2f}R\n"]
    adopt = []
    for name, f in features(fred).items():
        d = pd.merge_asof(base, f[["valid_from", "f"]].rename(columns={"valid_from": "time"}).sort_values("time"), on="time").dropna(subset=["f"])
        if len(d) < 60:
            lines.append(f"{name}: zu wenige Daten"); continue
        m = d["f"] > 0.5
        keep, drop = d[m], d[~m]
        t = welch_t(keep[COL], drop[COL])
        ok, per = True, []
        for pn, pm in (("vor 2023", d.time < DEV_START), ("ab 2023", d.time >= DEV_START)):
            k, a_ = d[m & pm][COL], d[pm][COL]
            per.append(f"{pn}: behalten {k.mean():+.2f}R (n={len(k)}) gegen alle {a_.mean():+.2f}R")
            ok = ok and len(k) >= 30 and k.mean() > a_.mean()
        passed = ok and t >= 2.0
        if passed: adopt.append(name)
        lines.append(f"{name}: behalten n={len(keep)} Ø {keep[COL].mean():+.2f}R | aussortiert n={len(drop)} Ø {drop[COL].mean():+.2f}R | Unterschied t={t:.1f}")
        lines.append("   " + " | ".join(per) + (" -> ÜBERNEHMEN" if passed else " -> nicht übernehmen"))
    lines.append("\nERGEBNIS: " + ("Filter mit Vorteil: " + ", ".join(adopt) + ". Diese baue ich in den Bot ein." if adopt
                                    else "Kein Makro-Filter verbessert die Gold-Signale belastbar. Keiner kommt in den Bot."))
    lines.append("Bedingung: in beiden Zeiträumen besser als alle Trades (mind. 30) und Unterschied t>=2. Bei 4 Tests ist ein einzelner Zufallstreffer möglich.")
    if notes: lines.append("Hinweise: " + " | ".join(notes))
    text = "\n".join(lines); print(text)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    S.telegram(text[:3900], dry)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    fred, notes = {}, []
    for sid in SERIES:
        try:
            fred[sid] = fetch_fred(sid); print(f"FRED {sid}: {len(fred[sid])} Werte bis {fred[sid].date.max():%Y-%m-%d}", flush=True)
        except Exception as e:
            notes.append(f"{sid}: {str(e)[:80]}")
    missing = [s for s in SERIES if s not in fred]
    if missing:
        S.telegram(f"⚠️ Makro-Test abgebrochen, FRED-Daten fehlen: {', '.join(missing)}. {' | '.join(notes)}", a.dry); sys.exit(1)
    try:
        raw = M.fetch_1h("XAU/USD", 12)
    except Exception as e:
        S.telegram(f"⚠️ Makro-Test abgebrochen: {S.scrub(e)[:200]}", a.dry); sys.exit(1)
    run(raw, fred, a.dry, notes)


if __name__ == "__main__":
    main()
