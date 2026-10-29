#!/usr/bin/env python3
"""Teilgewinn-Test: Gold V1 nur Long, Ziel 1,5R.
 A "simple": bis TP1 (+1,5R) oder Stop (-1R)
 B "half":   50 % beim halben Ziel (+0,75R) schließen, Stop auf Einstieg, Rest bis TP1 -> +1,125R / Break-even +0,375R / Stop -1R
Gleiche Signale, gleiche Kosten. Übernahme von B nur, wenn B in BEIDEN Zeiträumen besser ist und der gepaarte Unterschied t>=2 hat.
Start: GitHub -> Actions -> partial -> Run workflow. Ca. 12 Twelve-Data-Abrufe."""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanner as S
import gold2 as G
import multi as M
from cot import DEV_START

MAX_HOLD = 36


def outcomes(h, sig, mode):
    hi, lo, cl = h.high.values, h.low.values, h.close.values; rows = []
    for i, side, stop in sig:
        entry = cl[i]; risk = abs(entry - stop)
        if risk <= 0 or i + 5 >= len(cl): continue
        tp1 = entry + (1 if side == "long" else -1) * S.TP1_R * risk
        st, r = S.simulate_path(side, entry, stop, tp1, tp1, list(zip(hi[i + 1:i + 1 + MAX_HOLD], lo[i + 1:i + 1 + MAX_HOLD])), mode)
        if st == "open": r = 0.0
        rows.append({"time": h.time.iloc[i], "st": st, "r": r - M.cost_price("XAUUSD", entry) / risk})
    return pd.DataFrame(rows)


def run(raw, dry):
    h = G.prepare(raw)
    sig = [s for s in M.signals_v1(h) if s[1] == "long"]
    a, b = outcomes(h, sig, "simple"), outcomes(h, sig, "half")
    d = a[["time", "r"]].rename(columns={"r": "A"}).assign(B=b.r.values, stB=b.st.values, stA=a.st.values)
    lines = [f"Teilgewinn-Test Gold (V1 nur Long, Ziel {S.TP1_R:g}R), {len(d)} Trades {d.time.min():%Y-%m-%d} bis {d.time.max():%Y-%m-%d}\n"]
    passed = True
    for name, m in (("Gesamt", d.time.notna()), ("vor 2023", d.time < DEV_START), ("ab 2023", d.time >= DEV_START)):
        x = d[m]; diff = x.B - x.A
        t = float(diff.mean() / (diff.std() / np.sqrt(len(diff)))) if diff.std() > 0 else 0.0
        eqA, eqB = x.A.cumsum(), x.B.cumsum()
        ddA, ddB = (eqA.cummax() - eqA).max(), (eqB.cummax() - eqB).max()
        lines.append(f"{name}: A ohne Teilgewinn Ø {x.A.mean():+.3f}R, Quote {round((x.A > 0).mean() * 100)}%, Rückgang {ddA:.1f}R | "
                     f"B mit Teilgewinn Ø {x.B.mean():+.3f}R, Quote {round((x.B > 0).mean() * 100)}%, Rückgang {ddB:.1f}R | Unterschied t={t:.1f}")
        if name != "Gesamt":
            passed = passed and x.B.mean() > x.A.mean()
        else:
            t_all = t
    passed = passed and t_all >= 2
    sb = d.stB.value_counts()
    lines.append(f"\nB im Detail: {sb.get('tp1', 0)}x volles Ziel, {sb.get('be', 0)}x Break-even nach Teilgewinn, {sb.get('stop', 0)}x Stop, "
                 f"{sb.get('tp1_open', 0) + sb.get('open', 0)}x offen am Ende")
    lines.append("\nERGEBNIS: " + ("Teilgewinn + Break-even ist besser. Empfehlung: im Bot einschalten (TRADE_MODE = half)." if passed else
                                    "Teilgewinn + Break-even bringt im Schnitt nicht mehr. Höhere Trefferquote ja, aber kein höherer Ertrag. "
                                    "Einschalten nur, wenn dir die ruhigere Kurve wichtiger ist als der Ertrag."))
    lines.append("Annahme: Wird das halbe Ziel erreicht, zählt Break-even erst ab der nächsten Kerze (Reihenfolge innerhalb einer Kerze ist unbekannt).")
    text = "\n".join(lines); print(text)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    S.telegram(text[:3900], dry)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    try:
        raw = M.fetch_1h("XAU/USD", 12)
    except Exception as e:
        S.telegram(f"⚠️ Teilgewinn-Test abgebrochen: {S.scrub(e)[:200]}", a.dry); sys.exit(1)
    run(raw, a.dry)


if __name__ == "__main__":
    main()
