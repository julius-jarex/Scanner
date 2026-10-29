#!/usr/bin/env python3
"""Gold-Historie: testet Regel V1 UNVERÄNDERT auf Gold-Daten vor 2023 (nie zur Entwicklung benutzt) und vergleicht mit der Zeit ab 2023.
Start: GitHub -> Actions -> goldhist -> Run workflow. Braucht ca. 12 Twelve-Data-Abrufe."""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanner as S
import gold2 as G
import multi as M

DEV_START = pd.Timestamp("2023-02-17", tz="UTC")     # ab hier lagen die Daten der Entwicklung (Gold-Labor 2)
PAGES = 12
SIMS = 100
COL = f"r_{M.MAIN_TP}"


def block(name, h, d, sig, rng, lines):
    if len(d) < 20:
        lines.append(f"{name}: zu wenige Trades ({len(d)})"); return
    sims = []
    for _ in range(SIMS):
        tw = G.random_twins(h, {"cl": h.close.values, "atr": h.atr.values, "td": h.trend_d.values}, sig, [G.f_session], rng)
        r = [x[COL] for x in M.evaluate("XAUUSD", h, tw)]
        sims.append(np.mean(r) if r else np.nan)
    s = np.array([x for x in sims if not np.isnan(x)])
    p = float((s >= d[COL].mean()).mean()) if len(s) else np.nan
    q = d.groupby(d.time.dt.to_period("Q"))[COL].mean()
    lines.append(f"{name}: n={len(d)}, Quote {round((d[COL] > 0).mean() * 100)}%, Ø {d[COL].mean():+.2f}R, t={M.tstat(d[COL]):.1f}, "
                 f"Quartale im Plus {(q > 0).sum()}/{len(q)}, long {d[d.side == 'long'][COL].mean():+.2f} ({int((d.side == 'long').sum())}) / "
                 f"short {d[d.side == 'short'][COL].mean():+.2f} ({int((d.side == 'short').sum())}) | Zufall {np.nanmean(s):+.2f}, p={p:.2f}")


def run(raw, dry, notes=None):
    h = G.prepare(raw)
    sig = M.signals_v1(h)
    d = pd.DataFrame(M.evaluate("XAUUSD", h, sig))
    if d.empty:
        S.telegram("Gold-Historie: keine Trades.", dry); return
    idx = {t: i for i, t in enumerate(h.time)}
    rng = np.random.default_rng(11)
    lines = [f"Gold-Historie: Regel V1 unverändert (1h, nur 07-17 UTC, Ziel {M.MAIN_TP}R, Kosten 0,20 $), Daten {raw.time.min():%Y-%m-%d} bis {raw.time.max():%Y-%m-%d}\n"]
    for name, m in (("VOR 2023 (nie benutzt, echter Test)", d.time < DEV_START), ("AB 2023 (Entwicklungszeitraum)", d.time >= DEV_START)):
        sub = d[m]
        s_sub = [s for s in sig if (h.time.iloc[s[0]] < DEV_START) == (name.startswith("VOR"))]
        block(name, h, sub, s_sub, rng, lines)
    lines.append("\nJe Jahr (Ø R pro Trade, Anzahl, Gold-Preis Jahresanfang -> Jahresende):")
    for y, g in d.groupby(d.time.dt.year):
        px = raw[raw.time.dt.year == y].close
        lines.append(f"  {y}: {g[COL].mean():+.2f}R (n={len(g)}, Summe {g[COL].sum():+.0f}R) | Gold {px.iloc[0]:.0f} -> {px.iloc[-1]:.0f} $")
    old = d[d.time < DEV_START]
    if len(old) >= 100:
        dd, streak, worst, posm = M.dd_stats(old, COL)
        lines.append(f"\nVor 2023: größter Rückgang {dd:.1f}R, längste Verlustserie {streak}, schlechtester Monat {worst:+.1f}R, Monate im Plus {posm * 100:.0f}%")
    lines.append("\nLesart: Entscheidend ist VOR 2023. Positiv und besser als Zufall (p<0,05) = Hinweis auf echten Gold-Effekt. "
                 "Um null oder im Minus = das Plus ab 2023 kam vermutlich vom starken Gold-Anstieg. Kosten früher eher höher (Spreads), Gold-Preis niedriger.")
    if notes: lines.append("Hinweise: " + " | ".join(notes))
    text = "\n".join(lines); print(text)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    for i in range(0, len(text), 3800): S.telegram(text[i:i + 3800], dry)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    try:
        raw = M.fetch_1h("XAU/USD", PAGES)
    except Exception as e:
        S.telegram(f"⚠️ Gold-Historie abgebrochen: {S.scrub(e)[:200]}", a.dry); print(e); sys.exit(1)
    notes = []
    if raw.time.min() >= DEV_START:
        notes.append("Der Key liefert keine Daten vor 2023, der eigentliche Test ist daher nicht möglich.")
    run(raw, a.dry, notes)


if __name__ == "__main__":
    main()
