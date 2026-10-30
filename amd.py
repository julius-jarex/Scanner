#!/usr/bin/env python3
"""AMD-Labor Gold (Accumulation, Manipulation, Distribution), Regeln vorab festgelegt:
 Accumulation: Asia-Spanne 00-07 UTC, mindestens 1 ATR breit.
 Manipulation: 07-10 UTC sticht eine Kerze über das Asia-Hoch (unter das Tief) und schließt zurück in der Spanne (erste solche Kerze des Tages).
 Distribution: Einstieg in Gegenrichtung zum Schlusskurs, Stop 0,1 ATR hinter dem Extrem des Stichs (Stop 0,3-3 ATR).
   Z1 = Gegenseite der Spanne | Z2 = eine Spannenbreite über die Gegenseite hinaus. Spätestens 20 UTC raus.
Auf 15 min (ca. 1 Jahr) und 1 h (ca. 3,7 Jahre). Bestehen: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv, Zufall p<0,01.
Start: GitHub -> Actions -> amd -> Run workflow. Ca. 13 Twelve-Data-Abrufe."""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanner as S
import lab as L
import multi as M
import gold5 as G5

COST, SIMS, P_MAX = 0.20, 100, 0.01


def setups(h):
    """Je Tag höchstens ein Setup: (Index, Seite, Stop, Z1, Z2)."""
    out = []
    for day, g in h.groupby(h.time.dt.date):
        asia = g[g.time.dt.hour < 7]
        if len(asia) < 4: continue
        top, bot = asia.high.max(), asia.low.min(); width = top - bot
        for idx, r in g[(g.time.dt.hour >= 7) & (g.time.dt.hour < 10)].iterrows():
            a = r.atr
            if np.isnan(a) or width < a: break
            up = r.high > top and r.close < top; dn = r.low < bot and r.close > bot
            if up == dn: continue
            if up: side, stop, z1, z2 = "short", r.high + 0.1 * a, bot, bot - width
            else:  side, stop, z1, z2 = "long", r.low - 0.1 * a, top, top + width
            risk = abs(r.close - stop)
            if 0.3 * a <= risk <= 3 * a and (z1 - r.close) * (1 if side == "long" else -1) > 0:
                out.append((idx, side, stop, z1, z2))
            break
    return out


def outcome(h, i, side, stop, target):
    hi, lo, cl = h.high.values, h.low.values, h.close.values
    day = h.time.iloc[i].date(); j_end = i
    while j_end + 1 < len(h) and h.time.iloc[j_end + 1].date() == day and h.time.iloc[j_end + 1].hour < 20: j_end += 1
    entry = cl[i]; sg = 1 if side == "long" else -1; risk = abs(entry - stop)
    if j_end <= i or risk <= 0: return None
    st, _ = S.simulate_path(side, entry, stop, target, target, list(zip(hi[i + 1:j_end + 1], lo[i + 1:j_end + 1])), "simple")
    k = abs(target - entry) / risk
    g = k if st == "tp1" else -1.0 if st == "stop" else (cl[j_end] - entry) * sg / risk
    return g, risk, k


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


def analyse(label, h, rng):
    h = L.indicators(h).reset_index(drop=True)
    su = setups(h); lines, passed = [f"{label}: {len(su)} Setups ({h.time.min():%Y-%m-%d} bis {h.time.max():%Y-%m-%d})"], []
    window = np.where((h.time.dt.hour >= 7) & (h.time.dt.hour < 10))[0]
    for zname, zi in (("Z1 Gegenseite", 3), ("Z2 eine Breite darüber hinaus", 4)):
        rows = []
        for s in su:
            o = outcome(h, s[0], s[1], s[2], s[zi])
            if o: rows.append({"time": h.time.iloc[s[0]], "side": s[1], "r": o[0] - COST / o[1], "r2": o[0] - 2 * COST / o[1], "k": o[2],
                               "geo": (abs(h.close.values[s[0]] - s[2]) / h.atr.values[s[0]], abs(s[zi] - h.close.values[s[0]]) / h.atr.values[s[0]])})
        if len(rows) < 20:
            lines.append(f"  {zname}: zu wenige Trades ({len(rows)})"); continue
        d = pd.DataFrame(rows).sort_values("time"); half = d.time.iloc[len(d) // 2]
        a_, b_ = d[d.time < half].r.mean(), d[d.time >= half].r.mean(); t = tstat(d.r)
        sims = []
        for _ in range(SIMS):
            res = []
            for side, (ds, dt) in zip(d.side, d.geo):
                i = int(rng.choice(window)); a = h.atr.values[i]; sg = 1 if side == "long" else -1
                if np.isnan(a): continue
                o = outcome(h, i, side, h.close.values[i] - sg * ds * a, h.close.values[i] + sg * dt * a)
                if o: res.append(o[0] - COST / o[1])
            sims.append(np.mean(res) if res else np.nan)
        sims = np.array([x for x in sims if not np.isnan(x)]); p = float((sims >= d.r.mean()).mean())
        ok = len(d) >= 100 and t >= 2.5 and a_ > 0 and b_ > 0 and d.r2.mean() > 0 and p < P_MAX
        if ok: passed.append(f"{label} {zname}")
        lines.append(f"  {zname}: n={len(d)}, Quote {round((d.r > 0).mean() * 100)}%, Ø Chance/Risiko 1:{d.k.median():.2f} (Median), Ø {d.r.mean():+.2f}R, t={t:.1f}, "
                     f"1. Hälfte {a_:+.2f} / 2. Hälfte {b_:+.2f}, Kosten x2 {d.r2.mean():+.2f}R | long {d[d.side == 'long'].r.mean():+.2f} / "
                     f"short {d[d.side == 'short'].r.mean():+.2f} | Zufall {sims.mean():+.2f}, p={p:.2f}")
    return lines, passed


def run(frames, dry, notes):
    rng = np.random.default_rng(9); lines = ["AMD-Labor Gold (Accumulation 00-07 UTC, Manipulation 07-10 UTC, Distribution bis 20 UTC), Kosten 0,20 $\n"]; passed = []
    for label, h in frames.items():
        l, p = analyse(label, h, rng); lines += l; passed += p
    lines.append("\nERGEBNIS: " + ("Bestanden: " + ", ".join(passed) + ". Kann als Signal in den Bot." if passed else
                                    "Keine Variante besteht alle Bedingungen. Nicht in den Bot."))
    lines.append("Bedingungen: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv, besser als Zufall (p<0,01). Zeiten in UTC (deutsche Winterzeit +1 h).")
    if notes: lines.append("Hinweise: " + " | ".join(notes))
    text = "\n".join(lines); print(text)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    S.telegram(text[:3900], dry)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    frames, notes = {}, []
    try: frames["15 min"] = G5.fetch("15min", 8)
    except Exception as e: notes.append(f"15 min: {S.scrub(e)[:80]}")
    try: frames["1 h"] = M.fetch_1h("XAU/USD", 5)
    except Exception as e: notes.append(f"1 h: {S.scrub(e)[:80]}")
    if not frames:
        S.telegram("⚠️ AMD-Labor abgebrochen: " + " | ".join(notes), a.dry); sys.exit(1)
    run(frames, a.dry, notes)


if __name__ == "__main__":
    main()
