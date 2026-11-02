#!/usr/bin/env python3
"""
Labor: Intraday-Momentum mit "Noise Area" (nach Zarattini, Aziz & Barbon 2024)
Märkte: Nasdaq 100 (Daten aus dem ORB-Labor), S&P 500 falls Daten vorhanden.
Gleiche Bestehensbedingungen wie die anderen Labore.

REGELN (vorab festgelegt, EINE Hauptvariante, nicht optimieren!)
----------------------------------------------------------------
 - Handelszeit New York 09:30-16:00.
 - Für jede Minute des Tages: sigma = Durchschnitt der letzten 14 Tage von
   |Kurs zu dieser Minute / Tageseröffnung - 1|.
 - Noise Area: Oberkante = max(Eröffnung, Vortagesschluss) x (1 + sigma)
               Unterkante = min(Eröffnung, Vortagesschluss) x (1 - sigma)
 - Prüfzeitpunkte: 10:00, 10:30, ... 15:30 (Schlusskurs der Minute davor).
 - Kurs über Oberkante -> long, unter Unterkante -> short.
   Einstieg zum Eröffnungskurs der nächsten Minute.
 - Stop (nachgezogen, nur an den Prüfzeitpunkten geprüft):
   long:  Kurs < max(Oberkante, VWAP)   short: Kurs > min(Unterkante, VWAP)
   Danach ist ein neuer Einstieg möglich (auch in Gegenrichtung).
 - Spätestens Ausstieg zum Schlusskurs 16:00, nie über Nacht.
 - R = Abstand Einstieg zum Stop-Niveau beim Einstieg (mind. 3x Kosten).
 - VWAP ohne echtes Volumen: Durchschnitt des typischen Preises seit 09:30.

BEDINGUNGEN: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv,
besser als Zufall (p<0,01). Zufall = gleiche Trades (gleiche Uhrzeit, Dauer
und Richtung), aber an zufälligen anderen Tagen (2000 Durchläufe).
Zeigt, ob das Signal etwas taugt oder nur der Trend mitverdient.

AUFRUF:  python momentum_nasdaq_labor.py
"""
import glob
import os

import numpy as np
import pandas as pd

LOOKBACK = 14
CHECK_MIN = list(range(30, 390, 30))      # 10:00 ... 15:30 (Minuten nach 09:30)
MIN_RISK_COST = 3.0
N_RANDOM = 2000
SEED = 42
MARKETS = {
    "NASDAQ": dict(kind="orb", path=os.path.join("daten", "USATECHIDXUSD"), cost=1.5),
    "SP500": dict(kind="duka", path=os.path.join("daten", "duka", "USA500IDXUSD"), cost=0.5),
}


def _scale(X):
    med = np.nanmedian(X[:, :, 3])
    for div in (1000, 100, 10, 1, 10000, 100000):
        if 1000 <= med / div <= 60000:
            return X / div
    return X


def load_days(kind, path):
    files = sorted(glob.glob(os.path.join(path, "*.npy")))
    days, mats = [], []
    for f in files:
        a = np.load(f)
        if len(a) < 200:
            continue
        d = pd.Timestamp(os.path.basename(f)[:-4])
        if d.dayofweek >= 5:
            continue
        if kind == "duka":          # Sekunden seit Mitternacht UTC -> Minuten seit 09:30 NY
            ts = (pd.Timestamp(d, tz="UTC") + pd.to_timedelta(a[:, 0].astype(np.int64), unit="s")
                  ).tz_convert("America/New_York")
            mins = (ts.hour * 60 + ts.minute - 570).values
            ok = (mins >= 0) & (mins < 390)
            if ok.sum() < 300:
                continue
            a = np.column_stack([mins[ok], a[ok, 1:]])
        m = np.full((390, 4), np.nan)
        m[a[:, 0].astype(int)] = a[:, 1:5]
        if np.isnan(m[:, 3]).mean() > 0.2 or np.isnan(m[0, 0]):
            continue
        # Lücken mit letztem Schlusskurs füllen
        for k in range(1, 390):
            if np.isnan(m[k, 3]):
                m[k] = m[k - 1, 3]
        days.append(d)
        mats.append(m)
    if not mats:
        return [], None
    return days, _scale(np.stack(mats))


def backtest(days, M, cost):
    O, H, L, C = M[..., 0], M[..., 1], M[..., 2], M[..., 3]
    nd = len(days)
    move = np.abs(C / O[:, :1] - 1)
    trades = []
    for i in range(LOOKBACK, nd):
        if (days[i] - days[i - 1]).days > 5:
            continue
        sigma = move[i - LOOKBACK:i].mean(axis=0)
        op, pc = O[i, 0], C[i - 1, -1]
        ub = max(op, pc) * (1 + sigma)
        lb = min(op, pc) * (1 - sigma)
        tp = (H[i] + L[i] + C[i]) / 3
        vwap = np.cumsum(tp) / np.arange(1, 391)
        pos, entry, e_min, risk = 0, 0.0, 0, 0.0
        for k in CHECK_MIN:
            px = C[i, k - 1]
            # Stop prüfen
            if pos == 1 and px < max(ub[k - 1], vwap[k - 1]):
                trades.append((i, e_min, k, 1, entry, O[i, k], risk))
                pos = 0
            elif pos == -1 and px > min(lb[k - 1], vwap[k - 1]):
                trades.append((i, e_min, k, -1, entry, O[i, k], risk))
                pos = 0
            # Einstieg
            if pos == 0:
                if px > ub[k - 1]:
                    r = px - max(ub[k - 1], vwap[k - 1])
                    if r >= MIN_RISK_COST * cost:
                        pos, entry, e_min, risk = 1, O[i, k], k, r
                elif px < lb[k - 1]:
                    r = min(lb[k - 1], vwap[k - 1]) - px
                    if r >= MIN_RISK_COST * cost:
                        pos, entry, e_min, risk = -1, O[i, k], k, r
        if pos != 0:
            trades.append((i, e_min, 390, pos, entry, C[i, -1], risk))
    if not trades:
        return None
    t = np.array(trades)
    di, em, xm, d = t[:, 0].astype(int), t[:, 1].astype(int), t[:, 2].astype(int), t[:, 3]
    entry, exitp, risk = t[:, 4], t[:, 5], t[:, 6]
    gross = d * (exitp - entry)
    r = (gross - cost) / risk
    r2 = (gross - 2 * cost) / risk

    # Zufall: gleiche Uhrzeit/Dauer/Richtung an zufälligen Tagen
    rng = np.random.default_rng(SEED)
    pool = np.arange(LOOKBACK, nd)
    rd = rng.choice(pool, size=(N_RANDOM, len(t)))
    ent = O[rd, em]
    ex = np.where(xm == 390, C[rd, 389], O[rd, np.minimum(xm, 389)])
    rand = ((d * (ex - ent) - cost) / risk).mean(axis=1)
    return dict(r=r, r2=r2, d=d, rand=rand, ndays=nd - LOOKBACK,
                first=days[LOOKBACK], last=days[-1], per_day=len(t) / max(1, nd - LOOKBACK))


def fmt(x):
    return "nan" if not np.isfinite(x) else f"{x:.2f}"


def report(label, res):
    r = res["r"]
    n = len(r)
    half = n // 2
    sd = r.std(ddof=1)
    t = r.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    h1, h2 = r[:half].mean(), r[half:].mean()
    c2 = res["r2"].mean()
    p = (np.sum(res["rand"] >= r.mean()) + 1) / (len(res["rand"]) + 1)
    lg, sh = r[res["d"] > 0], r[res["d"] < 0]
    ok = n >= 100 and t >= 2.5 and h1 > 0 and h2 > 0 and c2 > 0 and p < 0.01
    print(f"  {label}: n={n}, Quote {(r > 0).mean()*100:.0f}%, Ø {fmt(r.mean())}R, "
          f"Median {fmt(np.median(r))}R, t={t:.1f}, 1. Hälfte {fmt(h1)} / 2. Hälfte {fmt(h2)}, "
          f"Kosten x2 {fmt(c2)}R | long {fmt(lg.mean() if len(lg) else np.nan)} (n={len(lg)}) / "
          f"short {fmt(sh.mean() if len(sh) else np.nan)} (n={len(sh)}) | "
          f"Zufall {fmt(res['rand'].mean())}, p={p:.2f}  -> {'BESTANDEN' if ok else 'durchgefallen'}")
    return ok


def main():
    print("LABOR Intraday-Momentum (Noise Area), Handelszeit New York 09:30-16:00\n")
    passed = []
    for name, cfg in MARKETS.items():
        days, M = load_days(cfg["kind"], cfg["path"])
        if not days:
            print(f"{name}: keine Daten in {cfg['path']}, übersprungen\n")
            continue
        res = backtest(days, M, cfg["cost"])
        if res is None:
            print(f"{name}: keine Trades\n")
            continue
        print(f"{name}: {res['ndays']} Handelstage ({res['first']:%Y-%m-%d} bis {res['last']:%Y-%m-%d}), "
              f"Kurs zuletzt {M[-1, -1, 3]:.0f}, Kosten {cfg['cost']} Punkte, "
              f"Ø {res['per_day']:.2f} Trades pro Tag")
        if report("Hauptvariante", res):
            passed.append(name)
        print()
    if passed:
        print("ERGEBNIS: Bestanden: " + ", ".join(passed))
        print("Nächster Schritt: Papertrading, bevor echtes Geld eingesetzt wird.")
    else:
        print("ERGEBNIS: Keine Variante besteht alle Bedingungen. Nicht in den Bot.")
    print("Bedingungen: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv, "
          "besser als Zufall (p<0,01).")


if __name__ == "__main__":
    main()
