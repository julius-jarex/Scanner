#!/usr/bin/env python3
"""
Labor: Trend + Retest in die Imbalance ("Buy Volume back to Buy Volume")
Märkte: EURUSD, GBPUSD, Gold (XAUUSD), Filter mit dem Dollar-Index (DXY).
Gleiche Bestehensbedingungen wie die anderen Labore.

REGELN (so besprochen, vorab festgelegt, nicht nachträglich optimieren!)
------------------------------------------------------------------------
Long (Short spiegelbildlich):
 1. Trend 4h UND 1h: höhere Hochs und höhere Tiefs (letzte zwei bestätigten
    Swing-Hochs und Swing-Tiefs steigen; Swing = Hoch/Tief mit je 2 Kerzen
    links und rechts, erst nach 2 Kerzen bestätigt -> kein Blick in die Zukunft).
 2. DXY-Filter: Dollar-Index im 4h-Abwärtstrend (für Long EURUSD/GBPUSD/Gold).
 3. Imbalance (1h): drei Kerzen K1, K2, K3. K2 bullisch, zwischen dem Körper
    von K1 und dem Körper von K3 bleibt eine Lücke = Imbalance-Zone.
    Mindestgröße 0,1 x ATR(14).
 4. Einstieg (Limit-Order, Hauptvariante):
      - reicht der obere Docht von K1 mindestens 1/3 in die Zone -> an die
        Dochtspitze ("Buy Volume back to Buy Volume"),
      - sonst an die 50-%-Linie der Zone.
    Die Order darf höchstens 2 x ATR vom aktuellen Kurs entfernt sein und
    gilt 24 Stunden.
 5. Stop: letztes bestätigtes 1h-Swing-Tief unter dem Einstieg.
 6. Ziel: letztes Hoch = höchster Kurs seit K1 bis zum Einstieg (steigt der Kurs
    vor dem Rücksetzer weiter, wandert das Ziel mit). Nur wenn mindestens 1R.
 7. Ausstieg: Ziel oder Stop, spätestens nach 48 Stunden zum Marktpreis.
    Trades dürfen über Nacht laufen (Swap-Kosten nicht eingerechnet).
 8. Stop und Ziel in derselben Kerze -> Stop zählt. In der Einstiegskerze
    zählt nur der Stop (konservativ). Eine Position pro Markt gleichzeitig.

Zum Vergleich (zählen NICHT fürs Bestehen): immer Docht, immer 50 %,
Hauptvariante ohne DXY-Filter.

BEDINGUNGEN: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv,
besser als Zufall (p<0,01). Gezählt wird die Hauptvariante, alle Märkte
zusammen. Zufall = gleich viele Trades zu zufälligen Zeitpunkten, gleiche
Richtung, gleicher Stop- und Zielabstand (2000 Durchläufe).

DATEN: daten/duka/<SYMBOL>/ (Minutenkerzen, geladen mit
  python range_sweep_mss_labor.py --laden SYMBOL).
Dollar-Index: daten/duka/DOLLARIDXUSD. Fehlt er, wird ein Näherungs-DXY aus
EURUSD, GBPUSD (und USDJPY, falls vorhanden) berechnet.

AUFRUF:  python imbalance_labor.py
"""
import glob
import os

import numpy as np
import pandas as pd

MARKETS = {"EURUSD": 0.00008, "GBPUSD": 0.00012, "XAUUSD": 0.20}   # Kosten je Trade (Kauf+Verkauf)
DXY_SYM = "DOLLARIDXUSD"
SWING_K = 2
ATR_N = 14
MIN_ZONE_ATR = 0.1
MAX_DIST_ATR = 2.0
ORDER_BARS = 24
MAX_HOLD = 48
MIN_RR = 1.0
MIN_RISK_COST = 3.0
WICK_SHARE = 1 / 3
N_RANDOM = 2000
SEED = 42


# ------------------------------------------------------------------ Daten
def load_minutes(sym):
    parts = []
    for f in sorted(glob.glob(os.path.join("daten", "duka", sym, "*.npy"))):
        a = np.load(f)
        if len(a) == 0:
            continue
        day = pd.Timestamp(os.path.basename(f)[:-4], tz="UTC")
        parts.append(pd.DataFrame(a[:, 1:5], columns=["open", "high", "low", "close"],
                                  index=day + pd.to_timedelta(a[:, 0].astype(np.int64), unit="s")))
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated()]


def resample(m1, rule):
    b = m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    b["end"] = b.index + pd.Timedelta(rule)
    return b


def atr(b, n=ATR_N):
    pc = b.close.shift(1)
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy()


def swings(H, L, k=SWING_K):
    n = len(H)
    sh = np.zeros(n, bool)
    sl = np.zeros(n, bool)
    for i in range(k, n - k):
        if H[i] > H[i - k:i].max() and H[i] >= H[i + 1:i + k + 1].max():
            sh[i] = True
        if L[i] < L[i - k:i].min() and L[i] <= L[i + 1:i + k + 1].min():
            sl[i] = True
    return sh, sl


def trend(H, L, k=SWING_K):
    """Trend je Kerze (nach ihrem Schluss): 1 = HH+HL, -1 = LH+LL, 0 = unklar. Nur bestätigte Swings."""
    sh, sl = swings(H, L, k)
    n = len(H)
    out = np.zeros(n, int)
    highs, lows = [], []
    for t in range(n):
        j = t - k
        if j >= 0:
            if sh[j]:
                highs.append(H[j])
            if sl[j]:
                lows.append(L[j])
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                out[t] = 1
            elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                out[t] = -1
    return out


def map_trend(h1_end, htf_end, htf_trend):
    """Trend der höheren Zeiteinheit, der zum Schluss der 1h-Kerze bekannt war."""
    idx = np.searchsorted(htf_end.values, h1_end.values, side="right") - 1
    out = np.where(idx >= 0, htf_trend[np.clip(idx, 0, None)], 0)
    return out


def dxy_h4():
    m = load_minutes(DXY_SYM)
    if m is not None and len(m) > 10000:
        b = resample(m, "4h")
        return b, trend(b.high.values, b.low.values), "Dollar-Index (Dukascopy)"
    # Näherung aus den Währungspaaren (Gewichte der DXY-Formel)
    comps = {"EURUSD": -0.576, "GBPUSD": -0.119, "USDJPY": 0.136}
    logs = []
    for s, w in comps.items():
        mm = load_minutes(s)
        if mm is not None:
            logs.append(w * np.log(resample(mm, "4h").close))
    if not logs:
        return None, None, "kein Dollar-Index verfügbar"
    lp = pd.concat(logs, axis=1).dropna().sum(axis=1)
    b = pd.DataFrame({"close": np.exp(lp)})
    b["high"] = b["low"] = b["close"]
    b["end"] = b.index + pd.Timedelta("4h")
    return b, trend(b.high.values, b.low.values), "Näherungs-DXY aus EURUSD/GBPUSD" + ("/USDJPY" if len(logs) == 3 else "")


# ------------------------------------------------------------------ Setups
def find_trades(h1, t1, t4, tdxy, cost, entry_mode, use_dxy):
    O, H, L, C = (h1[c].to_numpy() for c in ("open", "high", "low", "close"))
    A = atr(h1)
    sh, sl = swings(H, L)
    n = len(C)
    trades = []
    busy_until = -1
    i = 2
    while i < n - 1:
        if i <= busy_until or not np.isfinite(A[i]):
            i += 1
            continue
        found = None
        for d in (1, -1):
            if t1[i] != d or t4[i] != d or (use_dxy and tdxy[i] != -d):
                continue
            k1, k2, k3 = i - 2, i - 1, i
            if d * (C[k2] - O[k2]) <= 0:
                continue
            if d > 0:
                zb, zt = max(O[k1], C[k1]), min(O[k3], C[k3])
                wick = H[k1]
            else:
                zb, zt = max(O[k3], C[k3]), min(O[k1], C[k1])
                wick = L[k1]
            if zt - zb < MIN_ZONE_ATR * A[i]:
                continue
            mid = (zb + zt) / 2
            depth = (wick - zb) / (zt - zb) if d > 0 else (zt - wick) / (zt - zb)
            wick_lvl = min(max(wick, zb), zt)
            if entry_mode == "docht":
                entry = wick_lvl
            elif entry_mode == "mitte":
                entry = mid
            else:
                entry = wick_lvl if depth >= WICK_SHARE else mid
            if abs(entry - C[i]) > MAX_DIST_ATR * A[i]:
                continue
            # Stop: letztes bestätigtes Swing-Tief (bzw. -Hoch) jenseits des Einstiegs
            stop = None
            for j in range(i - SWING_K, max(0, i - 200), -1):
                if d > 0 and sl[j] and L[j] < entry:
                    stop = L[j]; break
                if d < 0 and sh[j] and H[j] > entry:
                    stop = H[j]; break
            if stop is None:
                continue
            risk = d * (entry - stop)
            if risk <= 0 or risk < MIN_RISK_COST * cost:
                continue
            target = H[k1:i + 1].max() if d > 0 else L[k1:i + 1].min()
            found = (d, entry, stop, risk, target)
            break
        if not found:
            i += 1
            continue
        d, entry, stop, risk, target = found
        filled = None
        for k in range(i + 1, min(i + 1 + ORDER_BARS, n)):
            # Ziel wandert mit, solange der Kurs vor dem Rücksetzer noch höher steigt
            if (d > 0 and L[k] <= entry) or (d < 0 and H[k] >= entry):
                filled = k
                break
            target = max(target, H[k]) if d > 0 else min(target, L[k])
        if filled is None:
            i += 1
            continue
        if d * (target - entry) < MIN_RR * risk:
            busy_until = filled
            i = filled + 1
            continue
        trades.append(dict(k=filled, d=d, entry=entry, stop=stop, risk=risk, target=target, time=h1.index[filled]))
        # Position belegt den Markt bis zum Ausstieg
        busy_until = filled + MAX_HOLD
        i = filled + 1
    return trades


def simulate(O, H, L, C, k0, d, entry, stop, tgt, first_bar_stop):
    """Vektorisiert. Brutto-PnL in Preiseinheiten und Ausstiegskerze."""
    n = len(C)
    k0 = np.asarray(k0); d = np.asarray(d, float)
    entry, stop, tgt = (np.asarray(x, float) for x in (entry, stop, tgt))
    pnl = np.full(k0.shape, np.nan)
    done = np.zeros(k0.shape, bool)
    xk = np.minimum(k0 + MAX_HOLD, n - 1)
    long = d > 0
    if first_bar_stop:
        hs = np.where(long, L[k0] <= stop, H[k0] >= stop)
        pnl = np.where(hs, d * (stop - entry), pnl); done |= hs; xk = np.where(hs, k0, xk)
    for s in range(1, MAX_HOLD + 1):
        j = k0 + s
        act = (j < n) & ~done
        if not act.any():
            break
        jj = np.minimum(j, n - 1)
        o, h, l = O[jj], H[jj], L[jj]
        hs = act & np.where(long, l <= stop, h >= stop)
        ht = act & ~hs & np.where(long, h >= tgt, l <= tgt)
        sf = np.where(long, np.minimum(o, stop), np.maximum(o, stop))
        pnl = np.where(hs, d * (sf - entry), pnl)
        pnl = np.where(ht, d * (tgt - entry), pnl)
        xk = np.where(hs | ht, jj, xk)
        done |= hs | ht
    pnl = np.where(done, pnl, d * (C[xk] - entry))
    return pnl, xk


def evaluate(h1, trades, cost, rng):
    if len(trades) < 2:
        return None
    O, H, L, C = (h1[c].to_numpy() for c in ("open", "high", "low", "close"))
    t = pd.DataFrame(trades)
    g, _ = simulate(O, H, L, C, t.k.values, t.d.values, t.entry.values, t.stop.values, t.target.values, True)
    r = (g - cost) / t.risk.values
    r2 = (g - 2 * cost) / t.risk.values
    m = len(t)
    rk = rng.integers(ATR_N + 5, len(C) - MAX_HOLD - 1, size=(N_RANDOM, m)).ravel()
    D = np.tile(t.d.values, N_RANDOM)
    R = np.tile(t.risk.values, N_RANDOM)
    dist = np.tile((t.d * (t.target - t.entry)).values, N_RANDOM)
    e = C[rk]
    gr, _ = simulate(O, H, L, C, rk, D, e, e - D * R, e + D * dist, False)
    rand = ((gr - cost) / R).reshape(N_RANDOM, m).mean(axis=1)
    return dict(r=r, r2=r2, d=t.d.values, rand=rand, dates=t.time.values,
                rr=(t.d * (t.target - t.entry)).values / t.risk.values)


def fmt(x):
    return "nan" if not np.isfinite(x) else f"{x:.2f}"


def report(label, res):
    order = np.argsort(res["dates"], kind="stable")
    r = res["r"][order]
    n = len(r)
    half = n // 2
    sd = r.std(ddof=1)
    t = r.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    h1, h2 = r[:half].mean(), r[half:].mean()
    c2 = res["r2"].mean()
    p = (np.sum(res["rand"] >= r.mean()) + 1) / (len(res["rand"]) + 1)
    d = res["d"][order]
    lg, sh = r[d > 0], r[d < 0]
    dates = pd.to_datetime(res["dates"])
    years = max((dates.max() - dates.min()).days / 365.25, 0.1)
    ok = n >= 100 and t >= 2.5 and h1 > 0 and h2 > 0 and c2 > 0 and p < 0.01
    print(f"  {label}: n={n} (~{n / years:.0f}/Jahr), Quote {(r > 0).mean()*100:.0f}%, "
          f"Chance/Risiko 1:{np.median(res['rr']):.2f} (Median), Ø {fmt(r.mean())}R, t={t:.1f}, "
          f"1. Hälfte {fmt(h1)} / 2. Hälfte {fmt(h2)}, Kosten x2 {fmt(c2)}R | "
          f"long {fmt(lg.mean() if len(lg) else np.nan)} / short {fmt(sh.mean() if len(sh) else np.nan)} | "
          f"Zufall {fmt(res['rand'].mean())}, p={p:.2f}  -> {'BESTANDEN' if ok else 'durchgefallen'}")
    return ok


VARIANTS = (("HAUPT: Docht ab 1/3 sonst 50 %, mit DXY", "regel", True),
            ("Vergleich: immer Docht, mit DXY", "docht", True),
            ("Vergleich: immer 50 %, mit DXY", "mitte", True),
            ("Vergleich: Hauptregel OHNE DXY-Filter", "regel", False))


def main():
    print("LABOR Trend + Retest in die Imbalance (1h-Einstieg, 4h-Trend, DXY-Filter)\n")
    db, dt, dname = dxy_h4()
    print(f"DXY-Filter: {dname}" + ("" if db is None else
          f" ({db.index[0]:%Y-%m-%d} bis {db.index[-1]:%Y-%m-%d})"))
    if db is not None and "Näherung" in dname:
        print("  Hinweis: Der Näherungs-DXY enthält EURUSD/GBPUSD selbst, der Filter ist dort also teilweise doppelt.")
    print()
    pooled = {v[0]: [] for v in VARIANTS}
    for sym, cost in MARKETS.items():
        m1 = load_minutes(sym)
        if m1 is None or len(m1) < 10000:
            print(f"{sym}: keine Daten in daten/duka/{sym}, übersprungen\n")
            continue
        h1, h4 = resample(m1, "1h"), resample(m1, "4h")
        t1 = trend(h1.high.values, h1.low.values)
        t4 = map_trend(h1.end, h4.end, trend(h4.high.values, h4.low.values))
        tdxy = map_trend(h1.end, db.end, dt) if db is not None else np.zeros(len(h1), int)
        days = m1.index.normalize().nunique()
        print(f"{sym}: {days} Handelstage ({m1.index[0]:%Y-%m-%d} bis {m1.index[-1]:%Y-%m-%d}), "
              f"Kurs zuletzt {m1.close.iloc[-1]:.5g}, Kosten {cost}")
        for label, mode, use_dxy in VARIANTS:
            if use_dxy and db is None:
                print(f"  {label}: kein DXY, übersprungen")
                continue
            tr = find_trades(h1, t1, t4, tdxy, cost, mode, use_dxy)
            res = evaluate(h1, tr, cost, np.random.default_rng(SEED))
            if res is None:
                print(f"  {label}: zu wenig Trades ({len(tr)})")
                continue
            report(label, res)
            pooled[label].append(res)
        print()
    print("ALLE MÄRKTE ZUSAMMEN")
    passed = False
    for label, _, _ in VARIANTS:
        lst = pooled[label]
        if not lst:
            continue
        comb = dict(r=np.concatenate([x["r"] for x in lst]), r2=np.concatenate([x["r2"] for x in lst]),
                    d=np.concatenate([x["d"] for x in lst]), dates=np.concatenate([x["dates"] for x in lst]),
                    rr=np.concatenate([x["rr"] for x in lst]),
                    rand=np.average(np.vstack([x["rand"] for x in lst]), axis=0, weights=[len(x["r"]) for x in lst]))
        ok = report(label, comb)
        if label.startswith("HAUPT"):
            passed = ok
    print()
    if passed:
        print("ERGEBNIS: Hauptvariante BESTANDEN (alle Märkte zusammen). Nächster Schritt: Papertrading.")
    else:
        print("ERGEBNIS: Hauptvariante besteht nicht alle Bedingungen. Nicht in den Bot.")
    print("Gezählt wird nur die Hauptvariante. Bedingungen: n>=100, t>=2,5, beide Hälften positiv, "
          "Kosten x2 positiv, besser als Zufall (p<0,01).")


if __name__ == "__main__":
    main()
