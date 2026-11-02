#!/usr/bin/env python3
"""
Labor: Index-Strategien auf mehreren Indizes (Nasdaq 100, S&P 500, Dow, DAX)
Gleiche Bestehensbedingungen wie die anderen Labore. Gezählt wird je Strategie
die Zeile "ALLE INDIZES ZUSAMMEN".

STRATEGIEN (Regeln vorab festgelegt, nicht optimieren!)
-------------------------------------------------------
Handelszeiten: US-Indizes 09:30-16:00 New York, DAX 09:00-17:30 Frankfurt.

A) ORB (Opening Range Breakout, wie Studie Zarattini/Aziz)
   Erste 5 Minuten grün -> long, rot -> short. Einstieg Eröffnung Minute 6,
   Stop am anderen Ende der 5-Minuten-Range, Ziel 10R, sonst Ausstieg zum
   Handelsschluss.
B) Momentum "Noise Area" (wie Studie Zarattini/Aziz/Barbon)
   14-Tage-Zone je Uhrzeit, Prüfung alle 30 Min. ab 30 Min. nach Eröffnung,
   Ausbruch -> Einstieg, Stop nachgezogen an max/min(Zonenrand, VWAP),
   Ausstieg spätestens zum Handelsschluss.
C) Mean Reversion (Swing, mehrere Tage)
   5 Min. vor Handelsschluss: Kurs > 200-Tage-Schnitt und RSI(2) < 25 -> long.
   Ausstieg 5 Min. vor Schluss am ersten Tag mit Kurs > 5-Tage-Schnitt,
   spätestens nach 10 Tagen. Übernachtfinanzierung 5 % p.a. eingerechnet.
   R = ATR(10).

BEDINGUNGEN: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv,
besser als Zufall (p<0,01).
Zufall: A = gleiche Tage, Richtung per Münzwurf; B = gleiche Trades an
zufälligen anderen Tagen; C = gleiche Haltedauer an zufälligen Tagen im
Aufwärtstrend.

DATEN: Nasdaq aus dem ORB-Labor (daten/USATECHIDXUSD), S&P 500, Dow, DAX aus
daten/duka/<SYMBOL> (geladen mit: python range_sweep_mss_labor.py --laden SYMBOL).

AUFRUF:  python index_labor.py
"""
import glob
import os

import numpy as np
import pandas as pd

N_RANDOM = 2000
SEED = 42
MIN_RISK_COST = 3.0
FINANCING = 0.05

INDICES = {
    "NASDAQ": dict(kind="orb", path=os.path.join("daten", "USATECHIDXUSD"), cost=1.5,
                   tz="America/New_York", open_min=570, length=390),
    "SP500": dict(kind="duka", path=os.path.join("daten", "duka", "USA500IDXUSD"), cost=0.5,
                  tz="America/New_York", open_min=570, length=390),
    "DOW": dict(kind="duka", path=os.path.join("daten", "duka", "USA30IDXUSD"), cost=2.5,
                tz="America/New_York", open_min=570, length=390),
    "DAX": dict(kind="duka", path=os.path.join("daten", "duka", "DEUIDXEUR"), cost=1.5,
                tz="Europe/Berlin", open_min=540, length=510),
}


# ------------------------------------------------------------------ Daten
def load_days(cfg):
    L = cfg["length"]
    days, mats = [], []
    for f in sorted(glob.glob(os.path.join(cfg["path"], "*.npy"))):
        a = np.load(f)
        if len(a) < 100:
            continue
        d = pd.Timestamp(os.path.basename(f)[:-4])
        if d.dayofweek >= 5:
            continue
        if cfg["kind"] == "duka":
            ts = (pd.Timestamp(d, tz="UTC") + pd.to_timedelta(a[:, 0].astype(np.int64), unit="s")
                  ).tz_convert(cfg["tz"])
            mins = (ts.hour * 60 + ts.minute - cfg["open_min"]).values
            same_day = (ts.normalize().tz_localize(None) == d)
            ok = (mins >= 0) & (mins < L) & np.asarray(same_day)
            a = np.column_stack([mins[ok], a[ok, 1:5]])
        m = np.full((L, 4), np.nan)
        if len(a):
            m[a[:, 0].astype(int)] = a[:, 1:5]
        if np.isnan(m[:, 3]).mean() > 0.2 or np.isnan(m[0, 0]):
            continue
        for k in range(1, L):
            if np.isnan(m[k, 3]):
                m[k] = m[k - 1, 3]
        days.append(d)
        mats.append(m)
    if not mats:
        return [], None
    M = np.stack(mats)
    # Nasdaq-Daten aus dem ORB-Labor sind Rohwerte (x1000), Dukascopy-Daten schon umgerechnet
    if cfg["kind"] == "orb" and np.nanmedian(M[:, :, 3]) > 500000:
        M = M / 1000
    return days, M


# ------------------------------------------------------------------ A) ORB
def strat_orb(days, M, cost, rng, n_or=5, tgt_r=10.0):
    O, H, L, C = M[..., 0], M[..., 1], M[..., 2], M[..., 3]
    out = []
    for i in range(len(days)):
        o, c = O[i, 0], C[i, n_or - 1]
        hi, lo = H[i, :n_or].max(), L[i, :n_or].min()
        if c == o:
            continue
        entry = O[i, n_or]
        res = []
        for d in (1, -1):
            stop = lo if d > 0 else hi
            risk = d * (entry - stop)
            if risk <= 0 or risk < MIN_RISK_COST * cost:
                res.append(None)
                continue
            tgt = entry + d * tgt_r * risk
            hh, ll = H[i, n_or:], L[i, n_or:]
            hs = (ll <= stop) if d > 0 else (hh >= stop)
            ht = (hh >= tgt) if d > 0 else (ll <= tgt)
            n = len(hh)
            i_s = np.argmax(hs) if hs.any() else n
            i_t = np.argmax(ht) if ht.any() else n
            ex = stop if (i_s <= i_t and i_s < n) else (tgt if i_t < n else C[i, -1])
            res.append(((d * (ex - entry)), risk))
        if res[0] is None or res[1] is None:
            continue
        sig = 1 if c > o else -1
        out.append((i, sig, res[0][0], res[0][1], res[1][0], res[1][1]))
    if len(out) < 2:
        return None
    t = np.array(out)
    sig = t[:, 1]
    gl, rl, gs, rs = t[:, 2], t[:, 3], t[:, 4], t[:, 5]
    g = np.where(sig > 0, gl, gs)
    rk = np.where(sig > 0, rl, rs)
    D = rng.random((N_RANDOM, len(t))) < 0.5
    rand = np.where(D, (gl - cost) / rl, (gs - cost) / rs).mean(axis=1)
    return dict(r=(g - cost) / rk, r2=(g - 2 * cost) / rk, d=sig, rand=rand,
                dates=np.array([days[int(i)] for i in t[:, 0]]))


# ------------------------------------------------------------------ B) Momentum
def strat_momentum(days, M, cost, rng, lookback=14):
    O, H, L, C = M[..., 0], M[..., 1], M[..., 2], M[..., 3]
    Lm = M.shape[1]
    checks = list(range(30, Lm, 30))
    move = np.abs(C / O[:, :1] - 1)
    trades = []
    for i in range(lookback, len(days)):
        if (days[i] - days[i - 1]).days > 5:
            continue
        sigma = move[i - lookback:i].mean(axis=0)
        op, pc = O[i, 0], C[i - 1, -1]
        ub, lb = max(op, pc) * (1 + sigma), min(op, pc) * (1 - sigma)
        vwap = np.cumsum((H[i] + L[i] + C[i]) / 3) / np.arange(1, Lm + 1)
        pos, entry, e_min, risk = 0, 0.0, 0, 0.0
        for k in checks:
            px = C[i, k - 1]
            if pos == 1 and px < max(ub[k - 1], vwap[k - 1]):
                trades.append((i, e_min, k, 1, entry, O[i, k], risk)); pos = 0
            elif pos == -1 and px > min(lb[k - 1], vwap[k - 1]):
                trades.append((i, e_min, k, -1, entry, O[i, k], risk)); pos = 0
            if pos == 0:
                if px > ub[k - 1]:
                    r = px - max(ub[k - 1], vwap[k - 1])
                    if r >= MIN_RISK_COST * cost:
                        pos, entry, e_min, risk = 1, O[i, k], k, r
                elif px < lb[k - 1]:
                    r = min(lb[k - 1], vwap[k - 1]) - px
                    if r >= MIN_RISK_COST * cost:
                        pos, entry, e_min, risk = -1, O[i, k], k, r
        if pos:
            trades.append((i, e_min, Lm, pos, entry, C[i, -1], risk))
    if len(trades) < 2:
        return None
    t = np.array(trades)
    di, em, xm, d = t[:, 0].astype(int), t[:, 1].astype(int), t[:, 2].astype(int), t[:, 3]
    entry, exitp, risk = t[:, 4], t[:, 5], t[:, 6]
    gross = d * (exitp - entry)
    rd = rng.choice(np.arange(lookback, len(days)), size=(N_RANDOM, len(t)))
    ent = O[rd, em]
    ex = np.where(xm == Lm, C[rd, Lm - 1], O[rd, np.minimum(xm, Lm - 1)])
    rand = ((d * (ex - ent) - cost) / risk).mean(axis=1)
    return dict(r=(gross - cost) / risk, r2=(gross - 2 * cost) / risk, d=d, rand=rand,
                dates=np.array([days[i] for i in di]))


# ------------------------------------------------------------------ C) Mean Reversion
def strat_meanrev(days, M, cost, rng, rsi_max=25.0, max_hold=10):
    Lm = M.shape[1]
    df = pd.DataFrame({"high": M[:, :, 1].max(axis=1), "low": M[:, :, 2].min(axis=1),
                       "close": M[:, -1, 3], "sig": M[:, Lm - 6, 3]}, index=pd.DatetimeIndex(days))
    s, pcl = df["sig"], df["close"].shift(1)
    sma200 = (df["close"].shift(1).rolling(199).sum() + s) / 200
    sma5 = (df["close"].shift(1).rolling(4).sum() + s) / 5
    chg = df["close"].diff()
    a = 0.5
    up = chg.clip(lower=0).shift(1).ewm(alpha=a, adjust=False).mean()
    dn = (-chg.clip(upper=0)).shift(1).ewm(alpha=a, adjust=False).mean()
    ct = s - pcl
    up_t = (1 - a) * up + a * ct.clip(lower=0)
    dn_t = (1 - a) * dn + a * (-ct.clip(upper=0))
    rsi = 100 - 100 / (1 + up_t / dn_t.replace(0, np.nan))
    tr = pd.concat([df.high - df.low, (df.high - pcl).abs(), (df.low - pcl).abs()], axis=1).max(axis=1)
    atr = tr.shift(1).rolling(10).mean()
    sig = s.values
    trend = (s > sma200).values & np.isfinite(sma200.values)
    entry_ok = trend & (rsi < rsi_max).values & np.isfinite(atr.values)
    exit_ok = (s > sma5).values
    idx = df.index
    trades = []
    i = 0
    while i < len(df) - 1:
        if entry_ok[i]:
            j = i + 1
            while j < len(df) - 1 and not exit_ok[j] and j - i < max_hold:
                j += 1
            nights = (idx[j] - idx[i]).days
            trades.append((i, j, sig[j] - sig[i], cost + FINANCING / 360 * sig[i] * nights, atr.values[i]))
            i = j + 1
        else:
            i += 1
    if len(trades) < 2:
        return None
    t = np.array(trades)
    ei, xi, gross, c, unit = t[:, 0].astype(int), t[:, 1].astype(int), t[:, 2], t[:, 3], t[:, 4]
    hold = xi - ei
    pool = np.where(trend)[0]
    pool = pool[pool < len(df) - max_hold - 1]
    if len(pool) == 0:
        return None
    rs = rng.choice(pool, size=(N_RANDOM, len(t)))
    rand = ((sig[rs + hold] - sig[rs] - c) / unit).mean(axis=1)
    return dict(r=(gross - c) / unit, r2=(gross - 2 * c) / unit, d=np.ones(len(t)), rand=rand,
                dates=np.array([days[i] for i in ei]))


# ------------------------------------------------------------------ Auswertung
def fmt(x):
    return "nan" if not np.isfinite(x) else f"{x:.2f}"


def report(label, res):
    order = np.argsort(res["dates"], kind="stable")
    r = res["r"][order]
    n = len(r)
    if n < 2:
        print(f"  {label}: zu wenig Trades")
        return False
    half = n // 2
    sd = r.std(ddof=1)
    t = r.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    h1, h2 = r[:half].mean(), r[half:].mean()
    c2 = res["r2"].mean()
    p = (np.sum(res["rand"] >= r.mean()) + 1) / (len(res["rand"]) + 1)
    d = res["d"][order]
    lg, sh = r[d > 0], r[d < 0]
    years = (res["dates"].max() - res["dates"].min()).days / 365.25
    ok = n >= 100 and t >= 2.5 and h1 > 0 and h2 > 0 and c2 > 0 and p < 0.01
    print(f"  {label}: n={n} (~{n / max(years, 0.1):.0f}/Jahr), Quote {(r > 0).mean()*100:.0f}%, "
          f"Ø {fmt(r.mean())}R, t={t:.1f}, 1. Hälfte {fmt(h1)} / 2. Hälfte {fmt(h2)}, "
          f"Kosten x2 {fmt(c2)}R | long {fmt(lg.mean() if len(lg) else np.nan)} / "
          f"short {fmt(sh.mean() if len(sh) else np.nan)} | Zufall {fmt(res['rand'].mean())}, "
          f"p={p:.2f}  -> {'BESTANDEN' if ok else 'durchgefallen'}")
    return ok


STRATS = (("A) ORB 5 min, Ziel 10R", strat_orb),
          ("B) Momentum Noise Area", strat_momentum),
          ("C) Mean Reversion RSI(2)<25", strat_meanrev))


def main():
    print("LABOR Index-Strategien: Nasdaq 100, S&P 500, Dow, DAX\n")
    data = {}
    for name, cfg in INDICES.items():
        days, M = load_days(cfg)
        if not days:
            print(f"{name}: keine Daten in {cfg['path']}, übersprungen")
            continue
        data[name] = (days, M, cfg)
        print(f"{name}: {len(days)} Handelstage ({days[0]:%Y-%m-%d} bis {days[-1]:%Y-%m-%d}), "
              f"Kurs zuletzt {M[-1, -1, 3]:.0f}, Kosten {cfg['cost']} Punkte")
    passed = []
    for label, fn in STRATS:
        print(f"\n{label}")
        pool = []
        for name, (days, M, cfg) in data.items():
            res = fn(days, M, cfg["cost"], np.random.default_rng(SEED))
            if res is None:
                print(f"  {name}: zu wenig Trades")
                continue
            report(name, res)
            pool.append(res)
        if len(pool) >= 2:
            comb = dict(r=np.concatenate([x["r"] for x in pool]),
                        r2=np.concatenate([x["r2"] for x in pool]),
                        d=np.concatenate([x["d"] for x in pool]),
                        dates=np.concatenate([x["dates"] for x in pool]),
                        rand=np.average(np.vstack([x["rand"] for x in pool]), axis=0,
                                        weights=[len(x["r"]) for x in pool]))
            if report("ALLE INDIZES ZUSAMMEN", comb):
                passed.append(label)
    print()
    if passed:
        print("ERGEBNIS: Bestanden (alle Indizes zusammen): " + "; ".join(passed))
        print("Nächster Schritt: Papertrading, bevor echtes Geld eingesetzt wird.")
    else:
        print("ERGEBNIS: Keine Strategie besteht alle Bedingungen. Nicht in den Bot.")
    print("Bedingungen: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv, "
          "besser als Zufall (p<0,01).")


if __name__ == "__main__":
    main()
