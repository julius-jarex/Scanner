#!/usr/bin/env python3
"""
Labor: Range -> Liquidity Sweep -> MSS -> FVG-Rücksetzer -> Ziel Gegenseite
Märkte: Nasdaq 100 (Daten aus dem ORB-Labor), BTC, ETH, SOL, XRP, BNB (Binance),
Gold, EURUSD, GBPUSD, USDJPY, S&P 500, DAX (Dukascopy).
Gleiche Bestehensbedingungen wie die anderen Labore.

REGELN (vorab festgelegt, nicht nachträglich optimieren!)
---------------------------------------------------------
Long (Short spiegelbildlich), Zeiteinheiten 5 min und 15 min:
 1. Range: die letzten 12 Kerzen. Gilt nur als Seitwärtsphase, wenn
    Range-Höhe <= 3,5 x durchschnittliche Kerzengröße dieser 12 Kerzen.
 2. Sweep: eine Kerze sticht unter das Range-Tief.
 3. MSS: innerhalb von 8 Kerzen schließt eine Kerze über dem höchsten Hoch
    der 3 Kerzen vor dem Sweep (bis einschließlich Sweep-Kerze).
    Das tiefste Tief bis dahin ist das Sweep-Tief.
 4. FVG: bullisches FVG (Tief[b] > Hoch[b-2]) zwischen Sweep-Tief und MSS,
    das zuletzt entstandene zählt.
 5. Einstieg: Limit-Order an der FVG-Oberkante, gültig 12 Kerzen.
    Wird vorher das Range-Hoch erreicht -> Setup verworfen.
 6. Stop: Sweep-Tief. Risiko mindestens 3x Kosten.
 7. Ziele: Z1 = Range-Hoch (Gegenseite, wie im Bild), Z2 = fest 2R.
    Sonst Ausstieg nach 6 Stunden bzw. Nasdaq zum Handelsschluss 16:00 NY.
 8. Stop und Ziel in derselben Kerze -> Stop zählt. In der Einstiegskerze
    zählt nur der Stop (konservativ).
 Nasdaq: nur Handelszeit 09:30-16:00 New York, kein Trade über Nacht.
 Andere Märkte: Sweeps nur 07-20 UTC.

BEDINGUNGEN: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv,
besser als Zufall (p<0,01). Zufall = gleiche Anzahl Trades zu zufälligen
Zeitpunkten, gleiche Richtung, gleicher Stop- und Zielabstand.

AUFRUF:  python range_sweep_mss_labor.py
         python range_sweep_mss_labor.py --laden EURUSD   (nur Daten laden)
"""
import glob
import io
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

RANGE_BARS = 12
RANGE_MAX_RATIO = 3.5
MSS_WINDOW = 8
FILL_WINDOW = 12
MIN_RISK_COST = 3.0
Z2_R = 2.0
MAX_HOLD_HOURS = 6
TIMEFRAMES = (5, 15)
NASDAQ_DIR = os.path.join("daten", "USATECHIDXUSD")
NASDAQ_COST = 1.5
CRYPTO = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT")
# Dukascopy-Märkte: Round-Turn-Kosten in Preiseinheiten (an Broker anpassen)
DUKA = {"XAUUSD": 0.20, "EURUSD": 0.00008, "GBPUSD": 0.00012, "USDJPY": 0.012,
        "USA500IDXUSD": 0.5, "DEUIDXEUR": 1.5}
DUKA_START = "2023-01-01"
DUKA_THREADS = 6
DUKA_BUDGET_MIN = 50
SWEEP_HOURS_UTC = (7, 20)
CRYPTO_REL_COST = 0.0006
CRYPTO_START = "2023-01-01"
CRYPTO_HOURS = (7, 20)
N_RANDOM = 2000
SEED = 42


# ------------------------------------------------------------------ Daten
def _get(url, tries=3):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
        except Exception:
            pass
        time.sleep(1 + a)
    return None


def _duka_div(sym):
    return 1000 if (sym.startswith("XA") or sym.endswith("JPY") or "IDX" in sym) else 100000


def _duka_day(sym, day, folder):
    import lzma
    url = (f"https://datafeed.dukascopy.com/datafeed/{sym}/{day.year}/"
           f"{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5")
    raw = _get(url)
    if raw is None:
        return False
    arr = np.empty((0, 5))
    if raw:
        rec = np.frombuffer(lzma.decompress(raw), dtype=np.dtype(
            [("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")]))
        rec = rec[rec["v"] > 0]
        dv = _duka_div(sym)
        arr = np.column_stack([rec["t"], rec["o"] / dv, rec["h"] / dv,
                               rec["l"] / dv, rec["c"] / dv]).astype(float)
    np.save(os.path.join(folder, f"{day}.npy"), arr)
    return True


def duka_folder(sym):
    return os.path.join("daten", "duka", sym)


def duka_days(sym):
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    return [d.date() for d in pd.date_range(DUKA_START, end, freq="D") if d.dayofweek != 5]


def duka_download(sym):
    from concurrent.futures import as_completed
    folder = duka_folder(sym)
    os.makedirs(folder, exist_ok=True)
    days = duka_days(sym)
    todo = [d for d in days if not os.path.exists(os.path.join(folder, f"{d}.npy"))]
    print(f"{sym}: {len(days) - len(todo)}/{len(days)} Tage vorhanden, lade {len(todo)} ...", flush=True)
    t0, failed = time.time(), 0
    with ThreadPoolExecutor(DUKA_THREADS) as ex:
        futs = [ex.submit(_duka_day, sym, d, folder) for d in todo]
        for i, f in enumerate(as_completed(futs), 1):
            if not f.result():
                failed += 1
            if i % 50 == 0 or i == len(todo):
                print(f"   {i}/{len(todo)} ({time.time()-t0:.0f}s), {failed} fehlgeschlagen", flush=True)
            if time.time() - t0 > DUKA_BUDGET_MIN * 60:
                print("   Zeitlimit erreicht, Rest beim nächsten Lauf.", flush=True)
                for g in futs:
                    g.cancel()
                break
    have = sum(os.path.exists(os.path.join(folder, f"{d}.npy")) for d in days)
    print(f"{sym}: jetzt {have}/{len(days)} Tage vorhanden", flush=True)


def duka_minutes(sym):
    folder = duka_folder(sym)
    days = duka_days(sym)
    parts = []
    for d in days:
        f = os.path.join(folder, f"{d}.npy")
        if os.path.exists(f):
            a = np.load(f)
            if len(a):
                parts.append(pd.DataFrame(a[:, 1:], columns=["open", "high", "low", "close"],
                                          index=pd.Timestamp(d, tz="UTC")
                                          + pd.to_timedelta(a[:, 0].astype(np.int64), unit="s")))
    have = sum(os.path.exists(os.path.join(folder, f"{d}.npy")) for d in days)
    if not parts:
        return None, have, len(days)
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated()], have, len(days)


def _binance_month(sym, ym):
    raw = _get(f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/{sym}-1m-{ym}.zip")
    if not raw:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not pd.api.types.is_numeric_dtype(df[0]):
        df = df.iloc[1:].astype({0: np.int64})
    t = df[0].astype(np.int64)
    unit = "us" if t.iloc[0] > 1e14 else "ms"
    return pd.DataFrame({"time": pd.to_datetime(t, unit=unit, utc=True),
                         "open": df[1].astype(float), "high": df[2].astype(float),
                         "low": df[3].astype(float), "close": df[4].astype(float)})


def crypto_minutes(sym):
    path = os.path.join("daten", f"{sym}_1m.parquet.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    end = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=1)
    months = list(pd.period_range(CRYPTO_START, end, freq="M").strftime("%Y-%m"))
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda m: _binance_month(sym, m), months) if p is not None and len(p)]
    if not parts:
        return None
    df = pd.concat(parts).sort_values("time").drop_duplicates("time").set_index("time")
    os.makedirs("daten", exist_ok=True)
    df.to_pickle(path)
    return df


class Market:
    """Kerzen als flache Arrays + Segment-Ende je Kerze (kein Trade über Segmentgrenzen)."""
    def __init__(self, name, t, o, h, l, c, seg_end, allowed, cost_fn):
        self.name, self.t = name, t
        self.O, self.H, self.L, self.C = o, h, l, c
        self.seg_end, self.allowed, self.cost_fn = seg_end, allowed, cost_fn
        self.seg_start = np.zeros(len(c), int)
        s = 0
        for i in range(len(c)):
            if i > 0 and seg_end[i - 1] == i - 1:
                s = i
            self.seg_start[i] = s


def nasdaq_market(tf):
    files = sorted(glob.glob(os.path.join(NASDAQ_DIR, "*.npy")))
    rows, times = [], []
    for f in files:
        a = np.load(f)
        if len(a) < 200:
            continue
        day = pd.Timestamp(os.path.basename(f)[:-4])
        b = (a[:, 0] // tf).astype(int)
        nb = 390 // tf
        o = np.full(nb, np.nan); h = np.full(nb, -np.inf); l = np.full(nb, np.inf); c = np.full(nb, np.nan)
        for k in range(len(a)):           # Minuten sind sortiert
            j = b[k]
            if np.isnan(o[j]):
                o[j] = a[k, 1]
            h[j] = max(h[j], a[k, 2]); l[j] = min(l[j], a[k, 3]); c[j] = a[k, 4]
        ok = ~np.isnan(o)
        if ok.sum() < nb * 0.8:
            continue
        rows.append(np.column_stack([o[ok], h[ok], l[ok], c[ok]]))
        times.append(day + pd.Timedelta(hours=9, minutes=30)
                     + pd.to_timedelta(np.where(ok)[0] * tf, unit="m"))
    if not rows:
        return None, 0
    X = np.vstack(rows)
    med = np.median(X[:, 3])
    for div in (1000, 100, 10, 1, 10000, 100000):
        if 3000 <= med / div <= 60000:
            X = X / div
            break
    seg_end, pos = np.empty(len(X), int), 0
    for r in rows:
        seg_end[pos:pos + len(r)] = pos + len(r) - 1
        pos += len(r)
    t = pd.DatetimeIndex(np.concatenate([np.asarray(x) for x in times]))
    return Market("NASDAQ", t, X[:, 0], X[:, 1], X[:, 2], X[:, 3], seg_end,
                  np.ones(len(X), bool), lambda p: NASDAQ_COST), len(rows)


def minute_market(name, m1, tf, cost_fn):
    """Aus Minuten-Daten: Kerzen, Segmente bei Lücken > 3 Kerzen (Wochenende etc.)."""
    bars = m1.resample(f"{tf}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    n = len(bars)
    gap = np.diff(bars.index.values).astype("timedelta64[m]").astype(int) > 3 * tf
    seg_end = np.empty(n, int)
    ends = np.r_[np.where(gap)[0], n - 1]
    start = 0
    for e in ends:
        seg_end[start:e + 1] = e
        start = e + 1
    hrs = bars.index.hour
    return Market(name, bars.index, bars.open.values, bars.high.values, bars.low.values,
                  bars.close.values, seg_end,
                  (hrs >= SWEEP_HOURS_UTC[0]) & (hrs < SWEEP_HOURS_UTC[1]), cost_fn)


# ------------------------------------------------------------------ Setups
def find_setups(m):
    O, H, L, C = m.O, m.H, m.L, m.C
    n = len(C)
    trades = []
    i = 0
    while i < n:
        s = m.seg_start[i]
        e = m.seg_end[i]
        if i - RANGE_BARS < s or i + 2 > e or not m.allowed[i]:
            i += 1
            continue
        w = slice(i - RANGE_BARS, i)
        r_hi, r_lo = H[w].max(), L[w].min()
        avg = (H[w] - L[w]).mean()
        if avg <= 0 or (r_hi - r_lo) > RANGE_MAX_RATIO * avg:
            i += 1
            continue
        nxt = i + 1
        for d in (1, -1):
            if not ((d == 1 and L[i] < r_lo) or (d == -1 and H[i] > r_hi)):
                continue
            lo3 = max(s, i - 3)
            mss = H[lo3:i + 1].max() if d == 1 else L[lo3:i + 1].min()
            ext, ext_i, mss_i = (L[i] if d == 1 else H[i]), i, None
            for j in range(i + 1, min(i + MSS_WINDOW, e) + 1):
                if (d == 1 and L[j] < ext) or (d == -1 and H[j] > ext):
                    ext, ext_i = (L[j] if d == 1 else H[j]), j
                if (d == 1 and C[j] > mss) or (d == -1 and C[j] < mss):
                    mss_i = j
                    break
            if mss_i is None:
                continue
            fvg = None
            for b in range(max(ext_i + 2, s + 2), mss_i + 1):
                if d == 1 and L[b] > H[b - 2]:
                    fvg = L[b]
                if d == -1 and H[b] < L[b - 2]:
                    fvg = H[b]
            nxt = mss_i + 1
            if fvg is None:
                break
            entry, stop = fvg, ext
            risk = d * (entry - stop)
            cost = m.cost_fn(entry)
            z1 = r_hi if d == 1 else r_lo
            if risk <= 0 or risk < MIN_RISK_COST * cost or d * (z1 - entry) <= 0:
                break
            fill = None
            for k in range(mss_i + 1, min(mss_i + FILL_WINDOW, e) + 1):
                if (d == 1 and L[k] <= entry) or (d == -1 and H[k] >= entry):
                    fill = k
                    break
                if (d == 1 and H[k] >= z1) or (d == -1 and L[k] <= z1):
                    break
            if fill is None:
                break
            trades.append(dict(idx=fill, d=d, entry=entry, stop=stop, risk=risk, cost=cost,
                               z1=z1, z2=entry + d * Z2_R * risk, time=m.t[fill]))
            nxt = fill + 1
            break
        i = max(nxt, i + 1)
    return pd.DataFrame(trades)


def simulate(m, idx, d, entry, stop, tgt, maxhold, stop_in_first=False):
    """Vektorisiert. Brutto-PnL in Preiseinheiten. Ab Kerze idx+1 (bzw. idx bei stop_in_first)."""
    O, H, L, C = m.O, m.H, m.L, m.C
    idx = np.asarray(idx); d = np.asarray(d, float)
    entry, stop, tgt = (np.asarray(x, float) for x in (entry, stop, tgt))
    last = np.minimum(idx + maxhold, m.seg_end[idx])
    pnl = np.full(idx.shape, np.nan)
    done = np.zeros(idx.shape, bool)
    long = d > 0
    if stop_in_first:
        hs = np.where(long, L[idx] <= stop, H[idx] >= stop)
        pnl = np.where(hs, d * (stop - entry), pnl)
        done |= hs
    for k in range(1, maxhold + 1):
        j = idx + k
        valid = (j <= last) & ~done
        if not valid.any():
            break
        jj = np.minimum(j, len(C) - 1)
        o, h, l = O[jj], H[jj], L[jj]
        hs = valid & np.where(long, l <= stop, h >= stop)
        ht = valid & ~hs & np.where(long, h >= tgt, l <= tgt)
        sfill = np.where(long, np.minimum(o, stop), np.maximum(o, stop))
        pnl = np.where(hs, d * (sfill - entry), pnl)
        pnl = np.where(ht, d * (tgt - entry), pnl)
        done |= hs | ht
    pnl = np.where(done, pnl, d * (C[last] - entry))
    return pnl


def evaluate(m, tr, key, maxhold, rng):
    tr = tr.reset_index(drop=True)
    gross = simulate(m, tr.idx.values, tr.d.values, tr.entry.values, tr.stop.values,
                     tr[key].values, maxhold, stop_in_first=True)
    r = (gross - tr.cost.values) / tr.risk.values
    r2 = (gross - 2 * tr.cost.values) / tr.risk.values
    rr = (tr.d * (tr[key] - tr.entry)).values / tr.risk.values
    pool = np.where(m.allowed & (np.arange(len(m.C)) < m.seg_end - 2))[0]
    k = len(tr)
    ridx = rng.choice(pool, size=(N_RANDOM, k))
    D = np.broadcast_to(tr.d.values, (N_RANDOM, k)).ravel()
    risk = np.broadcast_to(tr.risk.values, (N_RANDOM, k)).ravel()
    rew = np.broadcast_to((tr.d * (tr[key] - tr.entry)).values, (N_RANDOM, k)).ravel()
    cost = np.broadcast_to(tr.cost.values, (N_RANDOM, k)).ravel()
    e = m.C[ridx.ravel()]
    g = simulate(m, ridx.ravel(), D, e, e - D * risk, e + D * rew, maxhold)
    rand = ((g - cost) / risk).reshape(N_RANDOM, k).mean(axis=1)
    return dict(r=r, r2=r2, rr=rr, d=tr.d.values, times=pd.Series(tr.time.values), rand=rand)


def fmt(x):
    return "nan" if not np.isfinite(x) else f"{x:.2f}"


def report(label, res):
    r = res["r"]
    n = len(r)
    if n < 2:
        print(f"  {label}: n={n} (zu wenig)")
        return False
    rs = r[np.argsort(res["times"].values, kind="stable")]
    half = n // 2
    sd = r.std(ddof=1)
    t = r.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    h1, h2 = rs[:half].mean(), rs[half:].mean()
    c2 = res["r2"].mean()
    p = (np.sum(res["rand"] >= r.mean()) + 1) / (len(res["rand"]) + 1)
    lg, sh = r[res["d"] > 0], r[res["d"] < 0]
    ok = n >= 100 and t >= 2.5 and h1 > 0 and h2 > 0 and c2 > 0 and p < 0.01
    print(f"  {label}: n={n}, Quote {(r > 0).mean()*100:.0f}%, Ø Chance/Risiko 1:{np.median(res['rr']):.2f} "
          f"(Median), Ø {fmt(r.mean())}R, t={t:.1f}, 1. Hälfte {fmt(h1)} / 2. Hälfte {fmt(h2)}, "
          f"Kosten x2 {fmt(c2)}R | long {fmt(lg.mean() if len(lg) else np.nan)} / "
          f"short {fmt(sh.mean() if len(sh) else np.nan)} | Zufall {fmt(res['rand'].mean())}, "
          f"p={p:.2f}  -> {'BESTANDEN' if ok else 'durchgefallen'}")
    return ok


def main():
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--laden":
        duka_download(sys.argv[2])
        return
    rng = np.random.default_rng(SEED)
    print("LABOR Range -> Sweep -> MSS -> FVG-Rücksetzer\n")
    sources = {}      # name -> (minuten-df, kostenfunktion)
    for sym in CRYPTO:
        df = crypto_minutes(sym)
        if df is None:
            print(f"{sym}: keine Daten, übersprungen")
        else:
            sources[sym] = (df, lambda p: CRYPTO_REL_COST * p)
            print(f"{sym}: {len(df)} Minuten-Kerzen, letzter Kurs {df.close.iloc[-1]:.5g}")
    incomplete = []
    for sym, cost in DUKA.items():
        df, have, total = duka_minutes(sym)
        if df is None:
            print(f"{sym}: keine Daten, übersprungen")
            continue
        flag = ""
        if have < 0.98 * total:
            flag = f"  ACHTUNG: nur {have}/{total} Tage, Ergebnis vorläufig"
            incomplete.append(sym)
        print(f"{sym}: {have}/{total} Tage, letzter Kurs {df.close.iloc[-1]:.5g}{flag}")
        sources[sym] = (df, lambda p, c=cost: c)
    pooled, passed, tests = {}, [], 0
    for tf in TIMEFRAMES:
        markets = []
        nm, ndays = nasdaq_market(tf)
        if nm is None:
            print("NASDAQ: keine Daten im Ordner daten/USATECHIDXUSD (erst ORB-Labor laufen lassen)")
        else:
            markets.append((nm, f"{ndays} Handelstage"))
        for sym, (df, cfn) in sources.items():
            markets.append((minute_market(sym, df, tf, cfn), ""))
        for m, info in markets:
            print(f"\n{m.name} {tf}m: ", end="", flush=True)
            tr = find_setups(m)
            maxhold = MAX_HOLD_HOURS * 60 // tf
            print(f"{len(tr)} Setups ({m.t[0]:%Y-%m-%d} bis {m.t[-1]:%Y-%m-%d}) {info}")
            if len(tr) < 2:
                continue
            for key, label in (("z1", "Z1 Range-Gegenseite"), ("z2", "Z2 fest 2R")):
                res = evaluate(m, tr, key, maxhold, rng)
                tests += 1
                if report(label, res):
                    passed.append(f"{m.name} {tf}m {label}")
                pooled.setdefault((tf, label), []).append(res)
    if incomplete:
        print("\nUnvollständige Daten: " + ", ".join(incomplete) + " -> Lauf nochmal starten.")

    print("\nALLE MÄRKTE ZUSAMMEN (je Zeiteinheit)")
    for (tf, label), lst in sorted(pooled.items()):
        if len(lst) < 2:
            continue
        res = dict(r=np.concatenate([x["r"] for x in lst]), r2=np.concatenate([x["r2"] for x in lst]),
                   rr=np.concatenate([x["rr"] for x in lst]), d=np.concatenate([x["d"] for x in lst]),
                   times=pd.concat([x["times"] for x in lst], ignore_index=True),
                   rand=np.average(np.vstack([x["rand"] for x in lst]), axis=0,
                                   weights=[len(x["r"]) for x in lst]))
        tests += 1
        if report(f"{tf}m {label}", res):
            passed.append(f"ALLE {tf}m {label}")

    print()
    if passed:
        print("ERGEBNIS: Bestanden: " + "; ".join(passed))
        print(f"ACHTUNG: {tests} Tests gelaufen, ~{tests*0.01:.1f} Zufallstreffer zu erwarten. "
              f"Nur ernst nehmen, wenn es auch in 'Alle Märkte zusammen' besteht.")
    else:
        print("ERGEBNIS: Keine Variante besteht alle Bedingungen. Nicht in den Bot.")
    print("Bedingungen: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv, "
          "besser als Zufall (p<0,01).")


if __name__ == "__main__":
    main()
