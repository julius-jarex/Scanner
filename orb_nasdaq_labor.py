#!/usr/bin/env python3
"""
Labor: Opening Range Breakout (ORB) - Nasdaq 100
Gleiche Bestehensbedingungen wie die anderen Labore.

REGELN (vorab festgelegt, nicht nachträglich optimieren!)
---------------------------------------------------------
 - Handelszeit New York 09:30-16:00 (Sommer-/Winterzeit wird automatisch beachtet).
 - Opening Range (OR) = die ersten 5 bzw. 15 Minuten nach 09:30.
 - Richtung: OR-Kerze grün -> long, rot -> short, Doji -> kein Trade.
 - Einstieg: Eröffnungskurs der ersten Minute nach der OR.
 - Stop: Long am OR-Tief, Short am OR-Hoch. Risiko mindestens 3x Kosten.
 - Ziele: Z1 = 10R (wie in der Studie), Z2 = 2R.
   Wird das Ziel nicht erreicht: Ausstieg zum Schlusskurs um 16:00.
 - Stop und Ziel in derselben Minute -> Stop zählt (konservativ).
 - 1 Trade pro Tag.

BEDINGUNGEN fürs Bestehen: n>=100, t>=2,5, beide Hälften positiv,
Kosten x2 positiv, besser als Zufall (p<0,01).
Zufall = dieselben Tage, gleiche Stop-/Zielregel, aber Richtung per
Münzwurf (2000 Durchläufe). Zeigt, ob die Farbe der OR-Kerze etwas taugt
oder ob nur der Aufwärtstrend des Nasdaq mitverdient.

DATEN: werden automatisch von Dukascopy geladen (1-Minuten-Kerzen) und im
Ordner daten/ gespeichert. Fehlende Tage werden beim nächsten Lauf
nachgeladen. Ergebnis gilt erst, wenn "Daten vollständig" dasteht.

AUFRUF
  python orb_nasdaq_labor.py
  python orb_nasdaq_labor.py --kosten 1.5 --start 2021-01-01
"""
import argparse
import lzma
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

SYMBOL = "USATECHIDXUSD"     # Nasdaq 100 bei Dukascopy
START = "2021-01-01"
COST = 1.5                   # Round-Turn in Indexpunkten (Spread + Kommission)
OR_MINUTES = (5, 15)
TARGETS = (("Z1 10R oder Schluss", 10.0), ("Z2 2R oder Schluss", 2.0))
MIN_RISK_COST = 3.0
SESSION_MIN = 390            # 09:30-16:00
N_RANDOM = 2000
SEED = 42
THREADS = 6
DOWNLOAD_BUDGET_MIN = 60     # danach wird mit vorhandenen Daten gerechnet
CACHE = os.path.join("daten", SYMBOL)
REC = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"),
                ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])


# ---------------------------------------------------------------- Daten
def fetch_day(day):
    """Lädt einen Tag, speichert nur die US-Handelszeit. True=ok, False=Fehler."""
    path = os.path.join(CACHE, f"{day}.npy")
    url = (f"https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{day.year}/"
           f"{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5")
    raw = None
    for a in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raw = b""
                break
        except Exception:
            pass
        time.sleep(2 * (a + 1))
    if raw is None:
        return False
    arr = np.empty((0, 5))
    if raw:
        rec = np.frombuffer(lzma.decompress(raw), dtype=REC)
        rec = rec[rec["v"] > 0]
        if len(rec):
            ts = (pd.Timestamp(day, tz="UTC")
                  + pd.to_timedelta(rec["t"].astype(np.int64), unit="s")).tz_convert("America/New_York")
            mins = (ts.hour * 60 + ts.minute - (9 * 60 + 30)).values
            keep = (mins >= 0) & (mins < SESSION_MIN)
            arr = np.column_stack([mins[keep], rec["o"][keep], rec["h"][keep],
                                   rec["l"][keep], rec["c"][keep]]).astype(float)
    np.save(path, arr)
    return True


def download(start):
    os.makedirs(CACHE, exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=1)
    days = [d.date() for d in pd.date_range(start, end, freq="B")]
    todo = [d for d in days if not os.path.exists(os.path.join(CACHE, f"{d}.npy"))]
    print(f"Daten: {len(days) - len(todo)}/{len(days)} Tage schon vorhanden, lade {len(todo)} ...",
          flush=True)
    t0 = time.time()
    failed = 0
    if todo:
        with ThreadPoolExecutor(THREADS) as ex:
            futs = {ex.submit(fetch_day, d): d for d in todo}
            for i, f in enumerate(as_completed(futs), 1):
                if not f.result():
                    failed += 1
                if i % max(1, len(todo) // 10) == 0:
                    print(f"   {i}/{len(todo)} ({time.time() - t0:.0f}s, {failed} Fehler)", flush=True)
                if time.time() - t0 > DOWNLOAD_BUDGET_MIN * 60:
                    print("   Zeitlimit für Download erreicht, Rest beim nächsten Lauf.", flush=True)
                    for g in futs:
                        g.cancel()
                    break
    missing = sum(1 for d in days if not os.path.exists(os.path.join(CACHE, f"{d}.npy")))
    return days, missing


def load_matrix(days):
    """Tage x 390 Minuten x (o,h,l,c), fehlende Minuten = NaN."""
    rows, used = [], []
    for d in days:
        p = os.path.join(CACHE, f"{d}.npy")
        if not os.path.exists(p):
            continue
        a = np.load(p)
        if len(a) < 200:          # Feiertag / kaum Daten
            continue
        m = np.full((SESSION_MIN, 4), np.nan)
        m[a[:, 0].astype(int)] = a[:, 1:]
        rows.append(m)
        used.append(d)
    if not rows:
        return np.empty((0, SESSION_MIN, 4)), []
    M = np.stack(rows)
    # Preis-Skalierung automatisch (Dukascopy speichert Ganzzahlen)
    med = np.nanmedian(M[:, :, 3])
    for div in (1000, 100, 10, 1, 10000, 100000):
        if 3000 <= med / div <= 60000:
            M /= div
            break
    return M, used


# ---------------------------------------------------------------- Test
def trade_r(day, n_or, d, tgt_r, cost):
    """R-Ergebnis eines Tages für Richtung d (+1/-1). None wenn ungültig."""
    orb = day[:n_or]
    if np.isnan(orb[:, 0]).all():
        return None
    o = orb[~np.isnan(orb[:, 0])][0, 0]
    c = orb[~np.isnan(orb[:, 3])][-1, 3]
    hi, lo = np.nanmax(orb[:, 1]), np.nanmin(orb[:, 2])
    rest = day[n_or:]
    rest = rest[~np.isnan(rest[:, 0])]
    if len(rest) < 10:
        return None
    entry = rest[0, 0]
    stop = lo if d > 0 else hi
    risk = d * (entry - stop)
    if risk <= 0 or risk < MIN_RISK_COST * cost:
        return None
    tgt = entry + d * tgt_r * risk
    H, L = rest[:, 1], rest[:, 2]
    hs = (L <= stop) if d > 0 else (H >= stop)
    ht = (H >= tgt) if d > 0 else (L <= tgt)
    i_s = np.argmax(hs) if hs.any() else len(rest)
    i_t = np.argmax(ht) if ht.any() else len(rest)
    if i_s <= i_t and i_s < len(rest):
        ex = stop
    elif i_t < len(rest):
        ex = tgt
    else:
        ex = rest[-1, 3]
    gross = d * (ex - entry)
    return gross / risk, cost / risk, np.sign(c - o)


def run(M, used, n_or, tgt_r, cost, rng):
    rl, rs, cl, sig, dates = [], [], [], [], []
    for day, dt in zip(M, used):
        a = trade_r(day, n_or, 1, tgt_r, cost)
        b = trade_r(day, n_or, -1, tgt_r, cost)
        if a is None or b is None or a[2] == 0:
            continue
        rl.append(a[0]); rs.append(b[0]); cl.append((a[1], b[1])); sig.append(a[2]); dates.append(dt)
    rl, rs, cl, sig = np.array(rl), np.array(rs), np.array(cl), np.array(sig)
    if len(sig) < 2:
        return None
    gross = np.where(sig > 0, rl, rs)
    cr = np.where(sig > 0, cl[:, 0], cl[:, 1])
    r, r2 = gross - cr, gross - 2 * cr
    D = rng.choice([True, False], size=(N_RANDOM, len(sig)))
    rand = np.where(D, rl - cl[:, 0], rs - cl[:, 1]).mean(axis=1)
    always_long = (rl - cl[:, 0]).mean()
    return dict(r=r, r2=r2, sig=sig, rand=rand, always_long=always_long, dates=dates)


def fmt(x):
    return "nan" if not np.isfinite(x) else f"{x:.2f}"


def report(label, res, tgt_r):
    r = res["r"]
    n = len(r)
    half = n // 2
    sd = r.std(ddof=1)
    t = r.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    h1, h2 = r[:half].mean(), r[half:].mean()
    c2 = res["r2"].mean()
    p = (np.sum(res["rand"] >= r.mean()) + 1) / (len(res["rand"]) + 1)
    lg, sh = r[res["sig"] > 0], r[res["sig"] < 0]
    ok = n >= 100 and t >= 2.5 and h1 > 0 and h2 > 0 and c2 > 0 and p < 0.01
    print(f"  {label}: n={n}, Quote {(r > 0).mean()*100:.0f}%, Chance/Risiko 1:{tgt_r:.0f}, "
          f"Ø {fmt(r.mean())}R, t={t:.1f}, 1. Hälfte {fmt(h1)} / 2. Hälfte {fmt(h2)}, "
          f"Kosten x2 {fmt(c2)}R | long {fmt(lg.mean())} / short {fmt(sh.mean())} | "
          f"Zufall {fmt(res['rand'].mean())}, p={p:.2f} | immer long {fmt(res['always_long'])}"
          f"  -> {'BESTANDEN' if ok else 'durchgefallen'}")
    return ok


def main():
    global SYMBOL, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosten", type=float, default=COST)
    ap.add_argument("--start", default=START)
    ap.add_argument("--symbol", default=SYMBOL, help="z.B. USA500IDXUSD für S&P 500")
    args = ap.parse_args()
    SYMBOL = args.symbol
    CACHE = os.path.join("daten", SYMBOL)

    days, missing = download(args.start)
    M, used = load_matrix(days)
    complete = missing <= 0.02 * len(days)
    print()
    print(f"ORB-Labor {SYMBOL}, Kosten {args.kosten} Punkte, Handelszeit New York 09:30-16:00")
    if not len(used):
        print("Keine Daten geladen. Bitte Lauf nochmal starten.")
        return
    print(f"Handelstage mit Daten: {len(used)} ({used[0]} bis {used[-1]}), Kurs zuletzt "
          f"{np.nanmax(M[-1, :, 3]):.0f}")
    if complete:
        print("Daten vollständig.")
    else:
        print(f"ACHTUNG: {missing} von {len(days)} Tagen fehlen noch. Ergebnis vorläufig, "
              f"bitte Lauf nochmal starten.")
    print()

    rng = np.random.default_rng(SEED)
    passed = []
    for n_or in OR_MINUTES:
        print(f"Opening Range {n_or} Minuten:")
        for label, tgt_r in TARGETS:
            res = run(M, used, n_or, tgt_r, args.kosten, rng)
            if res is None:
                print(f"  {label}: zu wenig Trades")
                continue
            if report(label, res, tgt_r):
                passed.append(f"OR {n_or} min {label}")
        print()

    if passed and complete:
        print("ERGEBNIS: Bestanden: " + "; ".join(passed))
        print("Nächster Schritt: Papertrading, bevor echtes Geld eingesetzt wird.")
    elif passed:
        print("ERGEBNIS: Vorläufig bestanden: " + "; ".join(passed) +
              ". Erst gültig, wenn die Daten vollständig sind.")
    else:
        print("ERGEBNIS: Keine Variante besteht alle Bedingungen. Nicht in den Bot.")
    print("Bedingungen: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv, "
          "besser als Zufall (p<0,01).")


if __name__ == "__main__":
    main()
