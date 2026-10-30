#!/usr/bin/env python3
"""
Labor: Liquidity Sweep -> IFVG -> Trendline Break -> Einstieg
Mehrere Märkte, gleiche Bestehensbedingungen wie das AMD-Labor.

REGELN (vorab festgelegt, nicht nachträglich optimieren!)
---------------------------------------------------------
Long (Short spiegelbildlich):
 1. Liquidität: unberührte Swing-Tiefs (Fraktal, 3 Kerzen je Seite, erst nach
    3 Kerzen bestätigt -> kein Blick in die Zukunft), max. 100 Kerzen alt.
 2. Sweep: Kerze sticht unter ein solches Tief und SCHLIESST wieder darüber.
    Nur Sweeps zwischen 07:00 und 17:00 UTC (London + New York).
 3. IFVG: bärisches Fair Value Gap (Hoch[b] < Tief[b-2]) aus den 10 Kerzen
    bis zum Sweep; invertiert, wenn eine Kerze ÜBER der FVG-Oberkante schließt.
 4. Trendlinie: Linie durch die letzten zwei fallenden Swing-Hochs
    (Fraktal 2/2) vor dem Sweep; Bruch = Schlusskurs über der Linie.
 5. Einstieg: Schlusskurs der Kerze, an der IFVG UND Trendlinienbruch erfüllt
    sind, spätestens 10 Kerzen nach dem Sweep. Neues Tief unter dem Sweep-Tief
    vorher -> Setup verworfen.
 6. Stop: Sweep-Tief. Risiko muss mindestens 3x Kosten sein.
 7. Ziele: Z1 = nächstes unberührtes Swing-Hoch (Gegenliquidität),
           Z2 = fest 2R. Zeitstopp nach 24 Stunden.
 8. Stop und Ziel in derselben Kerze -> Stop zählt (konservativ).
    Kurslücken über den Stop -> Füllung zum Eröffnungskurs.

BEDINGUNGEN fürs Bestehen: n>=100, t>=2,5, beide Hälften positiv,
Kosten x2 positiv, besser als Zufall (p<0,01).
Zufall = gleiche Anzahl Trades an zufälligen Zeitpunkten (07-17 UTC),
gleiche Richtung, gleicher Stop- und Zielabstand, 2000 Durchläufe.

DATEN
-----
Ordner mit CSV-Dateien, Name: SYMBOL_TIMEFRAME.csv, z.B.
  XAUUSD_15m.csv  EURUSD_1h.csv  BTCUSD_15m.csv
Spalten: time, open, high, low, close (Groß/Klein egal; Zeit in UTC
oder mit Zeitzone; Unix-Sekunden/Millisekunden gehen auch).

AUFRUF
------
  python3 sweep_ifvg_tl_labor.py            (lädt Daten selbst, ab 2023)
  python3 sweep_ifvg_tl_labor.py ORDNER
  python3 sweep_ifvg_tl_labor.py ORDNER --kosten XAUUSD=0.20 EURUSD=0.00008
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# ---------- feste Parameter (nicht anfassen, sonst Data-Mining) ----------
K_LIQ = 3          # Fraktal-Breite für Liquiditätslevel
K_TL = 2           # Fraktal-Breite für Trendlinienpunkte
LOOKBACK = 100     # max. Alter der Level in Kerzen
FVG_WINDOW = 10    # FVG muss in den 10 Kerzen bis zum Sweep entstanden sein
CONFIRM_WINDOW = 10
SESSION = (7, 17)  # UTC-Stunden für Sweep und Zufallseinstiege
MIN_RISK_COST = 3.0
Z2_R = 2.0
MAX_HOLD_HOURS = 24
N_RANDOM = 2000
SEED = 42

# Round-Turn-Kosten in Preiseinheiten (Spread + Kommission). Bitte an deinen
# Broker anpassen. Krypto: relativ zum Preis.
DEFAULT_COSTS = {
    "XAUUSD": 0.20, "XAGUSD": 0.03,
    "EURUSD": 0.00008, "GBPUSD": 0.00012, "AUDUSD": 0.00010,
    "NZDUSD": 0.00014, "USDCAD": 0.00012, "USDCHF": 0.00012,
    "EURGBP": 0.00012, "EURJPY": 0.015, "GBPJPY": 0.025, "USDJPY": 0.012,
    "US30": 3.0, "NAS100": 1.5, "SPX500": 0.6, "GER40": 1.5,
}
CRYPTO_REL_COST = 0.0006   # 0,06 % vom Preis
FALLBACK_REL_COST = 0.0001


# ---------- Daten selbst laden (Dukascopy für Forex/Gold, Binance für Krypto) ----------
DEFAULT_MARKETS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "BTCUSDT", "ETHUSDT"]
START = "2023-01-01"


def _get(url, tries=4):
    import time
    import urllib.error
    import urllib.request
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
        time.sleep(1.5 * (a + 1))
    return None


def _duka_point(sym):
    s = sym.upper()
    return 1000 if (s.startswith("XA") or s.endswith("JPY")) else 100000


def _duka_day(sym, day):
    import lzma
    url = (f"https://datafeed.dukascopy.com/datafeed/{sym.upper()}/{day.year}/"
           f"{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5")
    raw = _get(url)
    if not raw:
        return None
    data = lzma.decompress(raw)
    rec = np.frombuffer(data, dtype=np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"),
                                              ("l", ">i4"), ("h", ">i4"), ("v", ">f4")]))
    rec = rec[rec["v"] > 0]
    if len(rec) == 0:
        return None
    p = _duka_point(sym)
    return pd.DataFrame({
        "time": pd.Timestamp(day, tz="UTC") + pd.to_timedelta(rec["t"].astype(np.int64), unit="s"),
        "open": rec["o"] / p, "high": rec["h"] / p, "low": rec["l"] / p, "close": rec["c"] / p})


def _binance_month(sym, ym):
    import io
    import zipfile
    url = (f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/"
           f"{sym}-1m-{ym}.zip")
    raw = _get(url)
    if not raw:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not np.issubdtype(df[0].dtype, np.number):   # evtl. Kopfzeile
        df = df.iloc[1:].astype({0: np.int64})
    t = df[0].astype(np.int64)
    unit = "us" if t.iloc[0] > 1e14 else "ms"
    return pd.DataFrame({"time": pd.to_datetime(t, unit=unit, utc=True),
                         "open": df[1].astype(float), "high": df[2].astype(float),
                         "low": df[3].astype(float), "close": df[4].astype(float)})


def download(markets, folder):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(folder, exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    for sym in markets:
        if all(os.path.exists(os.path.join(folder, f"{sym}_{tf}.csv")) for tf in ("15m", "1h")):
            print(f"{sym}: Daten schon vorhanden")
            continue
        is_crypto = sym.upper().endswith("USDT")
        print(f"{sym}: lade Daten ab {START} ...", flush=True)
        if is_crypto:
            months = pd.period_range(START, end - pd.Timedelta(days=1), freq="M").strftime("%Y-%m")
            jobs = [lambda m=m: _binance_month(sym.upper(), m) for m in months]
        else:
            days = [d.date() for d in pd.date_range(START, end - pd.Timedelta(days=1), freq="D")
                    if d.dayofweek != 5]
            jobs = [lambda d=d: _duka_day(sym, d) for d in days]
        with ThreadPoolExecutor(16) as ex:
            parts = [p for p in ex.map(lambda f: f(), jobs) if p is not None and len(p)]
        if not parts:
            print(f"{sym}: KEINE DATEN erhalten, übersprungen")
            continue
        m1 = pd.concat(parts).sort_values("time").drop_duplicates("time").set_index("time")
        for tf, rule in (("15m", "15min"), ("1h", "1h")):
            bars = m1.resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            bars.reset_index().to_csv(os.path.join(folder, f"{sym}_{tf}.csv"), index=False)
        print(f"{sym}: {len(m1)} Minuten-Kerzen, letzter Kurs {m1['close'].iloc[-1]:.5g}")


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in df.columns if c in ("time", "timestamp", "date",
                                                 "datetime", "open_time")), df.columns[0])
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        t = pd.to_datetime(t, unit=unit, utc=True)
    else:
        t = pd.to_datetime(t, utc=True)
    out = pd.DataFrame({"time": t, "open": df["open"].astype(float),
                        "high": df["high"].astype(float), "low": df["low"].astype(float),
                        "close": df["close"].astype(float)})
    out = out.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return out


def cost_for(symbol, price, overrides):
    s = symbol.upper()
    if s in overrides:
        return overrides[s], False
    if s in DEFAULT_COSTS:
        return DEFAULT_COSTS[s], False
    if any(k in s for k in ("BTC", "ETH", "SOL", "XRP", "USDT", "DOGE", "ADA")):
        return CRYPTO_REL_COST * price, True
    return FALLBACK_REL_COST * price, True


def fractals(H, L, k):
    n = len(H)
    fh = np.zeros(n, bool)
    fl = np.zeros(n, bool)
    for h in range(k, n - k):
        if H[h] > H[h - k:h].max() and H[h] >= H[h + 1:h + k + 1].max():
            fh[h] = True
        if L[h] < L[h - k:h].min() and L[h] <= L[h + 1:h + k + 1].min():
            fl[h] = True
    return fh, fl


def find_setups(df, cost_fn):
    O, H, L, C = (df[c].values for c in ("open", "high", "low", "close"))
    hours = df["time"].dt.hour.values
    n = len(C)
    fh3, fl3 = fractals(H, L, K_LIQ)
    fh2, fl2 = fractals(H, L, K_TL)

    lows, highs = [], []   # unberührte Level: [preis, index]
    trades = []
    next_allowed = 0

    def tl_points(fr, i, want_desc):
        idx = [h for h in range(max(0, i - LOOKBACK), i - K_TL + 1) if fr[h]]
        if len(idx) < 2:
            return None
        h1, h2 = idx[-2], idx[-1]
        p1 = H[h1] if want_desc else L[h1]
        p2 = H[h2] if want_desc else L[h2]
        if want_desc and not p2 < p1:
            return None
        if not want_desc and not p2 > p1:
            return None
        slope = (p2 - p1) / (h2 - h1)
        return lambda j: p2 + slope * (j - h2)

    for i in range(n):
        # neu bestätigte Level aufnehmen
        h = i - K_LIQ
        if h >= 0:
            if fl3[h]:
                lows.append([L[h], h])
            if fh3[h]:
                highs.append([H[h], h])
        lows = [lv for lv in lows if i - lv[1] <= LOOKBACK]
        highs = [lv for lv in highs if i - lv[1] <= LOOKBACK]

        swept_lo = [lv for lv in lows if L[i] < lv[0]]
        swept_hi = [lv for lv in highs if H[i] > lv[0]]
        lows = [lv for lv in lows if not L[i] < lv[0]]
        highs = [lv for lv in highs if not H[i] > lv[0]]

        if i < next_allowed or not (SESSION[0] <= hours[i] < SESSION[1]):
            continue

        for d, swept in ((1, swept_lo), (-1, swept_hi)):
            if not swept:
                continue
            if d == 1 and not C[i] > max(lv[0] for lv in swept):
                continue
            if d == -1 and not C[i] < min(lv[0] for lv in swept):
                continue
            extreme = L[i] if d == 1 else H[i]

            # FVGs gegen die neue Richtung aus den letzten Kerzen
            fvgs = []
            for b in range(max(2, i - FVG_WINDOW), i + 1):
                if d == 1 and H[b] < L[b - 2]:
                    fvgs.append((b, L[b - 2]))      # Oberkante bärisches FVG
                if d == -1 and L[b] > H[b - 2]:
                    fvgs.append((b, H[b - 2]))      # Unterkante bullisches FVG
            if not fvgs:
                continue
            line = tl_points(fh2 if d == 1 else fl2, i, want_desc=(d == 1))
            if line is None:
                continue

            ifvg = tl = False
            entry_j = None
            for j in range(i, min(n, i + CONFIRM_WINDOW + 1)):
                if j > i and ((d == 1 and L[j] < extreme) or (d == -1 and H[j] > extreme)):
                    break
                if not ifvg:
                    ifvg = any(b < j and (C[j] > top if d == 1 else C[j] < top)
                               for b, top in fvgs)
                if not tl:
                    tl = C[j] > line(j) if d == 1 else C[j] < line(j)
                if ifvg and tl:
                    entry_j = j
                    break
            if entry_j is None:
                continue

            entry = C[entry_j]
            stop = extreme
            risk = d * (entry - stop)
            cost, _ = cost_fn(entry)
            if risk <= 0 or risk < MIN_RISK_COST * cost:
                continue
            # Z1: nächstes unberührtes Gegen-Level
            if d == 1:
                cand = [lv[0] for lv in highs if lv[0] > entry]
                z1 = min(cand) if cand else np.nan
            else:
                cand = [lv[0] for lv in lows if lv[0] < entry]
                z1 = max(cand) if cand else np.nan
            trades.append(dict(idx=entry_j, time=df["time"].iloc[entry_j], d=d,
                               entry=entry, stop=stop, risk=risk, cost=cost,
                               z1=z1, z2=entry + d * Z2_R * risk))
            next_allowed = entry_j + 1
            break
    return pd.DataFrame(trades)


def simulate(O, H, L, C, idx, d, entry, stop, tgt, maxhold):
    """Vektorisiert. Gibt Brutto-PnL in Preiseinheiten zurück."""
    n = len(C)
    idx = np.asarray(idx); d = np.asarray(d, float)
    entry = np.asarray(entry, float); stop = np.asarray(stop, float); tgt = np.asarray(tgt, float)
    pnl = np.full(idx.shape, np.nan)
    done = np.zeros(idx.shape, bool)
    long = d > 0
    for k in range(1, maxhold + 1):
        j = idx + k
        valid = (j < n) & ~done
        if not valid.any():
            break
        jj = np.minimum(j, n - 1)
        o, h, l = O[jj], H[jj], L[jj]
        hs = valid & np.where(long, l <= stop, h >= stop)
        ht = valid & ~hs & np.where(long, h >= tgt, l <= tgt)
        sfill = np.where(long, np.minimum(o, stop), np.maximum(o, stop))
        pnl = np.where(hs, d * (sfill - entry), pnl)
        pnl = np.where(ht, d * (tgt - entry), pnl)
        done |= hs | ht
    rest = ~done
    if rest.any():
        jj = np.minimum(idx + maxhold, n - 1)
        pnl = np.where(rest, d * (C[jj] - entry), pnl)
    return pnl


def stats(r, times):
    n = len(r)
    if n < 2:
        return dict(n=n, quote=np.nan, mean=np.nan, t=np.nan, h1=np.nan, h2=np.nan)
    order = np.argsort(times.values)
    rs = r[order]
    half = n // 2
    sd = rs.std(ddof=1)
    return dict(n=n, quote=(r > 0).mean(), mean=r.mean(),
                t=r.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan,
                h1=rs[:half].mean(), h2=rs[half:].mean())


def run_target(df, tr, key, maxhold, rng):
    """Liefert R-Werte und Zufalls-Mittelwerte für ein Ziel."""
    tr = tr[np.isfinite(tr[key])].reset_index(drop=True)
    if tr.empty:
        return None
    O, H, L, C = (df[c].values for c in ("open", "high", "low", "close"))
    gross = simulate(O, H, L, C, tr.idx.values, tr.d.values, tr.entry.values,
                     tr.stop.values, tr[key].values, maxhold)
    r = (gross - tr.cost.values) / tr.risk.values
    r2 = (gross - 2 * tr.cost.values) / tr.risk.values
    rr = (tr.d * (tr[key] - tr.entry)).values / tr.risk.values

    # Zufall: gleiche Richtung, gleicher Stop-/Zielabstand, zufälliger Zeitpunkt
    hours = df["time"].dt.hour.values
    pool = np.where((hours >= SESSION[0]) & (hours < SESSION[1]))[0]
    pool = pool[pool < len(C) - maxhold]
    m = len(tr)
    ridx = rng.choice(pool, size=(N_RANDOM, m))
    D = np.broadcast_to(tr.d.values, (N_RANDOM, m))
    risk = np.broadcast_to(tr.risk.values, (N_RANDOM, m))
    rew = np.broadcast_to((tr.d * (tr[key] - tr.entry)).values, (N_RANDOM, m))
    cost = np.broadcast_to(tr.cost.values, (N_RANDOM, m))
    e = C[ridx]
    g = simulate(O, H, L, C, ridx.ravel(), D.ravel(), e.ravel(),
                 (e - D * risk).ravel(), (e + D * rew).ravel(), maxhold).reshape(N_RANDOM, m)
    rand_means = ((g - cost) / risk).mean(axis=1)
    return dict(r=r, r2=r2, rr=rr, times=tr.time, d=tr.d.values, rand=rand_means)


def verdict(st, cost2, rand_mean, p):
    return (st["n"] >= 100 and st["t"] >= 2.5 and st["h1"] > 0 and st["h2"] > 0
            and cost2 > 0 and p < 0.01)


def fmt(x):
    return "nan" if not np.isfinite(x) else f"{x:.2f}"


def report(label, res):
    st = stats(res["r"], res["times"])
    if st["n"] < 2:
        print(f"  {label}: n={st['n']} (zu wenig)")
        return False
    p = (np.sum(res["rand"] >= st["mean"]) + 1) / (len(res["rand"]) + 1)
    c2 = np.mean(res["r2"])
    lg = res["r"][res["d"] > 0]; sh = res["r"][res["d"] < 0]
    ok = verdict(st, c2, res["rand"].mean(), p)
    print(f"  {label}: n={st['n']}, Quote {st['quote']*100:.0f}%, "
          f"Ø Chance/Risiko 1:{np.median(res['rr']):.2f} (Median), Ø {fmt(st['mean'])}R, "
          f"t={st['t']:.1f}, 1. Hälfte {fmt(st['h1'])} / 2. Hälfte {fmt(st['h2'])}, "
          f"Kosten x2 {fmt(c2)}R | long {fmt(lg.mean() if len(lg) else np.nan)} / "
          f"short {fmt(sh.mean() if len(sh) else np.nan)} | Zufall {fmt(res['rand'].mean())}, "
          f"p={p:.2f}  -> {'BESTANDEN' if ok else 'durchgefallen'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ordner", nargs="?", default=None,
                    help="Ordner mit CSVs. Ohne Angabe werden die Daten selbst geladen.")
    ap.add_argument("--maerkte", nargs="*", default=DEFAULT_MARKETS)
    ap.add_argument("--kosten", nargs="*", default=[], help="SYMBOL=wert (Round-Turn, Preiseinheiten)")
    args = ap.parse_args()
    overrides = {k.upper(): float(v) for k, v in (s.split("=") for s in args.kosten)}
    if args.ordner is None:
        args.ordner = "daten"
        download(args.maerkte, args.ordner)
        print()

    files = sorted(glob.glob(os.path.join(args.ordner, "*.csv")))
    if not files:
        sys.exit("Keine CSV-Dateien gefunden.")
    rng = np.random.default_rng(SEED)
    pooled = {}   # (tf, ziel) -> Liste von Ergebnissen
    passed = []
    tests = 0

    print("LABOR Liquidity Sweep -> IFVG -> Trendline Break (Zeiten UTC, Sweeps 07-17 UTC)\n")
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if "_" not in name:
            print(f"Übersprungen (Name SYMBOL_TF erwartet): {name}")
            continue
        sym, tf = name.rsplit("_", 1)
        df = load_csv(f)
        step = df["time"].diff().median()
        maxhold = max(1, int(pd.Timedelta(hours=MAX_HOLD_HOURS) / step))
        c0, rel = cost_for(sym, df["close"].iloc[-1], overrides)
        cost_fn = (lambda p, s=sym: cost_for(s, p, overrides))
        tr = find_setups(df, cost_fn)
        hint = " (Kosten geschätzt, bitte mit --kosten setzen)" if rel and sym.upper() not in overrides \
            and not any(k in sym.upper() for k in ("BTC", "ETH", "SOL", "XRP", "USDT", "DOGE", "ADA")) else ""
        print(f"{sym} {tf}: {len(tr)} Setups ({df.time.iloc[0]:%Y-%m-%d} bis {df.time.iloc[-1]:%Y-%m-%d}), "
              f"Kosten ~{c0:.5g}{hint}")
        if tr.empty:
            continue
        for key, label in (("z1", "Z1 Gegenliquidität"), ("z2", "Z2 fest 2R")):
            res = run_target(df, tr, key, maxhold, rng)
            if res is None:
                print(f"  {label}: keine Trades")
                continue
            tests += 1
            if report(label, res):
                passed.append(f"{sym} {tf} {label}")
            pooled.setdefault((tf, label), []).append(res)
        print()

    print("ALLE MÄRKTE ZUSAMMEN (je Zeiteinheit)")
    for (tf, label), lst in sorted(pooled.items()):
        if len(lst) < 2:
            continue
        res = dict(r=np.concatenate([x["r"] for x in lst]),
                   r2=np.concatenate([x["r2"] for x in lst]),
                   rr=np.concatenate([x["rr"] for x in lst]),
                   d=np.concatenate([x["d"] for x in lst]),
                   times=pd.concat([x["times"] for x in lst], ignore_index=True),
                   rand=np.average(np.vstack([x["rand"] for x in lst]), axis=0,
                                   weights=[len(x["r"]) for x in lst]))
        tests += 1
        if report(f"{tf} {label}", res):
            passed.append(f"ALLE {tf} {label}")

    print()
    if passed:
        print("ERGEBNIS: Bestanden: " + "; ".join(passed))
        print(f"ACHTUNG: {tests} Tests gelaufen. Bei p<0,01 ist ~{tests*0.01:.1f} Zufallstreffer "
              f"zu erwarten. Nur ernst nehmen, wenn es auch in der Zusammenfassung "
              f"aller Märkte besteht oder p<{0.01/tests:.4f} ist.")
    else:
        print("ERGEBNIS: Keine Variante besteht alle Bedingungen. Nicht in den Bot.")
    print("Bedingungen: n>=100, t>=2,5, beide Hälften positiv, Kosten x2 positiv, "
          "besser als Zufall (p<0,01).")


if __name__ == "__main__":
    main()
