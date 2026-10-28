#!/usr/bin/env python3
"""Strategie-Labor: testet mehrere Daytrading-Ideen gegeneinander (nach Kosten, mit Train/Test-Aufteilung).
Start: GitHub -> Actions -> lab -> Run workflow. Ergebnis per Telegram + CSV.
Einstieg immer zum Schlusskurs der Signalkerze. Stop-Treffer vor Ziel in derselben Kerze zählt als Stop."""
import os, time, argparse
import requests
import pandas as pd
import numpy as np
import scanner as S

TPS = (1.0, 1.5, 2.0, 3.0)         # Ziele in R
MAX_HOLD_H = 36
TEST_SHARE = 0.4                    # letzte 40 % der Zeit = Test (nicht zum Aussuchen benutzt)
MIN_N = 100
MIN_T = 2.5                         # Mindest-Signifikanz (t-Wert), da viele Varianten getestet werden


# ---------------- Daten ----------------
def hist_twelve(symbol, interval, n):
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": interval, "outputsize": n, "timezone": "UTC",
        "apikey": S.env("TWELVEDATA_KEY", required=True)}, timeout=60)
    j = r.json()
    if j.get("status") == "error":
        raise RuntimeError(j.get("message"))
    df = pd.DataFrame(j["values"])
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df.sort_values("time")[["time", "open", "high", "low", "close"]].iloc[:-1].reset_index(drop=True)


def hist_bitstamp(pair, step, days):
    start = int(time.time()) - days * 86400
    rows = []
    while start < time.time():
        r = requests.get(f"https://www.bitstamp.net/api/v2/ohlc/{pair}/", params={"step": step, "limit": 1000, "start": start}, timeout=30)
        r.raise_for_status()
        part = r.json()["data"]["ohlc"]
        if not part:
            break
        rows += part
        nxt = int(part[-1]["timestamp"]) + step
        if nxt <= start:
            break
        start = nxt
        time.sleep(0.5)
    df = pd.DataFrame(rows).drop_duplicates("timestamp")
    df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df.sort_values("time")[["time", "open", "high", "low", "close"]].iloc[:-1].reset_index(drop=True)


def indicators(df):
    df = S.add_ind(df.copy())
    return df


def trend_filter(df, rule="4h"):
    """Trend aus höherer Zeitebene, nur mit abgeschlossenen Kerzen (kein Blick in die Zukunft)."""
    h = df.set_index("time").resample(rule).agg({"close": "last"}).dropna()
    e20 = h.close.ewm(span=20, adjust=False).mean(); e50 = h.close.ewm(span=50, adjust=False).mean()
    tr = pd.Series(np.where((e20 > e50) & (h.close > e50), 1, np.where((e20 < e50) & (h.close < e50), -1, 0)), index=h.index)
    # Wert einer Kerze gilt erst, wenn sie abgeschlossen ist -> eine Periode verschieben
    tr.index = tr.index + pd.tseries.frequencies.to_offset(rule)
    return tr.reindex(df.time, method="ffill").fillna(0).values


# ---------------- Strategien: liefern Liste (index, side, stop) ----------------
def strat_orb(df, tf_min):
    """Eröffnungsbereich 07:00-08:00 UTC (London). Ausbruch bis 11:00, Stop auf der Gegenseite der Spanne."""
    out, t = [], df.time
    for day, g in df.groupby(df.time.dt.date):
        rng = g[(g.time.dt.hour == 7)]
        if len(rng) < 60 // tf_min:
            continue
        hi, lo = rng.high.max(), rng.low.min()
        atr = g.atr.iloc[0] if "atr" in g else np.nan
        post = g[(g.time.dt.hour >= 8) & (g.time.dt.hour < 11)]
        for idx, r in post.iterrows():
            a = r.atr
            if np.isnan(a) or not (0.5 * a <= hi - lo <= 3 * a):
                continue
            if r.close > hi:
                out.append((idx, "long", lo)); break
            if r.close < lo:
                out.append((idx, "short", hi)); break
    return out


def strat_pullback(df, tf_min, trend):
    """Trend-Rücksetzer: höherer Trend, Kerze berührt EMA20 und schließt wieder in Trendrichtung."""
    out = []
    for i in range(60, len(df)):
        r = df.iloc[i]
        a = r.atr
        if np.isnan(a):
            continue
        t = trend[i]
        w = df.iloc[i - 5:i + 1]
        if t == 1 and (w.low <= w.ema20).any() and r.close > r.ema20 > r.ema50 and r.close > r.open:
            stop = w.low.min() - 0.2 * a
            if 0.5 * a <= r.close - stop <= 3 * a: out.append((i, "long", stop))
        if t == -1 and (w.high >= w.ema20).any() and r.close < r.ema20 < r.ema50 and r.close < r.open:
            stop = w.high.max() + 0.2 * a
            if 0.5 * a <= stop - r.close <= 3 * a: out.append((i, "short", stop))
    return out


def strat_donchian(df, tf_min, trend):
    """Ausbruch über das 20-Kerzen-Hoch/-Tief in Richtung des höheren Trends, Stop 1,5 ATR."""
    out = []
    hh = df.high.rolling(20).max().shift(1); ll = df.low.rolling(20).min().shift(1)
    for i in range(60, len(df)):
        r = df.iloc[i]; a = r.atr
        if np.isnan(a): continue
        if trend[i] == 1 and r.close > hh.iloc[i]: out.append((i, "long", r.close - 1.5 * a))
        if trend[i] == -1 and r.close < ll.iloc[i]: out.append((i, "short", r.close + 1.5 * a))
    return out


def strat_fade(df, tf_min, trend):
    """Gegenbewegung: Kurs weit unter/über dem EMA50 (2,5 ATR) und Umkehrkerze, im Seitwärts- oder Gegentrend."""
    out = []
    for i in range(60, len(df)):
        r = df.iloc[i]; a = r.atr
        if np.isnan(a): continue
        if r.close < r.ema50 - 2.5 * a and r.close > r.open and trend[i] != -1: out.append((i, "long", df.low.iloc[i - 2:i + 1].min() - 0.2 * a))
        if r.close > r.ema50 + 2.5 * a and r.close < r.open and trend[i] != 1: out.append((i, "short", df.high.iloc[i - 2:i + 1].max() + 0.2 * a))
    return out


STRATS = {
    "ORB London": lambda df, tf, tr: strat_orb(df, tf),
    "Trend-Ruecksetzer": strat_pullback,
    "Donchian-Ausbruch": strat_donchian,
    "Gegenbewegung": strat_fade,
}


# ---------------- Auswertung ----------------
def evaluate(sym, df, tf_min, sig, name, tf_label):
    rows, last = [], {}
    hi, lo = df.high.values, df.low.values
    maxc = int(MAX_HOLD_H * 60 / tf_min)
    for i, side, stop in sig:
        if sym in S.FOREX and not (df.time.iloc[i].weekday() < 5 and S.FOREX_HOURS_UTC[0] <= df.time.iloc[i].hour < S.FOREX_HOURS_UTC[1]):
            continue
        if i - last.get(side, -999) < 4 * 60 // tf_min:      # nicht jede Kerze erneut handeln
            continue
        last[side] = i
        entry = float(df.close.iloc[i]); risk = abs(entry - stop)
        if risk <= 0:
            continue
        sgn = 1 if side == "long" else -1
        fut = list(zip(hi[i + 1:i + 1 + maxc], lo[i + 1:i + 1 + maxc]))
        if len(fut) < 4:
            continue
        cr = S.cost_r(sym, entry, risk)
        row = {"time": df.time.iloc[i], "symbol": sym, "strategie": name, "ebene": tf_label, "side": side, "cost_r": cr}
        for k in TPS:
            st, _ = S.simulate_path(side, entry, stop, entry + sgn * k * risk, entry + sgn * k * risk, fut, "simple")
            row[f"r_{k}"] = (k if st == "tp1" else -1.0 if st == "stop" else 0.0) - cr
        rows.append(row)
    return rows


def summary(df):
    lines = []
    for (name, tfl), g in df.groupby(["strategie", "ebene"]):
        g = g.sort_values("time"); cut = g.time.iloc[0] + (g.time.iloc[-1] - g.time.iloc[0]) * (1 - TEST_SHARE)
        tr, te = g[g.time < cut], g[g.time >= cut]
        lines.append(f"{name} ({tfl}): n={len(g)} (Train {len(tr)}, Test {len(te)})")
        for k in TPS:
            c = f"r_{k}"
            lines.append(f"  Ziel {k}R: Quote {round((g[c] > 0).mean() * 100)}%, Erw. gesamt {g[c].mean():+.2f} | Train {tr[c].mean():+.2f} | Test {te[c].mean():+.2f}")
    return "\n".join(lines)


def verdicts(df):
    """Nur Varianten, die bei n>=MIN_N sowohl im Train- als auch im Test-Teil positiv sind."""
    good = []
    for (name, tfl), g in df.groupby(["strategie", "ebene"]):
        g = g.sort_values("time"); cut = g.time.iloc[0] + (g.time.iloc[-1] - g.time.iloc[0]) * (1 - TEST_SHARE)
        tr, te = g[g.time < cut], g[g.time >= cut]
        for k in TPS:
            c = f"r_{k}"
            tstat = g[c].mean() / (g[c].std() / np.sqrt(len(g))) if g[c].std() > 0 else 0
            if len(g) >= MIN_N and tr[c].mean() > 0 and te[c].mean() > 0 and tstat >= MIN_T:
                good.append(f"{name} ({tfl}), Ziel {k}R: gesamt {g[c].mean():+.2f}R, Test {te[c].mean():+.2f}R, n={len(g)}, t={tstat:.1f}")
    return good


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    jobs = []     # (symbol, label, minutes, loader)
    for n, sym in S.FOREX.items():
        jobs.append((n, "15m", 15, lambda s=sym: hist_twelve(s, "15min", 5000)))
        jobs.append((n, "1h", 60, lambda s=sym: hist_twelve(s, "1h", 5000)))
    jobs.append(("BTCUSD", "15m", 15, lambda: hist_bitstamp("btcusd", 900, 90)))
    jobs.append(("BTCUSD", "1h", 60, lambda: hist_bitstamp("btcusd", 3600, 700)))
    rows = []
    for n, tfl, tf, load in jobs:
        try:
            df = indicators(load())
            if n in S.FOREX: time.sleep(8)
        except Exception as e:
            print(f"{n} {tfl}: {S.scrub(e)}"); continue
        print(f"{n} {tfl}: {len(df)} Kerzen ({df.time.min():%Y-%m-%d} bis {df.time.max():%Y-%m-%d})", flush=True)
        tr = trend_filter(df, "4h" if tf == 15 else "1D")
        for name, fn in STRATS.items():
            if name == "ORB London" and tf == 60: continue
            rows += evaluate(n, df, tf, fn(df, tf, tr), name, tfl)
    if not rows:
        print("Keine Signale."); return
    df = pd.DataFrame(rows)
    text = ("Strategie-Labor (nach Kosten, Einstieg zum Schlusskurs)\n\n" + summary(df) + "\n\n"
            + ("ERFÜLLEN BEIDE BEDINGUNGEN (n>=100, Train und Test positiv, t>=2,5):\n" + "\n".join(verdicts(df)) if verdicts(df)
               else "Keine Variante ist bei n>=100 im Train- und Test-Teil positiv und statistisch belastbar (t>=2,5).")
            + "\n\nHinweis: Viele getestete Varianten erhöhen die Gefahr von Zufallstreffern. Treffer bitte nur als Kandidaten für weitere Tests sehen.")
    print(text)
    df.to_csv("lab_trades.csv", index=False)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    for i in range(0, len(text), 3800):
        S.telegram(text[i:i + 3800], a.dry)


if __name__ == "__main__":
    main()
