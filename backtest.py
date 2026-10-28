#!/usr/bin/env python3
"""Backtest der Scanner-Regeln auf historischen 15m-Kerzen (nutzt dieselben Funktionen wie scanner.py).
Start: GitHub -> Actions -> backtest -> Run workflow. Ergebnis kommt per Telegram, als CSV und im Lauf-Protokoll.
NICHT simuliert: News-Filter, Korrelationsfilter, Tageslimits, Slippage. Kosten sind pauschal geschätzt."""
import os, sys, io, time, argparse
from datetime import datetime, timezone
import requests
import pandas as pd
import scanner as S

WINDOW = 400          # Kerzen, die der Scanner pro Auswertung "sieht"
FOREX_TARGET = 5000   # Kerzen je Forex-Symbol (Twelve Data Maximum je Abruf, ca. 10 Wochen)
CRYPTO_DAYS = 60      # Tage Historie für Krypto (Bitstamp)


def hist_bitstamp(pair, days):
    """Bitstamp OHLC, 15m, vorwärts blättern (je Abruf max. 1000 Kerzen)."""
    start = int(time.time()) - days * 86400
    rows = []
    while True:
        r = requests.get(f"https://www.bitstamp.net/api/v2/ohlc/{pair}/", params={"step": 900, "limit": 1000, "start": start}, timeout=30)
        r.raise_for_status()
        part = r.json()["data"]["ohlc"]
        if not part:
            break
        rows += part
        last = int(part[-1]["timestamp"])
        if last <= start or len(part) < 2:
            break
        start = last + 900
        if start > time.time():
            break
        time.sleep(0.5)
    df = pd.DataFrame(rows).drop_duplicates("timestamp")
    df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df.sort_values("time")[["time", "open", "high", "low", "close", "volume"]].iloc[:-1].reset_index(drop=True)


def hist_twelve(symbol):
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": "15min", "outputsize": FOREX_TARGET, "timezone": "UTC",
        "apikey": S.env("TWELVEDATA_KEY", required=True)}, timeout=60)
    j = r.json()
    if j.get("status") == "error":
        raise RuntimeError(j.get("message"))
    df = pd.DataFrame(j["values"])
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = float("nan")
    return df.sort_values("time")[["time", "open", "high", "low", "close", "volume"]].iloc[:-1].reset_index(drop=True)


GRID_TP = (0.5, 1.0, 1.5, 2.0, 3.0)    # Ziel in Vielfachen des (skalierten) Risikos
GRID_STOP = (1.0, 1.5, 2.0)            # Stop-Weite relativ zum Standard-Stop


def run_symbol(sym, df, modes=("simple", "partial")):
    trades, last_seen = [], {}
    hi, lo = df.high.values, df.low.values
    maxc = S.TRADE_MAX_H * 4
    for i in range(WINDOW, len(df) - 1):
        t = df.time.iloc[i - 1]
        if sym in S.FOREX and not (t.weekday() < 5 and S.FOREX_HOURS_UTC[0] <= t.hour < S.FOREX_HOURS_UTC[1]):
            continue
        for s in S.find_setups(sym, df.iloc[i - WINDOW:i]):
            ts = pd.Timestamp(s["candle_ts"])
            if sym in S.FOREX and not (ts.weekday() < 5 and S.FOREX_HOURS_UTC[0] <= ts.hour < S.FOREX_HOURS_UTC[1]):
                continue
            if s["id"] in last_seen and ts - last_seen[s["id"]] < pd.Timedelta(hours=S.COOLDOWN_H):
                continue
            last_seen[s["id"]] = ts
            q, _, _ = S.score_free(s)
            fut = list(zip(hi[i:i + maxc], lo[i:i + maxc]))
            if len(fut) < 8:
                continue
            sgn = 1 if s["side"] == "long" else -1
            aligned = (s["htf_trend_1h"] == "up") == (s["side"] == "long") and s["htf_trend_1h"] != "flat"
            row = {"time": ts.strftime("%Y-%m-%d %H:%M"), "symbol": sym, "setup": s["setup"], "side": s["side"], "quality": q,
                   "session": "aktiv" if (S.SESSION_UTC[0] <= ts.hour < S.SESSION_UTC[1]) else "ruhig",
                   "trend": "mit" if aligned else ("seitwaerts" if s["htf_trend_1h"] == "flat" else "gegen"),
                   "room_r": s.get("room_r"), "body_atr": s["body_atr"], "atr_ratio": s.get("atr_ratio"),
                   "vol_ratio": s.get("vol_ratio"), "stop_atr": round(s["risk_dist"] / s["atr15"], 2), "hour": ts.hour}
            cr = S.cost_r(sym, s["entry"], s["risk_dist"])
            row["cost_r"] = round(cr, 3)
            for m in modes:
                st, r = S.simulate_path(s["side"], s["entry"], s["stop"], s["tp1"], s["tp2"], fut, m)
                if st in ("open", "tp1_open"):
                    r = r if st == "tp1_open" else 0.0
                row[f"r_{m}"] = round(r, 3); row[f"net_{m}"] = round(r - cr, 3); row[f"st_{m}"] = st
            # Raster: Stop-Weite x Ziel, jeweils einfacher Modus, Ergebnis in R des skalierten Risikos, Kosten umgerechnet
            for sc in GRID_STOP:
                risk = s["risk_dist"] * sc
                stop = s["entry"] - sgn * risk
                crs = S.cost_r(sym, s["entry"], risk)
                for k in GRID_TP:
                    tp = s["entry"] + sgn * k * risk
                    st, r = S.simulate_path(s["side"], s["entry"], stop, tp, tp, fut, "simple")
                    r = k if st == "tp1" else (-1.0 if st == "stop" else 0.0)
                    row[f"g_{sc}_{k}"] = round(r - crs, 3)
            trades.append(row)
    return trades


def grid_report(df):
    lines = ["Raster Stop x Ziel (einfacher Modus, nach Kosten): Trefferquote / Erwartung in R pro Trade"]
    for sc in GRID_STOP:
        cells = []
        for k in GRID_TP:
            x = df[f"g_{sc}_{k}"]
            cells.append(f"Ziel {k}R: {round((x > 0).mean() * 100)}% / {x.mean():+.2f}")
        lines.append(f"  Stop x{sc}: " + " | ".join(cells))
    return "\n".join(lines)


def feature_report(df, col="net_simple"):
    lines = ["Merkmale (Standard 2R, nach Kosten): n, Quote, Erwartung"]
    def row(name, mask):
        x = df.loc[mask, col]
        if len(x) >= 20:
            lines.append(f"  {name}: n={len(x)}, {round((x > 0).mean() * 100)}%, {x.mean():+.2f}R")
    for v in ("mit", "gegen", "seitwaerts"):
        row(f"1h-Trend {v}", df.trend == v)
    row("Platz bis Level < 1R", df.room_r < 1); row("Platz 1-2R", (df.room_r >= 1) & (df.room_r < 2)); row("Platz >= 2R", df.room_r >= 2)
    row("kleiner Stop (<1 ATR)", df.stop_atr < 1); row("mittlerer Stop (1-2 ATR)", (df.stop_atr >= 1) & (df.stop_atr < 2)); row("grosser Stop (>2 ATR)", df.stop_atr >= 2)
    row("Signalkerze schwach (<0,6 ATR)", df.body_atr < 0.6); row("Signalkerze stark (>=0,6 ATR)", df.body_atr >= 0.6)
    row("Volatilitaet niedrig (<0,8)", df.atr_ratio < 0.8); row("Volatilitaet normal/hoch", df.atr_ratio >= 0.8)
    for lo_h, hi_h in ((0, 7), (7, 12), (12, 17), (17, 24)):
        row(f"Stunde {lo_h}-{hi_h} UTC", (df.hour >= lo_h) & (df.hour < hi_h))
    return "\n".join(lines)


def summarize(df, col, by):
    out = []
    for key, g in df.groupby(by):
        x = g.sort_values("time")[col]
        eq = x.cumsum(); dd = (eq.cummax() - eq).max()
        streak = cur = 0
        for v in x:
            cur = cur + 1 if v < 0 else 0; streak = max(streak, cur)
        out.append({by: key, "n": len(g), "trefferquote": round((x > 0).mean() * 100), "erwartung_R": round(x.mean(), 2),
                    "summe_R": round(x.sum(), 1), "max_drawdown_R": round(dd, 1), "max_verlustserie": streak})
    return pd.DataFrame(out)


def report(df):
    lines = [f"Backtest: {len(df)} Signale ({df.time.min()[:10]} bis {df.time.max()[:10]})"]
    for m in ("simple", "partial"):
        net, gross = df[f"net_{m}"], df[f"r_{m}"]
        lines.append(f"\nModus {m}: Summe {gross.sum():+.1f}R brutto, {net.sum():+.1f}R nach Kosten, Erwartung {net.mean():+.2f}R/Trade")
        for by in ("quality", "setup", "symbol", "session", "side"):
            sm = summarize(df, f"net_{m}", by)
            lines.append(f"  nach {by}: " + " | ".join(
                f"{r[by]}: n={r['n']}, {r['trefferquote']}%, {r['erwartung_R']:+.2f}R" + ("*" if r["n"] < 30 else "") for _, r in sm.iterrows()))
        dd = summarize(df.assign(all="alle"), f"net_{m}", "all").iloc[0]
        lines.append(f"  Max-Drawdown {dd['max_drawdown_R']}R, längste Verlustserie {dd['max_verlustserie']}")
    lines.append("\n" + grid_report(df))
    lines.append("\n" + feature_report(df))
    lines.append("\n* = weniger als 30 Trades, kaum aussagekräftig.")
    lines.append("Nicht simuliert: News-Filter, Korrelationsfilter, Tageslimits, Slippage. Vergangenheit ist keine Prognose; "
                 "bei vielen getesteten Regeln steigt die Gefahr, zufällig gute Zahlen zu sehen.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    S.load_env()
    data = {}
    for n, p in S.CRYPTO.items():
        if a.only and n not in a.only.split(","): continue
        try: data[n] = hist_bitstamp(n.lower(), CRYPTO_DAYS)
        except Exception as e: print(f"{n}: {S.scrub(e)}")
    if S.env("TWELVEDATA_KEY"):
        for n, sym in S.FOREX.items():
            if a.only and n not in a.only.split(","): continue
            try: data[n] = hist_twelve(sym); time.sleep(8)
            except Exception as e: print(f"{n}: {S.scrub(e)}")
    rows = []
    for n, df in data.items():
        print(f"{n}: {len(df)} Kerzen ({df.time.min():%Y-%m-%d} bis {df.time.max():%Y-%m-%d})", flush=True)
        rows += run_symbol(n, df)
    if not rows:
        print("Keine Signale/Daten."); S.telegram("Backtest: keine Daten oder keine Signale.", a.dry); return
    df = pd.DataFrame(rows)
    text = report(df)
    print(text)
    df.to_csv("backtest_trades.csv", index=False)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ: open(summ, "a").write("```\n" + text + "\n```\n")
    S.telegram(text[:3900], a.dry)
    S.telegram_file("sendDocument", df.to_csv(index=False).encode(), "backtest_trades.csv", "Alle simulierten Signale", a.dry)


if __name__ == "__main__":
    main()
