#!/usr/bin/env python3
"""Labor 2: weitere Ideen, alle nach Kosten und mit Train/Test-Prüfung.
 1) Zeit-Muster (Stunde/Wochentag)      2) BTC-Funding als Gegen-Signal
 3) Limit-Einstieg am Rücksetzer        4) Tages-Trendfolge (Jahre an Daten)
 5) weitere Märkte (ETFs, Öl) wenn der Key sie liefert
Start: GitHub -> Actions -> lab2 -> Run workflow."""
import os, time, argparse
import requests
import numpy as np
import pandas as pd
import scanner as S
import lab as L

EXTRA = {"SPY": "SPY", "QQQ": "QQQ", "GLD": "GLD", "USOIL": "WTI/USD"}   # Name -> Twelve-Data-Symbol
EXTRA_COST_PCT = 0.0003            # angenommener Spread/Gebühr für ETFs und Öl (0,03 %)


def cost_abs(sym, price):
    if sym in S.COST_ABS:
        return S.COST_ABS[sym]
    if sym == "BTCUSD":
        return price * float(S.env("COST_CRYPTO_PCT", S.COST_CRYPTO_PCT))
    return price * EXTRA_COST_PCT


# ---------------- 1) Zeit-Muster ----------------
def seasonality(sym, df):
    d = df.copy()
    d["ret"] = (d.close - d.open) / d.atr.shift(1)
    d = d.dropna(subset=["ret"])
    if len(d) < 500:
        return []
    cut = d.time.iloc[0] + (d.time.iloc[-1] - d.time.iloc[0]) * 0.6
    cost_atr = cost_abs(sym, float(d.close.iloc[-1])) / float(d.atr.mean())
    out = []
    for key, name in ((d.time.dt.hour, "Stunde"), (d.time.dt.weekday, "Wochentag")):
        for v, g in d.groupby(key):
            if len(g) < 100 or g.ret.std() == 0:
                continue
            m = g.ret.mean(); t = m / (g.ret.std() / np.sqrt(len(g)))
            a1, a2 = g[g.time < cut].ret.mean(), g[g.time >= cut].ret.mean()
            out.append({"symbol": sym, "typ": name, "wert": int(v), "n": len(g), "mean_atr": m, "t": t,
                        "train": a1, "test": a2, "cost_atr": cost_atr})
    return out


def season_report(rows):
    if not rows:
        return "Zeit-Muster: keine Daten."
    d = pd.DataFrame(rows)
    strong = d[(d.t.abs() >= 3.5) & (np.sign(d.train) == np.sign(d.test)) & (np.sign(d.test) == np.sign(d.mean_atr))]
    lines = [f"Zeit-Muster: {len(d)} Zellen getestet, {len(strong)} mit |t|>=3,5 und gleichem Vorzeichen in Train und Test "
             f"(zufällig erwartet: ca. {len(d) * 0.0005:.1f})"]
    for _, r in strong.reindex(strong.t.abs().sort_values(ascending=False).index).head(8).iterrows():
        netto = abs(r.mean_atr) - r.cost_atr
        lines.append(f"  {r.symbol} {r.typ} {r.wert}{' UTC' if r.typ == 'Stunde' else ''}: {r.mean_atr:+.3f} ATR pro Kerze, t={r.t:.1f}, "
                     f"Kosten ca. {r.cost_atr:.3f} ATR -> {'ueber' if netto > 0 else 'unter'} den Kosten")
    return "\n".join(lines)


# ---------------- 2) BTC-Funding ----------------
def fetch_funding():
    r = requests.get("https://futures.kraken.com/derivatives/api/v4/historicalfundingrates",
                     params={"symbol": "PF_XBTUSD"}, timeout=30)
    r.raise_for_status()
    f = pd.DataFrame(r.json()["rates"])
    f["time"] = pd.to_datetime(f["timestamp"], utc=True)
    f["rate"] = f["relativeFundingRate"].astype(float)
    return f.sort_values("time")[["time", "rate"]].reset_index(drop=True)


def funding_signals(df, f, z_th=2.0):
    f = f.copy()
    f["f24"] = f.rate.rolling(24).mean()
    f["z"] = (f.f24 - f.f24.rolling(720).mean()) / f.f24.rolling(720).std()
    m = pd.merge_asof(df[["time"]], f[["time", "z"]].dropna(), on="time")
    z = m.z.values
    sig = []
    for i in range(60, len(df)):
        if np.isnan(z[i]) or np.isnan(df.atr.iloc[i]):
            continue
        c, a = df.close.iloc[i], df.atr.iloc[i]
        if z[i] >= z_th: sig.append((i, "short", c + 2 * a))     # sehr hohes Funding: Longs überhitzt
        if z[i] <= -z_th: sig.append((i, "long", c - 2 * a))
    return sig


# ---------------- 3) Limit-Einstieg ----------------
def evaluate_limit(sym, df, tf_min, sig, name, label, pull=0.5, valid=4):
    rows, last = [], {}
    hi, lo = df.high.values, df.low.values
    maxc = int(L.MAX_HOLD_H * 60 / tf_min)
    for i, side, stop in sig:
        if sym in S.FOREX and not (df.time.iloc[i].weekday() < 5 and S.FOREX_HOURS_UTC[0] <= df.time.iloc[i].hour < S.FOREX_HOURS_UTC[1]):
            continue
        if i - last.get(side, -999) < 4 * 60 // tf_min:
            continue
        last[side] = i
        sgn = 1 if side == "long" else -1
        lim = float(df.close.iloc[i] - sgn * pull * df.atr.iloc[i])
        risk = abs(lim - stop)
        if risk < 0.3 * df.atr.iloc[i] or (sgn == 1 and lim <= stop) or (sgn == -1 and lim >= stop):
            continue
        fill = None
        for j in range(i + 1, min(i + 1 + valid, len(df))):
            if (sgn == 1 and lo[j] <= lim) or (sgn == -1 and hi[j] >= lim):
                fill = j; break
        if fill is None:
            continue
        first = (lim, lo[fill]) if sgn == 1 else (hi[fill], lim)     # in der Füllkerze kein Zielgewinn, nur Stop-Risiko
        fut = [first] + list(zip(hi[fill + 1:fill + 1 + maxc], lo[fill + 1:fill + 1 + maxc]))
        if len(fut) < 4:
            continue
        cr = cost_abs(sym, lim) / risk
        row = {"time": df.time.iloc[i], "symbol": sym, "strategie": name, "ebene": label, "side": side}
        for k in L.TPS:
            st, _ = S.simulate_path(side, lim, stop, lim + sgn * k * risk, lim + sgn * k * risk, fut, "simple")
            row[f"r_{k}"] = (k if st == "tp1" else -1.0 if st == "stop" else 0.0) - cr
        rows.append(row)
    return rows


# ---------------- 4) Tages-Trendfolge ----------------
def daily_trend(sym, df, n_entry, atr_mult=3.0):
    hh = df.close.rolling(n_entry).max().shift(1); ll = df.close.rolling(n_entry).min().shift(1)
    trades, pos = [], None
    for i in range(max(n_entry, 20) + 1, len(df)):
        r = df.iloc[i]
        if pos:
            long_ = pos["side"] == "long"
            hit = r.low <= pos["stop"] if long_ else r.high >= pos["stop"]
            if hit:
                ex = min(pos["stop"], r.open) if long_ else max(pos["stop"], r.open)
                g = (ex - pos["entry"]) if long_ else (pos["entry"] - ex)
                trades.append({"time": pos["time"], "symbol": sym, "side": pos["side"],
                               "r": g / pos["risk"] - cost_abs(sym, pos["entry"]) / pos["risk"]})
                pos = None
            else:
                pos["best"] = max(pos["best"], r.close) if long_ else min(pos["best"], r.close)
                ns = pos["best"] - atr_mult * pos["atr"] if long_ else pos["best"] + atr_mult * pos["atr"]
                pos["stop"] = max(pos["stop"], ns) if long_ else min(pos["stop"], ns)
        if pos is None and not np.isnan(r.atr):
            if r.close > hh.iloc[i]:
                pos = {"side": "long", "entry": r.close, "atr": r.atr, "risk": atr_mult * r.atr, "best": r.close,
                       "stop": r.close - atr_mult * r.atr, "time": r.time}
            elif r.close < ll.iloc[i]:
                pos = {"side": "short", "entry": r.close, "atr": r.atr, "risk": atr_mult * r.atr, "best": r.close,
                       "stop": r.close + atr_mult * r.atr, "time": r.time}
    if pos:      # offene Position zum letzten Kurs bewerten
        c = df.close.iloc[-1]; g = (c - pos["entry"]) if pos["side"] == "long" else (pos["entry"] - c)
        trades.append({"time": pos["time"], "symbol": sym, "side": pos["side"], "r": g / pos["risk"] - cost_abs(sym, pos["entry"]) / pos["risk"]})
    return trades


def trend_report(tr_by_variant):
    lines, good = ["Tages-Trendfolge (alle Märkte zusammen, nach Kosten, R = Ergebnis / Anfangsrisiko 3 ATR):"], []
    for name, rows in tr_by_variant.items():
        if not rows:
            lines.append(f"  {name}: keine Trades"); continue
        d = pd.DataFrame(rows).sort_values("time")
        cut = d.time.iloc[0] + (d.time.iloc[-1] - d.time.iloc[0]) * 0.6
        a, b = d[d.time < cut].r, d[d.time >= cut].r
        t = d.r.mean() / (d.r.std() / np.sqrt(len(d))) if d.r.std() > 0 else 0
        lines.append(f"  {name}: n={len(d)}, Quote {round((d.r > 0).mean() * 100)}%, Ø {d.r.mean():+.2f}R, Summe {d.r.sum():+.1f}R, "
                     f"Train {a.mean():+.2f} | Test {b.mean():+.2f}, t={t:.1f}, Zeitraum {d.time.min():%Y}-{d.time.max():%Y}")
        per = d.groupby("symbol").r.agg(["count", "mean"])
        lines.append("    je Markt: " + " | ".join(f"{s}: n={int(r['count'])}, {r['mean']:+.2f}R" for s, r in per.iterrows()))
        if len(d) >= 100 and a.mean() > 0 and b.mean() > 0 and t >= 2.5:
            good.append(f"{name}: Ø {d.r.mean():+.2f}R, Test {b.mean():+.2f}R, n={len(d)}, t={t:.1f}")
    return "\n".join(lines), good


# ---------------- Hauptprogramm ----------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    report, season, limit_rows, daily = [], [], [], {"Donchian 50 Tage": [], "Donchian 100 Tage": []}
    notes = []

    def load(label, fn):
        try:
            return L.indicators(fn())
        except Exception as e:
            notes.append(f"{label}: {S.scrub(e)[:120]}"); return None

    # Forex / Gold
    for n, sym in S.FOREX.items():
        d1h = load(f"{n} 1h", lambda s=sym: L.hist_twelve(s, "1h", 5000)); time.sleep(8)
        d15 = load(f"{n} 15m", lambda s=sym: L.hist_twelve(s, "15min", 5000)); time.sleep(8)
        d1d = load(f"{n} 1d", lambda s=sym: L.hist_twelve(s, "1day", 5000)); time.sleep(8)
        run_symbol(n, d15, d1h, d1d, season, limit_rows, daily)
    # BTC
    d1h = load("BTC 1h", lambda: L.hist_bitstamp("btcusd", 3600, 700))
    d15 = load("BTC 15m", lambda: L.hist_bitstamp("btcusd", 900, 90))
    d1d = load("BTC 1d", lambda: L.hist_bitstamp("btcusd", 86400, 3000))
    run_symbol("BTCUSD", d15, d1h, d1d, season, limit_rows, daily)
    funding_text = "Funding-Gegensignal (BTC): "
    try:
        f = fetch_funding()
        if d1h is None: raise RuntimeError("keine BTC-1h-Daten")
        fr = pd.DataFrame(L.evaluate("BTCUSD", d1h, 60, funding_signals(d1h, f), "Funding-Kontra", "1h"))
        funding_text += (f"{len(f)} Funding-Werte ({f.time.min():%Y-%m-%d} bis {f.time.max():%Y-%m-%d}), Signale: {len(fr)}\n" +
                         (L.summary(fr) if len(fr) else "keine Signale") + "\n" +
                         ("  Kandidaten: " + "; ".join(L.verdicts(fr)) if len(fr) and L.verdicts(fr) else "  kein belastbarer Kandidat"))
    except Exception as e:
        funding_text += f"nicht testbar ({S.scrub(e)[:120]})"
    # Zusatzmärkte (nur Tagesdaten)
    got = []
    for n, sym in EXTRA.items():
        d = load(f"{n} 1d", lambda s=sym: L.hist_twelve(s, "1day", 5000)); time.sleep(8)
        if d is not None and len(d) > 300:
            got.append(n)
            for nm, ne in (("Donchian 50 Tage", 50), ("Donchian 100 Tage", 100)):
                daily[nm] += daily_trend(n, d, ne)

    text = ["Labor 2 (nach Kosten)\n", season_report(season), ""]
    if limit_rows:
        ld = pd.DataFrame(limit_rows)
        text += ["Limit-Einstieg am Rücksetzer (0,5 ATR unter/über Signalkurs, 4 Kerzen gültig):", L.summary(ld),
                 ("Kandidaten: " + "; ".join(L.verdicts(ld))) if L.verdicts(ld) else "Kein belastbarer Kandidat.", ""]
    text += [funding_text, ""]
    tr_text, good = trend_report(daily)
    text += [tr_text, ("Kandidaten: " + "; ".join(good)) if good else "Kein belastbarer Kandidat.", ""]
    text.append("Zusatzmärkte geladen: " + (", ".join(got) if got else "keine (Key liefert sie nicht)"))
    if notes:
        text.append("Nicht geladen: " + " | ".join(notes))
    text.append("\nNicht getestet: Nachrichten-Reaktion (keine historischen Kalenderdaten) und Wirtschaftsdaten (nicht kostenlos verlässlich).")
    text.append("Viele getestete Varianten erhöhen die Gefahr von Zufallstreffern. Kandidaten brauchen eine zweite Prüfung.")
    out = "\n".join(text)
    print(out)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + out + "\n```\n")
    if daily["Donchian 50 Tage"]:
        pd.DataFrame(daily["Donchian 50 Tage"]).to_csv("lab2_daily_trades.csv", index=False)
    for i in range(0, len(out), 3800):
        S.telegram(out[i:i + 3800], a.dry)


def run_symbol(n, d15, d1h, d1d, season, limit_rows, daily):
    if d1h is not None:
        season += seasonality(n, d1h)
    for df, tf, lab in ((d15, 15, "15m"), (d1h, 60, "1h")):
        if df is None: continue
        tr = L.trend_filter(df, "4h" if tf == 15 else "1D")
        for name, fn in (("Trend-Ruecksetzer", L.strat_pullback), ("Donchian-Ausbruch", L.strat_donchian)):
            limit_rows += evaluate_limit(n, df, tf, fn(df, tf, tr), name + " (Limit)", lab)
    if d1d is not None and len(d1d) > 300:
        for nm, ne in (("Donchian 50 Tage", 50), ("Donchian 100 Tage", 100)):
            daily[nm] += daily_trend(n, d1d, ne)


if __name__ == "__main__":
    main()
