#!/usr/bin/env python3
"""Labor 3: Prüft die Tages-Trendfolge (Donchian) gründlich: Long/Short getrennt, pro Jahr, Portfolio-Verlauf mit Drawdown,
Kosten-Stresstest, Vergleich mit Kaufen-und-Halten, mehr Märkte. Nur Tagesdaten.
Start: GitHub -> Actions -> lab3 -> Run workflow."""
import os, time, argparse
import numpy as np
import pandas as pd
import scanner as S
import lab as L
from lab2 import cost_abs

ETFS = {"SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "IWM": "IWM", "EFA": "EFA", "EEM": "EEM",
        "TLT": "TLT", "GLD": "GLD", "SLV": "SLV", "USO": "USO"}
ATR_MULT = 3.0
RISK_PCT = 1.0          # Annahme für den Portfolio-Verlauf: 1 % Risiko je Trade, ohne Zinseszins


def load_daily(fn, tries=3):
    for k in range(tries):
        try:
            return L.indicators(fn())
        except Exception as e:
            if "credits" in str(e).lower() and k < tries - 1:
                time.sleep(65); continue
            raise


def trend_trades(sym, df, n_entry, atr_mult=ATR_MULT):
    hh = df.close.rolling(n_entry).max().shift(1); ll = df.close.rolling(n_entry).min().shift(1)
    out, pos = [], None

    def close_pos(p, t, ex):
        g = (ex - p["entry"]) if p["side"] == "long" else (p["entry"] - ex)
        out.append({"entry_time": p["time"], "exit_time": t, "symbol": sym, "side": p["side"],
                    "gross": g / p["risk"], "cost": cost_abs(sym, p["entry"]) / p["risk"]})

    for i in range(max(n_entry, 20) + 1, len(df)):
        r = df.iloc[i]
        if pos:
            long_ = pos["side"] == "long"
            if (r.low <= pos["stop"]) if long_ else (r.high >= pos["stop"]):
                close_pos(pos, r.time, min(pos["stop"], r.open) if long_ else max(pos["stop"], r.open)); pos = None
            else:
                pos["best"] = max(pos["best"], r.close) if long_ else min(pos["best"], r.close)
                ns = pos["best"] - atr_mult * pos["atr"] if long_ else pos["best"] + atr_mult * pos["atr"]
                pos["stop"] = max(pos["stop"], ns) if long_ else min(pos["stop"], ns)
        if pos is None and not np.isnan(r.atr):
            side = "long" if r.close > hh.iloc[i] else "short" if r.close < ll.iloc[i] else None
            if side:
                sg = 1 if side == "long" else -1
                pos = {"side": side, "entry": r.close, "atr": r.atr, "risk": atr_mult * r.atr, "best": r.close,
                       "stop": r.close - sg * atr_mult * r.atr, "time": r.time}
    if pos:
        close_pos(pos, df.time.iloc[-1], df.close.iloc[-1])
    return out


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


def portfolio(d, col="net"):
    """Portfolio-Verlauf: jeder Trade bringt RISK_PCT * R Prozent des Kontos, gebucht am Ausstiegstag (ohne Zinseszins)."""
    p = d.sort_values("exit_time")
    eq = (p[col] * RISK_PCT).cumsum()
    dd = (eq.cummax() - eq)
    years = max((p.exit_time.max() - p.entry_time.min()).days / 365.25, 0.1)
    # längste Zeit unter dem Höchststand
    peak_t, longest, last_peak = None, pd.Timedelta(0), p.exit_time.iloc[0]
    for t, e, m in zip(p.exit_time, eq, eq.cummax()):
        if e >= m: last_peak = t
        longest = max(longest, t - last_peak)
    # gleichzeitig offene Positionen
    ev = sorted([(t, 1) for t in d.entry_time] + [(t, -1) for t in d.exit_time])
    cur = mx = 0
    for _, v in ev:
        cur += v; mx = max(mx, cur)
    return {"total": eq.iloc[-1], "per_year": eq.iloc[-1] / years, "maxdd": dd.max(), "longest_days": longest.days, "max_open": mx, "years": years}


def report_variant(name, d):
    d = d.copy(); d["net"] = d.gross - d.cost; d["net2"] = d.gross - 2 * d.cost
    d["year"] = d.exit_time.dt.year
    lines = [f"== {name} =="]
    pf = portfolio(d)
    yr = d.groupby("year").net.sum()
    ty = tstat(yr.values)
    lines.append(f"Gesamt: n={len(d)}, Quote {round((d.net > 0).mean() * 100)}%, Ø {d.net.mean():+.2f}R, t(Trades)={tstat(d.net):.1f}, "
                 f"t(Jahre)={ty:.1f}, Jahre im Plus: {(yr > 0).sum()}/{len(yr)}")
    lines.append(f"Kosten verdoppelt: Ø {d.net2.mean():+.2f}R")
    lines.append(f"Portfolio bei {RISK_PCT:g}% Risiko je Trade (ohne Zinseszins): {pf['total']:+.0f}% in {pf['years']:.0f} Jahren "
                 f"= {pf['per_year']:+.1f}%/Jahr, größter Rückgang {pf['maxdd']:.0f}%, längste Zeit unter dem Höchststand {pf['longest_days'] / 365.25:.1f} Jahre, "
                 f"bis zu {pf['max_open']} Positionen gleichzeitig offen")
    for side in ("long", "short"):
        s = d[d.side == side]
        if len(s): lines.append(f"  {side}: n={len(s)}, Ø {s.net.mean():+.2f}R, Summe {s.net.sum():+.0f}R, t={tstat(s.net):.1f}")
    lines.append("  je Markt: " + " | ".join(f"{k}: n={len(g)}, {g.net.mean():+.2f}R" for k, g in d.groupby("symbol")))
    lines.append("  je Jahr (Summe R): " + " ".join(f"{int(y) % 100:02d}:{v:+.0f}" for y, v in yr.items()))
    half = d.entry_time.min() + (d.entry_time.max() - d.entry_time.min()) / 2
    lines.append(f"  erste Hälfte {d[d.entry_time < half].net.mean():+.2f}R | zweite Hälfte {d[d.entry_time >= half].net.mean():+.2f}R")
    return "\n".join(lines), d


def buyhold(frames):
    """Gleichgewichtetes Kaufen-und-Halten über die Börsen-/Rohstoff-Märkte (ohne Forex)."""
    rets = []
    for n, df in frames.items():
        if n in S.FOREX: continue
        s = df.set_index("time").close.pct_change().dropna(); s.index = s.index.tz_localize(None).normalize(); rets.append(s.groupby(level=0).last().rename(n))
    if not rets: return "Kaufen-und-halten: keine Daten."
    r = pd.concat(rets, axis=1).mean(axis=1, skipna=True)
    eq = (1 + r).cumprod(); years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1; dd = (1 - eq / eq.cummax()).max()
    return f"Vergleich Kaufen-und-halten (gleich gewichtet, {len(rets)} Märkte, {years:.0f} Jahre): {cagr * 100:+.1f}%/Jahr, größter Rückgang {dd * 100:.0f}%"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    frames, notes = {}, []
    for n, sym in list(S.FOREX.items()) + list(ETFS.items()):
        tw = sym if n in ETFS else sym
        try:
            frames[n] = load_daily(lambda s=tw: L.hist_twelve(s, "1day", 5000))
        except Exception as e:
            notes.append(f"{n}: {S.scrub(e)[:90]}")
        time.sleep(10)
    try:
        frames["BTCUSD"] = load_daily(lambda: L.hist_bitstamp("btcusd", 86400, 3000))
    except Exception as e:
        notes.append(f"BTCUSD: {S.scrub(e)[:90]}")
    for n, df in frames.items():
        print(f"{n}: {len(df)} Tage ({df.time.min():%Y-%m-%d} bis {df.time.max():%Y-%m-%d})", flush=True)
    text = ["Labor 3: Tages-Trendfolge (nach Kosten; R = Ergebnis / Anfangsrisiko 3 ATR)\n"]
    allv = []
    for nm, ne in (("Donchian 50 Tage", 50), ("Donchian 100 Tage", 100)):
        rows = []
        for n, df in frames.items():
            if len(df) > 300: rows += trend_trades(n, df, ne)
        if not rows: continue
        t, d = report_variant(nm, pd.DataFrame(rows)); text.append(t); text.append("")
        allv.append(d.assign(variante=nm))
        # ohne Forex-Paare, nur Aktien/Rohstoffe/Krypto
        nofx = pd.DataFrame(rows); nofx = nofx[~nofx.symbol.isin(S.FOREX)]
        if len(nofx) > 30:
            t2, _ = report_variant(nm + " (ohne Forex-Paare, mit Gold)", nofx[~nofx.symbol.isin(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"])])
            text.append(t2); text.append("")
    text.append(buyhold(frames))
    text.append("Geladen: " + ", ".join(frames))
    if notes: text.append("Nicht geladen: " + " | ".join(notes))
    text.append("\nHinweise: Positionen überlappen und sind zwischen Märkten korreliert (t-Werte eher optimistisch, deshalb zusätzlich t über Jahre). "
                "2007-2026 war ein langer Aufwärtstrend für Aktien, Gold und BTC. Einstieg zum Schlusskurs, Kosten pauschal. Vergangenheit ist keine Prognose.")
    out = "\n".join(text); print(out)
    if allv: pd.concat(allv).to_csv("lab3_trades.csv", index=False)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + out + "\n```\n")
    for i in range(0, len(out), 3800):
        S.telegram(out[i:i + 3800], a.dry)


if __name__ == "__main__":
    main()
