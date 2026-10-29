#!/usr/bin/env python3
"""Forex-Labor: vier vorab festgelegte Ideen, alle nach Kosten.
 1 Carry + Trend (Tag):  Long, wenn Zins Basiswährung > Gegenwährung + 0,5 % UND Kurs über 100-Tage-Schnitt (Short umgekehrt).
                         Ausstieg, wenn eine Bedingung wegfällt; Schutz-Stop 3 ATR; Zinsertrag/-kosten werden mitgerechnet.
 2 US-Daten (1h):        Nach Arbeitsmarkt (NFP) und Inflation (CPI): Richtung der Termin-Kerze handeln, Stop andere Seite der Kerze,
                         Ziel 1,5R, spätestens nach 8 Stunden raus. Kontrolle: dieselbe Regel an Tagen OHNE Termin zur selben Uhrzeit.
 3 Asia-Spanne (1h):     Spanne 00-06 UTC. 07-11 UTC: Docht über/unter die Spanne und Schluss zurück -> Gegenrichtung,
                         Stop hinter dem Docht, Ziel Mitte der Spanne (nur wenn Ziel >= 1R).
 4 EUR/USD vs GBP/USD (1h): Abstand der beiden (log) mehr als 2 Standardabw. vom 10-Tage-Schnitt -> zurück zur Mitte,
                         Stop bei 3,5 Standardabw., spätestens nach 5 Tagen raus. Zwei Positionen = doppelte Kosten.
Bestehen: n>=50, t>=2,5, beide Zeit-Hälften positiv, bei doppelten Kosten positiv (Idee 2 zusätzlich besser als die Kontrolle).
Start: GitHub -> Actions -> fxlab -> Run workflow. Braucht ca. 20 Twelve-Data-Abrufe und den FRED-Schlüssel."""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
import scanner as S
import lab as L
import multi as M

PAIRS_D = {"EURUSD": ("EUR/USD", "EZ", "US"), "GBPUSD": ("GBP/USD", "GB", "US"), "USDJPY": ("USD/JPY", "US", "JP"),
           "AUDUSD": ("AUD/USD", "AU", "US"), "USDCAD": ("USD/CAD", "US", "CA")}
PAIRS_H = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY"}
RATES = {"US": "IRSTCI01USM156N", "EZ": "IRSTCI01EZM156N", "GB": "IRSTCI01GBM156N", "JP": "IRSTCI01JPM156N",
         "AU": "IRSTCI01AUM156N", "CA": "IRSTCI01CAM156N"}           # kurzfristige Zinsen (OECD, monatlich)
RELEASES = {"NFP": 50, "CPI": 10}
CARRY_SHARE = 0.5      # nur die Hälfte des Zinsunterschieds anrechnen: Broker-Swaps sind meist deutlich ungünstiger als der reine Zinsunterschied                                   # FRED-Release-IDs: Employment Situation, CPI


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0


def fred_get(path, **params):
    key = S.env("FRED_API_KEY", required=True)
    for k in range(4):
        try:
            r = requests.get(f"https://api.stlouisfed.org/fred/{path}", params={**params, "api_key": key, "file_type": "json"}, timeout=60)
            r.raise_for_status(); return r.json()
        except Exception as e:
            last = e; time.sleep(10 * (k + 1))
    raise RuntimeError(S.scrub(last)[:120])


def fetch_rates():
    out = {}
    for c, sid in RATES.items():
        j = fred_get("series/observations", series_id=sid, observation_start="2000-01-01")
        d = pd.DataFrame(j["observations"]); d["date"] = pd.to_datetime(d["date"], utc=True)
        d["v"] = pd.to_numeric(d["value"], errors="coerce"); d = d.dropna(subset=["v"])
        d["valid_from"] = d.date + pd.Timedelta(days=45)            # Monatswert erst ca. 6 Wochen später sicher bekannt
        out[c] = d[["valid_from", "v"]].sort_values("valid_from")
    return out


def fetch_release_dates():
    out = {}
    for name, rid in RELEASES.items():
        j = fred_get("release/dates", release_id=rid, realtime_start="2015-01-01", include_release_dates_with_no_data="false", limit=1000)
        out[name] = sorted({pd.Timestamp(x["date"]) for x in j.get("release_dates", [])})
    return out


def report(name, d, col="r", extra=""):
    if len(d) < 10: return [f"{name}: zu wenige Trades ({len(d)})"], False
    d = d.sort_values("time"); half = d.time.iloc[len(d) // 2]
    a, b = d[d.time < half][col], d[d.time >= half][col]
    t = tstat(d[col]); c2 = (d[col] - d["cost"]).mean()
    ok = len(d) >= 50 and t >= 2.5 and a.mean() > 0 and b.mean() > 0 and c2 > 0
    line = (f"{name}: n={len(d)}, Quote {round((d[col] > 0).mean() * 100)}%, Ø {d[col].mean():+.2f}R, t={t:.1f}, "
            f"1. Hälfte {a.mean():+.2f} / 2. Hälfte {b.mean():+.2f}, Kosten x2 {c2:+.2f}R{extra}")
    return [line], ok


# ---------------- Idee 1: Carry + Trend (Tagesdaten) ----------------
def carry_trend(sym, df, rates, base, quote):
    df = L.indicators(df).reset_index(drop=True)
    rb = pd.merge_asof(df[["time"]], rates[base].rename(columns={"valid_from": "time"}), on="time").v.values
    rq = pd.merge_asof(df[["time"]], rates[quote].rename(columns={"valid_from": "time"}), on="time").v.values
    sma = df.close.rolling(100).mean().values
    cl, hi, lo, atr = df.close.values, df.high.values, df.low.values, df.atr.values
    trades, pos = [], None
    for i in range(101, len(df)):
        diff = rb[i] - rq[i]
        if np.isnan(diff) or np.isnan(sma[i]) or np.isnan(atr[i]): continue
        want = 1 if (diff > 0.5 and cl[i] > sma[i]) else -1 if (diff < -0.5 and cl[i] < sma[i]) else 0
        if pos:
            sg = pos["sg"]; pos["carry"] += CARRY_SHARE * sg * diff / 100 / 365 * pos["entry"] / pos["risk"] * (df.time.iloc[i] - df.time.iloc[i - 1]).days
            stopped = (lo[i] <= pos["stop"]) if sg == 1 else (hi[i] >= pos["stop"])
            if stopped or want != sg:
                ex = pos["stop"] if stopped else cl[i]
                g = (ex - pos["entry"]) * sg / pos["risk"] + pos["carry"]
                c = S.cost_r(sym, pos["entry"], pos["risk"])
                trades.append({"time": pos["time"], "r": g - c, "cost": c, "days": (df.time.iloc[i] - pos["time"]).days, "carry": pos["carry"]})
                pos = None
        if pos is None and want != 0:
            risk = 3 * atr[i]
            pos = {"sg": want, "entry": cl[i], "risk": risk, "stop": cl[i] - want * risk, "time": df.time.iloc[i], "carry": 0.0}
    return trades


# ---------------- Idee 2: US-Daten ----------------
def release_utc(day):
    """8:30 Uhr New York als UTC (Sommerzeit-genau)."""
    return pd.Timestamp(day.strftime("%Y-%m-%d") + " 08:30", tz="America/New_York").tz_convert("UTC")


def event_trades(sym, h, times):
    hi, lo, cl, op, atr = h.high.values, h.low.values, h.close.values, h.open.values, h.atr.values
    idx = pd.Index(h.time); out = []
    for t in times:
        bar = t.floor("1h")
        if bar not in idx: continue
        i = idx.get_loc(bar)
        if i + 9 >= len(h) or np.isnan(atr[i]) or cl[i] == op[i]: continue
        sg = 1 if cl[i] > op[i] else -1
        entry, stop = cl[i], (lo[i] if sg == 1 else hi[i])
        risk = abs(entry - stop)
        if not (0.5 * atr[i] <= risk <= 3 * atr[i]): continue
        tp = entry + sg * 1.5 * risk
        st, _ = S.simulate_path("long" if sg == 1 else "short", entry, stop, tp, tp, list(zip(hi[i + 1:i + 9], lo[i + 1:i + 9])), "simple")
        g = 1.5 if st == "tp1" else -1.0 if st == "stop" else (cl[min(i + 8, len(h) - 1)] - entry) * sg / risk
        c = S.cost_r(sym, entry, risk)
        out.append({"time": t, "r": g - c, "cost": c})
    return out


# ---------------- Idee 3: Asia-Spanne ----------------
def asia_fade(sym, h):
    out = []
    for day, g in h.groupby(h.time.dt.date):
        asia = g[g.time.dt.hour < 6]
        if len(asia) < 5: continue
        top, bot = asia.high.max(), asia.low.min(); mid = (top + bot) / 2
        done = set()
        for idx_, r in g[(g.time.dt.hour >= 7) & (g.time.dt.hour < 11)].iterrows():
            a = r.atr
            if np.isnan(a) or (top - bot) < 1.0 * a: continue
            for side, cond, stop in (("short", r.high > top and r.close < top, r.high + 0.2 * a), ("long", r.low < bot and r.close > bot, r.low - 0.2 * a)):
                if not cond or side in done: continue
                entry = r.close; risk = abs(entry - stop); k = abs(mid - entry) / risk if risk else 0
                if risk <= 0 or k < 1.0: continue
                done.add(side)
                i = h.index.get_loc(idx_)
                st, _ = S.simulate_path(side, entry, stop, mid, mid, list(zip(h.high.values[i + 1:i + 13], h.low.values[i + 1:i + 13])), "simple")
                g_ = k if st == "tp1" else -1.0 if st == "stop" else (h.close.values[min(i + 12, len(h) - 1)] - entry) * (1 if side == "long" else -1) / risk
                c = S.cost_r(sym, entry, risk)
                out.append({"time": r.time, "r": g_ - c, "cost": c})
    return out


# ---------------- Idee 4: EUR/USD gegen GBP/USD ----------------
def pair_trades(e, g):
    d = pd.merge(e[["time", "close"]], g[["time", "close"]], on="time", suffixes=("_e", "_g")).dropna()
    s = np.log(d.close_e) - np.log(d.close_g)
    mu, sd = s.rolling(240).mean(), s.rolling(240).std()
    z = ((s - mu) / sd).values; sv, sdv = s.values, sd.values
    cost = (S.COST_ABS["EURUSD"] / d.close_e.mean() + S.COST_ABS["GBPUSD"] / d.close_g.mean())   # in log-Einheiten
    out, i = [], 241
    while i < len(d) - 1:
        if np.isnan(z[i]) or abs(z[i]) < 2: i += 1; continue
        sg = -1 if z[i] > 0 else 1                        # Abstand zu groß -> auf Rückkehr setzen
        entry, risk = sv[i], 1.5 * sdv[i]; stop = entry - sg * risk; target = mu.values[i]
        res = None
        for j in range(i + 1, min(i + 121, len(d))):
            if (sv[j] - stop) * sg <= 0: res = -1.0; break
            if (sv[j] - target) * sg >= 0: res = abs(target - entry) / risk; break
        if res is None: j = min(i + 120, len(d) - 1); res = (sv[j] - entry) * sg / risk
        c = cost / risk
        out.append({"time": d.time.iloc[i], "r": res - c, "cost": c}); i = j + 1
    return out


def run(daily, hourly, rates, rel, dry, notes):
    lines = ["Forex-Labor (nach Kosten, Regeln vorab festgelegt)\n"]
    passed = []
    # 1
    ct = []
    for sym, (ts, b, q) in PAIRS_D.items():
        if sym in daily and rates: ct += [dict(x, sym=sym) for x in carry_trend(sym, daily[sym], rates, b, q)]
    if ct:
        d = pd.DataFrame(ct)
        l, ok = report("1 Carry + Trend (Tag, alle Paare)", d, extra=f", Ø Haltedauer {d.days.mean():.0f} Tage, Ø Zinsanteil {d.carry.mean():+.2f}R")
        lines += l + ["   je Paar: " + " | ".join(f"{k}: n={len(v)} {v.r.mean():+.2f}R" for k, v in d.groupby("sym"))]
        if ok: passed.append("1 Carry + Trend")
    # 2
    if rel and hourly:
        ev, ctl = [], []
        all_days = None
        for name, days in rel.items():
            times = [release_utc(x) for x in days]
            for sym, h in hourly.items():
                ev += [dict(x, sym=sym, typ=name) for x in event_trades(sym, h, times)]
        rel_days = {t.date() for days in rel.values() for t in days}
        for sym, h in hourly.items():
            cand = sorted({t.date() for t in h.time if t.weekday() < 5} - rel_days)
            ctl += event_trades(sym, h, [release_utc(pd.Timestamp(x)) for x in cand])
        if ev:
            d, c = pd.DataFrame(ev), pd.DataFrame(ctl)
            better = len(c) and d.r.mean() > c.r.mean() and abs((d.r.mean() - c.r.mean()) / np.sqrt(d.r.var() / len(d) + c.r.var() / len(c))) >= 2
            l, ok = report("2 US-Daten (NFP/CPI, 1h)", d, extra=f" | Kontrolle ohne Termin: Ø {c.r.mean():+.2f}R (n={len(c)})")
            lines += l + ["   je Termin: " + " | ".join(f"{k}: n={len(v)} {v.r.mean():+.2f}R" for k, v in d.groupby("typ"))
                          + " | je Paar: " + " | ".join(f"{k}: {v.r.mean():+.2f}R" for k, v in d.groupby("sym"))]
            if ok and better: passed.append("2 US-Daten")
    # 3
    af = []
    for sym, h in hourly.items(): af += [dict(x, sym=sym) for x in asia_fade(sym, h)]
    if af:
        d = pd.DataFrame(af); l, ok = report("3 Asia-Spanne Rücklauf (1h)", d)
        lines += l + ["   je Paar: " + " | ".join(f"{k}: n={len(v)} {v.r.mean():+.2f}R" for k, v in d.groupby("sym"))]
        if ok: passed.append("3 Asia-Spanne")
    # 4
    if "EURUSD" in hourly and "GBPUSD" in hourly:
        d = pd.DataFrame(pair_trades(hourly["EURUSD"], hourly["GBPUSD"]))
        l, ok = report("4 EUR/USD gegen GBP/USD (1h)", d); lines += l
        if ok: passed.append("4 Paar-Handel")
    lines.append("\nERGEBNIS: " + ("Bestanden: " + ", ".join(passed) + ". Diese baue ich als Signale ein." if passed
                                    else "Keine Idee besteht alle Bedingungen. Keine kommt als Signal in den Bot."))
    lines.append("Idee 1 rechnet nur die Hälfte des Zinsunterschieds als Ertrag (Broker-Swaps sind ungünstiger). Ergebnisse in R je Trade.")
    lines.append("Bedingungen: n>=50, t>=2,5, beide Hälften positiv, Kosten x2 positiv; Idee 2 zusätzlich klar besser als die Kontrolle.")
    if notes: lines.append("Hinweise: " + " | ".join(notes))
    text = "\n".join(lines); print(text)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    for i in range(0, len(text), 3800): S.telegram(text[i:i + 3800], dry)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    notes, rates, rel = [], {}, {}
    try: rates = fetch_rates()
    except Exception as e: notes.append(f"Zinsen nicht geladen: {S.scrub(e)[:80]}")
    try: rel = fetch_release_dates()
    except Exception as e: notes.append(f"Termine nicht geladen: {S.scrub(e)[:80]}")
    daily, hourly = {}, {}
    for sym, (ts, b, q) in PAIRS_D.items():
        for k in range(4):
            try:
                daily[sym] = L.hist_twelve(ts, "1day", 5000); break
            except Exception as e:
                if k == 3 or "for the day" in str(e): notes.append(f"{sym} Tag: {S.scrub(e)[:60]}"); break
                time.sleep(65 if "minute" in str(e).lower() else 20)
        time.sleep(10)
    for sym, ts in PAIRS_H.items():
        try: hourly[sym] = L.indicators(M.fetch_1h(ts, 5))
        except Exception as e: notes.append(f"{sym} 1h: {S.scrub(e)[:60]}")
        time.sleep(10)
    run(daily, hourly, rates, rel, a.dry, notes)


if __name__ == "__main__":
    main()
