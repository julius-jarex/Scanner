#!/usr/bin/env python3
"""COT-Test: verbessert die Positionierung der großen Spekulanten (CFTC, Gold-Futures COMEX) die Gold-Long-Signale (V1)?
Zwei Hypothesen, vorab festgelegt:
 H1 Kontra:   Long nur, wenn die Netto-Long-Position der Spekulanten NICHT im obersten 20 % der letzten 3 Jahre liegt
 H2 Momentum: Long nur, wenn die Netto-Long-Position gegenüber der Vorwoche gestiegen ist
Übernahme nur, wenn der Filter in BEIDEN Zeiträumen (vor/ab 2023) besser ist als alle Trades und der Unterschied t>=2 hat.
COT-Daten gelten erst ab dem Montag nach der Veröffentlichung (kein Blick in die Zukunft).
Start: GitHub -> Actions -> cot -> Run workflow. Braucht ca. 12 Twelve-Data-Abrufe."""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
import scanner as S
import gold2 as G
import multi as M

GOLD_CODE = "088691"
DEV_START = pd.Timestamp("2023-02-17", tz="UTC")
COL = f"r_{M.MAIN_TP}"
SOURCES = [("https://publicreporting.cftc.gov/resource/72hh-3qpy.json", "m_money_positions_long_all", "m_money_positions_short_all", "Managed Money"),
           ("https://publicreporting.cftc.gov/resource/6dca-aqww.json", "noncomm_positions_long_all", "noncomm_positions_short_all", "Non-Commercials")]


def fetch_cot():
    errs = []
    for url, lcol, scol, label in SOURCES:
        try:
            r = requests.get(url, params={"$where": f"cftc_contract_market_code='{GOLD_CODE}'", "$order": "report_date_as_yyyy_mm_dd",
                                          "$limit": 5000}, timeout=60)
            r.raise_for_status()
            d = pd.DataFrame(r.json())
            if d.empty or lcol not in d or scol not in d:
                errs.append(f"{label}: Felder fehlen ({', '.join(list(d.columns)[:8])})"); continue
            d["date"] = pd.to_datetime(d["report_date_as_yyyy_mm_dd"], utc=True)
            d["net"] = d[lcol].astype(float) - d[scol].astype(float)
            d = d.sort_values("date").drop_duplicates("date")[["date", "net"]].reset_index(drop=True)
            d["rank"] = d.net.rolling(156, min_periods=104).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
            d["chg"] = d.net.diff()
            d["valid_from"] = d.date + pd.Timedelta(days=6)       # Bericht Di, Veröffentlichung Fr, gilt ab Montag
            return d, label
        except Exception as e:
            errs.append(f"{label}: {str(e)[:100]}")
    raise RuntimeError("; ".join(errs))


def welch_t(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 5 or len(b) < 5: return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)))


def run(raw, cot, label, dry):
    h = G.prepare(raw)
    sig = [s for s in M.signals_v1(h) if s[1] == "long"]
    d = pd.DataFrame(M.evaluate("XAUUSD", h, sig)).sort_values("time")
    d = pd.merge_asof(d, cot[["valid_from", "rank", "chg"]].rename(columns={"valid_from": "time"}).sort_values("time"), on="time")
    d = d.dropna(subset=["rank", "chg"])
    tests = {"H1 Kontra (nicht im obersten 20 %)": d["rank"] < 0.8, "H2 Momentum (Netto-Long gestiegen)": d["chg"] > 0}
    lines = [f"COT-Test Gold (CFTC {label}), Basis: V1 nur Long, Ziel {M.MAIN_TP}R, {len(d)} Trades {d.time.min():%Y-%m-%d} bis {d.time.max():%Y-%m-%d}",
             f"Alle Trades: Ø {d[COL].mean():+.2f}R | vor 2023 {d[d.time < DEV_START][COL].mean():+.2f}R | ab 2023 {d[d.time >= DEV_START][COL].mean():+.2f}R\n"]
    adopt = []
    for name, m in tests.items():
        keep, drop = d[m], d[~m]
        t = welch_t(keep[COL], drop[COL])
        per = []
        ok = True
        for pn, pm in (("vor 2023", d.time < DEV_START), ("ab 2023", d.time >= DEV_START)):
            k, a_ = d[m & pm][COL], d[pm][COL]
            per.append(f"{pn}: behalten {k.mean():+.2f}R (n={len(k)}) gegen alle {a_.mean():+.2f}R")
            ok = ok and len(k) >= 30 and k.mean() > a_.mean()
        passed = ok and t >= 2.0
        if passed: adopt.append(name)
        lines.append(f"{name}: behalten n={len(keep)} Ø {keep[COL].mean():+.2f}R | aussortiert n={len(drop)} Ø {drop[COL].mean():+.2f}R | Unterschied t={t:.1f}")
        lines.append("   " + " | ".join(per) + (" -> ÜBERNEHMEN" if passed else " -> nicht übernehmen"))
    lines.append("\nERGEBNIS: " + ("Filter mit Vorteil: " + ", ".join(adopt) + ". Ich baue ihn in den Bot ein." if adopt
                                    else "Kein COT-Filter verbessert die Gold-Signale belastbar. Er kommt NICHT in den Bot."))
    lines.append("Bedingung: in beiden Zeiträumen besser als alle Trades (mind. 30 Trades) und Unterschied t>=2. Swing-Gold hat zu wenige Signale für diesen Test.")
    text = "\n".join(lines); print(text)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm: open(sm, "a").write("```\n" + text + "\n```\n")
    S.telegram(text[:3900], dry)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    S.load_env()
    try:
        cot, label = fetch_cot()
        print(f"COT: {len(cot)} Wochen ({cot.date.min():%Y-%m-%d} bis {cot.date.max():%Y-%m-%d}), {label}")
        raw = M.fetch_1h("XAU/USD", 12)
    except Exception as e:
        S.telegram(f"⚠️ COT-Test abgebrochen: {S.scrub(e)[:300]}", a.dry); print(e); sys.exit(1)
    run(raw, cot, label, a.dry)


if __name__ == "__main__":
    main()
