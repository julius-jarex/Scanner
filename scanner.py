#!/usr/bin/env python3
"""Daytrading-Scanner (Forex & Krypto): Kurse holen -> Setups per Regel finden -> bewerten -> Telegram.
Standard: komplett kostenlos (nur Regeln). Optional: Claude als Zusatzfilter (USE_CLAUDE=1 + API-Key).
Nur Hinweise, keine Orders. Aufruf: python scanner.py [--dry] [--test]"""
import os, sys, io, json, time, math, argparse
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import numpy as np

# ---------------- Einstellungen (hier anpassen) ----------------
CRYPTO = {"BTCUSD": "XBTUSD"}          # Kraken-Paare (kostenlos, ohne Key)
FOREX = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY", "XAUUSD": "XAU/USD",
         "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD"}  # Twelve Data (Gratis-Key)
FOREX_HOURS_UTC = (6, 20)      # Forex nur zwischen 06:00 und 20:00 UTC, Mo-Fr
SESSION_UTC = (7, 17)          # aktive Forex-Session (London + New York), ausserhalb davon kein Bonus
MODEL = "claude-sonnet-5-5"    # nur relevant bei USE_CLAUDE=1
STATE_FILE = "state.json"
COOLDOWN_H = 4                 # gleiches Setup nicht öfter als alle 4 Stunden melden
MIN_QUALITY = "A"              # im Gratis-Modus nur Setups mit dieser Qualität (A oder B) melden
BLACKOUT_MIN = 30              # Minuten vor/nach einem Termin keine Meldung
MAX_DATA_AGE_MIN = 50          # Kursdaten älter als das gelten als veraltet -> keine Signale
MAX_CHASE_R = 0.5              # Signal ungültig, wenn Preis mehr als 0,5R vom Einstieg weg ist
MAX_SIGNALS_PER_DAY = None     # kein Tageslimit für die Anzahl der Meldungen (Zahl eintragen, um es wieder zu aktivieren)
CORR_HOURS = 8                 # gleiche Marktrichtung (z. B. USD-short) nicht zweimal innerhalb 8 Stunden
ERR_COOLDOWN_H = 6             # gleiche Fehlermeldung höchstens alle 6 Stunden
HEARTBEAT_HOUR_UTC = 7         # tägliches Lebenszeichen ab 07:00 UTC
INFO_COOLDOWN_H = 2            # Folgesignal-Infos zu derselben offenen Meldung höchstens alle 2 Stunden
TRADE_MAX_H = 30               # Meldung ohne Ergebnis nach so vielen Stunden schliessen
DAILY_LOSS_R = 3.0             # nach diesem Tagesverlust (in R) keine neuen Meldungen mehr (Secret DAILY_LOSS_R überschreibt)
TP1_R, TP2_R = 1.5, 3.0        # Ziele (Intraday) in Vielfachen des Risikos
TP1_SHARE = 0.5                # Anteil, der bei TP1 geschlossen wird (nur TRADE_MODE=partial)
# runde Marken je Symbol (Abstand zwischen zwei "runden" Preisen)
ROUND_STEP = {"EURUSD": 0.005, "GBPUSD": 0.005, "USDJPY": 0.5, "XAUUSD": 10.0,
              "BTCUSD": 1000.0, "ETHUSD": 50.0, "SOLUSD": 5.0, "AUDUSD": 0.005, "USDCAD": 0.005,
              "XRPUSD": 0.1, "ADAUSD": 0.05, "LINKUSD": 1.0}
# angenommene Kosten (Spread/Gebühren) je Trade: Forex/Gold absolut in Preis, Krypto in % vom Preis (Secret COST_CRYPTO_PCT)
COST_ABS = {"EURUSD": 0.00010, "GBPUSD": 0.00012, "USDJPY": 0.010, "XAUUSD": 0.20, "AUDUSD": 0.00012, "USDCAD": 0.00015}
COST_CRYPTO_PCT = 0.001
COST_OTHER_PCT = 0.0003         # ETFs/Indizes/Öl: angenommener Spread in % vom Preis

# --- Signal-Bot v2: zwei klar getrennte Signalarten ---
INTRADAY = {"XAUUSD": "XAU/USD"}            # Gold, 1h-Trend-Rücksetzer im Tagestrend
INTRADAY_STALE_MIN = 25                      # Signal nur, wenn die 1h-Kerze höchstens so viele Minuten alt ist
INTRADAY_SHORTS = False                      # Gold-Intraday: False = nur Long-Signale, True = auch Short (Short lag in beiden Testzeiträumen im Minus)
INTRADAY_SESSION_UTC = (7, 17)               # nur Signalkerzen, die zwischen 07:00 und 17:00 UTC beginnen (Gold-Labor 2, V1)
SWING = {"XAUUSD": ("twelve", "XAU/USD"), "BTCUSD": ("kraken", "XBTUSD"), "QQQ": ("twelve", "QQQ"),
         "SPY": ("twelve", "SPY"), "SLV": ("twelve", "SLV"), "USO": ("twelve", "USO")}
SWING_ENTRY_DAYS = 100          # Einstieg bei neuem 100-Tage-Hoch (Schlusskurs), nur Long
SWING_ATR_MULT = 3.0            # Anfangsstop und Nachzieh-Abstand in ATR
SWING_HOUR_UTC = 7              # tägliche Prüfung ab 07:00 UTC (Tageskerzen sind dann abgeschlossen)
SWING_MAX_OPEN = 4              # Hinweis, wenn schon so viele Swing-Positionen offen sind
INTRADAY_EVIDENCE = ("Beleg: schwach. Gold 1h, 2020-2026, nur 07-17 UTC: ca. +0,06R (vor 2023, nie zur Entwicklung benutzt) bzw. "
                     "+0,10R (ab 2023) pro Trade nach Kosten, knapp besser als zufällige Einstiege, statistisch nicht gesichert. "
                     "Long in beiden Zeiträumen deutlich besser als Short. Auf 8 anderen Märkten nicht bestätigt, 2026 bisher leicht im Minus. "
                     "Demo oder max. 0,5 % Risiko.")
SWING_EVIDENCE = ("Beleg: schwach positiv. 20-Jahres-Test über viele Märkte, nur Long: etwa +0,2R pro Trade nach Kosten, "
                  "Trefferquote ca. 38 %. Im Portfolio nur ca. +4 bis +5 % pro Jahr bei 1 % Risiko, mit Rückgängen von 20-35 % "
                  "und Jahren ohne Gewinn. Kein Garant.")
# ---------------------------------------------------------------

NAMES = {"swing": "Trailing-Stop", "tp1": "TP1", "tp2": "TP2", "be": "Break-even nach Teilgewinn", "stop": "Stop", "expired": "ohne Ergebnis"}


def load_env():
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def scrub(text):
    """Zugangsdaten aus Fehlertexten entfernen (URLs enthalten sonst Keys/Token)."""
    text = str(text)
    for k in ("TWELVEDATA_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "MYFXBOOK_EMAIL", "MYFXBOOK_PASSWORD", "FRED_API_KEY"):
        v = os.environ.get(k)
        if v:
            text = text.replace(v, "***")
    return text


def env(name, default=None, required=False):
    v = os.environ.get(name) or default
    if required and not v:
        sys.exit(f"Fehlende Einstellung: {name}")
    return v


TRADE_MODE_DEFAULT = "simple"   # "simple" = bis TP1 oder Stop | "half" = 50 % beim halben TP1, Stop auf Einstieg, Rest bis TP1 | "partial" = 50 % bei TP1, Rest bis TP2


def trade_mode():
    m = (env("TRADE_MODE", TRADE_MODE_DEFAULT) or "simple").lower()
    return m if m in ("simple", "partial", "half") else "simple"


# ---------------- Daten ----------------
def fetch_kraken(pair, interval=15):
    r = requests.get("https://api.kraken.com/0/public/OHLC", params={"pair": pair, "interval": interval}, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(j["error"])
    rows = next(v for k, v in j["result"].items() if k != "last")
    raw = pd.DataFrame(rows)
    df = raw.iloc[:, :5].copy()
    df.columns = ["t", "open", "high", "low", "close"]
    df["time"] = pd.to_datetime(df["t"].astype(int), unit="s", utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["volume"] = raw.iloc[:, 6].astype(float) if raw.shape[1] > 6 else float("nan")
    return df[["time", "open", "high", "low", "close", "volume"]].iloc[:-1].reset_index(drop=True)  # letzte Kerze läuft noch


def fetch_twelve(symbol, outputsize=800):
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": "15min", "outputsize": outputsize, "timezone": "UTC",
        "apikey": env("TWELVEDATA_KEY", required=True)}, timeout=20)
    j = r.json()
    if j.get("status") == "error":
        raise RuntimeError(j.get("message"))
    df = pd.DataFrame(j["values"])
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["volume"] = float("nan")
    df = df.sort_values("time")[["time", "open", "high", "low", "close", "volume"]]
    return df.iloc[:-1].reset_index(drop=True)


def check_fresh(name, df, now):
    """Veraltete Kursdaten (API-Fehler, Marktpause) -> Fehler statt falscher Signale."""
    age = (now - df.time.iloc[-1].to_pydatetime()).total_seconds() / 60
    if age > MAX_DATA_AGE_MIN:
        raise RuntimeError(f"Kursdaten veraltet (letzte Kerze vor {int(age)} Min.)")
    return df


# ---------------- Setups ----------------
def add_ind(df):
    df["ema20"] = df.close.ewm(span=20, adjust=False).mean()
    df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


def htf_trend(df):
    h = df.set_index("time").resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    e20 = h.close.ewm(span=20, adjust=False).mean().iloc[-1]
    e50 = h.close.ewm(span=50, adjust=False).mean().iloc[-1]
    c = h.close.iloc[-1]
    if e20 > e50 and c > e50: return "up"
    if e20 < e50 and c < e50: return "down"
    return "flat"


def day_levels(df):
    """Vortageshoch/-tief (UTC-Tag). (None, None), wenn zu wenig Daten vorliegen."""
    d = df.set_index("time").resample("1D").agg({"high": "max", "low": "min"}).dropna()
    if len(d) < 2:
        return None, None
    return float(d.high.iloc[-2]), float(d.low.iloc[-2])


def round_levels(sym, price):
    step = ROUND_STEP.get(sym)
    if not step:
        return None, None
    down = math.floor(price / step + 1e-9) * step
    up = down + step
    if abs(down - price) < step * 1e-6:
        down -= step
    return round(down, 6), round(up, 6)


def next_level(side, entry, levels):
    """Nächstes Level in Handelsrichtung hinter dem Einstieg."""
    levels = [(n, p) for n, p in levels if p is not None]
    if side == "long":
        c = [(n, p) for n, p in levels if p > entry]
        return min(c, key=lambda x: x[1]) if c else (None, None)
    c = [(n, p) for n, p in levels if p < entry]
    return max(c, key=lambda x: x[1]) if c else (None, None)


def cost_r(sym, entry, risk):
    if not risk:
        return 0.0
    if sym in COST_ABS:
        c = COST_ABS[sym]
    elif sym in CRYPTO:
        c = entry * float(env("COST_CRYPTO_PCT", COST_CRYPTO_PCT))
    else:
        c = entry * COST_OTHER_PCT
    return c / risk


def size(sym, entry, stop):
    risk = float(env("ACCOUNT_SIZE", 1000)) * float(env("RISK_PERCENT", 1)) / 100
    dist = abs(entry - stop)
    if sym in FOREX:
        if sym == "XAUUSD": per_lot = dist * 100                 # Verlust je 1 Lot bei Stop
        elif sym == "USDJPY": per_lot = (dist / 0.01) * (1000 / entry)
        elif sym == "USDCAD": per_lot = (dist / 0.0001) * (10 / entry)
        else: per_lot = (dist / 0.0001) * 10
        lots = math.floor(risk / per_lot * 100) / 100          # immer abrunden
        if lots < 0.01:
            return f"unter 0.01 Lots möglich (bei 0.01 Lots Risiko {per_lot * 0.01:.2f} statt {risk:.2f}), eher auslassen"
        return f"{lots:.2f} Lots (Risiko ca. {lots * per_lot:.2f} von {risk:.2f})"
    if sym in CRYPTO:
        return f"{risk / dist:.4f} {sym[:3]} (Risiko {risk:.2f}, ohne Hebel)"
    return f"{risk / dist:.2f} Stück/Einheiten (Risiko {risk:.2f}; Kontraktgröße beim Broker prüfen)"


def find_setups(sym, df):
    df = add_ind(df.copy())
    if len(df) < 120 or df.atr.isna().iloc[-1]:
        return []
    trend = htf_trend(df)
    last = df.iloc[-1]
    prev = df.iloc[-21:-1]
    rec = df.iloc[-4:]
    atr = last.atr
    hi20, lo20 = prev.high.max(), prev.low.min()
    body = abs(last.close - last.open)
    pdh, pdl = day_levels(df)
    rdown, rup = round_levels(sym, float(last.close))
    atr_med = float(df.atr.iloc[-200:].median())
    atr_ratio = float(atr / atr_med) if atr_med else 1.0
    vol_ratio = None
    if "volume" in df and pd.notna(last.get("volume")):
        vmed = float(df.volume.iloc[-51:-1].median())
        if vmed > 0:
            vol_ratio = float(last.volume / vmed)
    out = []

    def add(name, side, stop):
        entry = float(last.close)
        risk = abs(entry - stop)
        if not (0.5 * atr <= risk <= 3 * atr):
            return
        sgn = 1 if side == "long" else -1
        levels = [("20-Kerzen-Hoch", float(hi20)), ("20-Kerzen-Tief", float(lo20)), ("Rundzahl", rup),
                  ("Rundzahl", rdown), ("Vortageshoch", pdh), ("Vortagestief", pdl)]
        lv_name, lv_price = next_level(side, entry, levels)
        out.append({
            "id": f"{sym}|{name}|{side}", "symbol": sym, "setup": name, "side": side,
            "candle_time": last.time.strftime("%Y-%m-%d %H:%M UTC"), "candle_ts": last.time.isoformat(),
            "entry": round(entry, 5), "stop": round(float(stop), 5),
            "tp1": round(entry + sgn * TP1_R * risk, 5), "tp2": round(entry + sgn * TP2_R * risk, 5),
            "htf_trend_1h": trend, "atr15": round(float(atr), 5),
            "ema20": round(float(last.ema20), 5), "ema50": round(float(last.ema50), 5),
            "high_20_candles": round(float(hi20), 5), "low_20_candles": round(float(lo20), 5),
            "body_atr": round(float(body / atr), 2), "risk_dist": float(risk),
            "next_level_name": lv_name, "next_level": None if lv_price is None else round(lv_price, 5),
            "room_r": None if lv_price is None else round(abs(lv_price - entry) / risk, 2),
            "atr_ratio": round(atr_ratio, 2), "vol_ratio": None if vol_ratio is None else round(vol_ratio, 2),
            "size": size(sym, entry, stop)})

    low6, high6 = df.low.iloc[-6:].min(), df.high.iloc[-6:].max()
    # 1) Trend-Pullback
    if trend == "up" and (rec.low <= rec.ema20).any() and last.close > last.ema20 > last.ema50 and last.close > last.open:
        add("Trend-Pullback", "long", low6 - 0.2 * atr)
    if trend == "down" and (rec.high >= rec.ema20).any() and last.close < last.ema20 < last.ema50 and last.close < last.open:
        add("Trend-Pullback", "short", high6 + 0.2 * atr)
    # 2) Range-Breakout
    if last.close > hi20 and trend != "down" and body > 0.5 * atr:
        add("Range-Breakout", "long", hi20 - atr)
    if last.close < lo20 and trend != "up" and body > 0.5 * atr:
        add("Range-Breakout", "short", lo20 + atr)
    # 3) Liquidity-Sweep
    if last.low < lo20 and last.close > lo20 and trend != "down":
        add("Liquidity-Sweep", "long", last.low - 0.2 * atr)
    if last.high > hi20 and last.close < hi20 and trend != "up":
        add("Liquidity-Sweep", "short", last.high + 0.2 * atr)
    return out


# ---------------- Verlauf eines Trades (auch vom Backtest genutzt) ----------------
def simulate_path(side, entry, stop, tp1, tp2, hl, mode="simple"):
    """hl: Liste (high, low) der Kerzen NACH dem Signal. Rückgabe (status, R).
    status: open | tp1_open | stop | tp1 | tp2 | be.
    simple : TP1 (+2R) oder Stop (-1R), was zuerst kommt.
    partial: bei TP1 Teilgewinn, Stop auf Einstieg, Rest bis TP2 oder Break-even.
    Berühren Stop und Ziel dieselbe Kerze, zählt vorsichtig der Stop."""
    long_ = side == "long"
    if mode == "half":                 # 50 % beim halben Weg zu TP1, Stop auf Einstieg, Rest bis TP1
        half = entry + (tp1 - entry) / 2
        r_half = TP1_SHARE * TP1_R / 2
        phase = 1
        for hi, lo in hl:
            if phase == 1:
                if (lo <= stop) if long_ else (hi >= stop):
                    return "stop", -1.0
                if (hi >= half) if long_ else (lo <= half):
                    phase = 2
                    if (hi >= tp1) if long_ else (lo <= tp1):      # halbes und ganzes Ziel in derselben Kerze
                        return "tp1", r_half + (1 - TP1_SHARE) * TP1_R
                    continue
            else:
                if (lo <= entry) if long_ else (hi >= entry):
                    return "be", r_half
                if (hi >= tp1) if long_ else (lo <= tp1):
                    return "tp1", r_half + (1 - TP1_SHARE) * TP1_R
        return ("tp1_open", r_half) if phase == 2 else ("open", 0.0)
    r_be = TP1_SHARE * TP1_R
    r_tp2 = TP1_SHARE * TP1_R + (1 - TP1_SHARE) * TP2_R
    phase = 1
    for hi, lo in hl:
        if phase == 1:
            hit_stop = lo <= stop if long_ else hi >= stop
            hit_tp1 = hi >= tp1 if long_ else lo <= tp1
            if hit_stop:
                return "stop", -1.0
            if hit_tp1:
                if mode == "simple":
                    return "tp1", TP1_R
                phase = 2
        else:
            hit_be = lo <= entry if long_ else hi >= entry
            hit_tp2 = hi >= tp2 if long_ else lo <= tp2
            if hit_be:
                return "be", r_be
            if hit_tp2:
                return "tp2", r_tp2
    return ("tp1_open", r_be) if phase == 2 else ("open", 0.0)


# ---------------- Gratis-News-Check ----------------
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"   # inoffizieller, kostenloser Wochenkalender
RSS_FEEDS = ["https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cointelegraph.com/rss"]
RISK_WORDS = ["hack", "exploit", "stolen", "sec ", "lawsuit", "ban ", "delist", "halt", "outage",
              "insolven", "bankrupt", "liquidation", "crackdown", "sanction"]
CCY = {"EURUSD": {"EUR", "USD"}, "GBPUSD": {"GBP", "USD"}, "USDJPY": {"USD", "JPY"}, "XAUUSD": {"USD"}, "AUDUSD": {"AUD", "USD"}, "USDCAD": {"USD", "CAD"}}
LOOKAHEAD_MIN = 120            # Termine in den nächsten 2 Stunden werden als Warnung angezeigt
HEADLINE_HOURS = 3             # Krypto-Schlagzeilen der letzten 3 Stunden


def fetch_news(now):
    """Holt kostenlos: High-Impact-Termine (Forex-Kalender) und Krypto-Schlagzeilen (RSS). Fehler -> leere Daten."""
    news = {"events": [], "headlines": [], "ok": True}
    try:
        r = requests.get(CAL_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        for e in r.json():
            if e.get("impact") != "High":
                continue
            t = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
            news["events"].append({"time": t, "ccy": e.get("country", ""), "title": e.get("title", ""),
                                   "forecast": e.get("forecast", ""), "previous": e.get("previous", "")})
    except Exception as ex:
        print(f"Kalender nicht verfügbar: {scrub(ex)}"); news["ok"] = False
    for url in RSS_FEEDS:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            for item in ET.fromstring(r.content).iter("item"):
                title = (item.findtext("title") or "").strip()
                pub = parsedate_to_datetime(item.findtext("pubDate"))
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                if now - pub.astimezone(timezone.utc) <= timedelta(hours=HEADLINE_HOURS) and \
                        any(w in (title.lower() + " ") for w in RISK_WORDS):
                    news["headlines"].append(title)
        except Exception as ex:
            print(f"RSS nicht verfügbar ({url}): {scrub(ex)}")
    return news


def news_check(sym, news, now):
    """-> (block, penalty, notizen). block = Setup nicht melden."""
    if not news:
        return False, 0, []
    ccys = CCY.get(sym, {"USD"})
    block, penalty, notes = False, 0, []
    for e in news["events"]:
        if e["ccy"] not in ccys:
            continue
        mins = (e["time"] - now).total_seconds() / 60
        label = f"{e['ccy']} {e['title']} um {e['time']:%H:%M} UTC"
        if e.get("forecast") or e.get("previous"):
            label += f" (Prognose {e.get('forecast') or '-'}, zuvor {e.get('previous') or '-'})"
        if -BLACKOUT_MIN <= mins <= BLACKOUT_MIN:
            block = True; notes.append("Termin jetzt: " + label)
        elif 0 < mins <= LOOKAHEAD_MIN:
            penalty = max(penalty, 1); notes.append(f"Termin in {int(mins)} Min.: " + label)
    if sym in CRYPTO and news["headlines"]:
        penalty = max(penalty, 1)
        notes.append("Krypto-Risikomeldung: " + news["headlines"][0][:90])
    if not news["ok"]:
        notes.append("Termin-Kalender nicht erreichbar, bitte manuell prüfen")
    return block, penalty, notes


# ---------------- Gratis-Bewertung (ohne KI) ----------------
def score_free(s, penalty=0):
    """Punktesystem 0-7 minus News-Abzug. A >= 5, B = 4. Nachvollziehbar, keine Prognose."""
    pts, why, risks = 0, [], []
    long_ = s["side"] == "long"
    aligned = (s["htf_trend_1h"] == "up") == long_ and s["htf_trend_1h"] != "flat"
    if aligned:
        pts += 2; why.append("1h-Trend passt zur Richtung")
    elif s["htf_trend_1h"] == "flat":
        pts += 1; risks.append("1h-Trend seitwärts")
    else:
        risks.append("gegen den 1h-Trend")

    room_r, name = s.get("room_r"), s.get("next_level_name") or "Level"
    if room_r is None or room_r >= 2:
        pts += 2; why.append("Platz bis zum nächsten Level" + ("" if room_r is None else f" ({name}, {room_r:.1f}R)"))
    elif room_r >= 1:
        pts += 1; risks.append(f"{name} bei {room_r:.1f}R, vor TP1")
    else:
        risks.append(f"{name} direkt voraus ({room_r:.1f}R)")

    if s["body_atr"] >= 0.6:
        pts += 1; why.append("kräftige Signalkerze")
    if s["setup"] != "Liquidity-Sweep" and aligned:
        pts += 1; why.append(f"{s['setup']} mit Trend")

    hour = pd.Timestamp(s["candle_ts"]).hour if s.get("candle_ts") else 12
    active = (s["symbol"] not in FOREX) or (SESSION_UTC[0] <= hour < SESSION_UTC[1])
    calm = s.get("atr_ratio", 1.0) < 0.8
    if active and not calm:
        pts += 1; why.append("aktive Marktphase")
    else:
        risks.append("ruhige Session" if not active else "flaue Volatilität")
    if s["setup"] == "Range-Breakout" and s.get("vol_ratio") is not None and s["vol_ratio"] < 0.8:
        pts -= 1; risks.append("Ausbruch ohne Volumen")

    pts -= penalty
    q = "A" if pts >= 5 else "B" if pts == 4 else "C"
    return q, why, risks


def in_blackout(now):
    for item in (env("NEWS_BLACKOUTS", "") or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            t = datetime.strptime(item, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Ungültiger Termin in NEWS_BLACKOUTS: {item}"); continue
        if abs(now - t) <= timedelta(minutes=BLACKOUT_MIN):
            return item
    return None


# ---------------- Claude (optional) ----------------
SYSTEM = """Du bist ein nüchterner Daytrading-Analyst für Forex und Krypto. Du bekommst Setup-Kandidaten als JSON (15m-Daten, 1h-Trend).
Spiele für jeden Kandidaten kurz Bull-, Base-, Bear- und Fakeout-Fall durch. Prüfe: Liegt TP1 realistisch vor dem nächsten gegnerischen Level
(next_level, high_20_candles / low_20_candles)? Ist der Stop strukturell sinnvoll? Passt der 1h-Trend? Suche per Websuche High-Impact-Termine in den nächsten
60 Minuten und aktuelle Meldungen zum Symbol. HANDELN nur bei klarem Setup, TP1 mit CRV >= 1:2 und ohne akutes Event-Risiko. Im Zweifel ABWARTEN.
Erfinde keine Daten. Antworte NUR mit JSON: {"results":[{"id":"<id>","verdict":"HANDELN|ABWARTEN|VERWERFEN","quality":"A|B|C","text":"max 3 kurze Sätze auf Deutsch"}]}"""


def ask_claude(setups, news=True):
    body = {"model": MODEL, "max_tokens": 2000, "system": SYSTEM,
            "messages": [{"role": "user", "content": json.dumps(setups, ensure_ascii=False)}]}
    if news:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    headers = {"x-api-key": env("ANTHROPIC_API_KEY", required=True),
               "anthropic-version": "2023-06-01", "content-type": "application/json"}
    r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=180)
    if r.status_code == 400 and news:
        return ask_claude(setups, news=False)
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json()["content"] if b["type"] == "text")
    return json.loads(text[text.find("{"): text.rfind("}") + 1])["results"]


# ---------------- Telegram ----------------
def chat_ids():
    return [c.strip() for c in (env("TELEGRAM_CHAT_ID", "") or "").split(",") if c.strip()]


def tg(method, payload=None, files=None):
    token = env("TELEGRAM_TOKEN", required=True)
    url = f"https://api.telegram.org/bot{token}/{method}"
    if files:
        r = requests.post(url, data=payload, files=files, timeout=60)
    else:
        r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def telegram(text, dry=False, chat=None, buttons=None):
    """Sendet an alle Empfänger (TELEGRAM_CHAT_ID, mehrere mit Komma) oder an einen bestimmten Chat."""
    if dry:
        print("---- TELEGRAM (dry) ----\n" + text + ("\n[Buttons: " + ", ".join(b["text"] for b in buttons) + "]" if buttons else "") + "\n")
        return
    targets = [chat] if chat else chat_ids()
    if not targets:
        sys.exit("Fehlende Einstellung: TELEGRAM_CHAT_ID")
    errors = []
    for c in targets:
        payload = {"chat_id": c, "text": text[:4000]}
        if buttons:
            payload["reply_markup"] = {"inline_keyboard": [buttons]}
        try:
            tg("sendMessage", payload)
        except Exception as e:
            errors.append(e)
    if errors and len(errors) == len(targets):
        raise errors[0]


def telegram_file(kind, data, filename, caption, dry=False, chat=None):
    """kind: 'sendPhoto' oder 'sendDocument'."""
    if dry:
        print(f"---- TELEGRAM {kind} (dry): {filename}, {len(data)} Bytes ----\n"); return
    field = "photo" if kind == "sendPhoto" else "document"
    for c in ([chat] if chat else chat_ids()):
        try:
            tg(kind, {"chat_id": c, "caption": caption[:1000]}, files={field: (filename, data)})
        except Exception as e:
            print(f"{kind} fehlgeschlagen: {scrub(e)}")


# ---------------- Chart ----------------
def make_chart(s, df, n=96):
    """Kerzenchart (PNG) mit EMA 20/50, Einstieg, Stop, TP1, TP2 und dem nächsten Level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = add_ind(df.copy()).iloc[-n:].reset_index(drop=True)
    bg, up, dn = "#0d1117", "#2ea043", "#f85149"
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    min_body = float(d.high.max() - d.low.min()) * 0.002
    for i, r in d.iterrows():
        col = up if r.close >= r.open else dn
        ax.vlines(i, r.low, r.high, color=col, linewidth=0.8)
        ax.bar(i, max(abs(r.close - r.open), min_body), bottom=min(r.open, r.close), width=0.6, color=col)
    ax.plot(d.ema20, color="#58a6ff", linewidth=1, label="EMA 20")
    ax.plot(d.ema50, color="#d29922", linewidth=1, label="EMA 50")
    x1 = len(d) + 1
    marks = [("Einstieg", s["entry"], "#c9d1d9", "-"), ("Stop", s["stop"], dn, "-"),
             ("TP1", s["tp1"], up, "-"), ("TP2", s["tp2"], up, "--")]
    if s.get("next_level"):
        marks.append((s.get("next_level_name") or "Level", s["next_level"], "#a371f7", ":"))
    for label, price, col, ls in marks:
        ax.axhline(price, color=col, linestyle=ls, linewidth=1)
        ax.text(x1, price, f" {label} {price}", color=col, va="center", fontsize=8)
    lo = min([d.low.min(), s["stop"], s["tp1"]]); hi = max([d.high.max(), s["stop"], s["tp1"]])
    pad = (hi - lo) * 0.06
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-1, len(d) + 14)
    ticks = list(range(0, len(d), 12))
    ax.set_xticks(ticks); ax.set_xticklabels([d.time.iloc[i].strftime("%H:%M") for i in ticks], color="#8b949e", fontsize=8)
    ax.tick_params(axis="y", colors="#8b949e", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.grid(color="#21262d", linewidth=0.5)
    ax.set_title(f"{s['symbol']} 15m  {s['side'].upper()}  {s['setup']}   (UTC)", color="#c9d1d9", fontsize=11)
    ax.legend(loc="upper left", facecolor=bg, edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def send_chart(s, df, no, dry):
    if df is None:
        return
    try:
        png = make_chart(s, df)
    except Exception as e:
        print(f"Chart fehlgeschlagen: {scrub(e)}"); return
    telegram_file("sendPhoto", png, f"{s['symbol']}.png", f"#{no} {s['symbol']} {s['side'].upper()} – {s['setup']}", dry)


# ---------------- Zustand ----------------
def load_state():
    try: st = json.load(open(STATE_FILE))
    except Exception: st = {}
    return st


def save_state(s):
    now = time.time()
    json.dump({k: v for k, v in s.items()
               if k.startswith("_") or not isinstance(v, (int, float)) or now - v < 86400},
              open(STATE_FILE, "w"))


def new_stats():
    return {"c": {}, "r": 0.0, "rn": 0.0, "by": {}, "sym": {}}


def stats_of(state, scope):
    key = "_stats" if scope == "all" else "_stats_mine"
    st = state.setdefault(key, new_stats())
    if "c" not in st:      # alte Struktur aus früheren Versionen übernehmen
        old = st
        st = state[key] = new_stats()
        st["c"] = {k: old.get(k, 0) for k in ("tp1", "stop", "expired") if old.get(k)}
        st["r"] = st["rn"] = float(old.get("r", 0.0))
    return st


def stats_line(state, scope="all"):
    label = "alle Meldungen" if scope == "all" else "deine Trades"
    st = state.get("_stats" if scope == "all" else "_stats_mine")
    if st and "c" not in st:
        st = stats_of(state, scope)
    if not st or not sum(st["c"].values()):
        return f"Bilanz ({label}): noch nichts abgeschlossen."
    parts = [f"{v}x {NAMES.get(k, k)}" for k, v in st["c"].items() if v]
    return f"Bilanz ({label}): {', '.join(parts)}. Summe {st['r']:+.1f}R (nach Kosten {st['rn']:+.1f}R)"


def stats_text(state):
    out = [stats_line(state, "all")]
    if state.get("_stats_mine"):
        out.append(stats_line(state, "mine"))
    st = state.get("_stats")
    if st and st.get("by"):
        out.append("Je Setup (nach Kosten):")
        out += [f"  {n}: {b['n']} Trades, {b['r']:+.1f}R" for n, b in st["by"].items()]
    if st and st.get("sym"):
        out.append("Je Symbol (nach Kosten):")
        out += [f"  {n}: {b['n']} Trades, {b['r']:+.1f}R" for n, b in st["sym"].items()]
    out.append(f"Annahmen: Einstieg zum gemeldeten Preis, Kosten pauschal, Modus {trade_mode()}.")
    return "\n".join(out)


def day_check(state, now):
    """(gesperrt, Tages-R). Zählt eigene Trades, wenn welche markiert wurden, sonst alle Meldungen."""
    d = state.get("_day")
    today = now.strftime("%Y-%m-%d")
    if not d or d.get("date") != today:
        return False, 0.0
    v = d["mine"] if d.get("mine_n") else d["all"]
    return v <= -float(env("DAILY_LOSS_R", DAILY_LOSS_R)), v


def book(state, tr, outcome, r, now):
    """Verbucht ein abgeschlossenes Ergebnis. Gibt R nach Kosten zurück."""
    risk = abs(tr["entry"] - tr["stop"])
    rn = r - cost_r(tr["symbol"], tr["entry"], risk)
    for scope in ["all"] + (["mine"] if tr.get("taken") else []):
        st = stats_of(state, scope)
        st["c"][outcome] = st["c"].get(outcome, 0) + 1
        st["r"] = round(st["r"] + r, 3); st["rn"] = round(st["rn"] + rn, 3)
        for grp, key in (("by", tr["setup"]), ("sym", tr["symbol"])):
            b = st.setdefault(grp, {}).setdefault(key, {"n": 0, "r": 0.0})
            b["n"] += 1; b["r"] = round(b["r"] + rn, 3)
    today = now.strftime("%Y-%m-%d")
    d = state.get("_day")
    if not d or d.get("date") != today:
        d = state["_day"] = {"date": today, "all": 0.0, "mine": 0.0, "mine_n": 0}
    d["all"] = round(d["all"] + rn, 3)
    if tr.get("taken"):
        d["mine"] = round(d["mine"] + rn, 3); d["mine_n"] += 1
    j = state.setdefault("_journal", [])
    j.append({"closed": now.strftime("%Y-%m-%d %H:%M"), "no": tr.get("no"), "symbol": tr["symbol"], "side": tr["side"],
              "setup": tr["setup"], "quality": tr.get("q", ""), "signal_candle": tr["t0"], "entry": tr["entry"],
              "stop": tr["stop"], "tp1": tr["tp1"], "outcome": outcome, "r": round(r, 2), "r_net": round(rn, 2),
              "taken": {True: "ja", False: "nein"}.get(tr.get("taken"), ""),
              "vix_hoch": {True: "ja", False: "nein"}.get(tr.get("vix_high"), ""),
              "dollar_stark": {True: "ja", False: "nein"}.get(tr.get("usd_strong"), "")})
    del j[:-500]
    return rn


def journal_csv(state):
    rows = state.get("_journal", [])
    if not rows:
        return None
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# ---------------- Meldungen ----------------
def format_msg(s, quality, text, no=None):
    arrow = "🟢 LONG" if s["side"] == "long" else "🔴 SHORT"
    r = s["risk_dist"]
    if s["side"] == "long":
        expiry = f"Ungültig, wenn Preis über {round(s['entry'] + MAX_CHASE_R * r, 5)} steigt oder Stop {s['stop']} berührt."
    else:
        expiry = f"Ungültig, wenn Preis unter {round(s['entry'] - MAX_CHASE_R * r, 5)} fällt oder Stop {s['stop']} berührt."
    level = ""
    if s.get("next_level"):
        level = f"Nächstes Level: {s.get('next_level_name')} bei {s['next_level']} ({s['room_r']:.1f}R)\n"
    plan = ""
    if trade_mode() == "partial":
        plan = f"Plan: bei TP1 {TP1_SHARE:.0%} schließen, Stop auf Einstieg, Rest bis TP2.\n"
    elif trade_mode() == "half":
        half = round(s['entry'] + (s['tp1'] - s['entry']) / 2, 5)
        plan = f"Plan: bei {half} (halbes Ziel) {TP1_SHARE:.0%} schließen, Stop auf Einstieg, Rest bis TP1.\n"
    nr = f"#{no} " if no else ""
    return (f"{arrow} {nr}{s['symbol']} | {s['setup']} | {quality}\n"
            f"Kerze: {s['candle_time']} | {s.get('trend_text') or '1h-Trend: ' + str(s.get('htf_trend_1h'))}\n\n"
            f"Einstieg: {s['entry']}\nStop: {s['stop']}\nTP1 ({TP1_R:g}R): {s['tp1']}\nTP2 ({TP2_R:g}R): {s['tp2']}\n"
            f"{level}Größe: {s['size']}\n{plan}{expiry}\n\n{text}\n\n"
            f"Nur Hinweis, keine Anlageberatung. Preis, Größe und News vor der Order selbst prüfen.")


def evaluate(candidates, now, news=None):
    """Gibt Liste (setup, qualität, text) der zu meldenden Setups zurück. Gratis-Modus, außer USE_CLAUDE=1."""
    if env("USE_CLAUDE", "0") == "1" and env("ANTHROPIC_API_KEY"):
        by_id = {s["id"]: s for s in candidates}
        out = []
        for res in ask_claude(candidates, news=env("NEWS_CHECK", "1") == "1"):
            s = by_id.get(res.get("id"))
            if s and res.get("verdict") == "HANDELN":
                out.append((s, res.get("quality", "?"), res.get("text", "")))
        return out
    out = []
    order = {"A": 2, "B": 1, "C": 0}
    for s in candidates:
        block, penalty, notes = news_check(s["symbol"], news, now)
        if block:
            print(f"{s['id']} unterdrückt: {notes[0]}"); continue
        q, why, risks = score_free(s, penalty)
        if order[q] >= order[MIN_QUALITY]:
            text = "Pro: " + "; ".join(why)
            if risks: text += "\nRisiko: " + "; ".join(risks)
            text += "\nNews: " + ("; ".join(notes) if notes else "keine High-Impact-Termine/Risikomeldungen gefunden")
            out.append((s, q, text))
        else:
            print(f"{s['id']}: Qualität {q}, nicht gemeldet ({'; '.join(risks) or 'zu wenig Punkte'})")
    out.sort(key=lambda x: -order.get(x[1], 0))
    return out


# ---------------- Sicherheitsregeln ----------------
def exposure(s):
    """Gleiche Marktrichtung zusammenfassen: EURUSD/GBPUSD/Gold long = USD short usw."""
    sym, side = s["symbol"], s["side"]
    if sym in CRYPTO:
        return f"KRYPTO-{side}"
    usd_long = (side == "long") == sym.startswith("USD")
    return "USD-long" if usd_long else "USD-short"


def sent_recent(state, hours):
    now = time.time()
    out = []
    for k, v in state.items():
        if k.startswith("sent|") and isinstance(v, (int, float)) and now - v < hours * 3600:
            out.append(k.split("|")[2])
    return out


def r_now(tr, df):
    """Aktueller Stand der offenen Meldung in R (letzte geschlossene Kerze)."""
    risk = abs(tr["entry"] - tr["stop"])
    if df is None or risk == 0:
        return None, None
    price = float(df.close.iloc[-1])
    r = (price - tr["entry"]) / risk if tr["side"] == "long" else (tr["entry"] - price) / risk
    return price, r


def follow_msg(s, q, text, tr, frames):
    price, r = r_now(tr, frames.get(tr["symbol"]))
    same = s["symbol"] == tr["symbol"]
    head = (f"🔁 Folgesignal zu deiner offenen Meldung (#{tr.get('no', '?')} {tr['symbol']} {tr['side'].upper()}, {tr['setup']})\n"
            f"{s['symbol']} {s['setup']} {s['side']} (Qualität {q}) zeigt erneut in dieselbe Richtung.\n")
    if same:
        head += "Gleiches Paar: Das Setup bleibt aktuell intakt.\n"
    else:
        head += "Anderes Paar, gleiche Marktrichtung: ähnliches Risiko wie dein offener Trade, keine Streuung.\n"
    stand = ""
    if price is not None:
        stand = (f"\nDein Trade: Einstieg {tr['entry']}, Stop {tr['stop']}, TP1 {tr['tp1']}\n"
                 f"Letzter 15m-Schlusskurs: {round(price, 5)} ({r:+.1f}R)")
    first = text.split("\n")[0] if text else ""
    return (head + stand + (f"\n{first}" if first else "") +
            "\n\nKein neues Einstiegssignal, nur eine Info. Kein Ersatz für Chart und eigene Planung.")


def apply_limits(results, state, frames=None, now=None):
    """Tagesverlust, Tageslimit und Korrelationsfilter. Gibt (erlaubte Meldungen, Infos) zurück."""
    frames = frames or {}
    now = now or datetime.now(timezone.utc)
    allowed, infos = [], []
    blocked, v = day_check(state, now)
    if blocked:
        key = f"dl|{now:%Y-%m-%d}"
        if results and not state.get(key):
            infos.append(f"⛔ Tagesverlustlimit erreicht ({v:+.1f}R, Grenze -{float(env('DAILY_LOSS_R', DAILY_LOSS_R)):g}R). "
                         "Bis morgen 00:00 UTC keine neuen Meldungen.")
            state[key] = time.time()
        for s, q, text in results:
            print(f"{s['id']} unterdrückt: Tagesverlustlimit erreicht ({v:+.1f}R)")
        return [], infos
    for s, q, text in results:
        ex = exposure(s)
        if ex in sent_recent(state, CORR_HOURS):
            # gleichgerichtet zu kürzlich Gemeldetem: Info nur, wenn dazu eine Meldung noch offen ist
            tr = next((t for t in state.get("_trades", []) if exposure(t) == ex and t.get("taken") is not False), None)
            key = f"info|{tr['id']}|{s['id']}" if tr else None
            if tr and time.time() - state.get(key, 0) > INFO_COOLDOWN_H * 3600:
                infos.append(follow_msg(s, q, text, tr, frames))
                state[key] = time.time()
                print(f"{s['id']}: Folgesignal-Info zu {tr['id']}")
            else:
                print(f"{s['id']} unterdrückt: korrelierte Position ({ex}) kürzlich gemeldet")
            continue
        if MAX_SIGNALS_PER_DAY and len(sent_recent(state, 24)) >= MAX_SIGNALS_PER_DAY:
            print(f"{s['id']} unterdrückt: Tageslimit erreicht"); continue
        key = f"sent|{time.time():.3f}|{ex}"
        state[key] = time.time()
        s["sent_key"] = key
        allowed.append((s, q, text))
    return allowed, infos


def alert_error(state, key, text, dry):
    """Fehlermeldung per Telegram, höchstens alle ERR_COOLDOWN_H Stunden je Quelle."""
    k = f"err|{key}"
    if time.time() - state.get(k, 0) > ERR_COOLDOWN_H * 3600:
        try:
            telegram("⚠️ Scanner-Problem\n" + text, dry)
            state[k] = time.time()
        except Exception as e:
            print(f"Fehlermeldung konnte nicht gesendet werden: {scrub(e)}")


def heartbeat(state, now, checked, errors, dry):
    if now.hour >= HEARTBEAT_HOUR_UTC and time.time() - state.get("hb", 0) > 20 * 3600:
        n = len(sent_recent(state, 24))
        try:
            mine = f"\n{stats_line(state, 'mine')}" if state.get("_stats_mine") else ""
            telegram(f"💚 Scanner läuft. Letzte 24h: {n} Meldung(en). Aktuell {checked} Märkte geprüft, {errors} Fehler, "
                     f"{len(state.get('_trades', []))} offene Meldung(en).\n{stats_line(state)}{mine}\n"
                     + f"{demo_summary_line(state)}\n{meme_summary_line(state)}\n" +
                     "Befehle: /status /bilanz /journal /demo /meme /hilfe\n"
                     "Bleibt diese Nachricht aus: GitHub → Actions prüfen (nach ~60 Tagen ohne Repo-Aktivität pausiert GitHub geplante Läufe).", dry)
            state["hb"] = time.time()
        except Exception as e:
            print(f"Heartbeat fehlgeschlagen: {scrub(e)}")


# ---------------- Trade-Verfolgung ----------------
def next_no(state):
    state["_seq"] = state.get("_seq", 0) + 1
    return state["_seq"]


def add_trade(state, s, q, no):
    tr = {"no": no, "id": s["id"], "symbol": s["symbol"], "side": s["side"], "setup": s["setup"], "q": q,
          "entry": s["entry"], "stop": s["stop"], "tp1": s["tp1"], "tp2": s["tp2"], "t0": s["candle_ts"],
          "opened": time.time(), "taken": None, "sent_key": s.get("sent_key"), "tp1_notified": False,
          "vix_high": s.get("vix_high"), "usd_strong": s.get("usd_strong")}
    state.setdefault("_trades", []).append(tr)
    return tr


def _upgrade_trade(state, tr):
    """Ältere gespeicherte Meldungen auf das aktuelle Format bringen."""
    if "no" not in tr:
        tr["no"] = next_no(state)
    if "tp2" not in tr:
        d = abs(tr["entry"] - tr["stop"]) * TP2_R
        tr["tp2"] = round(tr["entry"] + d if tr["side"] == "long" else tr["entry"] - d, 5)
    tr.setdefault("taken", None); tr.setdefault("tp1_notified", False)


def update_trades(state, frames, now, dry):
    keep, mode = [], trade_mode()
    for tr in state.get("_trades", []):
        _upgrade_trade(state, tr)
        df = frames.get(tr["symbol"])
        status, r = "open", 0.0
        if df is not None:
            sub = df[df.time > pd.Timestamp(tr["t0"])]
            status, r = simulate_path(tr["side"], tr["entry"], tr["stop"], tr["tp1"], tr["tp2"],
                                      list(zip(sub.high, sub.low)), mode)
        age_h = (time.time() - tr["opened"]) / 3600
        head = f"#{tr['no']} {tr['symbol']} {tr['side'].upper()} ({tr['setup']})" + (" – dein Trade" if tr.get("taken") else "")
        if status == "tp1_open":
            if not tr["tp1_notified"]:
                tr["tp1_notified"] = True
                try:
                    if mode == "half":
                        half = round(tr["entry"] + (tr["tp1"] - tr["entry"]) / 2, 5)
                        telegram(f"🎯 {head}: halbes Ziel {half} erreicht (+{TP1_R / 2:g}R). Jetzt {TP1_SHARE:.0%} schließen und Stop auf "
                                 f"Einstieg {tr['entry']} setzen (Break-even). Rest läuft bis TP1 {tr['tp1']}.", dry)
                    else:
                        telegram(f"🎯 {head}: TP1 erreicht (+{TP1_R:g}R). {TP1_SHARE:.0%} Teilgewinn, Stop auf Einstieg "
                                 f"{tr['entry']} (Break-even), Rest läuft bis TP2 {tr['tp2']}.", dry)
                except Exception as e:
                    print(f"TP1-Meldung fehlgeschlagen: {scrub(e)}")
            if age_h > TRADE_MAX_H:
                status, r = "expired", r
            else:
                keep.append(tr); continue
        elif status == "open":
            if age_h > TRADE_MAX_H:
                status, r = "expired", 0.0
            else:
                keep.append(tr); continue
        rn = book(state, tr, status, r, now)
        if status == "tp1":
            msg = f"✅ {head}: TP1 erreicht ({r:+.2f}R{' gesamt mit Teilgewinn' if mode == 'half' else ''})\nEinstieg {tr['entry']} → Ziel {tr['tp1']}"
        elif status == "tp2":
            msg = f"✅✅ {head}: TP2 erreicht ({r:+.1f}R gesamt, Teilgewinn + Rest)\nEinstieg {tr['entry']} → Ziel {tr['tp2']}"
        elif status == "be":
            msg = f"➖ {head}: nach Teilgewinn auf Break-even ausgestoppt ({r:+.2f}R, Teilgewinn gesichert)"
        elif status == "stop":
            msg = f"❌ {head}: Stop getroffen ({r:+.1f}R)\nEinstieg {tr['entry']} → Stop {tr['stop']}"
        else:
            msg = f"⏱ {head}: nach {int(age_h)} h ohne Ergebnis geschlossen ({r:+.1f}R)"
        msg += f"\nNach Kosten: {rn:+.2f}R (Kosten pauschal geschätzt)\n{stats_line(state)}"
        if state.get("_stats_mine"):
            msg += f"\n{stats_line(state, 'mine')}"
        try:
            telegram(msg, dry)
        except Exception as e:
            print(f"Ergebnis-Meldung fehlgeschlagen: {scrub(e)}")
    state["_trades"] = keep


# ---------------- Befehle und Buttons ----------------
HELP = ("Befehle:\n/status – offene Meldungen mit aktuellem Stand\n/bilanz – Auswertung\n/journal – abgeschlossene Meldungen als CSV\n"
        "/demo – Demo-Signale: offene Positionen und Bilanz\n"
        "/meme – Memecoin-Radar: Live-Test-Auswertung\n"
        "/rein <Nr> – Meldung als deinen Trade markieren\n/raus <Nr> – Meldung als nicht genommen markieren\n"
        "(Die Buttons unter jeder Meldung machen dasselbe.)\n"
        "Hinweis: Der Scanner liest Befehle nur bei jedem Lauf, die Antwort kommt also mit Verzögerung.")


def mark(state, no, taken):
    for tr in list(state.get("_trades", [])) + list(state.get("_swing", {}).values()):
        if tr.get("no") == no:
            tr["taken"] = taken
            if taken is False and tr.get("sent_key"):
                state.pop(tr["sent_key"], None)       # zählt nicht mehr für den Korrelationsfilter
            return tr
    return None


def status_text(state, frames):
    trs = state.get("_trades", [])
    sw = state.get("_swing", {})
    if not trs and not sw:
        return "Keine offenen Meldungen."
    lines = []
    for p in sw.values():
        r = (p.get("close", p["entry"]) - p["entry"]) / p["risk"]
        flag = {True: "dein Trade", False: "nicht genommen"}.get(p.get("taken"), "nicht markiert")
        lines.append(f"#{p['no']} SWING {p['symbol']} LONG seit {p['t0'][:10]}: {r:+.1f}R (Schluss {round(p.get('close', p['entry']), 4)}), Stop {round(p['stop'], 4)}, {flag}")
    for tr in trs:
        price, r = r_now(tr, frames.get(tr["symbol"]))
        stand = f"{r:+.1f}R (Kurs {round(price, 5)})" if price is not None else "Kurs aktuell nicht verfügbar"
        flag = {True: "dein Trade", False: "nicht genommen"}.get(tr.get("taken"), "nicht markiert")
        lines.append(f"#{tr.get('no', '?')} {tr['symbol']} {tr['side'].upper()} {tr['setup']}: {stand}, {flag}"
                     + (" | TP1 erreicht, Stop auf Einstieg" if tr.get("tp1_notified") else ""))
    return "Offene Meldungen:\n" + "\n".join(lines)


def process_updates(state, frames, now, dry, updates=None):
    """Liest Befehle und Button-Klicks aus Telegram. Nur Chats aus TELEGRAM_CHAT_ID werden beachtet."""
    allowed = set(chat_ids())
    if updates is None:
        if dry or not env("TELEGRAM_TOKEN") or not allowed:
            return
        payload = {"timeout": 0, "allowed_updates": ["message", "callback_query"]}
        if state.get("_offset"):
            payload["offset"] = state["_offset"]
        try:
            updates = tg("getUpdates", payload).get("result", [])
        except Exception as e:
            print(f"Telegram-Updates nicht lesbar: {scrub(e)}"); return
    for u in updates:
        state["_offset"] = max(state.get("_offset", 0), u["update_id"] + 1)
        try:
            cb = u.get("callback_query")
            msg = u.get("message")
            if cb and cb.get("message"):
                chat = str(cb["message"]["chat"]["id"])
                if chat not in allowed:
                    continue
                kind, _, num = (cb.get("data") or "").partition(":")
                if kind in ("in", "out") and num.isdigit():
                    tr = mark(state, int(num), kind == "in")
                    reply = (f"Gespeichert: #{num} ist {'dein Trade' if kind == 'in' else 'nicht genommen'}."
                             if tr else f"#{num} ist nicht mehr offen oder nicht bekannt.")
                    telegram(reply, dry, chat=chat)
                    if not dry:
                        for method, pl in (("answerCallbackQuery", {"callback_query_id": cb["id"], "text": "Gespeichert"}),
                                           ("editMessageReplyMarkup", {"chat_id": chat, "message_id": cb["message"]["message_id"],
                                                                       "reply_markup": {"inline_keyboard": []}})):
                            try: tg(method, pl)
                            except Exception: pass
            elif msg and msg.get("text"):
                chat = str(msg["chat"]["id"])
                if chat not in allowed:
                    print(f"Nachricht von fremdem Chat ignoriert ({chat})"); continue
                parts = msg["text"].strip().split()
                cmd = parts[0].split("@")[0].lower()
                if cmd in ("/start", "/hilfe", "/help"):
                    telegram(HELP, dry, chat=chat)
                elif cmd == "/status":
                    telegram(status_text(state, frames), dry, chat=chat)
                elif cmd == "/bilanz":
                    telegram(stats_text(state), dry, chat=chat)
                elif cmd == "/meme":
                    telegram(meme_stats_text(state), dry, chat=chat)
                elif cmd == "/demo":
                    telegram(demo_text(state), dry, chat=chat)
                elif cmd == "/journal":
                    data = journal_csv(state)
                    if data:
                        telegram_file("sendDocument", data, "journal.csv", "Abgeschlossene Meldungen", dry, chat=chat)
                    else:
                        telegram("Noch keine abgeschlossenen Meldungen.", dry, chat=chat)
                elif cmd in ("/rein", "/raus"):
                    if len(parts) > 1 and parts[1].lstrip("#").isdigit():
                        no = int(parts[1].lstrip("#"))
                        tr = mark(state, no, cmd == "/rein")
                        telegram((f"Gespeichert: #{no} ist {'dein Trade' if cmd == '/rein' else 'nicht genommen'}."
                                  if tr else f"#{no} ist nicht offen oder nicht bekannt."), dry, chat=chat)
                    else:
                        telegram(f"Bitte mit Nummer, z. B. {cmd} 3", dry, chat=chat)
        except Exception as e:
            print(f"Befehl fehlgeschlagen: {scrub(e)}")


# ---------------- Signal-Bot v2: Daten ----------------
def fetch_twelve_tf(symbol, interval, n, drop_last=True):
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": interval, "outputsize": n, "timezone": "UTC",
        "apikey": env("TWELVEDATA_KEY", required=True)}, timeout=30)
    j = r.json()
    if j.get("status") == "error":
        raise RuntimeError(j.get("message"))
    df = pd.DataFrame(j["values"])
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.sort_values("time")[["time", "open", "high", "low", "close"]].reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True) if drop_last else df


def fetch_daily(kind, sym, now):
    """Abgeschlossene Tageskerzen (die heutige, laufende wird entfernt)."""
    if kind == "kraken":
        df = fetch_kraken(sym, 1440)[["time", "open", "high", "low", "close"]]
    else:
        df = fetch_twelve_tf(sym, "1day", 160, drop_last=False)
    df = df[df.time.dt.date < now.date()].reset_index(drop=True)
    if df.empty or (now - df.time.iloc[-1].to_pydatetime()).days > 6:
        raise RuntimeError("Tagesdaten veraltet oder leer")
    return df


def daily_trend_value(dd):
    """+1 / -1 / 0 aus Tageskerzen (EMA20 gegen EMA50 und Schlusskurs), nur abgeschlossene Kerzen."""
    e20 = dd.close.ewm(span=20, adjust=False).mean().iloc[-1]
    e50 = dd.close.ewm(span=50, adjust=False).mean().iloc[-1]
    c = dd.close.iloc[-1]
    return 1 if (e20 > e50 and c > e50) else -1 if (e20 < e50 and c < e50) else 0


# ---------------- Signal-Bot v2: Intraday (Gold, 1h-Rücksetzer im Tagestrend) ----------------
def intraday_signals(sym, df, trend, now):
    d = add_ind(df.copy())
    if len(d) < 60 or trend == 0:
        return []
    r = d.iloc[-1]
    a = r.atr
    if pd.isna(a):
        return []
    if not (INTRADAY_SESSION_UTC[0] <= r.time.hour < INTRADAY_SESSION_UTC[1]):
        return []
    age_min = (pd.Timestamp(now) - (r.time + pd.Timedelta(hours=1))).total_seconds() / 60
    if age_min > INTRADAY_STALE_MIN:
        return []
    w = d.iloc[-6:]
    out = []

    def add(side, stop):
        entry = float(r.close)
        risk = abs(entry - stop)
        if not (0.5 * a <= risk <= 3 * a):
            return
        sg = 1 if side == "long" else -1
        out.append({"id": f"{sym}|Rücksetzer 1h|{side}", "symbol": sym, "setup": "Rücksetzer 1h (Tagestrend)", "side": side,
                    "candle_time": r.time.strftime("%Y-%m-%d %H:%M UTC"), "candle_ts": r.time.isoformat(),
                    "entry": round(entry, 5), "stop": round(float(stop), 5),
                    "tp1": round(entry + sg * TP1_R * risk, 5), "tp2": round(entry + sg * TP2_R * risk, 5),
                    "risk_dist": float(risk), "trend_text": "Tagestrend " + ("aufwärts" if trend == 1 else "abwärts"),
                    "size": size(sym, entry, stop)})

    if trend == 1 and (w.low <= w.ema20).any() and r.close > r.ema20 > r.ema50 and r.close > r.open:
        add("long", w.low.min() - 0.2 * a)
    short_on = str(os.environ.get("INTRADAY_SHORTS", INTRADAY_SHORTS)).lower() in ("1", "true", "ja")
    if short_on and trend == -1 and (w.high >= w.ema20).any() and r.close < r.ema20 < r.ema50 and r.close < r.open:
        add("short", w.high.max() + 0.2 * a)
    return out


def retail_sentiment(sym):
    """Info-Zeile: Positionierung der Privat-Trader laut Myfxbook (nur wenn MYFXBOOK_EMAIL/PASSWORD gesetzt). Nicht getestet."""
    email, pw = env("MYFXBOOK_EMAIL"), env("MYFXBOOK_PASSWORD")
    if not email or not pw:
        return None
    try:
        j = requests.get("https://www.myfxbook.com/api/login.json", params={"email": email, "password": pw}, timeout=20).json()
        if j.get("error") or not j.get("session"):
            print(f"Myfxbook-Login fehlgeschlagen: {scrub(j.get('message', ''))}"); return None
        sess = j["session"]
        o = requests.get("https://www.myfxbook.com/api/get-community-outlook.json", params={"session": sess}, timeout=20).json()
        try:
            requests.get("https://www.myfxbook.com/api/logout.json", params={"session": sess}, timeout=10)
        except Exception:
            pass
        for s in o.get("symbols", []):
            if str(s.get("name", "")).upper() == sym:
                return (f"Privat-Trader (Myfxbook): {s.get('longPercentage')} % long, {s.get('shortPercentage')} % short. "
                        "Nur Info, nicht getestet; extreme Werte gelten eher als Gegen-Signal.")
    except Exception as e:
        print(f"Myfxbook nicht verfügbar: {scrub(e)}")
    return None


def _fred_series(sid):
    key = env("FRED_API_KEY")
    if key:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": sid, "api_key": key, "file_type": "json", "observation_start": (datetime.now(timezone.utc) - timedelta(days=500)).strftime("%Y-%m-%d")}, timeout=30)
        r.raise_for_status()
        v = pd.to_numeric(pd.DataFrame(r.json()["observations"])["value"], errors="coerce")
    else:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=(10, 40), headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text)); v = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    return v.dropna().reset_index(drop=True)


def macro_context(state, now):
    """Umfeld für den Live-Test (beeinflusst keine Signale): VIX hoch/niedrig, Dollar stark/schwach. Einmal pro Tag gecacht."""
    today = now.strftime("%Y-%m-%d")
    c = state.get("_macro")
    if c and c.get("date") == today:
        return c
    c = {"date": today, "vix_high": None, "usd_strong": None}
    try:
        vix = _fred_series("VIXCLS")
        if len(vix) >= 200: c["vix_high"] = bool(vix.iloc[-1] > vix.iloc[-252:].median())
    except Exception as e:
        print(f"VIX nicht verfügbar: {scrub(e)}")
    try:
        usd = _fred_series("DTWEXBGS")
        if len(usd) >= 50: c["usd_strong"] = bool(usd.iloc[-1] > usd.iloc[-50:].mean())
    except Exception as e:
        print(f"Dollar-Index nicht verfügbar: {scrub(e)}")
    parts = [{True: "VIX hoch (über Jahresmittel)", False: "VIX niedrig (unter Jahresmittel)"}.get(c["vix_high"], "VIX unbekannt"),
             {True: "Dollar stark (über 50-Tage-Schnitt)", False: "Dollar schwach (unter 50-Tage-Schnitt)"}.get(c["usd_strong"], "Dollar unbekannt")]
    c["text"] = ("Umfeld: " + " | ".join(parts) + ". Nur Info für den Live-Test: Im Backtest liefen Signale bei VIX hoch und bei Dollar stark "
                 "besser, nicht gesichert, kein Grund ein Signal auszulassen.")
    state["_macro"] = c
    return c


def evaluate_intraday(cands, now, news, state=None):
    out = []
    for s in cands:
        block, penalty, notes = news_check(s["symbol"], news, now)
        if block:
            print(f"{s['id']} unterdrückt: {notes[0]}"); continue
        text = INTRADAY_EVIDENCE + "\nNews: " + ("; ".join(notes) if notes else "keine High-Impact-Termine gefunden")
        senti = retail_sentiment(s["symbol"])
        if senti:
            text += "\n" + senti
        if state is not None:
            mc = macro_context(state, now)
            s["vix_high"], s["usd_strong"] = mc.get("vix_high"), mc.get("usd_strong")
            text += "\n" + mc["text"]
        out.append((s, "Intraday", text))
    return out


# ---------------- Signal-Bot v2: Swing (Tages-Trend, nur Long) ----------------
def swing_signal(sym, df):
    d = add_ind(df.copy())
    if len(d) < SWING_ENTRY_DAYS + 20:
        return None
    hh = d.close.rolling(SWING_ENTRY_DAYS).max().shift(1)
    r = d.iloc[-1]
    if pd.isna(r.atr) or not (r.close > hh.iloc[-1]):
        return None
    risk = SWING_ATR_MULT * float(r.atr)
    entry = float(r.close)
    return {"symbol": sym, "entry": entry, "atr": float(r.atr), "risk": risk, "stop": entry - risk, "init_stop": entry - risk,
            "best": entry, "t0": r.time.isoformat(), "last": r.time.isoformat(), "close": entry,
            "size": size(sym, entry, entry - risk)}


def swing_msg(p, n_open):
    e, risk = p["entry"], p["risk"]
    note = (f"\nAchtung: schon {n_open} Swing-Positionen offen (Hinweisgrenze {SWING_MAX_OPEN}). Nur zusätzlich eingehen, "
            "wenn du das Gesamtrisiko bewusst trägst.") if n_open >= SWING_MAX_OPEN else ""
    return (f"📈 SWING LONG #{p['no']} {p['symbol']} | Tages-Trend (neues {SWING_ENTRY_DAYS}-Tage-Hoch)\n"
            f"Tageskerze: {p['t0'][:10]}\n\n"
            f"Einstieg: ca. {round(e, 5)} (Marktpreis, nicht über {round(e + MAX_CHASE_R * risk, 5)} einsteigen)\n"
            f"Stop: {round(p['stop'], 5)} ({SWING_ATR_MULT:g} ATR)\nGröße: {p['size']}\n\n"
            f"Kein festes Ziel. Stop täglich nachziehen: höchster Schlusskurs seit Einstieg minus {SWING_ATR_MULT:g} ATR "
            f"(ATR beim Einstieg {round(p['atr'], 5)}). Der Bot meldet jeden neuen Stop.{note}\n\n"
            f"{SWING_EVIDENCE}\n\nNur Hinweis, keine Anlageberatung. Preis, Größe und Termine vor der Order selbst prüfen.")


def swing_update(state, sym, df, now, dry):
    """Nachziehen oder Beenden einer offenen Swing-Position anhand neuer Tageskerzen."""
    p = state["_swing"].get(sym)
    if not p:
        return
    old_stop = p["stop"]
    for _, r in df[df.time > pd.Timestamp(p["last"])].iterrows():
        if r.low <= p["stop"]:
            ex = min(p["stop"], float(r.open))
            g = (ex - p["entry"]) / p["risk"]
            tr = {"symbol": sym, "side": "long", "setup": "Tages-Trend", "entry": p["entry"], "stop": p["init_stop"],
                  "t0": p["t0"], "q": "Swing", "no": p["no"], "taken": p.get("taken"), "tp1": 0}
            rn = book(state, tr, "swing", g, now)
            del state["_swing"][sym]
            msg = (f"🏁 SWING #{p['no']} {sym} beendet: Trailing-Stop {round(p['stop'], 5)} getroffen (Ausstieg ca. {round(ex, 5)})\n"
                   f"Ergebnis {g:+.1f}R, nach Kosten {rn:+.2f}R. Einstieg war {round(p['entry'], 5)} am {p['t0'][:10]}.\n{stats_line(state)}")
            try:
                telegram(msg, dry)
            except Exception as e:
                print(f"Swing-Ende-Meldung fehlgeschlagen: {scrub(e)}")
            return
        p["best"] = max(p["best"], float(r.close))
        p["stop"] = max(p["stop"], p["best"] - SWING_ATR_MULT * p["atr"])
        p["last"] = r.time.isoformat()
        p["close"] = float(r.close)
    p["moved"] = p["stop"] > old_stop + 1e-9


def swing_run(state, now, dry):
    """Einmal täglich: Tageskerzen holen, offene Positionen nachziehen, neue Signale prüfen."""
    today = now.strftime("%Y-%m-%d")
    done = state.setdefault("_swing_done", {})
    state.setdefault("_swing", {})
    errors, fetched = 0, 0
    for sym, (kind, ts) in SWING.items():
        if done.get(sym) == today:
            continue
        if kind == "twelve" and not env("TWELVEDATA_KEY"):
            continue
        try:
            if kind == "twelve" and fetched:
                time.sleep(9)                       # Twelve Data: höchstens 8 Abrufe pro Minute
            df = fetch_daily(kind, ts, now)
            fetched += 1 if kind == "twelve" else 0
        except Exception as e:
            errors += 1
            print(f"[Swing {sym}] Fehler: {scrub(e)}")
            alert_error(state, f"swing-{sym}", f"Swing {sym}: {scrub(e)[:150]}", dry)
            continue
        if sym in state["_swing"]:
            swing_update(state, sym, df, now, dry)
        if sym not in state["_swing"]:
            sig = swing_signal(sym, df)
            if sig:
                sig["no"] = next_no(state)
                sig["taken"] = None
                n_open = len(state["_swing"])
                state["_swing"][sym] = sig
                buttons = [{"text": "✅ Ich bin drin", "callback_data": f"in:{sig['no']}"},
                           {"text": "🚫 Nicht genommen", "callback_data": f"out:{sig['no']}"}]
                try:
                    telegram(swing_msg(sig, n_open), dry, buttons=buttons)
                except Exception as e:
                    print(f"Swing-Meldung fehlgeschlagen: {scrub(e)}")
        done[sym] = today
    # Tagesübersicht der offenen Positionen
    open_p = list(state["_swing"].values())
    if open_p and all(done.get(s) == today for s in SWING if (SWING[s][0] != "twelve" or env("TWELVEDATA_KEY"))) and state.get("_swing_msg") != today:
        lines = []
        for p in open_p:
            r = (p.get("close", p["entry"]) - p["entry"]) / p["risk"]
            lines.append(f"#{p['no']} {p['symbol']}: Schluss {round(p.get('close', p['entry']), 4)} ({r:+.1f}R), neuer Stop {round(p['stop'], 4)}"
                         + (" (nachgezogen)" if p.get("moved") else ""))
        try:
            telegram("📋 Swing-Übersicht (Tagesschluss)\n" + "\n".join(lines) + "\nStop jeweils beim Broker nachziehen, falls du die Position hältst.", dry)
            state["_swing_msg"] = today
        except Exception as e:
            print(f"Swing-Übersicht fehlgeschlagen: {scrub(e)}")
    return errors


# ================= Demo-Signale (im Labor NICHT bestanden, nur Demokonto) =================
# 
try:
    from zoneinfo import ZoneInfo
    DEMO_NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    DEMO_NY = "America/New_York"

DEMO_COST_PCT = 0.00006          # Kosten je Trade (Kauf+Verkauf) in % vom Preis: ca. 1,5 Nasdaq-Punkte bzw. 0,5 S&P-Punkte
DEMO_FINANCING = 0.05            # Übernachtfinanzierung p.a. (nur Mean Reversion)
DEMO_ORB_TARGET_R = 10.0
DEMO_ORB_LATEST_MIN = 30         # ORB-Signal nur bis 10:00 New York, danach verpasst
DEMO_MOM_LOOKBACK = 14
DEMO_MOM_CHECKS = list(range(30, 390, 30))   # Minuten nach 09:30 -> 10:00 ... 15:30
DEMO_MOM_LATE_MIN = 20           # Prüfzeitpunkt höchstens so viele Minuten verspätet auswerten
DEMO_MR_WINDOW = (360, 389)      # 15:30-15:59 New York
DEMO_MR_RSI_MAX, DEMO_MR_MAX_HOLD = 25.0, 10
DEMO_MIN_RISK_COST = 3.0
DEMO_CHASE_R = 0.5
DEMO_NAMES = {"QQQ": "Nasdaq 100 (QQQ)", "SPY": "S&P 500 (SPY)"}

DEMO_EVIDENCE = {
    "ORB": ("Labor: Nasdaq 2021-2026 +0,12R pro Trade (t=1,6), auf S&P 500, Dow und DAX nicht bestätigt "
            "(alle zusammen +0,01R). DURCHGEFALLEN."),
    "MOMENTUM": ("Labor: Nasdaq +0,19R pro Trade (t=1,2), S&P 500 -0,14R, Dow -0,14R, DAX +0,05R "
                 "(alle zusammen +0,02R). DURCHGEFALLEN."),
    "MEANREV": ("Labor: Nasdaq +0,32R (t=2,7), S&P 500 +0,53R (t=4,3), 4 Indizes zusammen +0,28R (t=3,6), "
                "aber nicht sicher besser als Zufall (p=0,04). NICHT BESTANDEN, bester Kandidat."),
}
DEMO_HEAD = "🧪 DEMO-SIGNAL (nicht bestanden, nur Demokonto)"
# Umrechnung ETF -> Index-CFD (Index ist etwa so viel Mal größer). Mit Variablen NDX_FAKTOR / SPX_FAKTOR anpassbar,
# falls die CFD-Werte deines Brokers dauerhaft abweichen. Die Prozent-Abstände stimmen unabhängig davon.
DEMO_INDEX = {"QQQ": ("Nasdaq-CFD", "NDX_FAKTOR", 41.2), "SPY": ("S&P-500-CFD", "SPX_FAKTOR", 10.0)}


def demo_cfd(sym, entry, stop=None, target=None):
    """Textblock mit den Werten für den Index-CFD (Indexpunkte)."""
    name, var, default = DEMO_INDEX[sym]
    try:
        f = float(os.environ.get(var) or default)
    except ValueError:
        f = default
    lines = [f"Für deinen {name} (umgerechnet, ca.):", f"Einstieg: ca. {entry * f:,.0f}".replace(",", ".")]
    if stop is not None:
        lines.append(f"Stop: ca. {stop * f:,.0f} ({abs(entry - stop) * f:,.0f} Punkte entfernt)".replace(",", "."))
    if target is not None:
        lines.append(f"Ziel: ca. {target * f:,.0f} ({abs(target - entry) * f:,.0f} Punkte entfernt)".replace(",", "."))
    lines.append("Weicht dein CFD-Kurs beim Einstieg ab: Abstände in Punkten bzw. Prozent übernehmen, nicht die Preise.")
    return "\n".join(lines)


# ------------------------------------------------------------------ Daten
def demo_fetch(symbol, interval, n, key, now):
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": interval, "outputsize": n, "timezone": "UTC", "apikey": key}, timeout=30)
    j = r.json()
    if j.get("status") == "error":
        raise RuntimeError(j.get("message"))
    df = pd.DataFrame(j["values"])
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    df = df.sort_values("time").reset_index(drop=True)
    if interval.endswith("min"):
        step = pd.Timedelta(minutes=int(interval[:-3]))
        df = df[df.time + step <= pd.Timestamp(now)]          # nur abgeschlossene Kerzen
    return df.reset_index(drop=True)


def demo_session(df):
    """Fügt New-York-Datum und Minute seit 09:30 hinzu, nur reguläre Handelszeit."""
    t = df.time.dt.tz_convert(DEMO_NY)
    df = df.assign(day=t.dt.date, m=(t.dt.hour * 60 + t.dt.minute - 570).astype(int))
    return df[(df.m >= 0) & (df.m < 390)].reset_index(drop=True)


def demo_ny_now(now):
    t = pd.Timestamp(now).tz_convert(DEMO_NY)
    return t.date(), t.hour * 60 + t.minute - 570, t.weekday()


# ------------------------------------------------------------------ Buchführung
def demo_st(state):
    d = state.setdefault("_demo", {})
    for k, v in (("open", {}), ("stats", {}), ("journal", []), ("done", {})):
        d.setdefault(k, v)
    return d


def demo_book(d, key, pos, exit_px, why, now, extra_cost=0.0):
    risk = pos["risk"]
    gross = pos["dir"] * (exit_px - pos["entry"]) / risk
    net = gross - (DEMO_COST_PCT * pos["entry"] + extra_cost) / risk
    s = d["stats"].setdefault(pos["strat"], {"n": 0, "win": 0, "r": 0.0})
    s["n"] += 1
    s["win"] += 1 if net > 0 else 0
    s["r"] = round(s["r"] + net, 3)
    d["journal"].append({"closed": pd.Timestamp(now).strftime("%Y-%m-%d %H:%M"), "strategie": pos["strat"],
                         "symbol": pos["sym"], "richtung": "long" if pos["dir"] > 0 else "short",
                         "einstieg": pos["entry"], "ausstieg": round(exit_px, 4), "grund": why,
                         "r": round(gross, 2), "r_netto": round(net, 2)})
    del d["journal"][:-300]
    d["open"].pop(key, None)
    return gross, net


def demo_summary_line(state):
    d = state.get("_demo")
    if not d or not d.get("stats"):
        return "Demo-Signale: noch nichts abgeschlossen."
    parts = [f"{k} {v['n']}x {v['r']:+.1f}R" for k, v in d["stats"].items()]
    return "Demo-Signale (nach Kosten): " + ", ".join(parts)


def demo_text(state):
    d = demo_st(state)
    out = ["🧪 Demo-Signale (nicht bestanden, Live-Test)"]
    if d["stats"]:
        for k, v in d["stats"].items():
            q = v["win"] / v["n"] * 100 if v["n"] else 0
            out.append(f"{k}: {v['n']} Trades, Quote {q:.0f} %, Summe {v['r']:+.1f}R, Ø {v['r'] / max(v['n'], 1):+.2f}R")
    else:
        out.append("Noch keine abgeschlossenen Demo-Trades.")
    if d["open"]:
        out.append("Offen:")
        for p in d["open"].values():
            out.append(f"  {p['strat']} {p['sym']} {'LONG' if p['dir'] > 0 else 'SHORT'} seit {p['t0']}, "
                       f"Einstieg {p['entry']}, Stop {p.get('stop', '-')}")
    out.append("Zum Vergleich mit dem Labor: ORB +0,12R, Momentum +0,19R, Mean Reversion +0,3 bis +0,5R pro Trade "
               "(jeweils Nasdaq/S&P). Aussagekräftig erst nach etwa 50-100 Trades je Strategie.")
    return "\n".join(out)


def demo_pct(a, b):
    return f"{abs(a - b) / b * 100:.2f} %"


# ------------------------------------------------------------------ ORB
def demo_orb(d, q, today, m_now, now, send):
    key = "ORB|QQQ"
    t = q[q.day == today]
    pos = d["open"].get(key)
    if pos:                                   # offene Position verfolgen (Kerzen ihres Handelstags)
        tp = q[q.day.astype(str) == pos["day"]]
        for _, b in tp[tp.time > pd.Timestamp(pos["last"])].iterrows():
            stop_hit = b.low <= pos["stop"] if pos["dir"] > 0 else b.high >= pos["stop"]
            tgt_hit = b.high >= pos["target"] if pos["dir"] > 0 else b.low <= pos["target"]
            pos["last"] = b.time.isoformat()
            if stop_hit:
                px = min(b.open, pos["stop"]) if pos["dir"] > 0 else max(b.open, pos["stop"])
                g, n = demo_book(d, key, pos, px, "Stop", now)
                cfd = f"{pos['stop'] * DEMO_INDEX['QQQ'][2]:,.0f}".replace(",", ".")
                send(f"🧪❌ DEMO ORB QQQ: Stop {pos['stop']} getroffen (Nasdaq-CFD ca. {cfd}) "
                     f"({g:+.1f}R, nach Kosten {n:+.2f}R)")
                return
            if tgt_hit:
                g, n = demo_book(d, key, pos, pos["target"], "Ziel 10R", now)
                send(f"🧪✅ DEMO ORB QQQ: Ziel 10R {pos['target']} erreicht ({g:+.1f}R, nach Kosten {n:+.2f}R)")
                return
            if b.m >= 385:
                g, n = demo_book(d, key, pos, b.close, "Handelsschluss", now)
                send(f"🧪⏱ DEMO ORB QQQ: zum Handelsschluss geschlossen bei {b.close} ({g:+.1f}R, nach Kosten {n:+.2f}R). "
                     "Position jetzt schließen, falls noch offen.")
                return
        if pos["day"] != str(today):                          # Vortag ohne Abschluss -> letzter Kurs des Tages
            px = float(tp.close.iloc[-1]) if not tp.empty else pos["entry"]
            g, n = demo_book(d, key, pos, px, "Handelsschluss (nachträglich)", now)
            send(f"🧪⏱ DEMO ORB QQQ vom {pos['day']}: nachträglich zum Schluss abgerechnet ({g:+.1f}R, nach Kosten {n:+.2f}R).")
        return
    if d["done"].get("ORB") == str(today) or t.empty or t.m.iloc[0] != 0:
        return
    d["done"]["ORB"] = str(today)
    if m_now > DEMO_ORB_LATEST_MIN:
        print("Demo ORB: Signalzeit verpasst (Lauf zu spät)")
        return
    f = t.iloc[0]
    if f.close == f.open:
        return
    di = 1 if f.close > f.open else -1
    stop = f.low if di > 0 else f.high
    entry = float(t.close.iloc[-1])
    risk0 = di * (f.close - stop)
    risk = di * (entry - stop)
    if risk0 <= 0 or risk <= 0 or di * (entry - f.close) > DEMO_CHASE_R * risk0 or risk < DEMO_MIN_RISK_COST * DEMO_COST_PCT * entry:
        print("Demo ORB: kein gültiger Einstieg mehr (Stop zu nah, schon getroffen oder Preis davongelaufen)")
        return
    target = round(entry + di * DEMO_ORB_TARGET_R * risk, 2)
    d["open"][key] = {"strat": "ORB", "sym": "QQQ", "dir": di, "entry": round(entry, 2), "stop": round(stop, 2),
                      "target": target, "risk": risk, "t0": f"{today} 09:35 NY", "last": t.time.iloc[-1].isoformat(),
                      "day": str(today)}
    side = "LONG 🟢" if di > 0 else "SHORT 🔴"
    send(f"{DEMO_HEAD}\nORB {side} Nasdaq 100 (QQQ) | erste 5 Min. {'grün' if di > 0 else 'rot'}\n\n"
         f"Einstieg: ca. {entry:.2f} (Marktpreis)\nStop: {stop:.2f} ({demo_pct(stop, entry)} entfernt)\n"
         f"Ziel 10R: {target:.2f} ({demo_pct(target, entry)} entfernt)\nSonst: Ausstieg um 16:00 New York\n\n"
         f"{demo_cfd('QQQ', entry, stop, target)}\n\n{DEMO_EVIDENCE['ORB']}\n"
         "Nur Demo, keine Anlageberatung.")


# ------------------------------------------------------------------ Momentum
def demo_momentum(d, q, today, m_now, now, send):
    key = "MOM|QQQ"
    days = sorted(q.day.unique())
    if not days or days[-1] != today:
        return
    pos = d["open"].get(key)
    t = q[q.day == today].set_index("m")
    if pos and pos["day"] != str(today):                      # Vortag ohne Abschluss -> letzter Kurs des Tages
        tp = q[q.day.astype(str) == pos["day"]]
        px = float(tp.close.iloc[-1]) if not tp.empty else pos.get("last_px", pos["entry"])
        g, n = demo_book(d, key, pos, px, "Handelsschluss (nachträglich)", now)
        send(f"🧪⏱ DEMO MOMENTUM QQQ vom {pos['day']}: nachträglich zum Schluss abgerechnet ({g:+.1f}R, nach Kosten {n:+.2f}R).")
        pos = None
    if pos and m_now >= 390 and 385 in t.index:               # Handelsschluss
        px = float(t.loc[385, "close"])
        g, n = demo_book(d, key, pos, px, "Handelsschluss", now)
        send(f"🧪⏱ DEMO MOMENTUM QQQ: zum Handelsschluss geschlossen bei {px:.2f} ({g:+.1f}R, nach Kosten {n:+.2f}R). "
             "Position jetzt schließen, falls noch offen.")
        return
    due = [k for k in DEMO_MOM_CHECKS if k <= m_now and (k - 5) in t.index]
    if not due:
        return
    k = due[-1]
    done = d["done"].setdefault("MOM", {})
    if done.get("day") != str(today):
        done.clear()
        done["day"] = str(today)
    if str(k) in done:
        return
    done[str(k)] = True
    if m_now - k > DEMO_MOM_LATE_MIN:
        print(f"Demo Momentum: Prüfzeitpunkt {k} verpasst (Lauf zu spät)")
        return
    prior = [x for x in days if x < today][-(DEMO_MOM_LOOKBACK + 1):]
    if len(prior) < DEMO_MOM_LOOKBACK + 1:
        print("Demo Momentum: zu wenig Historie")
        return
    moves = []
    for x in prior[1:]:
        dx = q[q.day == x].set_index("m")
        if 0 in dx.index and (k - 5) in dx.index:
            moves.append(abs(dx.loc[k - 5, "close"] / dx.loc[0, "open"] - 1))
    if len(moves) < DEMO_MOM_LOOKBACK - 3 or 0 not in t.index:
        return
    sigma = float(np.mean(moves))
    op = float(t.loc[0, "open"])
    pc = float(q[q.day == prior[-1]].close.iloc[-1])
    ub, lb = max(op, pc) * (1 + sigma), min(op, pc) * (1 - sigma)
    upto = t.loc[:k - 5]
    tp = (upto.high + upto.low + upto.close) / 3
    vwap = float((tp * upto.volume).sum() / upto.volume.sum()) if upto.volume.sum() > 0 else float(tp.mean())
    px = float(t.loc[k - 5, "close"])
    clock = (pd.Timestamp("2000-01-01 09:30") + timedelta(minutes=k)).strftime("%H:%M")
    if pos:
        pos["last_px"] = px
        lvl = max(ub, vwap) if pos["dir"] > 0 else min(lb, vwap)
        if (pos["dir"] > 0 and px < lvl) or (pos["dir"] < 0 and px > lvl):
            g, n = demo_book(d, key, pos, px, f"Stop {clock}", now)
            send(f"🧪❌ DEMO MOMENTUM QQQ: Ausstieg {clock} New York bei ca. {px:.2f} "
                 f"(Kurs hinter Stop-Niveau {lvl:.2f}). Ergebnis {g:+.1f}R, nach Kosten {n:+.2f}R. Jetzt schließen.")
            pos = None
        else:
            pos["stop"] = round(lvl, 2)
            return
    if pos is None and key not in d["open"]:
        di = 1 if px > ub else -1 if px < lb else 0
        if not di:
            return
        lvl = max(ub, vwap) if di > 0 else min(lb, vwap)
        risk = di * (px - lvl)
        if risk < DEMO_MIN_RISK_COST * DEMO_COST_PCT * px:
            return
        d["open"][key] = {"strat": "MOMENTUM", "sym": "QQQ", "dir": di, "entry": round(px, 2), "stop": round(lvl, 2),
                          "risk": risk, "t0": f"{today} {clock} NY", "day": str(today), "last_px": px}
        side = "LONG 🟢" if di > 0 else "SHORT 🔴"
        send(f"{DEMO_HEAD}\nMOMENTUM {side} Nasdaq 100 (QQQ) | Ausbruch aus der Noise Area um {clock} New York\n\n"
             f"Einstieg: ca. {px:.2f} (Marktpreis)\nStop-Niveau jetzt: {lvl:.2f} ({demo_pct(lvl, px)} entfernt)\n"
             f"Kein festes Ziel. Der Bot prüft alle 30 Min. (nächster Check "
             f"{(pd.Timestamp('2000-01-01 09:30') + timedelta(minutes=k + 30)).strftime('%H:%M')}) und meldet den Ausstieg; "
             "spätestens um 16:00 schließen.\n\n"
             f"{demo_cfd('QQQ', px, lvl)}\n(Stop als Notbremse setzen; der eigentliche Ausstieg kommt vom Bot.)\n\n"
             f"{DEMO_EVIDENCE['MOMENTUM']}\nNur Demo, keine Anlageberatung.")


# ------------------------------------------------------------------ Mean Reversion
def demo_meanrev(d, sym, daily, px, today, now, send):
    key = f"MR|{sym}"
    dd = daily[daily.time.dt.date < today].reset_index(drop=True)
    if len(dd) < 210:
        print(f"Demo Mean Reversion {sym}: zu wenig Tageskerzen ({len(dd)})")
        return
    closes = np.r_[dd.close.values, px]
    sma200, sma5 = closes[-200:].mean(), closes[-5:].mean()
    ch = np.diff(closes)
    a = 0.5
    up = dn = 0.0
    for i, c in enumerate(ch):
        if i == 0:
            up, dn = max(c, 0), max(-c, 0)
        else:
            up = (1 - a) * up + a * max(c, 0)
            dn = (1 - a) * dn + a * max(-c, 0)
    rsi = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
    pc = dd.close.shift(1)
    tr = pd.concat([dd.high - dd.low, (dd.high - pc).abs(), (dd.low - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.iloc[-10:].mean())
    pos = d["open"].get(key)
    name = DEMO_NAMES.get(sym, sym)
    if pos:
        held = int((dd.time.dt.date > pd.Timestamp(pos["day"]).date()).sum()) + 1
        if px > sma5 or held >= DEMO_MR_MAX_HOLD:
            nights = (pd.Timestamp(today) - pd.Timestamp(pos["day"])).days
            fin = DEMO_FINANCING / 360 * pos["entry"] * nights
            g, n = demo_book(d, key, pos, px, "über 5-Tage-Schnitt" if px > sma5 else "10 Tage", now, extra_cost=fin)
            send(f"🧪 DEMO MEAN REVERSION {name}: VERKAUFEN jetzt (vor 16:00 New York), Kurs ca. {px:.2f} "
                 f"({'über 5-Tage-Schnitt' if px > sma5 else 'nach 10 Tagen'}). Ergebnis {g:+.2f}R, "
                 f"nach Kosten und Finanzierung {n:+.2f}R ({held} Tage gehalten).")
        return
    if px > sma200 and rsi < DEMO_MR_RSI_MAX:
        d["open"][key] = {"strat": "MEANREV", "sym": sym, "dir": 1, "entry": round(px, 2), "risk": atr,
                          "t0": f"{today} 15:30-16:00 NY", "day": str(today), "stop": "-"}
        send(f"{DEMO_HEAD}\nMEAN REVERSION LONG 🟢 {name} | Rücksetzer im Aufwärtstrend\n\n"
             f"Einstieg: ca. {px:.2f}, jetzt kaufen (vor 16:00 New York)\n"
             f"RSI(2): {rsi:.0f} (unter {DEMO_MR_RSI_MAX:g}), Kurs über 200-Tage-Schnitt ({sma200:.2f})\n"
             f"Ausstieg: Der Bot meldet den Verkauf, sobald der Kurs über dem 5-Tage-Schnitt schließt, "
             f"spätestens nach {DEMO_MR_MAX_HOLD} Tagen. Kein Stop (wie im Test), 1R = ATR(10) = {atr:.2f} "
             f"({demo_pct(px - atr, px)}).\n\n{demo_cfd(sym, px)}\nHaltedauer meist 2-5 Tage, "
             f"Übernachtkosten beachten.\n\n{DEMO_EVIDENCE['MEANREV']}\nNur Demo, keine Anlageberatung.")


# ------------------------------------------------------------------ Hauptlauf
def demo_run(state, now, dry, telegram, alert_error, env):
    """Gibt die Anzahl Fehler zurück."""
    if str(env("DEMO_SIGNALS", "1")) == "0" or not env("TWELVEDATA_KEY"):
        return 0
    today, m_now, wd = demo_ny_now(now)
    if wd >= 5 or not (-5 <= m_now <= 450):
        return 0
    key = env("TWELVEDATA_KEY")
    d = demo_st(state)

    def send(msg):
        try:
            telegram(msg, dry)
        except Exception as e:
            print(f"Demo-Meldung fehlgeschlagen: {e}")

    errors = 0
    try:
        time.sleep(8)                                    # Twelve Data: max. 8 Abrufe pro Minute
        q = demo_session(demo_fetch("QQQ", "5min", 1500, key, now))
        if not q.empty and q.day.iloc[-1] == today and 0 <= m_now < 390:
            age = (pd.Timestamp(now) - q.time.iloc[-1]).total_seconds() / 60
            if age > 20:
                raise RuntimeError(f"QQQ-Daten {int(age)} Min. alt (verzögert?)")
        demo_orb(d, q, today, m_now, now, send)
        demo_momentum(d, q, today, m_now, now, send)
    except Exception as e:
        errors += 1
        print(f"[Demo QQQ] Fehler: {e}")
        alert_error(state, "demo-qqq", f"Demo-Signale QQQ: {str(e)[:150]}", dry)
        q = None

    if DEMO_MR_WINDOW[0] <= m_now <= DEMO_MR_WINDOW[1] and d["done"].get("MR") != str(today):
        d["done"]["MR"] = str(today)
        for sym in ("QQQ", "SPY"):
            try:
                time.sleep(8)
                if sym == "QQQ" and q is not None and not q.empty:
                    px = float(q.close.iloc[-1])
                else:
                    px = float(demo_fetch(sym, "5min", 5, key, now).close.iloc[-1])
                    time.sleep(8)
                daily = demo_fetch(sym, "1day", 260, key, now)
                demo_meanrev(d, sym, daily, px, today, now, send)
            except Exception as e:
                errors += 1
                print(f"[Demo Mean Reversion {sym}] Fehler: {e}")
                alert_error(state, f"demo-mr-{sym}", f"Demo Mean Reversion {sym}: {str(e)[:150]}", dry)
    return errors


# ================= Memecoin-Radar (Base) – Spielgeld, NICHT getestet =================
# Sucht alle 10 Minuten auf dem Netzwerk Base nach Memecoins mit plötzlichem Kauf-Andrang
# und prüft jeden Kandidaten mit GoPlus auf typische Betrugsmerkmale (Honeypot, hohe Steuern ...).
# Daten: GeckoTerminal und GoPlus, beide kostenlos und ohne Schlüssel.
# Kein Backtest möglich (keine freien historischen DEX-Daten). Stattdessen Live-Test:
# Der Bot misst den Kurs jedes Treffers nach 1 h, 6 h und 24 h und vergleicht mit einer
# Zufallsgruppe anderer gefragter Pools (/meme). Abschalten: Variable MEME_RADAR=0.
MEME_NETWORK, MEME_GOPLUS_CHAIN = "base", "8453"
MEME_GT = "https://api.geckoterminal.com/api/v2"
MEME_MIN_LIQ = 40_000          # Liquidität im Pool (USD)
MEME_MIN_MCAP = 150_000        # Marktwert bzw. FDV (USD)
MEME_MIN_AGE_H = 2             # frisch gestartete Pools (höchstes Rug-Risiko) auslassen
MEME_MIN_VOL_H1 = 30_000       # Volumen der letzten Stunde (USD)
MEME_VOL_ACCEL = 2.0           # Volumen 1 h mindestens 2x so hoch wie der Stundenschnitt der letzten 6 h
MEME_CHG_H1 = (5.0, 80.0)      # Kurs 1 h: steigend, aber noch nicht senkrecht
MEME_MIN_M5 = -5.0             # letzte 5 Min. nicht im Absturz
MEME_BUYER_RATIO = 1.3         # mehr verschiedene Käufer als Verkäufer
MEME_MIN_BUYERS = 50
MEME_MAX_TAX = 0.10            # Kauf-/Verkaufssteuer höchstens 10 %
MEME_TOP_HOLDER = 0.15         # Warnung, wenn eine einzelne Wallet mehr als 15 % hält
MEME_MAX_PER_RUN = 3
MEME_COOLDOWN_H = 24
MEME_CHECKS_H = (1, 6, 24)
MEME_SKIP = {"WETH", "ETH", "USDC", "USDT", "DAI", "CBBTC", "WBTC", "CBETH", "WSTETH", "EURC", "USDBC", "AERO"}


def meme_get(url, params=None):
    r = requests.get(url, params=params, timeout=20, headers={"Accept": "application/json"})
    r.raise_for_status()
    time.sleep(2.2)                     # GeckoTerminal: ca. 30 Abrufe pro Minute
    return r.json()


def meme_num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def meme_pools(now):
    """Gefragte und neue Pools auf Base -> Liste flacher Dicts."""
    out, seen = [], set()
    for path in ("trending_pools", "new_pools"):
        js = meme_get(f"{MEME_GT}/networks/{MEME_NETWORK}/{path}", {"page": 1})
        for p in js.get("data", []):
            a = p.get("attributes", {})
            tok = (((p.get("relationships") or {}).get("base_token") or {}).get("data") or {}).get("id", "")
            addr = tok.split("_", 1)[1].lower() if "_" in tok else ""
            name = a.get("name") or ""
            sym = name.split("/")[0].strip()
            if not addr or addr in seen:
                continue
            seen.add(addr)
            tx = (a.get("transactions") or {}).get("h1") or {}
            vol = a.get("volume_usd") or {}
            chg = a.get("price_change_percentage") or {}
            created = pd.to_datetime(a.get("pool_created_at"), utc=True, errors="coerce")
            out.append({"addr": addr, "pool": a.get("address", ""), "name": name, "sym": sym,
                        "price": meme_num(a.get("base_token_price_usd")),
                        "liq": meme_num(a.get("reserve_in_usd")),
                        "mcap": meme_num(a.get("market_cap_usd")) or meme_num(a.get("fdv_usd")),
                        "vol_h1": meme_num(vol.get("h1")), "vol_h6": meme_num(vol.get("h6")),
                        "chg_m5": meme_num(chg.get("m5")), "chg_h1": meme_num(chg.get("h1")),
                        "chg_h6": meme_num(chg.get("h6")), "chg_h24": meme_num(chg.get("h24")),
                        "buyers": int(meme_num(tx.get("buyers"))), "sellers": int(meme_num(tx.get("sellers"))),
                        "buys": int(meme_num(tx.get("buys"))), "sells": int(meme_num(tx.get("sells"))),
                        "age_h": (pd.Timestamp(now) - created).total_seconds() / 3600 if pd.notna(created) else 0.0})
    return out


def meme_basic_ok(p):
    return (p["sym"].upper() not in MEME_SKIP and p["price"] > 0 and p["liq"] >= MEME_MIN_LIQ
            and p["mcap"] >= MEME_MIN_MCAP and p["age_h"] >= MEME_MIN_AGE_H)


def meme_signal(p):
    """Gibt (ok, Gründe) zurück."""
    accel = p["vol_h1"] / (p["vol_h6"] / 6) if p["vol_h6"] > 0 else 0.0
    p["accel"] = accel
    checks = [(meme_basic_ok(p), "Grundfilter"),
              (p["vol_h1"] >= MEME_MIN_VOL_H1, "Volumen 1 h"),
              (accel >= MEME_VOL_ACCEL, "Volumen-Beschleunigung"),
              (MEME_CHG_H1[0] <= p["chg_h1"] <= MEME_CHG_H1[1], "Kurs 1 h"),
              (p["chg_m5"] >= MEME_MIN_M5, "Kurs 5 Min."),
              (p["buyers"] >= MEME_MIN_BUYERS and p["buyers"] >= MEME_BUYER_RATIO * max(p["sellers"], 1), "Käufer/Verkäufer")]
    return all(c for c, _ in checks), [n for c, n in checks if not c]


def meme_security(addr):
    """GoPlus-Check. (ok, Punkte, Warnungen). Ohne Ergebnis -> nicht ok (Sicherheit zuerst)."""
    r = requests.get(f"https://api.gopluslabs.io/api/v1/token_security/{MEME_GOPLUS_CHAIN}",
                     params={"contract_addresses": addr}, timeout=20)
    r.raise_for_status()
    time.sleep(2.0)
    res = (r.json().get("result") or {})
    s = res.get(addr) or res.get(addr.lower()) or next(iter(res.values()), None)
    if not s:
        return False, [], ["GoPlus kennt den Token (noch) nicht"]
    flag = lambda k: str(s.get(k, "0")) == "1"
    bt, st_ = meme_num(s.get("buy_tax")), meme_num(s.get("sell_tax"))
    bad = []
    if flag("is_honeypot"): bad.append("Honeypot (Verkauf blockiert)")
    if flag("cannot_sell_all"): bad.append("nicht alles verkaufbar")
    if flag("hidden_owner"): bad.append("versteckter Besitzer")
    if flag("owner_change_balance"): bad.append("Besitzer kann Guthaben ändern")
    if flag("transfer_pausable"): bad.append("Handel kann gestoppt werden")
    if flag("is_blacklisted"): bad.append("Sperrliste im Vertrag")
    if str(s.get("is_open_source", "0")) != "1": bad.append("Vertrag nicht offengelegt")
    if bt > MEME_MAX_TAX or st_ > MEME_MAX_TAX: bad.append(f"Steuer zu hoch ({bt:.0%}/{st_:.0%})")
    good = [f"kein Honeypot, Steuer {bt:.0%} Kauf / {st_:.0%} Verkauf, Vertrag offen"]
    warn = []
    if flag("is_mintable"): warn.append("neue Token können erzeugt werden (mintbar)")
    if flag("slippage_modifiable"): warn.append("Steuer nachträglich änderbar")
    top = 0.0
    for h in s.get("holders") or []:
        if str(h.get("is_locked", 0)) in ("1", "True", "true") or str(h.get("is_contract", 0)) in ("1", "True", "true"):
            continue
        top = max(top, meme_num(h.get("percent")))
    if top > MEME_TOP_HOLDER: warn.append(f"eine Wallet hält {top:.0%}")
    hc = s.get("holder_count")
    if hc: good.append(f"{int(meme_num(hc))} Halter")
    return not bad, good, bad + warn if bad else warn


def meme_prices(addrs):
    out = {}
    for i in range(0, len(addrs), 30):
        chunk = addrs[i:i + 30]
        js = meme_get(f"{MEME_GT}/simple/networks/{MEME_NETWORK}/token_price/{','.join(chunk)}")
        tp = ((js.get("data") or {}).get("attributes") or {}).get("token_prices") or {}
        out.update({k.lower(): meme_num(v, None) for k, v in tp.items()})
    return out


def meme_st(state):
    d = state.setdefault("_meme", {})
    for k, v in (("track", []), ("sent", {})):
        d.setdefault(k, v)
    return d


def meme_fmt_usd(x):
    return f"{x / 1e6:.2f} Mio. $" if x >= 1e6 else f"{x / 1e3:.0f} Tsd. $"


def meme_msg(p, good, warn):
    w = ("\n⚠️ " + "; ".join(warn)) if warn else ""
    return (f"🎰 MEMECOIN-RADAR (Base) | Spielgeld, NICHT getestet\n{p['sym']} ({p['name']})\n\n"
            f"Kurs: {p['price']:.8g} $ | Marktwert: {meme_fmt_usd(p['mcap'])} | Liquidität: {meme_fmt_usd(p['liq'])}\n"
            f"Kurs: 1 h {p['chg_h1']:+.0f} %, 6 h {p['chg_h6']:+.0f} %, 24 h {p['chg_h24']:+.0f} % | Alter {p['age_h']:.0f} h\n"
            f"Volumen 1 h: {meme_fmt_usd(p['vol_h1'])} ({p['accel']:.1f}x so viel wie üblich)\n"
            f"Käufer/Verkäufer 1 h: {p['buyers']}/{p['sellers']}\n\n"
            f"Sicherheit (GoPlus): ✅ {'; '.join(good)}{w}\n\n"
            f"Contract: {p['addr']}\n"
            f"https://dexscreener.com/base/{p['pool']}\nhttps://www.geckoterminal.com/base/pools/{p['pool']}\n\n"
            "Wichtig: Die meisten Memecoins fallen stark oder auf null. Nur Spielgeld (z. B. max. 10 €). In Coinbase Wallet "
            "den Token NUR über die Contract-Adresse oben suchen (Fälschungen mit gleichem Namen sind häufig), Netzwerk Base, "
            "Slippage niedrig halten. GoPlus erkennt nicht jeden Betrug. Der Bot misst nach 1 h, 6 h und 24 h, "
            "wie es weiterging (/meme).")


def meme_stats_text(state):
    d = meme_st(state)
    out = ["🎰 Memecoin-Radar: Live-Test (Kursänderung nach dem Treffer, ohne Gebühren)"]
    for grp, label in (("signal", "Radar-Treffer"), ("zufall", "Vergleich: zufällige andere Pools")):
        rows = [t for t in d["track"] if t["grp"] == grp]
        out.append(f"\n{label}: {len(rows)}")
        for h in MEME_CHECKS_H:
            v = [t["r"][str(h)] for t in rows if str(h) in t["r"]]
            if v:
                v = np.array(v)
                out.append(f"  nach {h} h (n={len(v)}): Median {np.median(v):+.0f} %, Ø {v.mean():+.0f} %, "
                           f"≥ +50 %: {(v >= 50).mean():.0%}, ≤ -50 %: {(v <= -50).mean():.0%}")
    out.append("\nAussagekräftig erst nach etwa 50-100 Treffern. Nur wenn die Treffer klar besser abschneiden als der "
               "Vergleich, hat der Radar einen Vorsprung. Swap-Gebühren und Slippage (oft 1-3 %) kommen noch dazu.")
    return "\n".join(out)


def meme_summary_line(state):
    d = state.get("_meme")
    rows = [t for t in (d or {}).get("track", []) if t["grp"] == "signal" and "24" in t["r"]]
    if not rows:
        return "Memecoin-Radar: noch keine 24-h-Auswertung."
    v = np.array([t["r"]["24"] for t in rows])
    return f"Memecoin-Radar: {len(v)} Treffer ausgewertet, nach 24 h Median {np.median(v):+.0f} %"


def meme_run(state, now, dry):
    """Gibt die Anzahl Fehler zurück."""
    if str(env("MEME_RADAR", "1")) == "0":
        return 0
    d = meme_st(state)
    errors = 0
    # 1) Laufende Messungen aktualisieren
    due = [t for t in d["track"] if any(str(h) not in t["r"] and now.timestamp() - t["t0"] >= h * 3600
                                        for h in MEME_CHECKS_H)]
    if due:
        try:
            px = meme_prices(sorted({t["addr"] for t in due}))
            for t in due:
                for h in MEME_CHECKS_H:
                    if str(h) not in t["r"] and now.timestamp() - t["t0"] >= h * 3600:
                        p = px.get(t["addr"])
                        if p:
                            t["r"][str(h)] = round((p / t["p0"] - 1) * 100, 1)
                        elif now.timestamp() - t["t0"] >= h * 3600 + 6 * 3600:
                            t["r"][str(h)] = -100.0          # dauerhaft kein Kurs mehr -> als Totalverlust werten
        except Exception as e:
            errors += 1
            print(f"[Meme] Kursabfrage fehlgeschlagen: {scrub(e)}")
    sig = [t for t in d["track"] if t["grp"] == "signal"][-400:]
    rnd = [t for t in d["track"] if t["grp"] == "zufall"][-400:]
    d["track"] = sorted(sig + rnd, key=lambda t: t["t0"])
    # 2) Neue Kandidaten suchen
    try:
        pools = meme_pools(now)
    except Exception as e:
        print(f"[Meme] GeckoTerminal nicht erreichbar: {scrub(e)}")
        alert_error(state, "meme-gt", f"Memecoin-Radar: GeckoTerminal nicht erreichbar ({scrub(e)[:120]})", dry)
        return errors + 1
    cands = []
    for p in pools:
        ok, why = meme_signal(p)
        if ok and now.timestamp() - d["sent"].get(p["addr"], 0) > MEME_COOLDOWN_H * 3600:
            cands.append(p)
    cands.sort(key=lambda p: -(p["accel"] * p["buyers"] / max(p["sellers"], 1)))
    sent = 0
    for p in cands:
        if sent >= MEME_MAX_PER_RUN:
            break
        try:
            ok, good, warn = meme_security(p["addr"])
        except Exception as e:
            print(f"[Meme] GoPlus-Fehler bei {p['sym']}: {scrub(e)}")
            continue
        if not ok:
            print(f"[Meme] {p['sym']} aussortiert: {'; '.join(warn)}")
            d["sent"][p["addr"]] = now.timestamp()          # nicht jede 10 Min. erneut prüfen
            continue
        try:
            telegram(meme_msg(p, good, warn), dry)
        except Exception as e:
            print(f"[Meme] Meldung fehlgeschlagen: {scrub(e)}")
            continue
        d["sent"][p["addr"]] = now.timestamp()
        d["track"].append({"grp": "signal", "addr": p["addr"], "sym": p["sym"], "p0": p["price"],
                           "t0": now.timestamp(), "r": {}})
        sent += 1
    # 3) Vergleichsgruppe: ein zufälliger anderer Pool, der nur den Grundfilter erfüllt
    others = [p for p in pools if meme_basic_ok(p) and p["addr"] not in d["sent"]]
    last_rnd = max([t["t0"] for t in d["track"] if t["grp"] == "zufall"] or [0])
    if others and now.timestamp() - last_rnd >= 3600:        # höchstens einer pro Stunde
        p = others[int(now.timestamp()) % len(others)]
        d["track"].append({"grp": "zufall", "addr": p["addr"], "sym": p["sym"], "p0": p["price"],
                           "t0": now.timestamp(), "r": {}})
    d["sent"] = {k: v for k, v in d["sent"].items() if now.timestamp() - v < 7 * 86400}
    print(f"[Meme] {len(pools)} Pools geprüft, {len(cands)} Kandidaten, {sent} gemeldet")
    return errors


# ---------------- Hauptprogramm ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="nichts senden, nur ausgeben")
    ap.add_argument("--test", action="store_true", help="Telegram-Testnachricht senden")
    a = ap.parse_args()
    load_env()
    if a.test:
        telegram("✅ Scanner verbunden. Telegram funktioniert.", a.dry); return

    now = datetime.now(timezone.utc)
    state = load_state()
    window = now.weekday() < 5 and FOREX_HOURS_UTC[0] <= now.hour < FOREX_HOURS_UTC[1]
    candidates, errors, frames, checked = [], 0, {}, 0

    # Intraday: Gold, 1h-Kerzen, Tagestrend als Filter
    if window and env("TWELVEDATA_KEY"):
        for name, ts in INTRADAY.items():
            checked += 1
            try:
                h1 = fetch_twelve_tf(ts, "1h", 300)
                if (now - h1.time.iloc[-1].to_pydatetime()).total_seconds() / 60 > 130:
                    raise RuntimeError("Kursdaten veraltet")
                frames[name] = h1
                time.sleep(2)
                dd = fetch_twelve_tf(ts, "1day", 120, drop_last=False)
                dd = dd[dd.time.dt.date < now.date()].reset_index(drop=True)
                trend = daily_trend_value(dd)
                for s in intraday_signals(name, h1, trend, now):
                    if time.time() - state.get(s["id"], 0) > COOLDOWN_H * 3600:
                        candidates.append(s)
            except Exception as e:
                errors += 1
                print(f"[{name}] Fehler: {scrub(e)}")
                alert_error(state, name, f"{name}: {scrub(e)[:150]}", a.dry)
    print(f"{now:%H:%M} UTC: {checked} Intraday-Märkte geprüft, {errors} Fehler, {len(candidates)} neue Kandidaten")

    try:
        process_updates(state, frames, now, a.dry)
    except Exception as e:
        print(f"Befehle fehlgeschlagen: {scrub(e)}")
    try:
        update_trades(state, frames, now, a.dry)
    except Exception as e:
        print(f"Trade-Verfolgung fehlgeschlagen: {scrub(e)}")

    b = in_blackout(now)
    if b:
        print(f"Pause wegen Termin {b} UTC (keine Meldungen)")
    elif candidates:
        for s in candidates:
            state[s["id"]] = time.time()
        news = fetch_news(now) if env("NEWS_CHECK", "1") == "1" else None
        results = evaluate_intraday(candidates, now, news, state)
        allowed, infos = apply_limits(results, state, frames, now)
        for s, q, text in allowed:
            no = next_no(state)
            buttons = [{"text": "✅ Ich bin drin", "callback_data": f"in:{no}"},
                       {"text": "🚫 Nicht genommen", "callback_data": f"out:{no}"}]
            telegram(format_msg(s, q, text, no), a.dry, buttons=buttons)
            add_trade(state, s, q, no)
            send_chart(s, frames.get(s["symbol"]), no, a.dry)
        for msg in infos:
            try:
                telegram(msg, a.dry)
            except Exception as e:
                print(f"Info-Meldung fehlgeschlagen: {scrub(e)}")

    # Swing: einmal täglich ab 07:00 UTC
    if now.hour >= SWING_HOUR_UTC:
        try:
            errors += swing_run(state, now, a.dry)
            checked += len(SWING)
        except Exception as e:
            print(f"Swing-Lauf fehlgeschlagen: {scrub(e)}")

    # Demo-Signale (ORB, Momentum, Mean Reversion auf Nasdaq/S&P 500), nur Demokonto
    try:
        errors += demo_run(state, now, a.dry, telegram, alert_error, env)
    except Exception as e:
        print(f"Demo-Lauf fehlgeschlagen: {scrub(e)}")

    # Memecoin-Radar (Base), rund um die Uhr, Spielgeld
    try:
        errors += meme_run(state, now, a.dry)
    except Exception as e:
        print(f"Memecoin-Radar fehlgeschlagen: {scrub(e)}")

    heartbeat(state, now, checked, errors, a.dry)
    save_state(state)


if __name__ == "__main__":
    main()
