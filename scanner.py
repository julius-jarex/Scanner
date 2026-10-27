#!/usr/bin/env python3
"""Daytrading-Scanner (Forex & Krypto): Kurse holen -> Setups per Regel finden -> Telegram.
Standard: komplett kostenlos (nur Regeln). Optional: Claude als Zusatzfilter (USE_CLAUDE=1 + API-Key).
Nur Hinweise, keine Orders. Aufruf: python scanner.py [--dry] [--test]"""
import os, sys, json, time, math, argparse
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

# ---------------- Einstellungen (hier anpassen) ----------------
CRYPTO = {"BTCUSD": "XBTUSD", "ETHUSD": "ETHUSD", "SOLUSD": "SOLUSD"}          # Kraken-Paare (kostenlos, ohne Key)
FOREX = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY", "XAUUSD": "XAU/USD"}  # Twelve Data (Gratis-Key)
FOREX_HOURS_UTC = (6, 20)      # Forex nur zwischen 06:00 und 20:00 UTC, Mo-Fr
MODEL = "claude-sonnet-5-5"    # nur relevant bei USE_CLAUDE=1
STATE_FILE = "state.json"
COOLDOWN_H = 4                 # gleiches Setup nicht öfter als alle 4 Stunden melden
MIN_QUALITY = "A"              # im Gratis-Modus nur Setups mit dieser Qualität (A oder B) melden
BLACKOUT_MIN = 30              # Minuten vor/nach einem Termin keine Meldung
MAX_DATA_AGE_MIN = 50          # Kursdaten älter als das gelten als veraltet -> keine Signale
MAX_CHASE_R = 0.5              # Signal ungültig, wenn Preis mehr als 0,5R vom Einstieg weg ist
MAX_SIGNALS_PER_DAY = 3        # höchstens so viele Meldungen pro 24 Stunden
CORR_HOURS = 8                 # gleiche Marktrichtung (z. B. USD-short) nicht zweimal innerhalb 8 Stunden
ERR_COOLDOWN_H = 6             # gleiche Fehlermeldung höchstens alle 6 Stunden
HEARTBEAT_HOUR_UTC = 7         # tägliches Lebenszeichen ab 07:00 UTC
TRADE_MAX_H = 30               # Meldung ohne TP1/Stop nach so vielen Stunden als "ohne Ergebnis" schließen
# ---------------------------------------------------------------


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
    for k in ("TWELVEDATA_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"):
        v = os.environ.get(k)
        if v:
            text = text.replace(v, "***")
    return text


def env(name, default=None, required=False):
    v = os.environ.get(name) or default
    if required and not v:
        sys.exit(f"Fehlende Einstellung: {name}")
    return v


# ---------------- Daten ----------------
def fetch_kraken(pair):
    r = requests.get("https://api.kraken.com/0/public/OHLC", params={"pair": pair, "interval": 15}, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(j["error"])
    rows = next(v for k, v in j["result"].items() if k != "last")
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["t", "open", "high", "low", "close"]
    df["time"] = pd.to_datetime(df["t"].astype(int), unit="s", utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df[["time", "open", "high", "low", "close"]].iloc[:-1].reset_index(drop=True)  # letzte Kerze läuft noch


def fetch_twelve(symbol):
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": "15min", "outputsize": 500, "timezone": "UTC",
        "apikey": env("TWELVEDATA_KEY", required=True)}, timeout=20)
    j = r.json()
    if j.get("status") == "error":
        raise RuntimeError(j.get("message"))
    df = pd.DataFrame(j["values"])
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.sort_values("time")[["time", "open", "high", "low", "close"]]
    return df.iloc[:-1].reset_index(drop=True)


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


def size(sym, entry, stop):
    risk = float(env("ACCOUNT_SIZE", 1000)) * float(env("RISK_PERCENT", 1)) / 100
    dist = abs(entry - stop)
    if sym in FOREX:
        if sym == "XAUUSD": per_lot = dist * 100                 # Verlust je 1 Lot bei Stop
        elif sym == "USDJPY": per_lot = (dist / 0.01) * (1000 / entry)
        else: per_lot = (dist / 0.0001) * 10
        lots = math.floor(risk / per_lot * 100) / 100          # immer abrunden
        if lots < 0.01:
            return f"unter 0.01 Lots möglich (bei 0.01 Lots Risiko {per_lot * 0.01:.2f} statt {risk:.2f}), eher auslassen"
        return f"{lots:.2f} Lots (Risiko ca. {lots * per_lot:.2f} von {risk:.2f})"
    return f"{risk / dist:.4f} {sym[:3]} (Risiko {risk:.2f}, ohne Hebel)"


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
    out = []

    def add(name, side, stop):
        entry = float(last.close)
        risk = abs(entry - stop)
        if not (0.5 * atr <= risk <= 3 * atr):
            return
        sgn = 1 if side == "long" else -1
        out.append({
            "id": f"{sym}|{name}|{side}", "symbol": sym, "setup": name, "side": side,
            "candle_time": last.time.strftime("%Y-%m-%d %H:%M UTC"), "candle_ts": last.time.isoformat(),
            "entry": round(entry, 5), "stop": round(float(stop), 5),
            "tp1": round(entry + sgn * 2 * risk, 5), "tp2": round(entry + sgn * 3 * risk, 5),
            "htf_trend_1h": trend, "atr15": round(float(atr), 5),
            "ema20": round(float(last.ema20), 5), "ema50": round(float(last.ema50), 5),
            "high_20_candles": round(float(hi20), 5), "low_20_candles": round(float(lo20), 5),
            "body_atr": round(float(body / atr), 2), "risk_dist": float(risk),
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


# ---------------- Gratis-News-Check ----------------
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"   # inoffizieller, kostenloser Wochenkalender
RSS_FEEDS = ["https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cointelegraph.com/rss"]
RISK_WORDS = ["hack", "exploit", "stolen", "sec ", "lawsuit", "ban ", "delist", "halt", "outage",
              "insolven", "bankrupt", "liquidation", "crackdown", "sanction"]
CCY = {"EURUSD": {"EUR", "USD"}, "GBPUSD": {"GBP", "USD"}, "USDJPY": {"USD", "JPY"}, "XAUUSD": {"USD"}}
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
            news["events"].append({"time": t, "ccy": e.get("country", ""), "title": e.get("title", "")})
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
    """Punktesystem 0-6 minus News-Abzug. A >= 5, B = 4. Nachvollziehbar, keine Prognose."""
    pts, why, risks = 0, [], []
    long_ = s["side"] == "long"
    aligned = (s["htf_trend_1h"] == "up") == long_ and s["htf_trend_1h"] != "flat"
    if aligned:
        pts += 2; why.append("1h-Trend passt zur Richtung")
    elif s["htf_trend_1h"] == "flat":
        pts += 1; risks.append("1h-Trend seitwärts")
    else:
        risks.append("gegen den 1h-Trend")

    opp = s["high_20_candles"] if long_ else s["low_20_candles"]
    room = (opp - s["entry"]) if long_ else (s["entry"] - opp)
    if room >= 2 * s["risk_dist"] or room <= 0:
        pts += 2; why.append("TP1 hat Platz bis zum nächsten Level" if room > 0 else "Preis über/unter dem 20-Kerzen-Level, freier Raum")
    elif room >= 1 * s["risk_dist"]:
        pts += 1; risks.append("nächstes Level liegt vor TP1")
    else:
        risks.append("Level direkt vor dem Einstieg")

    if s["body_atr"] >= 0.6:
        pts += 1; why.append("kräftige Signalkerze")
    if s["setup"] != "Liquidity-Sweep" and aligned:
        pts += 1; why.append(f"{s['setup']} mit Trend")
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
(high_20_candles / low_20_candles)? Ist der Stop strukturell sinnvoll? Passt der 1h-Trend? Suche per Websuche High-Impact-Termine in den nächsten
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


# ---------------- Telegram / Status ----------------
def telegram(text, dry=False):
    if dry:
        print("---- TELEGRAM (dry) ----\n" + text + "\n"); return
    r = requests.post(f"https://api.telegram.org/bot{env('TELEGRAM_TOKEN', required=True)}/sendMessage",
                      json={"chat_id": env("TELEGRAM_CHAT_ID", required=True), "text": text}, timeout=20)
    r.raise_for_status()


def load_state():
    try: return json.load(open(STATE_FILE))
    except Exception: return {}


def save_state(s):
    now = time.time()
    json.dump({k: v for k, v in s.items() if not isinstance(v, (int, float)) or now - v < 86400},
              open(STATE_FILE, "w"))


def format_msg(s, quality, text):
    arrow = "🟢 LONG" if s["side"] == "long" else "🔴 SHORT"
    r = s["risk_dist"]
    if s["side"] == "long":
        expiry = f"Ungültig, wenn Preis über {round(s['entry'] + MAX_CHASE_R * r, 5)} steigt oder Stop {s['stop']} berührt."
    else:
        expiry = f"Ungültig, wenn Preis unter {round(s['entry'] - MAX_CHASE_R * r, 5)} fällt oder Stop {s['stop']} berührt."
    return (f"{arrow} {s['symbol']} | {s['setup']} | Qualität {quality}\n"
            f"Kerze: {s['candle_time']} | 1h-Trend: {s['htf_trend_1h']}\n\n"
            f"Einstieg: {s['entry']}\nStop: {s['stop']}\nTP1 (2R): {s['tp1']}\nTP2 (3R): {s['tp2']}\n"
            f"Größe: {s['size']}\n{expiry}\n\n{text}\n\n"
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
    out.sort(key=lambda x: -order.get(x[1], 0))
    return out


# ---------------- Sicherheitsregeln ----------------
def check_fresh(name, df, now):
    """Veraltete Kursdaten (API-Fehler, Marktpause) -> Fehler statt falscher Signale."""
    age = (now - df.time.iloc[-1].to_pydatetime()).total_seconds() / 60
    if age > MAX_DATA_AGE_MIN:
        raise RuntimeError(f"Kursdaten veraltet (letzte Kerze vor {int(age)} Min.)")
    return df


def exposure(s):
    """Gleiche Marktrichtung zusammenfassen: EURUSD/GBPUSD/Gold long = USD short usw."""
    sym, side = s["symbol"], s["side"]
    if sym in CRYPTO:
        return f"KRYPTO-{side}"
    usd_long = (side == "long") == (sym == "USDJPY")
    return "USD-long" if usd_long else "USD-short"


def sent_recent(state, hours):
    now = time.time()
    out = []
    for k, v in state.items():
        if k.startswith("sent|") and now - v < hours * 3600:
            out.append(k.split("|", 2)[2])
    return out


def apply_limits(results, state):
    """Tageslimit und Korrelationsfilter. Gibt erlaubte Meldungen zurück und trägt sie im State ein."""
    allowed = []
    for s, q, text in results:
        ex = exposure(s)
        if len(sent_recent(state, 24)) >= MAX_SIGNALS_PER_DAY:
            print(f"{s['id']} unterdrückt: Tageslimit erreicht"); continue
        if ex in sent_recent(state, CORR_HOURS):
            print(f"{s['id']} unterdrückt: korrelierte Position ({ex}) kürzlich gemeldet"); continue
        state[f"sent|{time.time():.3f}|{ex}"] = time.time()
        allowed.append((s, q, text))
    return allowed


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
            lines = setup_lines(state)
            telegram(f"💚 Scanner läuft. Letzte 24h: {n} Meldung(en). Aktuell {checked} Märkte geprüft, {errors} Fehler, "
                     f"{len(state.get('_trades', []))} offene Meldung(en).\n{stats_line(state)}"
                     + (f"\n{lines}" if lines else "") +
                     "\nBleibt diese Nachricht aus: GitHub → Actions prüfen (nach ~60 Tagen ohne Repo-Aktivität pausiert GitHub geplante Läufe).", dry)
            state["hb"] = time.time()
        except Exception as e:
            print(f"Heartbeat fehlgeschlagen: {scrub(e)}")


# ---------------- Trade-Verfolgung ----------------
def add_trade(state, s):
    state.setdefault("_trades", []).append({
        "id": s["id"], "symbol": s["symbol"], "side": s["side"], "setup": s["setup"],
        "entry": s["entry"], "stop": s["stop"], "tp1": s["tp1"],
        "t0": s["candle_ts"], "opened": time.time()})


def _stats(state):
    return state.setdefault("_stats", {"tp1": 0, "stop": 0, "expired": 0, "r": 0.0, "by": {}})


def _book(state, tr, outcome, r):
    st = _stats(state)
    st[outcome] += 1
    st["r"] = round(st["r"] + r, 2)
    b = st.setdefault("by", {}).setdefault(tr["setup"], {"tp1": 0, "stop": 0})
    if outcome in b:
        b[outcome] += 1


def check_trade(tr, df):
    """Prüft Kerzen NACH der Signalkerze. -> ('tp1'|'stop'|None, Hinweis)"""
    t0 = pd.Timestamp(tr["t0"])
    for c in df[df.time > t0].itertuples():
        if tr["side"] == "long":
            hit_stop, hit_tp = c.low <= tr["stop"], c.high >= tr["tp1"]
        else:
            hit_stop, hit_tp = c.high >= tr["stop"], c.low <= tr["tp1"]
        if hit_stop and hit_tp:
            return "stop", "Stop und Ziel in derselben Kerze, vorsichtig als Stop gewertet"
        if hit_stop:
            return "stop", ""
        if hit_tp:
            return "tp1", ""
    return None, ""


def stats_line(state):
    st = state.get("_stats")
    if not st or not (st["tp1"] + st["stop"] + st["expired"]):
        return "Bilanz: noch keine abgeschlossene Meldung."
    n = st["tp1"] + st["stop"]
    quote = f" ({st['tp1'] / n:.0%} Treffer)" if n else ""
    return (f"Bilanz: {st['tp1']}x TP1, {st['stop']}x Stop, {st['expired']}x ohne Ergebnis{quote}, "
            f"Summe {st['r']:+.1f}R")


def setup_lines(state):
    out = []
    for name, b in (state.get("_stats", {}).get("by", {})).items():
        n = b["tp1"] + b["stop"]
        if n:
            out.append(f"  {name}: {b['tp1']} von {n} bis TP1")
    return "\n".join(out)


def update_trades(state, frames, now, dry):
    keep = []
    for tr in state.get("_trades", []):
        df = frames.get(tr["symbol"])
        res, note = check_trade(tr, df) if df is not None else (None, "")
        age_h = (time.time() - tr["opened"]) / 3600
        head = f"{tr['symbol']} {tr['side'].upper()} ({tr['setup']})"
        if res == "tp1":
            _book(state, tr, "tp1", 2.0)
            msg = f"✅ {head}: TP1 erreicht (+2R)\nEinstieg {tr['entry']} → Ziel {tr['tp1']}"
        elif res == "stop":
            _book(state, tr, "stop", -1.0)
            msg = f"❌ {head}: Stop getroffen (-1R)\nEinstieg {tr['entry']} → Stop {tr['stop']}"
            if note:
                msg += f"\n({note})"
        elif age_h > TRADE_MAX_H:
            _book(state, tr, "expired", 0.0)
            msg = f"⏱ {head}: nach {int(age_h)} h weder TP1 noch Stop, ohne Ergebnis geschlossen"
        else:
            keep.append(tr)
            continue
        try:
            telegram(msg + "\n" + stats_line(state), dry)
        except Exception as e:
            print(f"Ergebnis-Meldung fehlgeschlagen: {scrub(e)}")
    state["_trades"] = keep


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
    forex_open = now.weekday() < 5 and FOREX_HOURS_UTC[0] <= now.hour < FOREX_HOURS_UTC[1]
    jobs = [(n, lambda p=p: fetch_kraken(p)) for n, p in CRYPTO.items()]
    if forex_open and env("TWELVEDATA_KEY"):
        jobs += [(n, lambda s=s: fetch_twelve(s)) for n, s in FOREX.items()]

    candidates, errors, frames = [], 0, {}
    for name, fetch in jobs:
        try:
            df = check_fresh(name, fetch(), now)
            frames[name] = df
            for s in find_setups(name, df):
                if time.time() - state.get(s["id"], 0) > COOLDOWN_H * 3600:
                    candidates.append(s)
        except Exception as e:
            errors += 1
            print(f"[{name}] Fehler: {scrub(e)}")
            alert_error(state, name, f"{name}: {scrub(e)[:150]}", a.dry)
    print(f"{now:%H:%M} UTC: {len(jobs)} Märkte geprüft, {errors} Fehler, {len(candidates)} neue Kandidaten")

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
        try:
            results = evaluate(candidates, now, news)
        except Exception as e:
            print(f"Bewertung fehlgeschlagen: {scrub(e)}")
            alert_error(state, "claude", f"Claude-Bewertung fehlgeschlagen: {scrub(e)[:150]}", a.dry)
            results = []
        for s, q, text in apply_limits(results, state):
            telegram(format_msg(s, q, text), a.dry)
            add_trade(state, s)

    heartbeat(state, now, len(jobs), errors, a.dry)
    save_state(state)


if __name__ == "__main__":
    main()
