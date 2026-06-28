#!/usr/bin/env python3
"""
Competitor monitoring agent.
Reads companies from CSV → fetches Google News RSS → analyses with Gemini → sends to Telegram.
"""

import csv
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import requests

# ── Config from environment variables ──────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
COMPANIES_FILE = os.environ.get("COMPANIES_FILE", "companies.csv")
MAX_ARTICLES_PER_COMPANY = int(os.environ.get("MAX_ARTICLES_PER_COMPANY", "5"))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

# ── Google News RSS ─────────────────────────────────────────────────────────────

def fetch_news(query: str, lang: str = "ru", max_items: int = 5) -> list[dict]:
    """Fetch recent articles from Google News RSS for a search query."""
    encoded = urllib.parse.quote(query)
    gl = lang.upper()
    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl={lang}&gl={gl}&ceid={gl}:{lang}"
    )
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] RSS fetch failed for '{query}': {e}")
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"  [WARN] RSS parse error for '{query}': {e}")
        return []

    articles = []
    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None else "Unknown"
        articles.append({"title": title, "link": link, "date": pub_date, "source": source})

    return articles


# ── Gemini analysis ─────────────────────────────────────────────────────────────

def analyse_with_gemini(company: str, articles: list[dict]) -> str:
    """Send article headlines to Gemini and get a structured summary."""
    if not articles:
        return "Новостей не найдено."

    headlines = "\n".join(
        f"- [{a['source']}] {a['title']} ({a['date']})" for a in articles
    )
    prompt = f"""Ты — бизнес-аналитик. Проанализируй свежие новости о компании «{company}» и дай краткий структурированный отчёт на русском языке.

Новости:
{headlines}

Формат ответа:
📊 **Краткое резюме** (2–3 предложения)
⚠️ **Риски / проблемы** (если есть)
🚀 **Возможности / позитивные события** (если есть)
🔍 **На что обратить внимание**

Будь конкретным и лаконичным."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }
    for attempt in range(3):
        try:
            if attempt > 0:
                wait = 15 * attempt
                print(f"  Retrying Gemini in {wait}s (attempt {attempt+1})...")
                time.sleep(wait)
            resp = requests.post(GEMINI_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"  [WARN] Gemini error for '{company}': {e}")
    return "Анализ недоступен (ошибка Gemini API)."


# ── Telegram sender ─────────────────────────────────────────────────────────────

def send_telegram(text: str, parse_mode: str = "Markdown") -> bool:
    """Send a message via Telegram Bot API. Splits long messages automatically."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Telegram max message length is 4096 chars
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    ok = True
    for chunk in chunks:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": parse_mode}
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [ERROR] Telegram send failed: {e}")
            ok = False
        time.sleep(0.5)  # avoid hitting rate limits
    return ok


# ── CSV reader ──────────────────────────────────────────────────────────────────

def load_companies(path: str) -> list[dict]:
    """Load companies from CSV. Required columns: name, search_query. Optional: language."""
    companies = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            query = row.get("search_query", name).strip()
            lang = row.get("language", "ru").strip()
            if name:
                companies.append({"name": name, "query": query, "lang": lang})
    return companies


# ── Main ────────────────────────────────────────────────────────────────────────

def build_report(company: dict) -> str:
    """Build a full report block for one company."""
    name = company["name"]
    print(f"  Fetching news for: {name}")
    articles = fetch_news(company["query"], company["lang"], MAX_ARTICLES_PER_COMPANY)

    print(f"  Analysing {len(articles)} articles with Gemini...")
    analysis = analyse_with_gemini(name, articles)

    # Article links block
    if articles:
        links = "\n".join(f"• [{a['title'][:60]}…]({a['link']})" for a in articles[:3])
    else:
        links = "_нет свежих новостей_"

    block = (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏢 *{name}*\n\n"
        f"{analysis}\n\n"
        f"📎 *Источники:*\n{links}\n"
    )
    return block


def main():
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    print(f"[{now}] Starting competitor monitoring...")

    companies = load_companies(COMPANIES_FILE)
    if not companies:
        print("No companies found in CSV. Exiting.")
        return

    header = f"📡 *Мониторинг конкурентов* — {now}\n"
    send_telegram(header)

    for company in companies:
        try:
            report = build_report(company)
            send_telegram(report)
        except Exception as e:
            print(f"  [ERROR] Failed to process '{company['name']}': {e}")
        time.sleep(10)  # delay between companies to avoid Gemini rate limits

    send_telegram("✅ Мониторинг завершён.")
    print("Done.")


if __name__ == "__main__":
    main()
