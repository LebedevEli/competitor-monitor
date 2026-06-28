#!/usr/bin/env python3
"""
Competitor monitoring agent.
Reads companies from CSV → scrapes their websites → analyses with Groq → sends to Telegram.
"""

import csv
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# ── Config ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
COMPANIES_FILE = os.environ.get("COMPANIES_FILE", "companies.csv")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CompetitorBot/1.0)"}


# ── Website scraper ─────────────────────────────────────────────────────────────

def scrape_website(url: str, max_chars: int = 3000) -> str:
    """Fetch a website and extract readable text content."""
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "form", "iframe", "noscript", "meta", "link"]):
        tag.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up blank lines
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
    text = "\n".join(lines)

    return text[:max_chars]


# ── Google News RSS (fallback) ──────────────────────────────────────────────────

def fetch_news(query: str, lang: str = "ru", max_items: int = 3) -> list[dict]:
    """Fetch recent articles from Google News RSS."""
    encoded = urllib.parse.quote(query)
    gl = lang.upper()
    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl={lang}&gl={gl}&ceid={gl}:{lang}"
    )
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"  [WARN] News fetch failed: {e}")
        return []

    articles = []
    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None else "?"
        articles.append({"title": title, "link": link, "date": pub_date, "source": source})
    return articles


# ── Groq analysis ───────────────────────────────────────────────────────────────

def analyse_with_groq(company: str, website_text: str, news: list[dict]) -> str:
    """Analyse company website content and news with Groq."""
    parts = []

    if website_text:
        parts.append(f"=== Содержимое сайта {company} ===\n{website_text}")

    if news:
        news_text = "\n".join(f"- {a['title']} ({a['date']})" for a in news)
        parts.append(f"=== Новости в СМИ ===\n{news_text}")

    if not parts:
        return "Данные недоступны."

    content = "\n\n".join(parts)

    prompt = f"""Ты — бизнес-аналитик, анализирующий конкурента «{company}».

Вот актуальная информация с их сайта и из СМИ:

{content}

Дай краткий структурированный отчёт на русском языке:

📊 Чем сейчас занимается компания (2–3 предложения)
🚀 Новые продукты / проекты / направления (если есть)
👥 Найм и рост (если видны вакансии или упоминания роста)
⚠️ Проблемы или риски (если есть)
🔍 На что обратить внимание конкуренту

Будь конкретным. Опирайся только на предоставленные данные."""

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 600,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            if attempt > 0:
                wait = 15 * attempt
                print(f"  Retrying Groq in {wait}s...")
                time.sleep(wait)
            resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [WARN] Groq error for '{company}': {e}")

    return "Анализ недоступен (ошибка Groq API)."


# ── Telegram ────────────────────────────────────────────────────────────────────

def send_telegram(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = [text[i: i + 4000] for i in range(0, len(text), 4000)]
    ok = True
    for chunk in chunks:
        try:
            resp = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "Markdown"},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"  [ERROR] Telegram: {e}")
            ok = False
        time.sleep(0.5)
    return ok


# ── CSV loader ──────────────────────────────────────────────────────────────────

def load_companies(path: str) -> list[dict]:
    companies = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            if not name:
                continue
            companies.append({
                "name": name,
                "query": row.get("search_query", name).strip(),
                "lang": row.get("language", "ru").strip(),
                "website": row.get("website", "").strip(),
            })
    return companies


# ── Main ────────────────────────────────────────────────────────────────────────

def build_report(company: dict) -> str:
    name = company["name"]
    website = company.get("website", "")

    # Scrape website
    website_text = ""
    if website:
        print(f"  Scraping website: {website}")
        website_text = scrape_website(website)

    # Fetch news
    print(f"  Fetching news for: {name}")
    news = fetch_news(company["query"], company["lang"])

    # Analyse
    print(f"  Analysing with Groq...")
    analysis = analyse_with_groq(name, website_text, news)

    # News links
    if news:
        links = "\n".join(f"• [{a['title'][:55]}…]({a['link']})" for a in news[:3])
        links_block = f"\n📎 *Новости в СМИ:*\n{links}"
    else:
        links_block = ""

    website_link = f"🌐 [{website}]({website})" if website else ""

    block = (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏢 *{name}*  {website_link}\n\n"
        f"{analysis}"
        f"{links_block}\n"
    )
    return block


def main():
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    print(f"[{now}] Starting competitor monitoring...")

    companies = load_companies(COMPANIES_FILE)
    if not companies:
        print("No companies found. Exiting.")
        return

    send_telegram(f"📡 *Мониторинг конкурентов* — {now}\n")

    for company in companies:
        try:
            report = build_report(company)
            send_telegram(report)
        except Exception as e:
            print(f"  [ERROR] {company['name']}: {e}")
        time.sleep(5)

    send_telegram("✅ Мониторинг завершён.")
    print("Done.")


if __name__ == "__main__":
    main()
