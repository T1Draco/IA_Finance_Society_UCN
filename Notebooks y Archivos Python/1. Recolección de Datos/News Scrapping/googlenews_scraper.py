import time
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
]

def scrape_google_news(ticker: str, query: str, date_start: str, date_end: str, max_pages: int = 5):
    print(f"\n[+] Buscando noticias sobre: {query}")
    print(f"[+] Rango de fechas: {date_start} - {date_end}")

    fecha_inicio = datetime.strptime(date_start, "%d/%m/%Y").strftime("%m/%d/%Y")
    fecha_fin = datetime.strptime(date_end, "%d/%m/%Y").strftime("%m/%d/%Y")
    print(f"[+] Rango de fechas (formato Google): {fecha_inicio} - {fecha_fin}")

    user_agent = random.choice(USER_AGENTS)
    options = Options()
    options.headless = True
    options.set_preference("general.useragent.override", user_agent)
    options.set_preference("intl.accept_languages", "en-US")

    driver = webdriver.Firefox(options=options)
    all_news = []

    try:
        for page in range(max_pages):
            start_param = page * 10
            query_encoded = query.replace(" ", "+")
            date_filter = f"&tbs=cdr:1,cd_min:{fecha_inicio},cd_max:{fecha_fin}"
            url = f"https://www.google.com/search?q={query_encoded}&tbm=nws&hl=en&gl=US{date_filter}&start={start_param}"

            print(f"[→] Página {page + 1}: {url}")
            driver.get(url)
            time.sleep(3)

            cards = driver.find_elements(By.CSS_SELECTOR, "div.SoaBEf")

            if not cards:
                print("[ℹ️] No se encontraron noticias en esta página.")
                break

            for card in cards:
                try:
                    title_el = card.find_element(By.CSS_SELECTOR, "div.n0jPhd")
                    summary_el = card.find_element(By.CSS_SELECTOR, "div.GI74Re")
                    link_el = card.find_element(By.TAG_NAME, "a")

                    try:
                        date_el = card.find_element(By.CSS_SELECTOR, "span.WG9SHc span")
                        date_text = date_el.text.strip()
                    except Exception:
                        date_text = "Fecha no encontrada"

                    title = title_el.text.strip()
                    summary = summary_el.text.strip()
                    link = link_el.get_attribute("href")

                    if title and link:
                        all_news.append({
                            "ticker": ticker,
                            "title": title,
                            "summary": summary,
                            "body": summary,
                            "score": None,
                            "num_comments": None,
                            "date": date_text,
                            "source": "Google News",
                            "link": link,
                            "fecha_query_inicio": date_start,
                            "fecha_query_fin": date_end
                        })

                except Exception:
                    continue

    finally:
        driver.quit()

    return pd.DataFrame(all_news)

# === Parámetros ===
ticker = "AAPL"
query = "Apple stock"
fecha_inicio = "17/03/2025"
fecha_fin = "17/04/2025"

# === Ejecutar y guardar ===
df = scrape_google_news(ticker, query, fecha_inicio, fecha_fin, max_pages=5)

if df.empty:
    print("\n[⚠] No se encontraron noticias.")
else:
    print(df.head())

filename = f"googlenews_{ticker}_{fecha_inicio.replace('/', '-')}_to_{fecha_fin.replace('/', '-')}.json"
df.to_json(filename, orient="records", indent=4, force_ascii=False)
print(f"\n[✔] Noticias guardadas como: {filename}")
