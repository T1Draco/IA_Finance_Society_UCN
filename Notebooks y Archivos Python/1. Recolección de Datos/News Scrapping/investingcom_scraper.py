import time
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def scrape_investing_news(company_url_suffix: str, fecha_inicio: str, fecha_fin: str, max_pages: int = 5):
    url_base = f"https://www.investing.com/equities/{company_url_suffix}-news"
    print(f"\n[+] Scrapeando: {url_base}")
    print(f"[+] Rango: {fecha_inicio} a {fecha_fin}")

    fecha_inicio_dt = datetime.strptime(fecha_inicio, "%d/%m/%Y")
    fecha_fin_dt = datetime.strptime(fecha_fin, "%d/%m/%Y")

    options = Options()
    options.headless = False
    driver = webdriver.Firefox(options=options)

    noticias = []
    seen_links = set()

    for page in range(1, max_pages + 1):
        url = url_base if page == 1 else f"{url_base}/{page}"
        driver.get(url)
        time.sleep(5)

        # Cerrar popup si aparece
        if page == 1:
            try:
                close_btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, '[class*="popup"] button[class*="close"]'))
                )
                close_btn.click()
                print("[✔] Popup cerrado.")
            except:
                print("[ℹ️] No se encontró popup.")

        print(f"[→] Página {page}")
        soup = BeautifulSoup(driver.page_source, "html.parser")
        cards = soup.select("div.mediumTitle1 > article")

        for card in cards:
            try:
                title_tag = card.select_one(".title")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = "https://www.investing.com" + title_tag.get("href", "")

                summary_tag = card.select_one("p")
                summary = summary_tag.text.strip() if summary_tag else ""

                date_tag = card.select_one(".date")
                date_text = date_tag.text.strip() if date_tag else ""

                try:
                    post_date = datetime.strptime(date_text, "%b %d, %Y")
                except:
                    continue

                if not (fecha_inicio_dt <= post_date <= fecha_fin_dt):
                    continue

                if link in seen_links:
                    continue
                seen_links.add(link)

                noticias.append({
                    "title": title,
                    "summary": summary,
                    "date": post_date.strftime("%Y-%m-%d"),
                    "link": link,
                    "source": "Investing",
                    "fecha_query_inicio": fecha_inicio,
                    "fecha_query_fin": fecha_fin
                })

            except Exception as e:
                print(f"⚠️ Error en noticia: {e}")
                continue

    driver.quit()
    df = pd.DataFrame(noticias)
    filename = f"investing_{company_url_suffix}_news_{fecha_inicio.replace('/', '-')}_to_{fecha_fin.replace('/', '-')}.json"
    df.to_json(filename, orient="records", indent=4, force_ascii=False)
    print(f"[💾] Noticias guardadas como: {filename}")
    return df


# === Uso ===
scrape_investing_news("apple-computer-inc", "01/05/2025", "06/05/2025")
