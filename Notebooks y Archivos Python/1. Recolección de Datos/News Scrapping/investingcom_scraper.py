import time
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options

def scrape_investing_news(company_url_suffix: str, fecha_inicio: str, fecha_fin: str, max_pages: int = 5):
    base_url = f"https://www.investing.com/equities/{company_url_suffix}-news"
    print(f"\n[+] Iniciando scraping para: {company_url_suffix}")
    print(f"[+] Rango de fechas: {fecha_inicio} a {fecha_fin}")

    fecha_inicio_dt = datetime.strptime(fecha_inicio, "%d/%m/%Y")
    fecha_fin_dt = datetime.strptime(fecha_fin, "%d/%m/%Y")

    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)
    driver.get(base_url)
    time.sleep(5)

    noticias = []
    seen_links = set()

    for page in range(1, max_pages + 1):
        print(f"[→] Página {page}")
        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.select("div.largeTitle article")

        for article in articles:
            try:
                link_tag = article.find("a", href=True)
                title = link_tag.text.strip()
                link = "https://www.investing.com" + link_tag["href"]
                date_tag = article.select_one("span.date")
                summary_tag = article.select_one("p")

                if not date_tag or not title:
                    continue

                raw_date = date_tag.text.strip()
                post_date = datetime.strptime(raw_date, "%b %d, %Y")

                if not (fecha_inicio_dt <= post_date <= fecha_fin_dt):
                    continue

                if link in seen_links:
                    continue
                seen_links.add(link)

                noticias.append({
                    "title": title,
                    "summary": summary_tag.text.strip() if summary_tag else "",
                    "date": post_date.strftime("%Y-%m-%d"),
                    "link": link,
                    "source": "Investing",
                    "fecha_query_inicio": fecha_inicio,
                    "fecha_query_fin": fecha_fin
                })
            except Exception:
                continue

        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, 'a.pagination-next')
            if 'disabled' in next_btn.get_attribute('class'):
                break
            next_btn.click()
            time.sleep(3)
        except:
            break

    driver.quit()
    df = pd.DataFrame(noticias)
    filename = f"investing_{company_url_suffix}_{fecha_inicio.replace('/', '-')}_to_{fecha_fin.replace('/', '-')}.json"
    df.to_json(filename, orient="records", indent=4, force_ascii=False)
    print(f"[✔] Noticias guardadas como: {filename}")
    return df

# === Uso ejemplo con Apple ===
scrape_investing_news("apple-computer-inc", "17/03/2025", "17/04/2025")
