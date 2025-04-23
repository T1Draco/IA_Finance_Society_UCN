from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import time
import requests
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def scrape_yahoo_news(ticker: str, scrolls: int = 5, output_path: str = None):
    fecha_inicio = "17/03/2025"
    fecha_fin = "17/04/2025"

    # Configurar navegador headless
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)

    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news/"
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "mainContent"))
        )

        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(scrolls):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = BeautifulSoup(driver.page_source, "html.parser")
        main_content = soup.find("section", class_="mainContent")
        ul_element = main_content.find("ul", class_="stream-items")
        li_items = ul_element.find_all("li")

        noticias = {}
        for li in li_items:
            title_tag = li.find("h3")
            link_tag = li.find("a", href=True)
            if title_tag and link_tag:
                title = title_tag.text.strip()
                url = link_tag["href"]
                if url.startswith("/"):
                    url = "https://finance.yahoo.com" + url
                noticias[title] = url

        resultados = []
        for title, url in noticias.items():
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                article = soup.find("article", class_="gridLayout")
                date = soup.find("time", class_="byline-attr-meta-time")

                if article:
                    paragraphs = article.find_all("p")
                    contenido = "\n".join(p.get_text(strip=True) for p in paragraphs)
                else:
                    contenido = ""

                if contenido:
                    resultados.append({
                        "ticker": ticker,
                        "title": title,
                        "summary": contenido[:250],
                        "body": contenido,
                        "score": None,
                        "num_comments": None,
                        "date": date.get_text(strip=True) if date else "N/A",
                        "source": "Yahoo Finance",
                        "link": url,
                        "fecha_query_inicio": fecha_inicio,
                        "fecha_query_fin": fecha_fin
                    })
            except Exception as e:
                print(f"❌ Error en {title}: {e}")

        df = pd.DataFrame(resultados)

        if not output_path:
            output_path = Path(f"yahoofinance_{ticker}_{fecha_inicio.replace('/', '-')}_to_{fecha_fin.replace('/', '-')}.json").resolve()

        df.to_json(output_path, orient="records", indent=4, force_ascii=False)
        print(f"✅ Se guardaron {len(resultados)} noticias en {output_path}")
        return resultados

    finally:
        driver.quit()

# ➤ Ejemplo de uso
if __name__ == "__main__":
    scrape_yahoo_news("AAPL", scrolls=5)