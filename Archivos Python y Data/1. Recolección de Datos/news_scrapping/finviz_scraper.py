import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


def scrape_finviz_news(ticker: str, fecha_inicio: str, fecha_fin: str, nombre_empresa: str = None):
    print(f"\n[+] Buscando noticias en Finviz para: {ticker}")
    print(f"[+] Rango de fechas: {fecha_inicio} a {fecha_fin}")

    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    news_table = soup.find("table", class_="fullview-news-outer")
    rows = news_table.find_all("tr") if news_table else []

    fecha_inicio_dt = datetime.strptime(fecha_inicio, "%d/%m/%Y")
    fecha_fin_dt = datetime.strptime(fecha_fin, "%d/%m/%Y")

    noticias = []
    seen_links = set()
    last_valid_date = None

    for row in rows:
        try:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            raw_date = cols[0].text.strip()
            link_el = cols[1].find("a", href=True)

            if not link_el:
                continue

            title = link_el.text.strip()
            link = link_el["href"]

            if link.startswith("/"):
                link = f"https://finviz.com{link}"

            # Manejo de fecha parcial
            if ' ' in raw_date:
                date_str, _ = raw_date.split(' ')
                last_valid_date = datetime.strptime(date_str, "%b-%d-%y")
            else:
                # Solo hora: usar última fecha conocida
                if last_valid_date is None:
                    continue
                # no cambio de día
                pass  # usamos `last_valid_date` tal cual

            post_date = last_valid_date

            if not (fecha_inicio_dt <= post_date <= fecha_fin_dt):
                continue

            if link in seen_links:
                continue
            seen_links.add(link)

            noticias.append({
                "ticker": ticker,
                "title": title,
                "summary": title,
                "body": title,
                "score": None,
                "num_comments": None,
                "date": post_date.strftime("%Y-%m-%d"),
                "source": "Finviz",
                "link": link,
                "fecha_query_inicio": fecha_inicio,
                "fecha_query_fin": fecha_fin
            })

        except Exception as e:
            continue

    df = pd.DataFrame(noticias)

    filename = f"finviz_{ticker}_{fecha_inicio.replace('/', '-')}_to_{fecha_fin.replace('/', '-')}.json"
    df.to_json(filename, orient="records", indent=4, force_ascii=False)
    print(f"[✔] Noticias guardadas como: {filename}")
    return df


# === Uso ===
ticker = "AAPL"
nombre_empresa = "Apple"
fecha_inicio = "17/03/2025"
fecha_fin = "17/04/2025"
scrape_finviz_news(ticker, fecha_inicio, fecha_fin, nombre_empresa)
