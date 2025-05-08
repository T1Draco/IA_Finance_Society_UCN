import os
from datetime import datetime
import pandas as pd

# Importaciones locales
from scrapers.finviz_scraper import scrape_finviz_news
from scrapers.googlenews_scraper import scrape_google_news
from scrapers.reddit_scraper import reddit, subreddit_name
from scrapers.yfinance_scraper import scrape_yahoo_news

def archivo_ya_existe(path: str) -> bool:
    """Devuelve True si el archivo existe y no está vacío."""
    return os.path.exists(path) and os.path.getsize(path) > 0

def guardar_json(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", indent=4, force_ascii=False)

def ejecutar_scrapers(ticker: str, empresa: str, fecha_inicio: str, fecha_fin: str, save_dir="diarios_json"):
    base_dir = os.path.join("1_Recoleccion_Datos", "news_scrapping_data", save_dir, ticker)

    print(f"\n[🏁] Ejecutando scrapers para {ticker} de {fecha_inicio} a {fecha_fin}")

    # === Finviz ===
    path_finviz = os.path.join(base_dir, "finviz", f"finviz_{ticker}_{fecha_inicio.replace('/', '-')}.json")
    if not archivo_ya_existe(path_finviz):
        df_finviz = scrape_finviz_news(ticker, fecha_inicio, fecha_fin, empresa)
        guardar_json(df_finviz, path_finviz)
    else:
        print(f"[↪] Ya existe: {path_finviz}, se omite Finviz.")

    # === Google News ===
    path_google = os.path.join(base_dir, "googlenews", f"googlenews_{ticker}_{fecha_inicio.replace('/', '-')}.json")
    if not archivo_ya_existe(path_google):
        df_google = scrape_google_news(ticker, f"{empresa} stock", fecha_inicio, fecha_fin)
        guardar_json(df_google, path_google)
    else:
        print(f"[↪] Ya existe: {path_google}, se omite Google News.")

    # === Reddit ===
    path_reddit = os.path.join(base_dir, "reddit", f"reddit_{ticker}_{fecha_inicio.replace('/', '-')}.json")
    if not archivo_ya_existe(path_reddit):
        from datetime import datetime as dt
        fecha_inicio_dt = dt.strptime(fecha_inicio, "%d/%m/%Y")
        fecha_fin_dt = dt.strptime(fecha_fin, "%d/%m/%Y")

        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        for post in subreddit.search(empresa, sort='new', time_filter='all', limit=500):
            post_date = dt.fromtimestamp(post.created)
            if fecha_inicio_dt <= post_date <= fecha_fin_dt:
                posts.append({
                    "ticker": ticker,
                    "title": post.title,
                    "summary": post.selftext[:250],
                    "body": post.selftext[:500],
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "date": post_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": f"Reddit - r/{subreddit_name}",
                    "link": post.url,
                    "fecha_query_inicio": fecha_inicio,
                    "fecha_query_fin": fecha_fin
                })

        df_reddit = pd.DataFrame(posts)
        guardar_json(df_reddit, path_reddit)
    else:
        print(f"[↪] Ya existe: {path_reddit}, se omite Reddit.")

    # === Yahoo Finance ===
    path_yahoo = os.path.join(base_dir, "yahoofinance", f"yahoofinance_{ticker}_{fecha_inicio.replace('/', '-')}.json")
    if not archivo_ya_existe(path_yahoo):
        scrape_yahoo_news(ticker, scrolls=5, output_path=path_yahoo)
    else:
        print(f"[↪] Ya existe: {path_yahoo}, se omite Yahoo Finance.")

    print("\n✅ Scrapers ejecutados (o saltados) correctamente.")


# === Ejemplo de uso ===
if __name__ == "__main__":
    ejecutar_scrapers(
        ticker="AAPL",
        empresa="Apple",
        fecha_inicio="17/03/2025",
        fecha_fin="17/03/2025"
    )
