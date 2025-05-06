import praw
import pandas as pd
from datetime import datetime

# ==== CREDENCIALES ====
reddit = praw.Reddit(
    client_id='NoDVu9joldZigcTMYKkFCw',
    client_secret='O1Nes42qcod6ZuqMBrw3xp2Jqh2RjA',
    user_agent='script_noticias_reddit(by u/MartinPueblaRivera)'
)

# ==== PARÁMETROS MODIFICABLES ====
ticker = "AAPL"
query = "Apple stock"
subreddit_name = "WallStreetBets"
fecha_inicio = "17/03/2025"
fecha_fin = "17/04/2025"

# ==== CONVERSIÓN DE FECHAS ====
fecha_inicio_dt = datetime.strptime(fecha_inicio, "%d/%m/%Y")
fecha_fin_dt = datetime.strptime(fecha_fin, "%d/%m/%Y")

# ==== SCRAPING ====
print(f"\n[+] Buscando posts sobre: '{query}' en r/{subreddit_name}")
print(f"[+] Rango de fechas: {fecha_inicio} a {fecha_fin}")

posts = []
subreddit = reddit.subreddit(subreddit_name)

for post in subreddit.search(query, sort='new', time_filter='all', limit=500):
    post_date = datetime.fromtimestamp(post.created)

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

# ==== EXPORTAR ====
df = pd.DataFrame(posts)

if df.empty:
    print("[⚠] No se encontraron posts en ese rango.")
else:
    print(f"[✔] {len(df)} posts encontrados.")

filename = f"reddit_{ticker}_{fecha_inicio.replace('/', '-')}_to_{fecha_fin.replace('/', '-')}.json"
df.to_json(filename, orient="records", indent=4, force_ascii=False)
print(f"[💾] Archivo guardado: {filename}")
