from newsapi import NewsApiClient
import pandas as pd
from tabulate import tabulate

# Reemplaza con tu clave personal
API_KEY = "ed547c9f043048248b5b851a6451a2a5"

# Inicializar cliente
newsapi = NewsApiClient(api_key=API_KEY)

# Buscar noticias relacionadas a "apple" o lo que quieras
res = newsapi.get_everything(q="apple stock",
                             language="en",
                             sort_by="relevancy",
                             page_size=20)

# Extraer los datos
articles = res["articles"]
df = pd.DataFrame(articles)[["source", "author", "title", "description", "publishedAt", "url"]]

print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
