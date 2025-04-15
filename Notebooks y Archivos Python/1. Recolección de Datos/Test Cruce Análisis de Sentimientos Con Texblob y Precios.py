from newsapi import NewsApiClient
import pandas as pd
from textblob import TextBlob
import yfinance as yf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Inicializar cliente de NewsAPI
newsapi = NewsApiClient(api_key='ed547c9f043048248b5b851a6451a2a5')

# 2. Obtener múltiples páginas de noticias (hasta 500 noticias)
from datetime import timedelta, date

noticias = []
fecha_final = date(2025, 4, 14)
dias = 30  # puedes aumentar

for i in range(dias):
    dia = fecha_final - timedelta(days=i)
    res = newsapi.get_everything(
        q="Apple stock",
        from_param=str(dia),
        to=str(dia),
        language="en",
        sort_by="publishedAt",
        page_size=100
    )
    noticias.extend(res["articles"])


# 3. Crear DataFrame
df_news = pd.DataFrame(noticias)[["publishedAt", "title", "description"]]
df_news["publishedAt"] = pd.to_datetime(df_news["publishedAt"]).dt.date

# 4. Análisis de sentimiento
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df_news["sentiment_title"] = df_news["title"].apply(get_sentiment)
df_news["sentiment_desc"] = df_news["description"].apply(get_sentiment)

# 5. Agrupar sentimiento por día
df_sent = df_news.groupby("publishedAt")[["sentiment_title", "sentiment_desc"]].mean().reset_index()
df_sent.rename(columns={"publishedAt": "Date"}, inplace=True)

# 6. Obtener rango de fechas para precios
start_date = df_sent["Date"].min()
end_date = df_sent["Date"].max()
print("Descargando precios desde:", start_date, "hasta:", end_date)

# 7. Descargar precios de AAPL con yfinance
df_price = yf.download("AAPL", start=str(start_date), end=str(end_date + pd.Timedelta(days=1)))

# 🔧 Aplanar columnas si vienen en MultiIndex
if isinstance(df_price.columns, pd.MultiIndex):
    df_price.columns = df_price.columns.get_level_values(0)

df_price = df_price[["Close"]].copy()
df_price = df_price.rename_axis("Date").reset_index()
df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.date

# 8. Unir sentimiento y precios por columna 'Date'
df_final = pd.merge(df_price, df_sent, on="Date", how="inner")

# 9. Visualización
df_final.set_index("Date", inplace=True)
df_final.plot(subplots=True, figsize=(12, 8), marker='o', title=[
    "Precio Cierre", "Sentimiento Título", "Sentimiento Descripción"])
plt.tight_layout()
plt.show()

# 10. Correlaciones
print("\nCorrelaciones:")
print(df_final.corr())

# 11. Exportar a Excel
#df_final.to_excel("sentimiento_vs_precio_AAPL.xlsx")
#print("\n✅ Archivo exportado como 'sentimiento_vs_precio_AAPL.xlsx'")
