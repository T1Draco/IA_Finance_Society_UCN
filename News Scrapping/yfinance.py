from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup

# Opcional: modo sin interfaz
# options = Options()
# options.add_argument("--headless")
# driver = webdriver.Firefox(options=options)



#Obtencion de los enlaces

driver = webdriver.Firefox()

stock = input("Ingresa el stock que quieres scrapear Σ(っ °Д °;)っ: ")

try:
    url = f"https://finance.yahoo.com/quote/{stock}/news/"
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "mainContent"))
    )

    soup = BeautifulSoup(driver.page_source, "html.parser")
    main_content = soup.find("section", class_="mainContent")
    ul_element = main_content.find("ul", class_="stream-items")

    li_items = ul_element.find_all("li")

    noticias = {}

    for li in li_items:
        title_tag = li.find("h3")
        link_tag = li.find("a", href=True)

        # Validar que hay título y link
        if title_tag and link_tag:
            title = title_tag.text.strip()
            if title:  # Solo si el título no está vacío
                url = link_tag["href"]
                if url.startswith("/"):
                    url = "https://finance.yahoo.com" + url
                noticias[title] = url

    # Mostrar resultados
    for i, (title, link) in enumerate(noticias.items(), start=1):
        print(f"{i}. {title}\n   🔗 {link}")

except Exception as e:
    print("❌ Ocurrió un error:", e)

finally:
    driver.quit()


#Scrapping a los articulos
import requests
from bs4 import BeautifulSoup
import json

resultados = []

print("🔍 Iniciando scraping...\n")

for titulo, url in noticias.items():
    print(f"➡️ Procesando: {titulo}")
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
            print("⚠️ No se encontró <article class='gridLayout'>")
            contenido = ""

        if contenido:
            resultados.append({
                "titulo": titulo,
                "fecha": date.get_text(strip=True),
                "url": url,
                "contenido": contenido
            })
            print("✅ Artículo guardado.\n")
        else:
            print("❌ Artículo sin contenido, no se guardó.\n")

    except Exception as e:
        print(f"❌ Error al procesar el artículo: {e}\n")

# 🔒 Ruta personalizada para guardar el JSON
ruta_personalizada = f"C:/Users/ferna/OneDrive/Escritorio/Ferna/Programación/Molecule Scrapping/noticias_{stock}.json"

# Guardar JSON
with open(ruta_personalizada, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=4)

print("🎉 Proceso finalizado.")
print(f"📁 Archivo guardado en:\n{ruta_personalizada}")