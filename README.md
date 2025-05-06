# IA & Finance Society UCN – Análisis de Sentimiento y Trading Algorítmico

Este repositorio forma parte del proyecto **IA & Finance Society UCN**, una iniciativa para aplicar técnicas de machine learning, procesamiento de lenguaje natural y series de tiempo en el análisis del mercado financiero, con foco inicial en acciones como AAPL (Apple Inc.).

---

## Objetivo del Proyecto

Desarrollar una **plataforma automatizada** que:

1. **Recolecte datos financieros e informativos** de diversas fuentes.  
2. **Analice el sentimiento** asociado a las noticias y redes sociales.  
3. **Integre precios históricos de acciones**.  
4. **Implemente estrategias de trading algorítmico**, combinando análisis técnico (modelos) y de sentimiento.

---

## Fases y pasos del proyecto (Metodología KDD adaptada)

### Fase 1: Recolección de Datos
- ✅ Scraping de noticias (Finviz, Google News, Reddit, Yahoo Finance).
- 🔄 Descarga de precios históricos con `yfinance`.

### Fase 2: Preprocesamiento y Limpieza
- 🔄 Procesamiento de textos con librerías como TextBlob o spaCy.
- 🔄 Limpieza y alineamiento temporal de precios y sentimiento.

### Fase 3: Análisis de Sentimiento
- 🔄 Cálculo de polaridad y subjetividad.
- 🔄 Integración con los precios de acciones.

### Fase 4: Modelado y Estrategia
- 🔄 Entrenamiento de modelos de Series de Tiempo con data de precios históricos.
- 🔄 Diseño de estrategias de trading (ej. SMA + Sentimiento).
- 🔄 Evaluación de backtesting.

### Fase 5: Evaluación y Plataforma
- 🔄 Evaluación de métricas de rendimiento (ROI, Sharpe, Drawdown).
- ⏳ Desarrollo de una plataforma de visualización.

---

## Estructura de Carpetas Propuesta (Guía)
IA_Finance_Society_UCN/
│
├── .venv/                                  # Entorno virtual de Python
├── app/
│   ├── Archivos Python y Data/
│   │   ├── 1. Recolección de Datos/
│   │   │   ├── news_scrapping/
│   │   │   └── stock_data/
│   │   ├── 2. Preprocesamiento/
│   │   │   ├── clean_sentiment_data.py     # Limpieza de textos, fechas, duplicados, outliers
│   │   │   ├── normalize_prices.py         # Normalización y alineamiento temporal
│   │   │   └── utils/                      # Funciones auxiliares
│   │   ├── 3. Análisis de Sentimiento/
│   │   │   ├── sentiment_textblob.py
│   │   │   ├── sentiment_vader.py
│   │   │   ├── sentiment_comparador.py     # Comparación entre métodos
│   │   ├── 4. Modelado/
│   │   │   ├── model_features.py           # Feature engineering (SMA, media sentimiento, etc.)
│   │   │   ├── model_training.py           # Entrenamiento y validación
│   │   │   ├── backtesting.py              # Evaluación con estrategias de trading
│   │   │   └── saved_models/
│   │   ├── 5. Evaluación y Plataforma/
│   │   │   ├── metrics_report.py           # ROI, Sharpe Ratio, Drawdown
│   │   │   ├── visualizations.py           # Gráficos y dashboards
│   │   │   └── streamlit_app.py            # Plataforma web de visualización y predicción
│   │
│   └── README.md
│
├── Figures/                                # Figuras para informes o publicaciones
├── strategies/                             # Implementación modular de estrategias
│   ├── sma_sentiment_strategy.py
│   └── risk_control.py
├── requirements.txt                        # Librerías necesarias
└── data/                                   # Carpeta centralizada de datos procesados
    ├── sentiment_data/
    ├── price_data/
    ├── merged_data/
    └── reports/

Proyecto desarrollado por estudiantes de la Universidad Católica del Norte bajo el grupo IA & Finance Society UCN.

Martín P. y Fernando V.
