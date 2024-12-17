# Archivo: app/pages/dashboard.py
import streamlit as st
import pandas as pd
from backend.visualization.plots import plot_timeseries
from backend.data_handling.loader import load_yfinance_data
from backend.data_handling.preprocesing import remove_multindex
from backend.data_handling.loader import load_file_from_streamlit

def render_dashboard():
    st.title("AlgoTrade-Lab Dashboard")
    st.header("Cargar Datos Financieros")
    
    # Dividir en columnas
    col1, col2 = st.columns(2)
    
    # Primera columna: Desde Yahoo! Finance
    with col1:
        st.subheader("Desde Yahoo! Finance")
        ticker = st.text_input("Ticker del stock (e.g., AAPL, TSLA)", "AAPL")
        start_date = st.date_input("Fecha de inicio", pd.to_datetime("2020-01-01"))
        end_date = st.date_input("Fecha de fin", pd.to_datetime("2023-01-01"))

        # Botón para cargar datos
        if st.button("Cargar datos desde Yahoo! Finance"):
            with st.spinner("Descargando datos..."):
                try:
                    st.session_state["data"] = load_yfinance_data(ticker, start_date, end_date)
                    st.success("Datos descargados exitosamente.")
                except Exception as e:
                    st.error(f"Error al descargar datos: {e}")

    # Segunda columna: Desde tu dispositivo
    with col2:
        st.subheader("Desde tu dispositivo")
        uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            st.write("Datos cargados:")
            st.session_state["data"] = load_file_from_streamlit(uploaded_file) # Cargar el CSV en un DataFrame

    # Visualización general
    st.subheader("Visualización de datos")
    if st.session_state["data"] is not None:
        data = st.session_state["data"]
        st.plotly_chart(plot_timeseries(remove_multindex(data), "Close", x_name="Fecha", y_name="Precio de Cierre"))
        st.dataframe(data)
