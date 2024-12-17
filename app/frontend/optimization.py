# Archivo: app/pages/dashboard.py
import streamlit as st
from backend.visualization import plot_timeseries

def render_optimization():
    st.title("Optimization")
    st.write("Visualización de datos históricos")
    
    # Carga de datos y visualización
    data = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if data:
        st.write("Datos cargados:")
        st.dataframe(data)
        st.plotly_chart(plot_timeseries(data, "Close"))
