# Archivo: app/pages/dashboard.py
import streamlit as st
from backend.visualization.plots import plot_timeseries
from backend.data_handling.preprocesing import remove_multindex
from backend.data_handling.loader import load_multiple_strategies
from backend.backtesting.backtester import run_backtest_with_strategies
from backend.backtesting.report import display_backtest_report
from backend.backtesting.report import generate_quant_reports


def render_backtesting():
    st.title("Backtesting")
    # Visualización general
    st.subheader("Visualización de datos")
    
    if st.session_state["data"] is not None:
        data = st.session_state["data"]
        st.plotly_chart(plot_timeseries(remove_multindex(data), "Close", x_name="Fecha", y_name="Precio de Cierre"))
        st.dataframe(data)

        st.subheader("Carga de la estrategia")
        uploaded_file = st.file_uploader("Sube un archivo Python", type=["py"], accept_multiple_files=True)
        if uploaded_file:
            try:
                # Llamar a la función para cargar el módulo
                strategies = load_multiple_strategies(uploaded_file)
                st.success(f"Módulos cargados exitosamente.")
                
            except Exception as e:
                st.error(f"Error al cargar el módulo: {e}")
            
            # Dividir en columnas
            col1, col2 = st.columns(2)
            # Primera columna: Capital inicial
            with col1:
                capital = st.number_input("Capital inicial", 10000)
            # Segunda columna: Comisión de operación
            with col2:
                comision = st.number_input("Comisión de operación", 0.001)

            # Botón para cargar datos
            if st.button("Iniciar Backtesting con las estrategias cargadas"):
                with st.spinner("Procesando datos..."):
                    try:
                        st.session_state["backtrader_results"] = run_backtest_with_strategies(strategies, data, cash=capital, commission=comision, plot=True)
                        st.success("Backtesting exitoso.")

                        st.write(st.session_state["backtrader_results"])

                        # Generar Reportes
                        display_backtest_report(st.session_state["backtrader_results"])
                            
                    except Exception as e:
                        st.error(f"Error, algo ocurrio mal durante el backtesting: {e}")

            # Botón para generar reportes detallados con QuantStats
            if "backtrader_results" in st.session_state:
                if st.button("Generar Reportes Detallados (QuantStats)"):
                    with st.spinner("Generando reportes..."):
                        try:
                            # Llamar a la función para generar los reportes de QuantStats
                            # generate_quant_reports(st.session_state["backtrader_results"], output_dir="reports")
                            st.success("Reportes generados correctamente en la carpeta 'reports'.")
                        except Exception as e:
                            st.error(f"Error al generar reportes con QuantStats: {e}")
    else:
        st.error(f"Error al cargar datos, deben ser cargados en el dashboard antes")
