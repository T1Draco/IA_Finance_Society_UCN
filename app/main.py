import streamlit as st
from sidebar import render_sidebar
from frontend.dashboard import render_dashboard
from frontend.backtesting import render_backtesting
#from frontend.optimization import render_optimization

# Título de la aplicación
st.set_page_config(page_title="Trading App", layout="wide")

# Renderizar la barra lateral y seleccionar página
selected_page = render_sidebar()

# Inicializar sesión de estado
if "data" not in st.session_state:
    st.session_state["data"] = None

if "backtrader_results" not in st.session_state:
    st.session_state["backtrader_results"] = None

# Renderizar la página seleccionada
if selected_page == "Dashboard":
    render_dashboard()
elif selected_page == "Backtesting":
    render_backtesting()

elif selected_page == "Optimization":
    #render_optimization()
    print("next rime")