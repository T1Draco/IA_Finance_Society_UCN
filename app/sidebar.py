import streamlit as st

def render_sidebar():
    st.sidebar.title("Trading App")
    page = st.sidebar.radio("Selecciona una página:", ["Dashboard", "Backtesting", "Optimization"])
    return page