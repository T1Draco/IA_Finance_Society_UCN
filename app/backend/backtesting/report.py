import streamlit as st
import pandas as pd
from collections import defaultdict
import quantstats as qs
import os

def generate_quant_reports(results_dict, output_dir="reports"):
    """
    Genera reportes para cada estrategia basada en su series de retornos.

    Parámetros:
        results_dict (dict): Un diccionario donde las claves son nombres de estrategias
                             y los valores son otro diccionario con la clave "daily_returns"
                             que contiene pandas.Series con los retornos.
        output_dir (str): Directorio donde se guardarán los reportes.
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    for strategy_name, data in results_dict.items():
        if "daily_returns" in data and isinstance(data["daily_returns"], pd.Series):
            return_series = data["daily_returns"]
            print(return_series)
            st.dataframe(return_series)
            # Asegurarse de que el índice sea un DatetimeIndex
            if isinstance(return_series.index, pd.DatetimeIndex):
                # Resamplear a frecuencia diaria si es necesario
                if return_series.index.freq is None or return_series.index.freq != 'D':
                    return_series = return_series.resample('D').sum()  # Puedes usar .mean() si tiene sentido en tu contexto
                    print("resampleado ta weno:3")
                print("ta weno:3")
            if not isinstance(return_series.index, pd.DatetimeIndex):
                try:
                    return_series.index = pd.to_datetime(return_series.index)
                except Exception as e:
                    print(f"Error al convertir el índice a DatetimeIndex para {strategy_name}: {e}")
                    continue
            
            qs.reports.html(return_series, output=output_dir)
            print(f"Reporte generado exitosamente: {output_dir}")

        else:
            print(f"Advertencia: La estrategia {strategy_name} no tiene datos válidos en 'daily_returns'.")

        



def display_backtest_report(results):
    """
    Función para procesar un diccionario de resultados de backtesting y generar reportes en Streamlit.
    
    Parámetros:
        results (dict): Diccionario con los resultados del backtesting.
    """
    st.title("Reporte de Backtesting")

    # Resumen de resultados
    st.header("Resumen de Resultados")
    summary_data = []
    for strategy, metrics in results.items():
        final_value = metrics['final_value']
        max_drawdown = metrics['drawdown']['max']['drawdown']
        money_drawdown = metrics['drawdown']['max']['moneydown']
        summary_data.append({
            "Estrategia": strategy,
            "Capital Final": round(final_value, 2),
            "Máximo Drawdown (%)": round(max_drawdown * 100, 2),
            "Máxima Pérdida ($)": round(money_drawdown, 2)
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)

    # Detalle de cada estrategia
    st.header("Detalles por Estrategia")
    for strategy, metrics in results.items():
        st.subheader(f"Estrategia: {strategy}")
        
        # Capital Final
        st.metric("Capital Final", f"${metrics['final_value']:.2f}")

        # Gráfico de PyFolio: Retornos
        pyfolio_returns = metrics['pyfolio'][0]
        st.line_chart(pyfolio_returns, height=300, use_container_width=True)

        # Análisis de Operaciones
        trade_analysis = metrics['trade_analysis']
        st.write("**Análisis de Operaciones**")
        trade_stats = {
            "Total de Operaciones": trade_analysis['total']['total'],
            "Ganadoras": trade_analysis['won']['total'],
            "Perdedoras": trade_analysis['lost']['total'],
            "Promedio P&L Bruto ($)": trade_analysis['pnl']['gross']['average'],
            "Promedio P&L Neto ($)": trade_analysis['pnl']['net']['average'],
            "Mayor Ganancia ($)": trade_analysis['won']['pnl']['max'],
            "Mayor Pérdida ($)": trade_analysis['lost']['pnl']['max'],
        }
        trade_stats_df = pd.DataFrame(trade_stats, index=[0]).T.rename(columns={0: "Valor"})
        st.dataframe(trade_stats_df)

        # Detalle de Drawdown
        drawdown = metrics['drawdown']
        st.write("**Detalle del Drawdown**")
        drawdown_stats = {
            "Drawdown Máximo (%)": drawdown['max']['drawdown'] * 100,
            "Dinero Perdido Máximo ($)": drawdown['max']['moneydown'],
            "Duración del Drawdown Máximo (días)": drawdown['max']['len']
        }
        drawdown_stats_df = pd.DataFrame(drawdown_stats, index=[0]).T.rename(columns={0: "Valor"})
        st.dataframe(drawdown_stats_df)

        # Gráfico de Capital y Drawdowns
        st.write("**Gráficos de Capital y Drawdowns**")
        pyfolio_data = metrics['pyfolio'][1]
        st.line_chart(pyfolio_data["cash"], height=300, use_container_width=True)

        # Separador entre estrategias
        st.markdown("---")