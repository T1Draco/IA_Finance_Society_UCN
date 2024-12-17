# Archivo: backend/visualization.py
import plotly.graph_objects as go

def plot_timeseries(data, column, title="Serie de tiempo", x_name="Date", y_name="Value"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=column))
    fig.update_layout(title=title, xaxis_title=x_name, yaxis_title=y_name)
    return fig
