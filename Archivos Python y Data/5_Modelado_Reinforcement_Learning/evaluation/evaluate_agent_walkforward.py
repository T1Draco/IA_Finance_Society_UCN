# evaluation/evaluate_agent_walkforward.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO, DQN, SAC
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from env.stock_trading_env_walkforward import StockTradingEnvWithLSTM

# ================= CONFIGURACIÓN =================
TICKER = "AAPL"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_DIR, "3_Modelado_Precios_Futuros", "data",
                         "multivariable_walkforward_hibrid_data", f"{TICKER}_X.npy")
LSTM_MODEL_PATH = f"../../3_Modelado_Precios_Futuros/models/LSTM/models_pytorch/{TICKER}_lstm_walkforward.pth"
SCALER_PATH = f"../../3_Modelado_Precios_Futuros/models/LSTM/models_pytorch/{TICKER}_walkforward_scaler.pkl"
RL_MODEL_PATH = f"../../5_Modelado_Reinforcement_Learning/models/{TICKER}_rl_models_pred_reward/{TICKER}_rl_final_model_pred_reward.zip"

INITIAL_BALANCE = 10000
LOOKBACK = 60
TOTAL_STEPS = 500  # número de pasos a evaluar

# ================= VERIFICAR ARCHIVOS =================
print("Verificando archivos...")
files_to_check = [DATA_PATH, LSTM_MODEL_PATH, SCALER_PATH, RL_MODEL_PATH]
for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"✓ {file_path}")
    else:
        print(f"✗ {file_path}")
        raise FileNotFoundError(f"No se encontró: {file_path}")

# ================= INICIALIZAR ENVIRONMENT =================
print("Inicializando environment...")
env = StockTradingEnvWithLSTM(
    data_path=DATA_PATH,
    lstm_model_path=LSTM_MODEL_PATH,
    scaler_path=SCALER_PATH,
    ticker=TICKER,
    initial_balance=INITIAL_BALANCE,
    lookback=LOOKBACK
)

print(f"Environment inicializado. Datos disponibles: {env.n_steps} pasos")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# ================= CARGAR MODELO RL =================
print("Cargando modelo RL...")
# Cargar modelo sin env para evitar conflict de observation spaces
model = PPO.load(RL_MODEL_PATH)
# Luego establecer el env manualmente
model.set_env(env)

# ================= EVALUACIÓN =================
print("Iniciando evaluación...")
obs, info = env.reset()
done = False

history = {
    "step": [],
    "action": [],
    "balance": [],
    "positions": [],
    "total_assets": [],
    "price": [],
    "reward": [],
    "lstm_prediction": []
}

step_count = 0
max_steps = min(TOTAL_STEPS, env.n_steps)

while not done and step_count < max_steps:
    # Predecir acción
    action, _ = model.predict(obs, deterministic=True)

    # Ejecutar acción
    obs, reward, done, truncated, info = env.step(action)

    # Obtener predicción LSTM para logging
    try:
        lstm_pred = env._get_lstm_prediction()
    except:
        lstm_pred = 0.0

    # Guardar historia
    history["step"].append(step_count)
    history["action"].append(int(action))
    history["balance"].append(info.get('balance', 0))
    history["positions"].append(info.get('positions', 0))
    history["total_assets"].append(info.get('total_assets', 0))
    history["price"].append(info.get('current_price', 0))
    history["reward"].append(float(reward))
    history["lstm_prediction"].append(lstm_pred)

    step_count += 1

    # Log progreso cada 50 steps
    if step_count % 50 == 0:
        print(f"Step {step_count}/{max_steps}, Total Assets: {info.get('total_assets', 0):.2f}")

print(f"Evaluación completada. {step_count} pasos ejecutados.")

# ================= CONVERTIR A DATAFRAME =================
df_history = pd.DataFrame(history)

if len(df_history) == 0:
    print("Error: No se registraron datos durante la evaluación")
    exit(1)

# ================= CALCULAR MÉTRICAS =================
print("Calculando métricas...")

# Filtrar valores no válidos
df_history = df_history[df_history['total_assets'] > 0]

if len(df_history) < 2:
    print("Error: No hay suficientes datos válidos para calcular métricas")
    exit(1)

# Calcular retornos
df_history['returns'] = df_history['total_assets'].pct_change().fillna(0)
df_history['returns'] = df_history['returns'].replace([np.inf, -np.inf], 0)

# Métricas principales
initial_value = df_history['total_assets'].iloc[0]
final_value = df_history['total_assets'].iloc[-1]
cumulative_return = (final_value / initial_value) - 1

# Sharpe ratio (anualizado)
mean_return = df_history['returns'].mean()
std_return = df_history['returns'].std()
sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0

# Max drawdown
running_max = df_history['total_assets'].expanding().max()
drawdown = (df_history['total_assets'] - running_max) / running_max
max_drawdown = drawdown.min()

# Estadísticas de trading
total_trades = (df_history['action'] != 0).sum()
buy_actions = (df_history['action'] == 1).sum()
sell_actions = (df_history['action'] == 2).sum()

print("\n" + "=" * 50)
print("RESULTADOS DE EVALUACIÓN")
print("=" * 50)
print(f"Ticker: {TICKER}")
print(f"Pasos evaluados: {len(df_history)}")
print(f"Balance inicial: ${initial_value:,.2f}")
print(f"Balance final: ${final_value:,.2f}")
print(f"Retorno acumulado: {cumulative_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Total trades: {total_trades}")
print(f"  - Compras: {buy_actions}")
print(f"  - Ventas: {sell_actions}")
print("=" * 50)

# ================= GRÁFICOS =================
print("Generando gráficos...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1: Valor del portafolio vs Precio
ax1.plot(df_history['step'], df_history['total_assets'], label='Portfolio Value', linewidth=2)
ax1.plot(df_history['step'], df_history['price'], label='Stock Price', alpha=0.7)
ax1.set_title(f'{TICKER} - Portfolio Value vs Stock Price')
ax1.set_xlabel('Step')
ax1.set_ylabel('Value ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Acciones tomadas
actions_df = df_history['action'].value_counts().sort_index()
ax2.bar(['Hold', 'Buy', 'Sell'], [actions_df.get(i, 0) for i in range(3)])
ax2.set_title('Distribution of Actions')
ax2.set_ylabel('Count')

# Gráfico 3: Predicciones LSTM vs Precio real
ax3.plot(df_history['step'], df_history['price'], label='Actual Price', alpha=0.7)
ax3.plot(df_history['step'], df_history['lstm_prediction'], label='LSTM Prediction', alpha=0.7)
ax3.set_title('LSTM Predictions vs Actual Prices')
ax3.set_xlabel('Step')
ax3.set_ylabel('Price ($)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Gráfico 4: Drawdown
ax4.fill_between(df_history['step'], drawdown * 100, 0, alpha=0.3, color='red')
ax4.set_title('Portfolio Drawdown')
ax4.set_xlabel('Step')
ax4.set_ylabel('Drawdown (%)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'../results/{TICKER}_evaluation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ================= GUARDAR RESULTADOS =================
print("Guardando resultados...")
os.makedirs("../results/evaluation", exist_ok=True)
df_history.to_csv(f"../results/evaluation/{TICKER}_rl_eval_detailed.csv", index=False)

# Resumen de métricas
metrics_summary = {
    'ticker': TICKER,
    'steps_evaluated': len(df_history),
    'initial_balance': initial_value,
    'final_balance': final_value,
    'cumulative_return': cumulative_return,
    'sharpe_ratio': sharpe_ratio,
    'max_drawdown': max_drawdown,
    'total_trades': total_trades,
    'buy_actions': buy_actions,
    'sell_actions': sell_actions
}

pd.DataFrame([metrics_summary]).to_csv(f"../results/evaluation/{TICKER}_rl_metrics_summary.csv", index=False)

print(f"Resultados guardados en ../results/evaluation/{TICKER}_rl_eval_detailed.csv")
print("Evaluación completada exitosamente!")