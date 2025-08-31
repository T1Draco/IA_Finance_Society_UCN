# evaluation/evaluate_agent_walkforward_with_actions.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.stock_trading_env_walkforward import StockTradingEnvWithLSTM
import os

# ================= CONFIGURACIÓN =================
TICKER = "AAPL"
LSTM_MODEL_PATH = f"../3_Modelado_Precios_Futuros/models/LSTM/models_pytorch/{TICKER}_lstm_walkforward.pth"
SCALER_PATH = f"../3_Modelado_Precios_Futuros/models/LSTM/models_pytorch/{TICKER}_walkforward_scaler.pkl"
DATA_PATH = f"../3_Modelado_Precios_Futuros/data/multivariable_walkforward_hibrid_data/{TICKER}_X.csv"
RL_MODEL_PATH = f"../5_Modelado_Reinforcement_Learning/models/{TICKER}_ppo_rl_model"
INITIAL_BALANCE = 10000
LOOKBACK = 60
TOTAL_STEPS = 500

# ================= INICIALIZAR ENVIRONMENT =================
env = StockTradingEnvWithLSTM(
    data_path=DATA_PATH,
    lstm_model_path=LSTM_MODEL_PATH,
    scaler_path=SCALER_PATH,
    ticker=TICKER,
    initial_balance=INITIAL_BALANCE,
    lookback=LOOKBACK
)

# ================= CARGAR MODELO RL =================
model = PPO.load(RL_MODEL_PATH, env=env)

# ================= EVALUACIÓN =================
obs, _ = env.reset()
done = False

history = {
    "step": [],
    "action": [],
    "balance": [],
    "positions": [],
    "total_assets": [],
    "price": [],
    "reward": []
}

action_map = {0: "Hold", 1: "Buy", 2: "Sell"}

for step in range(TOTAL_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)

    current_price = env.df.iloc[env.current_step]['Close']

    history["step"].append(step)
    history["action"].append(action)
    history["balance"].append(info['balance'])
    history["positions"].append(info['positions'])
    history["total_assets"].append(info['total_assets'])
    history["price"].append(current_price)
    history["reward"].append(reward)

    if done:
        break

df_history = pd.DataFrame(history)

# ================= MÉTRICAS =================
df_history['returns'] = df_history['total_assets'].pct_change().fillna(0)
cumulative_return = (df_history['total_assets'].iloc[-1] / INITIAL_BALANCE) - 1
sharpe_ratio = df_history['returns'].mean() / (df_history['returns'].std() + 1e-9) * np.sqrt(252)
max_drawdown = (df_history['total_assets'].cummax() - df_history['total_assets']).max()

print(f"Cumulative Return: {cumulative_return:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.2f}")

# ================= GRÁFICOS =================
plt.figure(figsize=(14, 7))
plt.plot(df_history['price'], label='Price', color='blue')
plt.plot(df_history['total_assets'], label='Total Assets', color='green')

# Marcar acciones Buy/Sell
buys = df_history[df_history['action'] == 1]
sells = df_history[df_history['action'] == 2]

plt.scatter(buys['step'], buys['price'], marker='^', color='green', label='Buy', s=100)
plt.scatter(sells['step'], sells['price'], marker='v', color='red', label='Sell', s=100)

plt.title(f"{TICKER} RL Evaluation Walk-Forward with Actions")
plt.xlabel("Step")
plt.ylabel("USD")
plt.legend()
plt.grid()
plt.show()

# ================= GUARDAR RESULTADOS =================
os.makedirs("../5_Modelado_Reinforcement_Learning/results/evaluation", exist_ok=True)
df_history.to_csv(f"../5_Modelado_Reinforcement_Learning/results/evaluation/{TICKER}_rl_eval_actions.csv", index=False)
