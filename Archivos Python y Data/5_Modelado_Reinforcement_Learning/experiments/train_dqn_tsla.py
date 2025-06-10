import os
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import sys

# === Configuración de entorno de ejecución ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from env.stock_trading_env import StockTradingEnv
from rewards.reward_fn_tsla import reward_fn_tsla

# === Configuración ===
TICKER = "TSLA"
TOTAL_TIMESTEPS = 150_000

# === Rutas ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "results", "models")
LOG_DIR = os.path.join(ROOT_DIR, "results", "tb_logs", f"{TICKER}_tensorboard", "DQN_1")
DATA_PATH = os.path.join(ROOT_DIR, "data", "rl_input", f"{TICKER}_rl_input.csv")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Carga de datos ===
df = pd.read_csv(DATA_PATH)

# === Inicializa entorno con función de recompensa ===
def make_env():
    env = StockTradingEnv(df, transaction_cost=0.002, max_shares_per_trade=3, reward_fn=reward_fn_tsla)
    return Monitor(env)

env = DummyVecEnv([make_env])

# === Entrenamiento ===
model = DQN("MlpPolicy", env, verbose=1, device="cuda")
episode_rewards = []
model.learn(total_timesteps=TOTAL_TIMESTEPS)
# === Simulación posterior al entrenamiento (no durante) ===
obs = env.reset()
done = False
total_rewards = []
steps = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_rewards.append(reward)
    steps += 1

# === Guardar recompensas acumuladas ===
import matplotlib.pyplot as plt
import numpy as np

cum_rewards = np.cumsum(total_rewards)

plt.figure(figsize=(12, 5))
plt.plot(cum_rewards)
plt.title(f"Reward acumulado paso a paso – {TICKER}")
plt.xlabel("Paso")
plt.ylabel("Reward acumulado")
plt.grid(True)

PLOT_DIR = os.path.join(ROOT_DIR, "results", "plots","rewards", TICKER)
os.makedirs(PLOT_DIR, exist_ok=True)
plt.savefig(os.path.join(PLOT_DIR, f"{TICKER}_reward_acumulado.png"))
plt.close()

# === Guardar modelo ===
model_path = os.path.join(MODEL_DIR, f"{TICKER}_dqn_model.zip")
model.save(model_path)
print(f"Modelo guardado en: {model_path}")
