import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os
import sys

# === Configuración de entorno de ejecución ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Seleccionar entorno: base o pequeño inversor
from env.stock_trading_env import StockTradingEnv # Descomenta para usar inversor estandar
from env.small_investor_env import SmallInvestorEnv  # Descomenta para usar inversor pequeño

# === Configuración del experimento ===
TICKER = "AAPL"
USE_SMALL_INVESTOR = True  # Cambia esto a True para usar el entorno del inversor pequeño

# === Cargar datos ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "rl_input", f"{TICKER}_rl_input.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "results", f"{TICKER}_dqn_model")

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# === Función de entorno ===
def make_env():
    if USE_SMALL_INVESTOR:
        from env.small_investor_env import SmallInvestorEnv
        return SmallInvestorEnv(df)
    else:
        return StockTradingEnv(df)

env = DummyVecEnv([make_env])

# Apilar las últimas 10 observaciones
env = VecFrameStack(env, n_stack=10)

# === Configuración del agente ===
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=5000,
    learning_starts=500,
    batch_size=32,
    tau=1.0,
    gamma=0.95,
    train_freq=1,
    target_update_interval=250,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    tensorboard_log=os.path.join(ROOT_DIR, "results", "tb_logs"),
    device="cuda"
)

# === Entrenamiento ===
model.learn(total_timesteps=50_000)

# === Evaluación rápida ===
obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

print(f"Recompensa total obtenida por el agente: {total_reward}")

# === Guardar modelo ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"[✔] Modelo DQN guardado en: {MODEL_PATH}")
