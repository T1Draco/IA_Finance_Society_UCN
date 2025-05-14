import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
# === Configuración de entorno de ejecución ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Seleccionar entorno: base o pequeño inversor
from env.stock_trading_env import StockTradingEnv # Descomenta para usar inversor estandar
from env.small_investor_tsla_env import SmallInvestorEnv  # Descomenta para usar inversor pequeño
# === CONFIGURACIÓN ===
TICKER = "AAPL"
N_STACK = 10
SEED = 42
USE_SMALL_INVESTOR = True

# === Rutas ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "results", "models")
LOG_DIR = os.path.join(ROOT_DIR, "results", "logs")
DATA_PATH = os.path.join(ROOT_DIR, "data", "rl_input", f"{TICKER}_rl_input.csv")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Fijar semillas ===
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Cargar datos y entorno ===
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

def make_env():
    env = SmallInvestorEnv(df) if USE_SMALL_INVESTOR else StockTradingEnv(df)
    env.action_space.seed(SEED)
    return Monitor(env)

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=N_STACK)

# === Arquitectura personalizada ===
policy_kwargs = dict(
    net_arch=[256, 256],  # Más profunda
    activation_fn=torch.nn.ReLU
)

# === Callback para registrar pérdida ===
class LossLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []

    def _on_step(self) -> bool:
        if "train/loss" in self.logger.name_to_value:
            self.losses.append(self.logger.name_to_value["train/loss"])
        return True

loss_logger = LossLogger()

# === Inicializar modelo ===
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=5000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    max_grad_norm=10,
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    tensorboard_log=os.path.join(LOG_DIR, f"{TICKER}_tensorboard")
)

# === Entrenamiento ===
model.learn(
    total_timesteps=150_000,  # Puedes aumentar a 300k+ luego
    callback=loss_logger
)

# === Guardar modelo y pérdida ===
model.save(os.path.join(MODEL_DIR, f"{TICKER}_dqn_model"))
np.save(os.path.join(MODEL_DIR, f"{TICKER}_losses.npy"), loss_logger.losses)

print("\n✅ Entrenamiento finalizado y modelo guardado.")
