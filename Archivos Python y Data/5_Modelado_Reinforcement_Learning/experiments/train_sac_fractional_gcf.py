import os
import pandas as pd
from stable_baselines3 import SAC  # SAC es mejor para acciones continuas
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import sys

# === Configuración de entorno de ejecución ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from env.stock_trading_env_fractional import FractionalStockTradingEnv
from rewards.reward_fn_gold_fractional import reward_fn_gold_fractional

# === Configuración ===
TICKER = "GC_F"
TOTAL_TIMESTEPS = 300_000  # Más timesteps para convergencia con acciones continuas

# === Rutas ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "results", "models")
LOG_DIR = os.path.join(ROOT_DIR, "results", "tb_logs", f"{TICKER}_fractional_tensorboard", "SAC_1")
DATA_PATH = os.path.join(ROOT_DIR, "data", "rl_input", f"{TICKER}_rl_input.csv")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Carga de datos ===
df = pd.read_csv(DATA_PATH)
print(f"Datos cargados: {len(df)} registros para {TICKER}")
print(f"Rango de precios: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

# === Parámetros específicos para trading fraccionario de oro ===
INITIAL_BALANCE = 5000  # Balance más alto para operaciones fraccionarias
TRANSACTION_COST = 0.001  # Costos más bajos (trading moderno)
MAX_POSITION_PCT = 0.90  # Máximo 90% del portafolio en una posición
MIN_TRADE_AMOUNT = 50   # Mínimo $50 por operación

# === Inicializa entorno ===
def make_env():
    env = FractionalStockTradingEnv(
        df,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        max_position_pct=MAX_POSITION_PCT,
        min_trade_amount=MIN_TRADE_AMOUNT,
        reward_fn=reward_fn_gold_fractional
    )
    return Monitor(env)

env = DummyVecEnv([make_env])

print(f"\n=== Configuración del entorno ===")
print(f"Balance inicial: ${INITIAL_BALANCE}")
print(f"Costo de transacción: {TRANSACTION_COST*100:.1f}%")
print(f"Máx posición: {MAX_POSITION_PCT*100:.1f}%")
print(f"Mínimo por trade: ${MIN_TRADE_AMOUNT}")

# === Entrenamiento con SAC (mejor para acciones continuas) ===
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",
    learning_rate=0.0003,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    tensorboard_log=LOG_DIR
)

print(f"\n=== Iniciando entrenamiento SAC ===")
print(f"Timesteps totales: {TOTAL_TIMESTEPS:,}")
print("Usando SAC (Soft Actor-Critic) para acciones continuas...")

model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="gold_fractional_sac")

# === Simulación posterior al entrenamiento ===
obs = env.reset()
done = False
total_rewards = []
steps = 0

print("\n=== Simulación post-entrenamiento ===")

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_rewards.append(reward)
    steps += 1

# === Guardar recompensas acumuladas ===
import matplotlib.pyplot as plt
import numpy as np

cum_rewards = np.cumsum(total_rewards)

plt.figure(figsize=(12, 5))
plt.plot(cum_rewards)
plt.title(f"Reward acumulado - Trading Fraccionario {TICKER}")
plt.xlabel("Paso")
plt.ylabel("Reward acumulado")
plt.grid(True)

PLOT_DIR = os.path.join(ROOT_DIR, "results", "plots","rewards", f"{TICKER}_fractional")
os.makedirs(PLOT_DIR, exist_ok=True)
plt.savefig(os.path.join(PLOT_DIR, f"{TICKER}_fractional_reward_acumulado.png"))
plt.close()

# === Guardar modelo ===
model_path = os.path.join(MODEL_DIR, f"{TICKER}_sac_fractional_model.zip")
model.save(model_path)

print(f"\n=== Entrenamiento completado ===")
print(f"Modelo guardado en: {model_path}")
print(f"Gráfico guardado en: {os.path.join(PLOT_DIR, f'{TICKER}_fractional_reward_acumulado.png')}")
print(f"Logs TensorBoard: tensorboard --logdir {LOG_DIR}")

# === Estadísticas finales ===
env_instance = env.get_attr('unwrapped')[0]
print(f"\n=== Estadísticas finales ===")
print(f"Balance final: ${env_instance.balance:.2f}")
print(f"Shares finales: {env_instance.shares_held:.4f}")
print(f"Valor total final: ${env_instance.total_asset:.2f}")
print(f"Retorno: {((env_instance.total_asset/INITIAL_BALANCE-1)*100):.2f}%")