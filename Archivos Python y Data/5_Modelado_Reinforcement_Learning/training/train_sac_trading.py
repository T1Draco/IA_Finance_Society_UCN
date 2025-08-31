import os
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Definir la ruta absoluta del proyecto (raíz del proyecto)
PROJECT_ROOT = 'C:\\Users\\Admin\\PycharmProjects\\IA_Finance_Society_UCN'
ENV_PATH = os.path.join(PROJECT_ROOT, 'Archivos Python y Data', '5_Modelado_Reinforcement_Learning', 'env')
sys.path.append(ENV_PATH)

# Importar el entorno directamente desde el módulo
import stock_trading_env

# Configuración
TICKER = "AAPL"  # Cambia al ticker deseado
DATA_PATH = os.path.join(PROJECT_ROOT, 'Archivos Python y Data', '5_Modelado_Reinforcement_Learning','data', 'rl_input', f'{TICKER}_rl_input.csv')
INITIAL_BALANCE = 10000
TOTAL_TIMESTEPS = 100000  # Ajusta según tamaño del dataset
SAVE_DIR = os.path.join(PROJECT_ROOT, 'Archivos Python y Data', '5_Modelado_Reinforcement_Learning', 'models', f'{TICKER}_sac_simple')
os.makedirs(SAVE_DIR, exist_ok=True)

# Función para crear el entorno
def make_env():
    return stock_trading_env.SimpleTradingEnv(DATA_PATH, INITIAL_BALANCE)

# Crear entornos de entrenamiento y evaluación
train_env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

# Callbacks para guardar modelos y evaluar
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=SAVE_DIR, name_prefix=f"{TICKER}_sac")
eval_callback = EvalCallback(eval_env, best_model_save_path=SAVE_DIR, log_path=SAVE_DIR,
                             eval_freq=5000, deterministic=True, render=False, n_eval_episodes=1)

# Crear modelo SAC con configuración simple
model = SAC(
    "MlpPolicy",  # Política MLP (red neuronal simple)
    train_env,
    learning_rate=0.0003,
    buffer_size=100000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    verbose=1,
    device="cuda"  # Cambia a "cuda" si tienes GPU
)

# Entrenar
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

# Guardar modelo final
model.save(os.path.join(SAVE_DIR, f"{TICKER}_sac_final.zip"))
print(f"Modelo guardado en: {SAVE_DIR}")