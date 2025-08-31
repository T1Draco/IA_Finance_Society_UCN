# training/training_rl_walkforward_pred_reward.py

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import sys

# ================= CONFIGURACIÓN =================
TICKER = "AAPL"
LOOKBACK = 60
INITIAL_BALANCE = 10000
TOTAL_TIMESTEPS = 100_000
SEED = 42

print(f"Configuración:")
print(f"- Ticker: {TICKER}")
print(f"- Lookback: {LOOKBACK}")
print(f"- Balance inicial: {INITIAL_BALANCE}")
print(f"- Total timesteps: {TOTAL_TIMESTEPS}")
print(f"- Seed: {SEED}")

# ================= RUTAS CORREGIDAS =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Corregir las rutas para usar los archivos correctos
DATA_PATH = os.path.join(BASE_DIR, "3_Modelado_Precios_Futuros", "data",
                         "multivariable_walkforward_hibrid_data", f"{TICKER}_X.npy")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "3_Modelado_Precios_Futuros", "models",
                               "LSTM", "models_pytorch", f"{TICKER}_lstm_walkforward.pth")
SCALER_PATH = os.path.join(BASE_DIR, "3_Modelado_Precios_Futuros", "models",
                           "LSTM", "models_pytorch", f"{TICKER}_walkforward_scaler.pkl")

SAVE_DIR = os.path.join(BASE_DIR, "5_Modelado_Reinforcement_Learning", "models",
                        f"{TICKER}_rl_models_pred_reward_walkforward")
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= VERIFICAR ARCHIVOS =================
print("\nVerificando archivos necesarios...")
files_to_check = {
    "Datos (.npy)": DATA_PATH,
    "Modelo LSTM": LSTM_MODEL_PATH,
    "Scaler": SCALER_PATH
}

for name, path in files_to_check.items():
    if os.path.exists(path):
        print(f"✓ {name}: {path}")
    else:
        print(f"✗ {name}: {path}")
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

# ================= IMPORTAR ENVIRONMENT =================
BASE_DIR_ENV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR_ENV)

try:
    from env.stock_trading_env_walkforward import StockTradingEnvWithLSTM

    print("✓ Environment importado correctamente")
except ImportError as e:
    print(f"✗ Error importando environment: {e}")
    raise


# ================= FUNCIONES AUXILIARES =================
def make_env(rank=0):
    """
    Crear environment con Monitor wrapper para logging
    """

    def _init():
        env = StockTradingEnvWithLSTM(
            data_path=DATA_PATH,
            lstm_model_path=LSTM_MODEL_PATH,
            scaler_path=SCALER_PATH,
            ticker=TICKER,
            initial_balance=INITIAL_BALANCE,
            lookback=LOOKBACK,
            seed=SEED + rank
        )
        # Wrap con Monitor para logging
        env = Monitor(env, filename=os.path.join(SAVE_DIR, f"training_log_{rank}.csv"))
        return env

    return _init


def create_eval_env():
    """Crear environment para evaluación"""
    env = StockTradingEnvWithLSTM(
        data_path=DATA_PATH,
        lstm_model_path=LSTM_MODEL_PATH,
        scaler_path=SCALER_PATH,
        ticker=TICKER,
        initial_balance=INITIAL_BALANCE,
        lookback=LOOKBACK,
        seed=SEED + 999  # Seed diferente para eval
    )
    return Monitor(env, filename=os.path.join(SAVE_DIR, "eval_log.csv"))


# ================= INICIALIZAR ENVIRONMENTS =================
print("\nInicializando environments...")

# Environment de entrenamiento
env = DummyVecEnv([make_env(0)])
print(f"✓ Environment de entrenamiento creado")

# Environment de evaluación
eval_env = create_eval_env()
print(f"✓ Environment de evaluación creado")

# Verificar espacios de observación y acción
print(f"✓ Observation space: {env.observation_space}")
print(f"✓ Action space: {env.action_space}")

# ================= CONFIGURAR CALLBACKS =================
print("\nConfigurando callbacks...")

# Callback para guardar checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=SAVE_DIR,
    name_prefix=f"{TICKER}_rl_checkpoint"
)

# Callback para evaluación periódica
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    log_path=SAVE_DIR,
    eval_freq=5_000,  # Evaluar cada 5k steps
    deterministic=True,
    render=False,
    n_eval_episodes=10  # Número de episodios para evaluación
)

callbacks = [checkpoint_callback, eval_callback]
print(f"✓ Callbacks configurados: {len(callbacks)}")

# ================= CREAR MODELO RL =================
print("\nCreando modelo PPO...")

# Configuración del modelo
model_config = {
    'policy': 'MlpPolicy',
    'env': env,
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 256,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'verbose': 1,
    'seed': SEED,
    'device': 'cuda'  # Usar CPU para evitar issues con GPU
}

model = PPO(**model_config)
print(f"✓ Modelo PPO creado con configuración:")
for key, value in model_config.items():
    if key != 'env':  # No mostrar el env completo
        print(f"  - {key}: {value}")

# ================= ENTRENAMIENTO =================
print(f"\n{'=' * 50}")
print("INICIANDO ENTRENAMIENTO")
print(f"{'=' * 50}")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True
    )
    print("✓ Entrenamiento completado exitosamente")

except Exception as e:
    print(f"✗ Error durante el entrenamiento: {e}")
    raise

# ================= GUARDAR MODELO FINAL =================
print("\nGuardando modelo final...")
final_model_path = os.path.join(SAVE_DIR, f"{TICKER}_rl_final_model_pred_reward_walkforward")
model.save(final_model_path)
print(f"✓ Modelo final guardado en: {final_model_path}.zip")

# Guardar también los parámetros del modelo
import json

model_info = {
    'ticker': TICKER,
    'lookback': LOOKBACK,
    'initial_balance': INITIAL_BALANCE,
    'total_timesteps': TOTAL_TIMESTEPS,
    'seed': SEED,
    'observation_space': str(env.observation_space),
    'action_space': str(env.action_space),
    'model_config': {k: v for k, v in model_config.items() if k != 'env'}
}

with open(os.path.join(SAVE_DIR, f"{TICKER}_model_info.json"), 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"✓ Información del modelo guardada")

# ================= EVALUACIÓN RÁPIDA =================
print(f"\n{'=' * 50}")
print("EVALUACIÓN RÁPIDA DEL MODELO")
print(f"{'=' * 50}")

try:
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    max_eval_steps = 100  # Limitar evaluación rápida

    while not done and step_count < max_eval_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward[0] if hasattr(reward, '__len__') else reward
        step_count += 1

        if step_count % 20 == 0:
            print(f"  Step {step_count}/{max_eval_steps}, Reward acumulado: {total_reward:.2f}")

    print(f"✓ Evaluación completada:")
    print(f"  - Steps ejecutados: {step_count}")
    print(f"  - Reward total: {total_reward:.4f}")
    print(f"  - Reward promedio por step: {total_reward / step_count:.4f}")

except Exception as e:
    print(f"✗ Error en evaluación rápida: {e}")

# ================= RESUMEN FINAL =================
print(f"\n{'=' * 50}")
print("RESUMEN DEL ENTRENAMIENTO")
print(f"{'=' * 50}")
print(f"Ticker: {TICKER}")
print(f"Timesteps entrenados: {TOTAL_TIMESTEPS:,}")
print(f"Modelo guardado en: {SAVE_DIR}")
print(f"Archivos generados:")
print(f"  - {TICKER}_rl_final_model_pred_reward_walkforward.zip")
print(f"  - {TICKER}_model_info.json")
print(f"  - Logs de entrenamiento y evaluación")
print(f"{'=' * 50}")
print("¡ENTRENAMIENTO COMPLETADO!")
print(f"{'=' * 50}")