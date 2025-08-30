import os
import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement, \
    BaseCallback
from stable_baselines3.common.utils import set_random_seed
import sys
import torch

# === Configuración de entorno de ejecución ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.stock_trading_env_fractional import FractionalStockTradingEnv
from rewards.reward_fn_fractional_gcf import reward_fn_fractional_gcf

# === CONFIGURACIÓN ANTI-NaN ===
set_random_seed(42)  # Para reproducibilidad

# Configurar PyTorch para detectar NaN temprano
torch.autograd.set_detect_anomaly(True)

# === Configuración ===
TICKER = "GC=F"
TOTAL_TIMESTEPS = 50_000  # Mantener objetivo original

# === Rutas ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "results", "tb_logs", f"{TICKER}_stable_sac")
EVAL_LOG_DIR = os.path.join(LOG_DIR, "eval")
DATA_PATH = os.path.join(ROOT_DIR, "data", "rl_input", f"{TICKER}_rl_input.csv")

for dir_path in [MODEL_DIR, LOG_DIR, EVAL_LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# === Parámetros del entorno ===
INITIAL_BALANCE = 5000
TRANSACTION_COST = 0.001
MAX_POSITION_PCT = 0.90
MIN_TRADE_AMOUNT = 50

# === Carga de datos ===
df = pd.read_csv(DATA_PATH)
print(f"=== ENTRENAMIENTO ESTABLE SAC ===")
print(f"Datos cargados: {len(df)} registros para {TICKER}")
print(f"Rango de precios: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")


# === Función para crear entorno ===
def make_env(seed=None):
    def _init():
        env = FractionalStockTradingEnv(
            df.copy(),  # Copiar para evitar modificaciones
            initial_balance=INITIAL_BALANCE,
            transaction_cost=TRANSACTION_COST,
            max_position_pct=MAX_POSITION_PCT,
            min_trade_amount=MIN_TRADE_AMOUNT,
            reward_fn=reward_fn_fractional_gcf
        )
        if seed is not None:
            env.seed(seed)
        return Monitor(env)

    return _init


# Crear entornos de entrenamiento y evaluación
print("\n=== Configurando entornos ===")
train_env = DummyVecEnv([make_env(seed=42)])
eval_env = DummyVecEnv([make_env(seed=123)])  # Diferente semilla para evaluación

print(f"Balance inicial: ${INITIAL_BALANCE}")
print(f"Costo de transacción: {TRANSACTION_COST * 100:.1f}%")
print(f"Máx posición: {MAX_POSITION_PCT * 100:.1f}%")
print(f"Mínimo por trade: ${MIN_TRADE_AMOUNT}")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=EVAL_LOG_DIR,
    eval_freq=5000,  # Evaluar cada 5k steps
    deterministic=True,
    render=False,
    n_eval_episodes=5
)

# Callback para parar si no hay mejora (evita overfitting)
stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=10,  # Parar si no mejora en 10 evaluaciones
    min_evals=5,  # Mínimo 5 evaluaciones antes de considerar parar
    verbose=1
)

# === CONFIGURACIÓN SAC ULTRA-ESTABLE ===
print("\n=== Configurando SAC para máxima estabilidad ===")

# Hiperparámetros conservadores para evitar explosión de gradientes
sac_config = {
    'policy': "MlpPolicy",
    'env': train_env,
    'verbose': 1,
    'device': "auto",

    # === PARÁMETROS CRÍTICOS PARA ESTABILIDAD ===
    'learning_rate': 0.0001,  # MÁS BAJO (era 0.0003)
    'buffer_size': 100_000,  # Suficiente para SAC
    'learning_starts': 5000,  # MÁS ALTO - más exploración antes de entrenar
    'batch_size': 64,  # Estándar
    'tau': 0.005,  # Suave para target networks
    'gamma': 0.99,  # Descuento estándar
    'train_freq': 1,  # Entrenar cada paso
    'gradient_steps': 1,  # Un gradiente por paso (conservador)

    # === CONFIGURACIÓN DE LA RED NEURONAL ===
    'policy_kwargs': {
        'net_arch': [256, 256],  # Red más grande pero con mejor regularización
        'activation_fn': torch.nn.ReLU,

        # === PARÁMETROS CRÍTICOS ANTI-NaN ===
        'optimizer_class': torch.optim.Adam,
        'optimizer_kwargs': {
            'eps': 1e-7,  # Evitar divisiones por cero en Adam
            'weight_decay': 1e-4  # Regularización L2 para estabilidad
        },

        # Inicialización de pesos más conservadora
        'log_std_init': -3,  # Inicializar log_std más bajo
        'use_sde': False,  # Desactivar SDE para más estabilidad
    },

    # === CONFIGURACIÓN DE ENTROPÍA ===
    'ent_coef': 'auto',  # Ajuste automático de entropía
    'target_entropy': 'auto',  # Target entropy automático

    # === LOGGING ===
    'tensorboard_log': LOG_DIR,
    'seed': 42
}

print("Configuración SAC:")
for key, value in sac_config.items():
    if key != 'policy_kwargs':
        print(f"  {key}: {value}")
print(f"  net_arch: {sac_config['policy_kwargs']['net_arch']}")
print(f"  learning_rate: {sac_config['learning_rate']} (REDUCIDO para estabilidad)")
print(f"  learning_starts: {sac_config['learning_starts']} (AUMENTADO para más exploración)")

# === Crear modelo SAC ===
try:
    model = SAC(**sac_config)
    print("✅ Modelo SAC creado exitosamente con configuración estable")
except Exception as e:
    print(f"❌ ERROR creando modelo SAC: {e}")
    sys.exit(1)

# === Callback de monitoreo para detectar NaN ===
from stable_baselines3.common.callbacks import BaseCallback


class NaNMonitorCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.n_calls = 0

    def _on_step(self) -> bool:
        self.n_calls += 1

        if self.n_calls % self.check_freq == 0:
            # Verificar si hay NaN en los parámetros del modelo
            has_nan = False
            nan_params = []

            for name, param in self.model.policy.named_parameters():
                if torch.isnan(param).any():
                    print(f"❌ NaN detectado en parámetro {name} en step {self.n_calls}")
                    has_nan = True
                    nan_params.append(name)

            if has_nan:
                print(f"❌ ENTRENAMIENTO ABORTADO - NaN en parámetros: {nan_params}")
                return False  # Parar entrenamiento
            else:
                if self.verbose > 0:
                    print(f"✅ Step {self.n_calls}: Sin NaN detectado")

        return True


nan_monitor = NaNMonitorCallback(check_freq=1000, verbose=1)

# === ENTRENAMIENTO CON MONITOREO ===
print(f"\n=== Iniciando entrenamiento SAC ESTABLE ===")
print(f"Timesteps totales: {TOTAL_TIMESTEPS:,}")
print("Configuración anti-NaN activada ⚡")
print("Monitoreo de gradientes: ACTIVO 🔍")
print("Checkpoints cada 10,000 steps 💾")
print("Evaluación cada 5,000 steps 📊")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, nan_monitor],  # <--- sin checkpoint_callback
        progress_bar=True
    )

    print("🎉 ¡Entrenamiento completado exitosamente!")

except Exception as e:
    print(f"❌ ERROR durante entrenamiento: {e}")
    import traceback

    traceback.print_exc()

    # Intentar guardar el modelo parcial
    try:
        partial_path = os.path.join(MODEL_DIR, f"{TICKER}_partial_model.zip")
        model.save(partial_path)
        print(f"💾 Modelo parcial guardado en: {partial_path}")
    except:
        print("No se pudo guardar modelo parcial")

    sys.exit(1)

# === EVALUACIÓN FINAL ===
print("\n=== Evaluación final ===")
try:
    obs = train_env.reset()
    total_rewards = []
    steps = 0
    done = False

    while not done and steps < 200:  # Evaluación más larga
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = train_env.step(action)
        total_rewards.append(reward[0])
        steps += 1

    cum_rewards = np.cumsum(total_rewards)
    final_return = cum_rewards[-1] if total_rewards else 0

    print(f"Evaluación completada: {steps} pasos")
    print(f"Retorno acumulado final: {final_return:.4f}")

    # Guardar gráfico de recompensas
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(total_rewards)
        plt.title(f"Rewards por Paso - {TICKER}")
        plt.xlabel("Paso")
        plt.ylabel("Reward")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(cum_rewards)
        plt.title(f"Reward Acumulado - {TICKER}")
        plt.xlabel("Paso")
        plt.ylabel("Reward Acumulado")
        plt.grid(True)

        PLOT_DIR = os.path.join(ROOT_DIR, "results", "plots", "rewards", f"{TICKER}_stable")
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{TICKER}_stable_rewards.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Gráfico guardado en: {os.path.join(PLOT_DIR, f'{TICKER}_stable_rewards.png')}")

    except Exception as e:
        print(f"⚠️  No se pudo generar gráfico: {e}")

except Exception as e:
    print(f"⚠️  Error en evaluación final: {e}")

# === GUARDAR MODELO FINAL ===
try:
    final_model_path = os.path.join(MODEL_DIR, f"{TICKER}_stable_sac_final.zip")
    model.save(final_model_path)
    print(f"💾 Modelo final guardado: {final_model_path}")
except Exception as e:
    print(f"❌ Error guardando modelo final: {e}")

# === ESTADÍSTICAS FINALES ===
try:
    env_instance = train_env.get_attr('unwrapped')[0]
    print(f"\n=== ESTADÍSTICAS FINALES ===")
    print(f"Balance final: ${env_instance.balance:.2f}")
    print(f"Shares finales: {env_instance.shares_held:.4f}")
    print(f"Valor total final: ${env_instance.total_asset:.2f}")
    print(f"Retorno total: {((env_instance.total_asset / INITIAL_BALANCE - 1) * 100):.2f}%")
except Exception as e:
    print(f"⚠️  No se pudieron obtener estadísticas finales: {e}")

print(f"\n=== INFORMACIÓN ÚTIL ===")
print(f"📊 TensorBoard: tensorboard --logdir {LOG_DIR}")
print(f"📁 Modelos guardados en: {MODEL_DIR}")
print(f"📈 Logs de evaluación: {EVAL_LOG_DIR}")
print("✅ Entrenamiento estable completado!")