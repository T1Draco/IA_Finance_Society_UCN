import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

# Definir la ruta absoluta del proyecto (raíz del proyecto)
PROJECT_ROOT = 'C:\\Users\Admin\\PycharmProjects\\IA_Finance_Society_UCN'
ENV_PATH = os.path.join(PROJECT_ROOT, 'Archivos Python y Data', '5_Modelado_Reinforcement_Learning', 'env')
sys.path.append(ENV_PATH)

# Importar el entorno
import stock_trading_env

# Configuración
TICKER = "AAPL"
DATA_PATH = os.path.join(PROJECT_ROOT, 'Archivos Python y Data', '5_Modelado_Reinforcement_Learning', 'data', 'rl_input', f'{TICKER}_rl_input.csv')
INITIAL_BALANCE = 10000
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Archivos Python y Data', '5_Modelado_Reinforcement_Learning', 'models', f'{TICKER}_sac_simple', f'{TICKER}_sac_final.zip')

# Crear entorno
env = stock_trading_env.SimpleTradingEnv(DATA_PATH, INITIAL_BALANCE)

# Cargar modelo entrenado
model = SAC.load(MODEL_PATH)

# Evaluar el agente
obs = env.reset()[0]  # Obtener solo la observación (ignorar info)
done = False
total_reward = 0
steps = 0
net_worth_history = [env.initial_balance]
action_history = []  # Guardar acciones tomadas

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    net_worth_history.append(info['net_worth'])
    # Clasificar acción: > 0 (buy), < 0 (sell), ~0 (hold)
    action_type = 'Buy' if action[0] > 0.1 else 'Sell' if action[0] < -0.1 else 'Hold'
    action_history.append(action_type)
    done = terminated or truncated

# Resultados
print(f"Evaluación completada para {TICKER}:")
print(f" - Total Steps: {steps}")
print(f" - Total Reward: {total_reward:.2f}")
print(f" - Final Net Worth: {info['net_worth']:.2f}")
print(f" - Initial Balance: {INITIAL_BALANCE:.00f}")
print(f" - Return (%): {((info['net_worth'] - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.2f}%")

# Guardar historial para análisis
results_df = pd.DataFrame({
    'Step': range(steps + 1),
    'Net_Worth': net_worth_history,
    'Action': ['Start'] + action_history  # 'Start' para el paso inicial
})
results_df.to_csv(os.path.join(PROJECT_ROOT, 'Archivos Python y Data', '5_Modelado_Reinforcement_Learning', 'results', f'{TICKER}_evaluation_results.csv'), index=False)
print(f"Historial de resultados guardado en: {TICKER}_evaluation_results.csv")

# Graficar Net Worth
plt.figure(figsize=(12, 6))
plt.plot(results_df['Step'], results_df['Net_Worth'], label='Net Worth', color='blue')
plt.title(f'Net Worth Evolution for {TICKER}')
plt.xlabel('Step')
plt.ylabel('Net Worth ($)')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PROJECT_ROOT, 'Archivos Python y Data', '5_Modelado_Reinforcement_Learning', 'results', f'{TICKER}_net_worth_plot.png'))
plt.close()

# Graficar Acciones
plt.figure(figsize=(12, 6))
actions_numeric = {'Buy': 1, 'Hold': 0, 'Sell': -1}  # Diccionario de mapeo
action_values = [actions_numeric.get(action, 0) for action in results_df['Action']]  # Usar .get() con valor por defecto 0 para 'Start'
plt.step(results_df['Step'], action_values, where='post', label='Action', color='green')
plt.title(f'Actions Taken for {TICKER}')
plt.xlabel('Step')
plt.ylabel('Action (1=Buy, 0=Hold, -1=Sell)')
plt.yticks([-1, 0, 1], ['Sell', 'Hold', 'Buy'])  # Etiquetas personalizadas
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PROJECT_ROOT, 'Archivos Python y Data', '5_Modelado_Reinforcement_Learning', 'results', f'{TICKER}_actions_plot.png'))
plt.close()

print(f"Gráficas guardadas en: {TICKER}_net_worth_plot.png y {TICKER}_actions_plot.png")