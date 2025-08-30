import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# === Configuración ===
TICKER_MODEL = "GC=F_stable_sac_final.zip"
TICKER_DATA = "GC=F"
INITIAL_BALANCE = 5000
N_STACK = 1  # SAC fractional normalmente no usa frame stacking

# === Rutas ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", TICKER_MODEL)
DATA_PATH = os.path.join(ROOT_DIR, "data", "rl_input", f"{TICKER_DATA}_rl_input.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots", f"{TICKER_MODEL}_on_{TICKER_DATA}")
os.makedirs(PLOT_DIR, exist_ok=True)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.stock_trading_env_fractional import FractionalStockTradingEnv
from rewards.reward_fn_fractional_gcf import reward_fn_fractional_gcf

# === Cargar datos ===
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# === Crear entorno de evaluación ===
def make_env():
    return FractionalStockTradingEnv(
        df.copy(),
        initial_balance=INITIAL_BALANCE,
        transaction_cost=0.001,
        max_position_pct=0.90,
        min_trade_amount=50,
        reward_fn=reward_fn_fractional_gcf
    )

eval_env = DummyVecEnv([make_env])
train_env = eval_env  # Para compatibilidad con tu código de simulación
env_ = train_env.get_attr('unwrapped')[0]  # acceder al entorno real

# === Cargar modelo SAC ===
model = SAC.load(MODEL_PATH, env=train_env)

# === Simulación ===
obs = train_env.reset()
done = False

dates = df["Date"].tolist()
capital_history = []
rewards = []
actions_taken = []
portfolio_log = []

for i in range(len(dates)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = train_env.step(action)

    price = df.loc[i, "Close"]
    capital = env_.balance + env_.shares_held * price
    capital_history.append(capital)
    rewards.append(float(reward[0]))  # extraer elemento del array
    actions_taken.append(int(action[0]))  # extraer elemento del array

    portfolio_log.append({
        "step": i,
        "date": dates[i],
        "action": int(action[0]),
        "balance": env_.balance,
        "shares_held": env_.shares_held,
        "price": price,
        "avg_buy_price": env_.avg_buy_price,
        "total_asset": capital,
        "reward": float(reward[0]),
    })

    if done:
        break

# === Guardar CSV con actividad ===
portfolio_df = pd.DataFrame(portfolio_log)
csv_path = os.path.join(RESULTS_DIR, f"{TICKER_MODEL}_portfolio_tracking.csv")
portfolio_df.to_csv(csv_path, index=False)

# === Gráficos ===
portfolio_df["delta_asset"] = portfolio_df["total_asset"].diff()
buy_idx = [i for i, a in enumerate(actions_taken) if a == 1]
sell_idx = [i for i, a in enumerate(actions_taken) if a == 2]
hold_idx = [i for i, a in enumerate(actions_taken) if a == 0]

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 20), gridspec_kw={'height_ratios':[2,1,1,1]})

# Capital con acciones
axes[0].plot(capital_history, label="Capital del Agente", linewidth=2)
axes[0].scatter(buy_idx, [capital_history[i] for i in buy_idx], marker='^', color='green', label='Buy', zorder=5)
axes[0].scatter(sell_idx, [capital_history[i] for i in sell_idx], marker='v', color='red', label='Sell', zorder=5)
axes[0].scatter(hold_idx, [capital_history[i] for i in hold_idx], marker='.', color='gray', label='Hold', zorder=3)
axes[0].set_title(f"Capital usando modelo {TICKER_MODEL} sobre datos de {TICKER_DATA}")
axes[0].set_ylabel("Total Asset ($)")
axes[0].legend()
axes[0].grid(True)

# Reward acumulado
axes[1].plot(dates[:len(rewards)], np.cumsum(rewards), color="green")
axes[1].set_title("Reward acumulado paso a paso")
axes[1].set_ylabel("Reward")
axes[1].set_xlabel("Fecha")
axes[1].grid(True)

# Delta capital vs reward
axes[2].scatter(portfolio_df["reward"], portfolio_df["delta_asset"], alpha=0.4, s=20, color="blue")
axes[2].axhline(0, color="gray", linestyle="--")
axes[2].axvline(0, color="gray", linestyle="--")
axes[2].set_title("Relación entre Reward y Δ Capital")
axes[2].set_xlabel("Reward obtenido en t")
axes[2].set_ylabel("Δ Capital total")
axes[2].grid(True)

# Distribución de acciones
axes[3].hist(actions_taken, bins=[-0.5,0.5,1.5,2.5], rwidth=0.7, color="purple")
axes[3].set_xticks([0,1,2])
axes[3].set_xticklabels(["Hold","Buy","Sell"])
axes[3].set_title("Distribución de acciones tomadas")
axes[3].set_ylabel("Frecuencia")
axes[3].grid(True)

plot_file = os.path.join(PLOT_DIR, f"{TICKER_MODEL}_evaluation_panel.png")
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.close()

# === Métricas finales ===
duracion = len(capital_history)
ganancia_neta = capital_history[-1] - INITIAL_BALANCE
retorno_pct = (capital_history[-1] / INITIAL_BALANCE - 1) * 100
action_counts = [actions_taken.count(i) for i in range(3)]
acciones_mantenidas_prom = np.mean([p["shares_held"] for p in portfolio_log])

print(f"\n📊 Evaluación SAC Fractional - Modelo {TICKER_MODEL}")
print(f"Capital inicial           : ${INITIAL_BALANCE:.2f}")
print(f"Capital final             : ${capital_history[-1]:.2f}")
print(f"Retorno porcentual        : {retorno_pct:.2f}%")
print(f"Acciones compradas        : {action_counts[1]}")
print(f"Acciones vendidas         : {action_counts[2]}")
print(f"Veces que mantuvo (hold)  : {action_counts[0]}")
print(f"Promedio de acciones mantenidas: {acciones_mantenidas_prom:.4f}")
print(f"Gráfico guardado en: {plot_file}")
