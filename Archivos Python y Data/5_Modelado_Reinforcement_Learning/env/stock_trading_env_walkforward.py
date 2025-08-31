# env/stock_trading_env_walkforward.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from datetime import datetime


# ================= MODELO LSTM (igual al que entrenaste) =================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ================= ENV =================
class StockTradingEnvWithLSTM(gym.Env):
    metadata = {'render.modes': ['human']}

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)

    def __init__(self, data_path, lstm_model_path, scaler_path,
                 ticker="AAPL", initial_balance=10000, lookback=60, seed=42):
        super().__init__()
        self.ticker = ticker
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.lookback = lookback
        self._seed = seed
        np.random.seed(seed)

        # ================= CARGAR DATOS =================
        # Cargar datos desde .npy
        arr = np.load(data_path, allow_pickle=True)  # shape=(num_samples, lookback, features)
        self.data = arr  # mantener como array
        self.n_steps = self.data.shape[0]  # número total de pasos
        self.lookbook_window = lookback

        # Variables de features y escalador
        # Solo usar las 11 features originales
        self.features = [
            "Close", "High", "Low", "Volume",
            "retorno_simple", "retorno_log",
            "SMA_20", "SMA_50",
            "RSI_14", "MACD", "MACD_signal"
        ]
        self.n_features = len(self.features)
        self.scaler = joblib.load(scaler_path)

        # Cargar modelo LSTM
        self.model = LSTMModel(input_size=self.n_features)
        state_dict = torch.load(lstm_model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Acciones: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observación: ventana de lookback + predicción LSTM como feature adicional
        # MANTENER el formato original del modelo entrenado
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.lookback, self.n_features + 1),
                                            # lookback x (features + LSTM pred)
                                            dtype=np.float32)

        self.current_step = 0  # Empezar desde 0
        self.position = 0  # 1 si comprado, 0 si no
        self.total_reward = 0
        self.prev_total_value = self.initial_balance

    def reset(self, seed=None, options=None):
        # Compatibilidad con gymnasium API
        if seed is not None:
            self.seed(seed)

        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0  # Empezar desde el principio
        self.total_reward = 0
        self.prev_total_value = self.initial_balance
        obs = self._get_observation()
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Gymnasium API requiere retornar (observation, info)
        info = {
            "total_value": self.initial_balance,
            "balance": self.balance,
            "positions": self.position,
            "total_assets": self.initial_balance,
            "current_price": self._get_current_price() if self.current_step < self.n_steps else 0
        }

        return obs, info

    def _get_current_price(self):
        """Extraer el precio actual de los datos con validación"""
        try:
            if self.current_step >= self.n_steps:
                # Si estamos al final, usar el último precio válido
                return self.prev_total_value / max(self.position, 1) if self.position > 0 else 100.0

            # Asumiendo que Close es el primer feature (índice 0)
            current_window = self.data[self.current_step]  # shape: (lookback, n_features)
            # Tomar el último precio de la ventana (el más reciente)
            current_price = current_window[-1, 0]  # último timestep, feature 0 (Close)

            # Validar que el precio sea un número válido y positivo
            if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
                # Buscar el precio válido más reciente en la ventana
                for i in range(len(current_window) - 1, -1, -1):
                    price = current_window[i, 0]
                    if np.isfinite(price) and price > 0:
                        return float(price)

                # Si no encontramos ningún precio válido, usar un precio por defecto
                return 100.0

            return float(current_price)
        except Exception as e:
            print(f"Warning: Error getting current price at step {self.current_step}: {e}")
            return 100.0  # Precio por defecto

    def _get_lstm_prediction(self):
        """Generar predicción LSTM usando la ventana actual de forma segura"""
        try:
            if self.current_step >= self.n_steps:
                return 0.0

            current_window = self.data[self.current_step]  # shape: (lookback, n_features)

            # Validar que la ventana no tenga valores inválidos
            current_window = np.nan_to_num(current_window, nan=0.0, posinf=1.0, neginf=-1.0)

            # Escalar los datos de forma segura
            scaled_window = []
            for step in current_window:
                try:
                    step_scaled = self.scaler.transform(step.reshape(1, -1))[0]
                    step_scaled = np.nan_to_num(step_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
                    scaled_window.append(step_scaled)
                except:
                    # Si hay error en escalado, usar ceros
                    scaled_window.append(np.zeros(self.n_features))

            scaled_window = np.array(scaled_window)

            # Convertir a tensor y agregar dimensión de batch
            input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0)

            # Predecir de forma segura
            with torch.no_grad():
                prediction = self.model(input_tensor).item()

            # Validar predicción
            if not np.isfinite(prediction):
                prediction = self._get_current_price()  # Usar precio actual como fallback

            return float(prediction)

        except Exception as e:
            # En caso de cualquier error, retornar el precio actual
            try:
                return self._get_current_price()
            except:
                return 100.0  # Valor por defecto

    def _get_observation(self):
        """Crear observación en formato de ventana + predicción LSTM"""
        if self.current_step >= self.n_steps:
            # Si estamos al final, devolver observación vacía
            return np.zeros((self.lookback, self.n_features + 1), dtype=np.float32)

        try:
            # Obtener la ventana completa actual
            current_window = self.data[self.current_step]  # shape: (lookback, n_features)

            # Validar que la ventana no contenga valores inválidos
            current_window = np.nan_to_num(current_window, nan=0.0, posinf=1.0, neginf=-1.0)

            # Escalar toda la ventana de forma segura
            scaled_window = []
            for step in current_window:
                try:
                    step_reshaped = step.reshape(1, -1)
                    step_scaled = self.scaler.transform(step_reshaped)[0]
                    step_scaled = np.nan_to_num(step_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
                    scaled_window.append(step_scaled)
                except Exception as e:
                    # Si hay error en el escalado, usar valores por defecto
                    scaled_window.append(np.zeros(self.n_features))

            scaled_window = np.array(scaled_window)

            # Obtener predicción LSTM de forma segura
            try:
                lstm_pred = self._get_lstm_prediction()
                current_price = self._get_current_price()

                # Crear diferencia porcentual entre predicción y precio actual
                if current_price != 0 and np.isfinite(current_price) and np.isfinite(lstm_pred):
                    pred_diff = (lstm_pred - current_price) / current_price
                    pred_diff = np.clip(pred_diff, -1.0, 1.0)  # Limitar a [-1, 1]
                else:
                    pred_diff = 0.0

            except Exception as e:
                pred_diff = 0.0

            # Validar pred_diff
            if not np.isfinite(pred_diff):
                pred_diff = 0.0

            # Agregar la predicción como una columna adicional a toda la ventana
            pred_column = np.full((self.lookback, 1), pred_diff, dtype=np.float32)

            # Observación final: ventana escalada + columna de predicción
            obs = np.concatenate([scaled_window, pred_column], axis=1)

            # Validación final de la observación
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

            return obs.astype(np.float32)

        except Exception as e:
            print(f"Warning: Error creating observation at step {self.current_step}: {e}")
            # Retornar observación por defecto en caso de error
            return np.zeros((self.lookback, self.n_features + 1), dtype=np.float32)

    def step(self, action):
        if self.current_step >= self.n_steps:
            return self._get_observation(), 0.0, True, True, {}

        done = False
        truncated = False  # Para gymnasium API
        reward = 0.0

        # Obtener precio actual con validación
        try:
            current_price = self._get_current_price()
            # Validar que el precio sea válido
            if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
                current_price = self.prev_total_value / max(self.position, 1) if self.position > 0 else 100.0
        except:
            current_price = 100.0  # Precio por defecto

        # Validar que current_price es un número válido
        current_price = float(current_price)
        if not np.isfinite(current_price) or current_price <= 0:
            current_price = 100.0

        # Ejecutar acción con validaciones
        if action == 1:  # Buy
            if self.position == 0 and self.balance > 0 and current_price > 0:
                self.position = self.balance / current_price
                self.balance = 0.0
        elif action == 2:  # Sell
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0.0

        # Calcular valor total con validaciones
        total_value = self.balance + (self.position * current_price)

        # Validar que todos los valores sean finitos
        if not np.isfinite(total_value):
            total_value = self.prev_total_value

        if not np.isfinite(self.prev_total_value):
            self.prev_total_value = self.initial_balance

        # Calcular reward de forma segura
        try:
            reward = total_value - self.prev_total_value

            # Validar reward
            if not np.isfinite(reward):
                reward = 0.0

            # Limitar reward a un rango razonable (-1000, 1000)
            reward = np.clip(reward, -1000.0, 1000.0)

        except:
            reward = 0.0

        # Actualizar prev_total_value solo si es válido
        if np.isfinite(total_value):
            self.prev_total_value = total_value

        self.total_reward += reward

        # Avanzar al siguiente step
        self.current_step += 1

        # Verificar si terminamos
        if self.current_step >= self.n_steps:
            done = True

        # Obtener siguiente observación
        obs = self._get_observation()

        info = {
            "total_value": float(total_value),
            "balance": float(self.balance),
            "positions": float(self.position),
            "total_assets": float(total_value),
            "current_price": float(current_price)
        }

        # Gymnasium API requiere 5 valores: obs, reward, terminated, truncated, info
        return obs, float(reward), done, truncated, info

    def render(self, mode='human'):
        if self.current_step > 0:
            current_price = self._get_current_price()
            total_value = self.balance + self.position * current_price
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
                  f"Position: {self.position:.4f}, Total: {total_value:.2f}, "
                  f"Price: {current_price:.2f}")