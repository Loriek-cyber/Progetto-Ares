"""
main.py — Ambiente RL per Assetto Corsa
Logica pura: Env Gymnasium + Training PPO.
Tutti i dati vengono letti tramite utils.driver.
"""

import os
import time

import numpy as np
import vgamepad as vg
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from utils.driver import (
    AssettoCorsaData,
    get_track_name,
    get_car_position,
    load_ai_line,
    build_kdtree,
)

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

BASE_TRACK_PATH = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks"

# ---------------------------------------------------------------------------
# Parametri di training — modifica qui per fare tuning
# ---------------------------------------------------------------------------

MAX_TYRES_OUT   = 3        # N° ruote fuori che scatena reset + penalità
PENALTY_TYRES   = -1000.0    # Penalità istantanea per troppe ruote fuori pista
MAX_DIST_RESET  = 6.0      # Distanza (m) dalla linea ideale che forza il reset
MAX_RPM_REF     = 8500.0   # RPM massimi di riferimento per normalizzazione
STEP_DELAY      = 0.005    # Secondi tra uno step e il successivo (basso = più veloce)

# Pesi del reward
W_SPEED         = 1.2      # Premia la velocità
W_RPM           = 0.8      # Premia RPM alti
W_LINE_PENALTY  = 4.0      # Penalizza la deviazione dalla linea ideale (aumentato)
LINE_EXP        = 2.5      # Esponente per la penalità di distanza (>2 = molto aggressivo)


# ---------------------------------------------------------------------------
# Ambiente Gymnasium
# ---------------------------------------------------------------------------

class AssettoCorsaEnv(gym.Env):
    """
    Ambiente Gymnasium per Assetto Corsa.
    Observation (7,): [speed_norm, dist_norm, g_lat, g_long, steer_norm, rpm_norm, tyres_out_norm]
    Action      (3,): [steer ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]]
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # --- Gamepad ---
        self.gamepad = vg.VX360Gamepad()

        # --- Telemetria (driver) ---
        self.asm = AssettoCorsaData()
        self.asm.start()

        # --- Pista e linea ideale ---
        self.track_name = get_track_name()
        print(f"[Env] Pista rilevata: {self.track_name}")

        ai_path = os.path.join(BASE_TRACK_PATH, self.track_name, "ai", "fast_lane.ai")
        if not os.path.exists(ai_path):
            raise FileNotFoundError(f"File AI non trovato: {ai_path}")

        self.ai_line = load_ai_line(ai_path)
        self.kdtree  = build_kdtree(self.ai_line)

        # --- Spazi ---
        self.action_space = spaces.Box(
            low  = np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high = np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Lettura stato
    # ------------------------------------------------------------------

    def _read_state(self) -> dict:
        """Legge telemetria + posizione e ritorna un dizionario con i valori grezzi."""
        self.asm.update()
        x, z     = get_car_position()
        dist, _  = self.kdtree.query([x, z])

        return {
            "speed"    : self.asm.speed,          # km/h
            "rpm"      : self.asm.rpm,
            "steer"    : self.asm.steerAngle,     # gradi [-180, 180]
            "g_lat"    : self.asm.accGX,
            "g_long"   : self.asm.accGY,
            "tyres_out": self.asm.tyres_out,      # int 0..4
            "dist"     : float(dist),             # metri dalla linea ideale
        }

    def _make_obs(self, state: dict) -> np.ndarray:
        """Costruisce il vettore di osservazione normalizzato."""
        return np.array([
            state["speed"]     / 300.0,          # 0..1 (auto raramente > 300 km/h)
            state["dist"]      / MAX_DIST_RESET,  # 0..1+ (>1 → reset imminente)
            state["g_lat"],                       # raw (già in g)
            state["g_long"],                      # raw
            state["steer"]     / 180.0,           # -1..1
            state["rpm"]       / MAX_RPM_REF,     # 0..1
            state["tyres_out"] / 4.0,             # 0..1
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, state: dict) -> float:
        """
        Reward shaping:
          + velocità alta      →  guida aggressiva
          + RPM alti           →  incentiva gear alto e motore al limite
          - (dist^exp) * peso  →  penalità FORTE per deviazione dalla traiettoria
          - penalità flat      →  se > MAX_TYRES_OUT ruote fuori
        """
        speed_r = state["speed"] / 10.0 * W_SPEED
        rpm_r   = (state["rpm"] / MAX_RPM_REF) * W_RPM
        dist_p  = -(state["dist"] ** LINE_EXP) * W_LINE_PENALTY

        print(dist_p)

        reward = speed_r + rpm_r + dist_p

        if state["tyres_out"] > MAX_TYRES_OUT:
            reward += PENALTY_TYRES

        return float(reward)

    # ------------------------------------------------------------------
    # Interfaccia Gymnasium
    # ------------------------------------------------------------------

    def step(self, action):
        # Applica i comandi al gamepad
        self.gamepad.left_joystick_float(x_value_float=float(action[0]), y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=float(action[1]))  # acceleratore
        self.gamepad.left_trigger_float (value_float=float(action[2]))  # freno
        self.gamepad.update()

        time.sleep(STEP_DELAY)

        state  = self._read_state()
        obs    = self._make_obs(state)
        reward = self._compute_reward(state)

        # --- Condizioni di terminazione ---
        terminated = False

        if state["tyres_out"] > MAX_TYRES_OUT:
            print(f"[RESET] {state['tyres_out']} ruote fuori | penalty={PENALTY_TYRES}")
            terminated = True

        if state["dist"] > MAX_DIST_RESET:
            print(f"[RESET] Distanza dalla traiettoria: {state['dist']:.2f} m")
            terminated = True

        return obs, reward, terminated, False, state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gamepad.reset()
        self.gamepad.update()
        time.sleep(0.5)          # dai tempo al gioco di recepire il reset
        state = self._read_state()
        return self._make_obs(state), {}

    def close(self):
        self.asm.stop()
        super().close()


# ---------------------------------------------------------------------------
# Entry point — Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = AssettoCorsaEnv()

    model_name = f"model_{env.track_name}"
    model_path = f"models/{model_name}.zip"
    os.makedirs("models", exist_ok=True)

    if os.path.exists(model_path):
        print(f"[Main] Caricamento modello salvato: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print(f"[Main] Nuovo modello per: {env.track_name}")
        model = PPO(
            policy        = "MlpPolicy",
            env           = env,
            verbose       = 1,
            learning_rate = 3e-4,
            n_steps       = 2048,
            batch_size    = 64,
            gamma         = 0.99,
            ent_coef      = 0.01,    # esplorazione
            clip_range    = 0.2,
        )

    try:
        model.learn(total_timesteps=1_000_000)
    except KeyboardInterrupt:
        print("\n[Main] Interruzione manuale — salvataggio in corso...")
    finally:
        model.save(model_path)
        env.close()
        print(f"[Main] Modello salvato in: {model_path}")