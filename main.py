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
    CheckpointSystem,
    get_track_name,
    get_car_position,
    load_ai_line,
    build_kdtree,
    send_reset_to_ac,
)

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

BASE_TRACK_PATH = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks"

# ---------------------------------------------------------------------------
# Parametri di training — modifica qui per fare tuning
# ---------------------------------------------------------------------------

MAX_TYRES_OUT   = 3          # N° ruote fuori che scatena reset + penalità
PENALTY_TYRES   = -2000.0   # Penalità istantanea per troppe ruote fuori pista
MAX_DIST_RESET  = 6.0        # Distanza (m) dalla linea ideale che forza il reset
MAX_RPM_REF     = 8500.0     # RPM massimi di riferimento per normalizzazione
STEP_DELAY      = 0.001      # Secondi tra uno step e il successivo

# Pesi del reward principale
W_SPEED         = 1.2        # Premia la velocità
W_RPM           = 0.8        # Premia RPM alti
W_LINE_PENALTY  = 4.0        # Penalizza la deviazione dalla linea ideale
LINE_EXP        = 2.5        # Esponente penalità distanza (>2 = aggressivo)

# Pesi reward checkpoint / corner braking
W_PROGRESS      = 5.0        # Moltiplica il reward di avanzamento checkpoint
W_BACKTRACK     = 3.0        # Moltiplica la penalità di retrocessione
W_BRAKING       = 6.0        # Penalità per entrare troppo veloce in curva
CORNER_DIST_REF = 200.0      # Distanza di riferimento per normalizzazione curva (m)

# Decellerazione fisica massima ragionevole per calcolo braking (m/s²)
MAX_BRAKE_DECEL = 18.0       # ~1.8g di frenata massima
CAR_DAMAGE_MAX = 1

# ---------------------------------------------------------------------------
# Ambiente Gymnasium
# ---------------------------------------------------------------------------

class AssettoCorsaEnv(gym.Env):
    """
    Ambiente Gymnasium per Assetto Corsa con sistema a checkpoint e corner braking.

    Observation (10,):
      [0] speed_norm          velocità normalizzata (0..1)
      [1] dist_norm           distanza dalla linea ideale normalizzata
      [2] g_lat               accelerazione laterale (g)
      [3] g_long              accelerazione longitudinale (g)
      [4] steer_norm          sterzo normalizzato (-1..1)
      [5] rpm_norm            RPM normalizzati (0..1)
      [6] tyres_out_norm      ruote fuori (0..1)
      [7] corner_dist_norm    distanza alla prossima curva normalizzata (0..1)
      [8] corner_speed_norm   velocità target alla curva normalizzata (0..1)
      [9] speed_excess_norm   eccesso di velocità rispetto al braking point (0..1+)

    Action (3,):
      [0] steer      ∈ [-1, 1]
      [1] throttle   ∈ [ 0, 1]
      [2] brake      ∈ [ 0, 1]
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # --- Gamepad ---
        self.gamepad = vg.VX360Gamepad()

        # --- Telemetria ---
        self.asm = AssettoCorsaData()
        self.asm.start()

        # --- Pista e linea ideale ---
        self.track_name = get_track_name()
        print(f"[Env] Pista rilevata: {self.track_name}")

        ai_path = os.path.join(BASE_TRACK_PATH, self.track_name, "ai", "fast_lane.ai")
        if not os.path.exists(ai_path):
            raise FileNotFoundError(f"File AI non trovato: {ai_path}")

        self.ai_line   = load_ai_line(ai_path)
        self.kdtree    = build_kdtree(self.ai_line)
        self.checkpoints = CheckpointSystem(self.ai_line, self.kdtree)

        # --- Spazi ---
        self.action_space = spaces.Box(
            low  = np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high = np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Lettura stato
    # ------------------------------------------------------------------

    def _read_state(self) -> dict:
        """Legge telemetria, posizione e info checkpoint. Ritorna stato grezzo."""
        self.asm.update()

        x, z = get_car_position()

        # KDTree query per distanza dalla linea ideale
        dist, _ = self.kdtree.query([x, z])

        # Aggiorna sistema checkpoint (avanzamento + rilevamento curva)
        cp_info = self.checkpoints.update(x, z)

        return {
            "speed"           : self.asm.speed,       # km/h
            "rpm"             : self.asm.rpm,
            "steer"           : self.asm.steerAngle,  # gradi
            "g_lat"           : self.asm.accGX,
            "g_long"          : self.asm.accGY,
            "tyres_out"       : self.asm.tyres_out,   # int 0..4
            "car_damage"      : self.asm.car_damage_total,  # somma danni carrozzeria
            "dist"            : float(dist),           # m dalla linea
            "progress_reward" : cp_info["progress_reward"],
            "backtrack_penalty": cp_info["backtrack_penalty"],
            "checkpoint_hit"  : cp_info["checkpoint_hit"],
            "corner_dist_m"   : cp_info["corner_dist_m"],   # m alla prossima curva
            "corner_speed"    : cp_info["corner_speed"],     # km/h target alla curva
        }

    def _make_obs(self, state: dict) -> np.ndarray:
        """Costruisce il vettore di osservazione normalizzato (10,)."""
        speed_ms       = state["speed"] / 3.6         # km/h → m/s
        corner_speed_ms = state["corner_speed"] / 3.6

        # Calcolo eccesso di velocità rispetto al braking point:
        # Per frenare da v_now a v_corner in dist_corner metri servono
        # a = (v_now² - v_corner²) / (2 * dist_corner)
        # Se a > MAX_BRAKE_DECEL → stai entrando troppo forte in curva
        d = max(state["corner_dist_m"], 1.0)  # evita divisione per 0
        required_decel = max(
            (speed_ms**2 - corner_speed_ms**2) / (2.0 * d), 0.0
        )
        speed_excess = required_decel / MAX_BRAKE_DECEL  # 0 = ok, >1 = missile

        return np.array([
            state["speed"]      / 300.0,           # [0] speed_norm
            state["dist"]       / MAX_DIST_RESET,  # [1] dist_norm
            state["g_lat"],                         # [2] g_lat
            state["g_long"],                        # [3] g_long
            state["steer"]      / 180.0,            # [4] steer_norm
            state["rpm"]        / MAX_RPM_REF,      # [5] rpm_norm
            state["tyres_out"]  / 4.0,              # [6] tyres_out_norm
            state["corner_dist_m"] / CORNER_DIST_REF,  # [7] corner_dist_norm
            state["corner_speed"]  / 300.0,         # [8] corner_speed_norm
            np.clip(speed_excess, 0.0, 3.0),        # [9] speed_excess_norm
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, state: dict) -> float:
        """
        Reward shaping completo:
          + velocità alta          → incentiva spingere forte
          + RPM alti               → incentiva gear alto
          + progresso checkpoint   → segui la traiettoria in avanti
          - penalità linea         → non uscire dalla traiettoria
          - backtrack              → non tornare indietro
          - braking penalty        → rallenta prima delle curve
          - tyres out              → non uscire dalla pista
        """
        speed_ms       = state["speed"] / 3.6
        corner_speed_ms = state["corner_speed"] / 3.6
        d              = max(state["corner_dist_m"], 1.0)

        # --- Componenti positivi ---
        speed_r    = state["speed"] / 10.0 * W_SPEED
        rpm_r      = (state["rpm"] / MAX_RPM_REF) * W_RPM
        progress_r = state["progress_reward"] * W_PROGRESS

        # --- Componenti negativi ---
        dist_p      = -(state["dist"] ** LINE_EXP) * W_LINE_PENALTY
        backtrack_p = state["backtrack_penalty"] * W_BACKTRACK

        # Penalità braking: quanto stai eccedendo la decelerazione fisica massima
        required_decel = max(
            (speed_ms**2 - corner_speed_ms**2) / (2.0 * d), 0.0
        )
        speed_excess = required_decel / MAX_BRAKE_DECEL
        # Penalità graduata: entra solo quando >0.5, cresce esponenzialmente
        if speed_excess > 0.5:
            braking_p = -((speed_excess - 0.5) ** 2) * W_BRAKING
        else:
            braking_p = 0.0

        reward = speed_r + rpm_r + progress_r + dist_p + backtrack_p + braking_p

        # --- Penalità piatta per uscita di pista ---
        if state["tyres_out"] > MAX_TYRES_OUT:
            reward += PENALTY_TYRES

        return float(reward)

    # ------------------------------------------------------------------
    # Interfaccia Gymnasium
    # ------------------------------------------------------------------

    def step(self, action):
        self.gamepad.left_joystick_float(x_value_float=float(action[0]), y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=float(action[1]))  # gas
        self.gamepad.left_trigger_float(value_float=float(action[2]))   # freno
        self.gamepad.update()

        time.sleep(STEP_DELAY)

        state  = self._read_state()
        obs    = self._make_obs(state)
        reward = self._compute_reward(state)

        # --- Log di debug ---
        if state["checkpoint_hit"]:
            print(f"[CP] Checkpoint! reward_progress={state['progress_reward']:.3f} "
                  f"| corner_in={state['corner_dist_m']:.0f}m "
                  f"@ {state['corner_speed']:.0f}km/h")

        # --- Condizioni di terminazione ---
        terminated = False

        if state["tyres_out"] >= MAX_TYRES_OUT:
            print(f"[RESET] {state['tyres_out']} ruote fuori pista")
            terminated = True

        if state["car_damage"] >= CAR_DAMAGE_MAX:
            print(f"[RESET] Danno carrozzeria: {state['car_damage']:.1f}")
            terminated = True

        if state["dist"] > MAX_DIST_RESET:
            print(f"[RESET] Distanza traiettoria: {state['dist']:.2f} m")
            terminated = True

        if terminated:
            send_reset_to_ac()   # Ctrl+R in game

        return obs, reward, terminated, False, state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gamepad.reset()
        self.gamepad.update()
        self.checkpoints.reset()
        time.sleep(0.5)
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
            ent_coef      = 0.01,
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