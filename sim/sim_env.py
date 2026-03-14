"""
sim/sim_env.py — Ambiente Gymnasium simulato (senza Assetto Corsa)
==================================================================
Replica la struttura di main.py (stessa obs, action, reward) ma usa
AcSimulator invece della shared memory del gioco reale.

Vantaggi:
  - Training 100x più veloce (nessun bottleneck grafico/real-time)
  - Eseguibile in parallelo su decine di processi (SubprocVecEnv)
  - Stessa interfaccia di main.py → modello trasferibile direttamente al gioco

Observation (10,):
  [0] speed_norm         velocita' normalizzata     (0..1)
  [1] dist_norm          distanza dalla linea        (0..1+)
  [2] g_lat              accelerazione laterale      (g)
  [3] g_long             accelerazione longitudinale (g)
  [4] rpm_norm           RPM normalizzati            (0..1)
  [5] tyres_out_norm     ruote fuori                 (0..1)
  [6] heading_err_norm   heading error normalizzato  (-1..1)
  [7] corner_dist_norm   distanza prossima curva     (0..1)
  [8] speed_excess_norm  eccesso vel rispetto curva  (0..3)
  [9] blend_factor       quanto il PD sta correggendo(0..1)

Action (3,):
  [0] throttle  in [0, 1]
  [1] brake     in [0, 1]
  [2] steer     in [-1, 1]
"""

import os
import math
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.driver import (
    load_ai_line,
    build_kdtree,
    CheckpointSystem,
)
from sim.ac_sim import AcSimulator, DT, TRACK_HALF_WIDTH

# ---------------------------------------------------------------------------
# Percorso AI line (stesso di main.py)
# ---------------------------------------------------------------------------

BASE_TRACK_PATH = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks"
DEFAULT_TRACK   = "monza"

# ---------------------------------------------------------------------------
# Parametri (allineati a main.py)
# ---------------------------------------------------------------------------

MAX_TYRES_OUT  = 3
PENALTY_TYRES  = -200.0
MAX_DIST_RESET = 8.0        # m — distanza dalla AI line che causa reset
MAX_RPM_REF    = 8500.0
MAX_STEPS      = 4000       # step massimi per episodio (evita loop infiniti)

# --- Safety blend ---
STEER_KP              = 0.4
STEER_KD              = 0.15
STEER_MAX             = 1.0
STEER_BLEND_DIST_START = 0.4
STEER_BLEND_DIST_FULL  = 0.85
STEER_BLEND_HEAD_START = 0.35
STEER_BLEND_HEAD_FULL  = 0.75

# --- Pesi reward ---
W_SPEED        = 1.6
W_RPM          = 0.6
W_PROGRESS     = 6.0
W_BACKTRACK    = 6.0
W_LINE         = 100.0
LINE_EXP       = 2.5
W_HEADING      = 5.0
W_BRAKING      = 5.0
MAX_BRAKE_DECEL = 18.0
CORNER_DIST_REF = 200.0

# Perturbazione posizione al reset (simulazione start variabile)
RESET_HEADING_NOISE = 0.15   # rad — rumore sull'heading iniziale
RESET_SPEED_RANGE   = (0.0, 30.0)  # m/s — velocità iniziale random


# ---------------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------------

class SimACEnv(gym.Env):
    """
    Ambiente simulato Assetto Corsa per training parallelo.
    Non richiede il simulatore in esecuzione.
    """

    metadata = {"render_modes": []}

    def __init__(self, track_name: str = DEFAULT_TRACK):
        super().__init__()

        # --- Caricamento AI line ---
        ai_path = os.path.join(BASE_TRACK_PATH, track_name, "ai", "fast_lane.ai")
        if not os.path.exists(ai_path):
            raise FileNotFoundError(f"AI line non trovata: {ai_path}")

        self.ai_line    = load_ai_line(ai_path)
        self.kdtree     = build_kdtree(self.ai_line)
        self.checkpoints = CheckpointSystem(self.ai_line, self.kdtree)
        self.n_points   = len(self.ai_line)

        # --- Simulatore fisico ---
        self.sim = AcSimulator()

        # --- Stato PD ---
        self._prev_heading_err = 0.0

        # --- Contatore step per episodio ---
        self._step_count = 0

        # --- Spazi ---
        self.action_space = spaces.Box(
            low  = np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high = np.array([1.0, 1.0,  1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # PD + Safety blend (identico a main.py)
    # ------------------------------------------------------------------

    def _compute_pd_steer(self, heading_err: float) -> float:
        p_term = STEER_KP * heading_err
        d_term = STEER_KD * (heading_err - self._prev_heading_err)
        self._prev_heading_err = heading_err
        return float(np.clip(p_term + d_term, -STEER_MAX, STEER_MAX))

    def _apply_safety_blend(self, agent_steer: float, pd_steer: float,
                            dist: float, heading_err: float) -> tuple[float, float]:
        dist_norm = dist / MAX_DIST_RESET
        blend_dist = float(np.clip(
            (dist_norm - STEER_BLEND_DIST_START) /
            (STEER_BLEND_DIST_FULL - STEER_BLEND_DIST_START),
            0.0, 1.0
        ))
        head_norm = abs(heading_err) / math.pi
        blend_head = float(np.clip(
            (head_norm - STEER_BLEND_HEAD_START) /
            (STEER_BLEND_HEAD_FULL - STEER_BLEND_HEAD_START),
            0.0, 1.0
        ))
        blend_factor = max(blend_dist, blend_head)
        steer_final  = (1.0 - blend_factor) * agent_steer + blend_factor * pd_steer
        return float(np.clip(steer_final, -1.0, 1.0)), blend_factor

    # ------------------------------------------------------------------
    # Stato completo
    # ------------------------------------------------------------------

    def _read_state(self, sim_state: dict, blend_factor: float) -> dict:
        x, z = sim_state["x"], sim_state["z"]
        dist, _ = self.kdtree.query([x, z])
        cp_info = self.checkpoints.update(x, z)

        raw_err     = sim_state["heading"] - cp_info["ideal_heading_rad"]
        heading_err = (raw_err + math.pi) % (2 * math.pi) - math.pi

        return {
            "speed"            : sim_state["speed"],        # km/h
            "speed_ms"         : sim_state["speed_ms"],
            "rpm"              : sim_state["rpm"],
            "g_lat"            : sim_state["g_lat"],
            "g_long"           : sim_state["g_long"],
            "tyres_out"        : sim_state["tyres_out"],
            "car_damage"       : 0.0,                       # simulatore non ha danni
            "dist"             : float(dist),
            "heading_err"      : heading_err,
            "ideal_heading_rad": cp_info["ideal_heading_rad"],
            "progress_reward"  : cp_info["progress_reward"],
            "backtrack_penalty": cp_info["backtrack_penalty"],
            "checkpoint_hit"   : cp_info["checkpoint_hit"],
            "corner_dist_m"    : cp_info["corner_dist_m"],
            "corner_speed"     : cp_info["corner_speed"],
            "blend_factor"     : blend_factor,
        }

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _make_obs(self, state: dict) -> np.ndarray:
        speed_ms        = state["speed_ms"]
        corner_speed_ms = state["corner_speed"] / 3.6
        d               = max(state["corner_dist_m"], 1.0)

        required_decel = max(
            (speed_ms**2 - corner_speed_ms**2) / (2.0 * d), 0.0
        )
        speed_excess = required_decel / MAX_BRAKE_DECEL

        return np.array([
            state["speed"]        / 300.0,
            state["dist"]         / MAX_DIST_RESET,
            state["g_lat"],
            state["g_long"],
            state["rpm"]          / MAX_RPM_REF,
            state["tyres_out"]    / 4.0,
            float(np.clip(state["heading_err"] / math.pi, -1.0, 1.0)),
            state["corner_dist_m"] / CORNER_DIST_REF,
            float(np.clip(speed_excess, 0.0, 3.0)),
            state["blend_factor"],
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward (identico a main.py)
    # ------------------------------------------------------------------

    def _compute_reward(self, state: dict) -> float:
        speed_ms        = state["speed_ms"]
        corner_speed_ms = state["corner_speed"] / 3.6
        d               = max(state["corner_dist_m"], 1.0)

        speed_r    = state["speed"] / 10.0 * W_SPEED
        rpm_r      = (state["rpm"] / MAX_RPM_REF) * W_RPM
        progress_r = state["progress_reward"] * W_PROGRESS

        dist_p      = -(state["dist"] ** LINE_EXP) * W_LINE
        backtrack_p = state["backtrack_penalty"] * W_BACKTRACK
        heading_p   = -(state["heading_err"] ** 2) * W_HEADING

        required_decel = max(
            (speed_ms**2 - corner_speed_ms**2) / (2.0 * d), 0.0
        )
        speed_excess = required_decel / MAX_BRAKE_DECEL
        braking_p    = -max((speed_excess - 0.5), 0.0) ** 2 * W_BRAKING

        reward = speed_r + rpm_r + progress_r + dist_p + backtrack_p + heading_p + braking_p

        if state["tyres_out"] >= MAX_TYRES_OUT:
            reward += PENALTY_TYRES

        return float(reward)

    # ------------------------------------------------------------------
    # Interfaccia Gymnasium
    # ------------------------------------------------------------------

    def step(self, action):
        self._step_count += 1

        # Calcola heading error corrente (per PD e blend)
        x, z = self.sim.x, self.sim.z
        dist_pre, _ = self.kdtree.query([x, z])
        cp_info_pre = self.checkpoints.update(x, z)
        raw_err = self.sim.heading - cp_info_pre["ideal_heading_rad"]
        heading_err_pre = (raw_err + math.pi) % (2 * math.pi) - math.pi

        # Safety blend sterzo
        agent_steer = float(np.clip(action[2], -1.0, 1.0))
        pd_steer    = self._compute_pd_steer(heading_err_pre)
        steer_final, blend_factor = self._apply_safety_blend(
            agent_steer, pd_steer, dist_pre, heading_err_pre
        )

        # Aggiorna simulatore
        sim_state = self.sim.step(
            throttle       = float(action[0]),
            brake          = float(action[1]),
            steer          = steer_final,
            dist_from_line = dist_pre,
        )

        # Leggi stato completo
        state  = self._read_state(sim_state, blend_factor)
        obs    = self._make_obs(state)
        reward = self._compute_reward(state)

        # Condizioni di terminazione
        terminated = False
        if state["tyres_out"] >= MAX_TYRES_OUT:
            terminated = True
        if state["dist"] > MAX_DIST_RESET:
            terminated = True
        if self._step_count >= MAX_STEPS:
            terminated = True   # truncated potremmo usarlo, ma terminated va bene

        return obs, reward, terminated, False, state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_heading_err = 0.0
        self.checkpoints.reset()

        # Posizione random sulla AI line
        idx = random.randint(0, self.n_points - 1)
        x_start = float(self.ai_line[idx, 0])
        z_start = float(self.ai_line[idx, 2])

        # Heading ideale con rumore
        ideal_heading = self.checkpoints.get_ideal_heading(idx)
        heading_noise = random.uniform(-RESET_HEADING_NOISE, RESET_HEADING_NOISE)

        # Velocità iniziale random (favorisce esplorazione)
        speed_ms = random.uniform(*RESET_SPEED_RANGE)

        self.sim.reset(
            x       = x_start,
            z       = z_start,
            heading = ideal_heading + heading_noise,
            speed_ms = speed_ms,
        )

        dist_init, _ = self.kdtree.query([x_start, z_start])
        sim_state = self.sim._state()
        state = self._read_state(sim_state, 0.0)
        return self._make_obs(state), {}

    def close(self):
        super().close()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = SimACEnv()
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    total_r = 0.0
    for i in range(500):
        action = env.action_space.sample()
        action[0] = 0.7   # throttle fisso per test
        obs, r, done, _, info = env.step(action)
        total_r += r
        if done:
            print(f"  Episode terminato a step {i} | reward totale: {total_r:.1f}")
            obs, _ = env.reset()
            total_r = 0.0
    print("Self-test completato.")
