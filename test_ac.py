"""
test_ac.py — Testa modelli RL su Assetto Corsa in modalità inference
=====================================================================
Carica uno o più modelli PPO e li esegue su AC raccogliendo statistiche:
  - Tempo giro / distanza percorsa
  - Velocità media e massima
  - Distanza media dalla AI line
  - Numero di reset

Uso:
    python test_ac.py                          # testa tutti i modelli trovati
    python test_ac.py --model models/model_monza.zip
    python test_ac.py --model models_sim/ppo_sim_monza_best/best_model.zip
    python test_ac.py --episodes 3 --steps 2000
"""

import os
import glob
import argparse
import time
import math

import numpy as np
import vgamepad as vg
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
# Configurazione
# ---------------------------------------------------------------------------

BASE_TRACK_PATH = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks"
MAX_DIST_RESET  = 8.0      # m — distanza reset (identica a main.py)
MAX_TYRES_OUT   = 3
CAR_DAMAGE_MAX  = 1
STEP_DELAY      = 0.001    # secondi tra step

# Dove cercare i modelli
MODEL_SEARCH_PATHS = [
    "models/*.zip",
    "models_sim/*.zip",
    "models_sim/*_best/best_model.zip",
]


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def find_all_models() -> list[str]:
    """Cerca tutti i file .zip di modelli nelle cartelle standard."""
    found = []
    for pattern in MODEL_SEARCH_PATHS:
        found.extend(glob.glob(pattern))
    # Deduplica e ordina
    return sorted(set(found))


def print_separator(char="-", width=60):
    print(char * width)


def format_stats(stats: dict) -> str:
    return (
        f"  Steps        : {stats['steps']}\n"
        f"  Velocità med : {stats['speed_avg']:.1f} km/h\n"
        f"  Velocità max : {stats['speed_max']:.1f} km/h\n"
        f"  Dist. AI med : {stats['dist_avg']:.3f} m\n"
        f"  Dist. AI max : {stats['dist_max']:.3f} m\n"
        f"  Checkpoints  : {stats['checkpoints']}\n"
        f"  Reset        : {stats['resets']}\n"
        f"  Reward tot   : {stats['reward_total']:.1f}\n"
        f"  Durata       : {stats['elapsed']:.1f} s"
    )


# ---------------------------------------------------------------------------
# Classe tester
# ---------------------------------------------------------------------------

class ACModelTester:
    """Esegue un modello PPO su AC e raccoglie statistiche."""

    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        self.asm     = AssettoCorsaData()
        self.asm.start()

        track_name = get_track_name()
        print(f"[Tester] Pista: {track_name}")

        ai_path = os.path.join(BASE_TRACK_PATH, track_name, "ai", "fast_lane.ai")
        if not os.path.exists(ai_path):
            raise FileNotFoundError(f"AI line non trovata: {ai_path}")

        self.ai_line     = load_ai_line(ai_path)
        self.kdtree      = build_kdtree(self.ai_line)
        self.checkpoints = CheckpointSystem(self.ai_line, self.kdtree)

        # PD state (safety blend identico a main.py)
        self._prev_heading_err = 0.0

    # ------------------------------------------------------------------
    # PD + safety blend
    # ------------------------------------------------------------------

    def _pd_steer(self, heading_err: float) -> float:
        p = 0.4 * heading_err
        d = 0.15 * (heading_err - self._prev_heading_err)
        self._prev_heading_err = heading_err
        return float(np.clip(p + d, -1.0, 1.0))

    def _blend_steer(self, agent: float, pd: float,
                     dist: float, herr: float) -> tuple[float, float]:
        dn = dist / MAX_DIST_RESET
        bd = float(np.clip((dn - 0.4) / 0.45, 0.0, 1.0))
        hn = abs(herr) / math.pi
        bh = float(np.clip((hn - 0.35) / 0.40, 0.0, 1.0))
        bf = max(bd, bh)
        return float(np.clip((1 - bf) * agent + bf * pd, -1.0, 1.0)), bf

    # ------------------------------------------------------------------
    # Observation builder (identico a main.py)
    # ------------------------------------------------------------------

    def _build_obs(self, speed_kmh: float, dist: float, g_lat: float,
                   g_long: float, rpm: float, tyres_out: int,
                   heading_err: float, corner_dist: float,
                   corner_speed: float, blend: float) -> np.ndarray:
        speed_ms        = speed_kmh / 3.6
        corner_speed_ms = corner_speed / 3.6
        d               = max(corner_dist, 1.0)
        req_decel       = max((speed_ms**2 - corner_speed_ms**2) / (2.0*d), 0.0)
        speed_excess    = req_decel / 18.0

        return np.array([
            speed_kmh  / 300.0,
            dist       / MAX_DIST_RESET,
            g_lat,
            g_long,
            rpm        / 8500.0,
            tyres_out  / 4.0,
            float(np.clip(heading_err / math.pi, -1.0, 1.0)),
            corner_dist / 200.0,
            float(np.clip(speed_excess, 0.0, 3.0)),
            blend,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Esegui un episodio
    # ------------------------------------------------------------------

    def run_episode(self, model: PPO, max_steps: int = 2000) -> dict:
        """Esegue un episodio e ritorna le statistiche."""
        self.checkpoints.reset()
        self._prev_heading_err = 0.0

        speeds, dists = [], []
        total_reward  = 0.0
        resets        = 0
        checkpoints   = 0
        t0            = time.time()

        # Observation iniziale
        self.asm.update()
        x, z = get_car_position()
        dist0, _ = self.kdtree.query([x, z])
        cp0 = self.checkpoints.update(x, z)
        obs = self._build_obs(
            float(self.asm.speed), float(dist0),
            float(self.asm.accGX), float(self.asm.accGY),
            float(self.asm.rpm), int(self.asm.tyres_out),
            0.0, cp0["corner_dist_m"], cp0["corner_speed"], 0.0
        )

        for step in range(max_steps):
            # Inferenza
            action, _ = model.predict(obs, deterministic=True)

            # Heading error
            self.asm.update()
            x, z = get_car_position()
            dist_pre, _ = self.kdtree.query([x, z])
            cp_pre = self.checkpoints.update(x, z)
            raw_err = self.asm.heading - cp_pre["ideal_heading_rad"]
            herr = (raw_err + math.pi) % (2 * math.pi) - math.pi

            # Steer blend
            pd    = self._pd_steer(herr)
            steer, blend = self._blend_steer(
                float(action[2]), pd, float(dist_pre), herr
            )

            # Invia comandi
            self.gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
            self.gamepad.right_trigger_float(value_float=float(action[0]))
            self.gamepad.left_trigger_float(value_float=float(action[1]))
            self.gamepad.update()
            time.sleep(STEP_DELAY)

            # Leggi telemetria
            self.asm.update()
            x, z = get_car_position()
            dist, _ = self.kdtree.query([x, z])
            cp = self.checkpoints.update(x, z)
            raw_err = self.asm.heading - cp["ideal_heading_rad"]
            herr_post = (raw_err + math.pi) % (2 * math.pi) - math.pi

            speed  = float(self.asm.speed)
            speeds.append(speed)
            dists.append(float(dist))
            if cp["checkpoint_hit"]:
                checkpoints += 1

            # Reward semplificato per statistiche
            total_reward += speed / 10.0 - (dist ** 2.5) * 100.0

            # Check reset
            terminated = False
            if int(self.asm.tyres_out) >= MAX_TYRES_OUT:
                terminated = True
                resets += 1
            if float(self.asm.car_damage_total) >= CAR_DAMAGE_MAX:
                terminated = True
                resets += 1
            if float(dist) > MAX_DIST_RESET:
                terminated = True
                resets += 1

            if terminated:
                send_reset_to_ac(2)
                self.checkpoints.reset()
                self._prev_heading_err = 0.0
                time.sleep(1.0)

            # Nuova obs
            obs = self._build_obs(
                speed, float(dist),
                float(self.asm.accGX), float(self.asm.accGY),
                float(self.asm.rpm), int(self.asm.tyres_out),
                herr_post, cp["corner_dist_m"], cp["corner_speed"], blend
            )

        elapsed = time.time() - t0
        return {
            "steps"        : max_steps,
            "speed_avg"    : float(np.mean(speeds)) if speeds else 0.0,
            "speed_max"    : float(np.max(speeds)) if speeds else 0.0,
            "dist_avg"     : float(np.mean(dists)) if dists else 0.0,
            "dist_max"     : float(np.max(dists)) if dists else 0.0,
            "checkpoints"  : checkpoints,
            "resets"       : resets,
            "reward_total" : total_reward,
            "elapsed"      : elapsed,
        }

    def close(self):
        self.gamepad.reset()
        self.gamepad.update()
        self.asm.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test modelli PPO su AC")
    parser.add_argument("--model",    type=str, default=None,
                        help="Percorso del modello .zip da testare. "
                             "Se omesso, testa tutti i modelli trovati.")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Episodi di test per modello (default: 2)")
    parser.add_argument("--steps",    type=int, default=3000,
                        help="Step per episodio (default: 3000)")
    args = parser.parse_args()

    # Trova modelli da testare
    if args.model:
        models_to_test = [args.model]
    else:
        models_to_test = find_all_models()
        if not models_to_test:
            print("[Test] Nessun modello trovato in:", MODEL_SEARCH_PATHS)
            return

    print_separator("=")
    print(f"  Modelli da testare : {len(models_to_test)}")
    print(f"  Episodi per modello: {args.episodes}")
    print(f"  Step per episodio  : {args.steps}")
    print_separator("=")

    tester = ACModelTester()
    results = {}

    for model_path in models_to_test:
        if not os.path.exists(model_path):
            print(f"[Test] SKIP (non trovato): {model_path}")
            continue

        print(f"\n[Test] Caricamento: {model_path}")
        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"[Test] ERRORE caricamento: {e}")
            continue

        ep_stats = []
        for ep in range(args.episodes):
            print(f"  Episodio {ep+1}/{args.episodes}...")
            stats = tester.run_episode(model, max_steps=args.steps)
            ep_stats.append(stats)
            print(f"    → v_avg={stats['speed_avg']:.1f} km/h  "
                  f"dist_avg={stats['dist_avg']:.3f} m  "
                  f"cp={stats['checkpoints']}  reset={stats['resets']}")

        # Media su tutti gli episodi
        avg = {
            k: float(np.mean([s[k] for s in ep_stats]))
            for k in ep_stats[0] if isinstance(ep_stats[0][k], (int, float))
        }
        results[model_path] = avg

        print(f"\n  [Riepilogo] {os.path.basename(model_path)}")
        print(format_stats(avg))

    # Ranking finale
    if len(results) > 1:
        print()
        print_separator("=")
        print("  CLASSIFICA (per velocità media)")
        print_separator("=")
        ranked = sorted(results.items(), key=lambda x: x[1]["speed_avg"], reverse=True)
        for i, (path, s) in enumerate(ranked, 1):
            name = os.path.basename(os.path.dirname(path) + "/" + os.path.basename(path))
            print(f"  #{i} {name}")
            print(f"     v_avg={s['speed_avg']:.1f} km/h  "
                  f"dist_avg={s['dist_avg']:.3f} m  "
                  f"cp={s['checkpoints']:.0f}  reset={s['resets']:.0f}")

    tester.close()


if __name__ == "__main__":
    main()
