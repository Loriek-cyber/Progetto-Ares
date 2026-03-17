"""
ac_agent.py — Agente AI per Assetto Corsa
==========================================
Legge la telemetria da AC via Shared Memory (driver.py),
passa le osservazioni al modello addestrato (pilot_model.pth),
e converte l'output in input del controller virtuale Xbox (vgamepad).

Dipendenze:
    pip install torch vgamepad pyautogui scipy numpy

Avvio:
    python ac_agent.py --model pilot_model.pth --ai path/to/fast_lane.ai
    python ac_agent.py --model pilot_model.pth --ai path/to/fast_lane.ai --dashboard

Struttura osservazione (7 valori, uguale al training):
    [0] speed normalizzata          (speed_kmh / 100)
    [1] accel_g longitudinale       (accGX, già in g)
    [2] heading normalizzato        (heading * 0.1)
    [3] lat_vel normalizzata        (localVelocityX / 10)
    [4] dist dal centro pista norm  (dist_m / 50)
    [5] angolo rispetto AI line     (-pi..pi / pi)
    [6] direzione prossima curva    (-1, 0, +1)
"""

import argparse
import math
import os
import sys
import time
import json
import socket
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import vgamepad as vg

# Driver AC (file fornito dall'utente)
from utils.driver import (
    AssettoCorsaData,
    CheckpointSystem,
    get_car_position,
    load_ai_line,
    build_kdtree,
    send_reset_to_ac,
)

# ---------------------------------------------------------------------------
# Rete neurale (identica al training)
# ---------------------------------------------------------------------------

class PilotNet(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )
        self.actor  = nn.Sequential(nn.Linear(128, 2), nn.Tanh())
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        return self.actor(x), self.critic(x)


# ---------------------------------------------------------------------------
# Controller virtuale
# ---------------------------------------------------------------------------

class VirtualController:
    """
    Wrappa vgamepad (Xbox 360 virtuale via ViGEm).
    throttle/brake → trigger sinistro/destro (0..1)
    steer          → stick sinistro X (-1..1)
    """
    def __init__(self):
        self.pad = vg.VX360Gamepad()
        print("[Ctrl] Controller virtuale Xbox attivo.")

    def apply(self, throttle: float, brake: float, steer: float):
        # Trigger: 0..255
        self.pad.left_trigger(val=int(max(0.0, min(1.0, brake))    * 255))
        self.pad.right_trigger(val=int(max(0.0, min(1.0, throttle)) * 255))
        # Stick sinistro X: -32768..32767
        steer_clamped = max(-1.0, min(1.0, steer))
        self.pad.left_joystick(x_value=int(steer_clamped * 32767), y_value=0)
        self.pad.update()

    def release(self):
        self.pad.left_trigger(0)
        self.pad.right_trigger(0)
        self.pad.left_joystick(0, 0)
        self.pad.update()


# ---------------------------------------------------------------------------
# Conversione output modello → throttle / brake / steer
# ---------------------------------------------------------------------------

def decode_action(action: np.ndarray) -> tuple[float, float, float]:
    """
    action[0] ∈ [-1, 1]:  > 0 = accelera, < 0 = frena
    action[1] ∈ [-1, 1]:  sterzo (- sinistra, + destra)
    """
    raw_accel = float(action[0])
    steer     = float(action[1])

    throttle = max(0.0, raw_accel)
    brake    = max(0.0, -raw_accel)

    return throttle, brake, steer


# ---------------------------------------------------------------------------
# Costruzione osservazione (deve specchiarsi esattamente col training)
# ---------------------------------------------------------------------------

def build_observation(
    ac: AssettoCorsaData,
    cp: CheckpointSystem,
    x: float,
    z: float,
) -> tuple[np.ndarray, dict]:
    """
    Costruisce il vettore di osservazione a 7 elementi dal checkpoint system
    e dalla telemetria AC, nel medesimo formato usato durante il training.
    """
    cp_info = cp.update(x, z)

    speed_norm   = ac.speed / 100.0
    accel_g      = float(ac.accGX)

    # Heading auto (AC usa radianti, campo 'heading' in rad)
    heading_ac   = float(ac.heading)

    # Velocità laterale (localVelocityX in m/s nel ref frame auto)
    lat_vel_norm = float(ac.localVelocityX) / 10.0

    # Distanza dal centro pista (stima dall'indice AI line più vicino)
    nearest_idx  = cp_info["nearest_idx"]
    ai_pos_xz    = cp.ai_line[nearest_idx, [0, 2]]
    dist_m       = float(np.sqrt((x - ai_pos_xz[0])**2 + (z - ai_pos_xz[1])**2))
    dist_norm    = dist_m / 50.0

    # Angolo rispetto alla AI line
    ideal_h      = cp_info["ideal_heading_rad"]
    angle        = (ideal_h - heading_ac + math.pi) % (2 * math.pi) - math.pi
    angle_norm   = angle / math.pi

    # Direzione prossima curva
    corner_dist  = cp_info["corner_dist_m"]
    corner_speed = cp_info["corner_speed"]
    current_speed_target = float(cp.ai_line[nearest_idx, 3]) if nearest_idx < len(cp.ai_line) else 100.0
    if corner_dist < 200.0 and current_speed_target > 1.0:
        drop = (current_speed_target - corner_speed) / current_speed_target
        curve_dir = float(np.sign(drop)) if drop > 0.05 else 0.0
    else:
        curve_dir = 0.0

    obs = np.array([
        speed_norm,
        accel_g,
        heading_ac * 0.1,
        lat_vel_norm,
        dist_norm,
        angle_norm,
        curve_dir,
    ], dtype=np.float32)

    telemetry = {
        "speed":        ac.speed,
        "gear":         ac.gear,
        "rpm":          ac.rpm,
        "throttle":     ac.throttle,
        "brake":        ac.brake,
        "dist_m":       dist_m,
        "angle_deg":    math.degrees(angle),
        "nearest_idx":  nearest_idx,
        "progress_r":   cp_info["progress_reward"],
        "corner_dist":  corner_dist,
        "corner_speed": corner_speed,
        "tyres_out":    ac.tyres_out,
        "damage":       ac.car_damage_total,
    }

    return obs, telemetry


# ---------------------------------------------------------------------------
# Server telemetria per dashboard (UDP JSON broadcast)
# ---------------------------------------------------------------------------

class TelemetryServer:
    """Invia i dati telemetrici via UDP alla dashboard locale."""
    def __init__(self, port: int = 5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ("127.0.0.1", port)

    def send(self, data: dict):
        try:
            payload = json.dumps(data).encode()
            self.sock.sendto(payload, self.addr)
        except Exception:
            pass

    def close(self):
        self.sock.close()


# ---------------------------------------------------------------------------
# Loop principale agente
# ---------------------------------------------------------------------------

def run_agent(
    model_path: str,
    ai_line_path: str,
    dashboard: bool = False,
    reset_on_stuck: bool = True,
    step_hz: int = 30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Agent] Device: {device}")

    # --- Caricamento modello ---
    model = PilotNet(input_dim=7).to(device)
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        print(f"[Agent] Modello caricato (epoca {checkpoint.get('epoch','?')}, "
              f"reward {checkpoint.get('reward', 0):.2f})")
    else:
        model.load_state_dict(checkpoint)
        print("[Agent] Modello legacy caricato.")
    model.eval()

    # --- AI Line ---
    ai_line = load_ai_line(ai_line_path)
    kdtree  = build_kdtree(ai_line)
    cp      = CheckpointSystem(ai_line, kdtree)
    print(f"[Agent] AI line caricata: {len(ai_line)} punti.")

    # --- Telemetria AC ---
    ac = AssettoCorsaData()
    ac.start()

    # --- Controller ---
    ctrl = VirtualController()

    # --- Dashboard UDP ---
    telem_server = TelemetryServer() if dashboard else None

    # --- Anti-stuck ---
    STUCK_SPEED_THRESHOLD = 5.0    # km/h sotto cui si considera stuck
    STUCK_TIME_LIMIT      = 5.0    # secondi prima di resettare
    stuck_timer           = 0.0
    step_dt               = 1.0 / step_hz

    print(f"\n[Agent] Avvio loop @ {step_hz} Hz — Ctrl+C per fermare\n")

    try:
        with torch.no_grad():
            while True:
                t_start = time.perf_counter()

                # 1. Lettura telemetria
                ac.update()
                x, z = get_car_position()

                # 2. Osservazione
                obs, telem = build_observation(ac, cp, x, z)

                # 3. Inferenza modello
                obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_t, value_t = model(obs_t)
                action = action_t.squeeze(0).cpu().numpy()

                # 4. Decodifica e invio al controller
                throttle, brake, steer = decode_action(action)
                ctrl.apply(throttle, brake, steer)

                # 5. Anti-stuck
                if reset_on_stuck:
                    if ac.speed < STUCK_SPEED_THRESHOLD:
                        stuck_timer += step_dt
                        if stuck_timer >= STUCK_TIME_LIMIT:
                            print("[Agent] Auto bloccata — reset sessione AC...")
                            ctrl.release()
                            send_reset_to_ac()
                            cp.reset()
                            stuck_timer = 0.0
                            time.sleep(3.0)
                    else:
                        stuck_timer = 0.0

                # 6. Telemetria dashboard
                if telem_server:
                    telem_server.send({
                        **telem,
                        "throttle_ai": throttle,
                        "brake_ai":    brake,
                        "steer_ai":    steer,
                        "value":       float(value_t.item()),
                        "action_raw":  action.tolist(),
                    })

                # 7. Log console ogni 60 step (~2s)
                if int(time.perf_counter() * step_hz) % 60 == 0:
                    print(
                        f"  spd={telem['speed']:5.1f}km/h  "
                        f"gear={telem['gear']}  "
                        f"thr={throttle:.2f}  "
                        f"brk={brake:.2f}  "
                        f"str={steer:+.2f}  "
                        f"dist={telem['dist_m']:.1f}m  "
                        f"ang={telem['angle_deg']:+.1f}°  "
                        f"V={float(value_t.item()):+.3f}"
                    )

                # 8. Rate limiting
                elapsed = time.perf_counter() - t_start
                sleep_t = step_dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Agent] Interruzione utente.")
    finally:
        ctrl.release()
        ac.stop()
        if telem_server:
            telem_server.close()
        print("[Agent] Spento.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente AI per Assetto Corsa")
    parser.add_argument("--model",     required=True, help="Percorso del file .pth del modello")
    parser.add_argument("--ai",        required=True, help="Percorso del file fast_lane.ai")
    parser.add_argument("--dashboard", action="store_true", help="Abilita broadcast UDP per la dashboard")
    parser.add_argument("--no-reset",  action="store_true", help="Disabilita il reset automatico se bloccato")
    parser.add_argument("--hz",        type=int, default=30, help="Frequenza loop in Hz (default: 30)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[Agent] ERRORE: modello non trovato: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.ai):
        print(f"[Agent] ERRORE: AI line non trovata: {args.ai}")
        sys.exit(1)

    run_agent(
        model_path    = args.model,
        ai_line_path  = args.ai,
        dashboard     = args.dashboard,
        reset_on_stuck= not args.no_reset,
        step_hz       = args.hz,
    )