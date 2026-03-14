"""
sim/ac_sim.py — Simulatore fisico macchina da corsa
====================================================
Modello semplificato basato su:
  - Bicycle model per lo sterzo (yaw rate da angolo ruote + velocità)
  - Dinamica longitudinale: throttle → forza motore, brake → decelerazione
  - Marce automatiche: cambio su/giù basato su RPM
  - RPM: calcolati da velocità e rapporto di trasmissione corrente
  - G-force: derivate dall'accelerazione lineare e centripeta

Parametri calibrati su una berlina sportiva generica (~250 CV, ~1200 kg).
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Parametri fisici macchina
# ---------------------------------------------------------------------------

MASS          = 1200.0      # kg
WHEELBASE     = 2.55        # m  — passo
WHEEL_RADIUS  = 0.315       # m  — raggio ruota
MAX_STEER_ANG = 0.42        # rad  ≈ 24°  (1.0 → angolo massimo ruote)

# Marce: [rapporto_trasmissione_totale (gear * final_drive)]
# final_drive = 3.45, gear ratios tipici 6-speed
GEAR_RATIOS = [
    0.0,     # neutro (unused)
    13.19,   # 1ª  (3.82 * 3.45)
     8.14,   # 2ª  (2.36 * 3.45)
     5.83,   # 3ª  (1.69 * 3.45)
     4.55,   # 4ª  (1.32 * 3.45)
     3.59,   # 5ª  (1.04 * 3.45)
     2.73,   # 6ª  (0.79 * 3.45)
]

# Soglie RPM per cambio automatico
RPM_UPSHIFT   = 6800.0      # scala in su
RPM_DOWNSHIFT = 2200.0      # scala in giù
RPM_MIN       = 900.0       # minima (folle)
RPM_MAX       = 8500.0      # limitatore

# Motore: forza massima alla ruota (N)
# F_max ≈ M * a_max_1a  (es. 1200 * 8.5 = 10200 N)
ENGINE_FORCE_MAX = 10_500.0  # N  (in 1ª marcia)

# Freno massimo
BRAKE_FORCE_MAX  = 28_000.0  # N  ≈ 23 m/s² a 1200 kg

# Resistenza aerodinamica: F_drag = k_drag * v²
# A 83 m/s (~300 km/h) bilanciata da ENGINE_FORCE_MAX in 6ª
K_DRAG = ENGINE_FORCE_MAX / (83.0 ** 2)   # ≈ 1.52

# Resistenza al rotolamento
K_ROLL = 120.0   # N circa

# Simulazione timestep
DT = 0.05        # secondi per step (~20 Hz)

# Larghezza approssimativa della pista oltre la AI line (m)
# Oltre questa soglia → ruote fuori
TRACK_HALF_WIDTH = 5.0


# ---------------------------------------------------------------------------
# Classe simulatore
# ---------------------------------------------------------------------------

class AcSimulator:
    """
    Simula la dinamica di una macchina da corsa su un tracciato 2D (XZ).

    Stato interno:
      x, z      — posizione world (float)
      heading   — angolo di yaw (rad, atan2(dx,dz))
      speed_ms  — velocità in m/s
      gear      — marcia attuale (1..6)
      rpm       — giri/min
      g_lat     — accelerazione laterale (g)
      g_long    — accelerazione longitudinale (g)
      tyres_out — numero di ruote fuori (0..4, semplificato 0 o 4)
    """

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self,
              x: float = 0.0,
              z: float = 0.0,
              heading: float = 0.0,
              speed_ms: float = 0.0):
        """Inizializza/resetta lo stato della macchina."""
        self.x        = float(x)
        self.z        = float(z)
        self.heading  = float(heading)
        self.speed_ms = float(speed_ms)
        self.gear     = 1
        self.rpm      = RPM_MIN
        self.g_lat    = 0.0
        self.g_long   = 0.0
        self.tyres_out = 0

        # Accelerazioni precedenti (per g-force smoothing)
        self._prev_vx = speed_ms * math.sin(heading)
        self._prev_vz = speed_ms * math.cos(heading)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, throttle: float, brake: float, steer: float,
             dist_from_line: float) -> dict:
        """
        Avanza la simulazione di DT secondi.

        Args:
          throttle      : [0, 1]
          brake         : [0, 1]
          steer         : [-1, 1]  (positivo = destra)
          dist_from_line: distanza dalla AI line (m), usata per tyres_out

        Returns dict con tutti i valori di telemetria.
        """
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake    = float(np.clip(brake,    0.0, 1.0))
        steer    = float(np.clip(steer,   -1.0, 1.0))

        v = self.speed_ms

        # --- Forze longitudinali ---
        # Forza motore ridotta con la velocità (curva di coppia semplificata)
        speed_ratio = max(0.0, 1.0 - v / 120.0)   # satura a ~430 km/h
        f_engine = throttle * ENGINE_FORCE_MAX * speed_ratio
        f_brake  = brake * BRAKE_FORCE_MAX
        f_drag   = K_DRAG * v * v
        f_roll   = K_ROLL

        f_net  = f_engine - f_brake - f_drag - f_roll
        accel  = f_net / MASS                        # m/s²

        # Blocca retromarcia
        v_new = max(0.0, v + accel * DT)

        # --- Cambio automatico ---
        self._auto_shift(v_new)

        # --- Heading (bicycle model) ---
        steer_angle = steer * MAX_STEER_ANG
        if abs(steer_angle) > 1e-6 and v_new > 0.5:
            turn_radius = WHEELBASE / math.tan(abs(steer_angle))
            yaw_rate    = v_new / turn_radius
            if steer_angle < 0:
                yaw_rate = -yaw_rate
            self.heading += yaw_rate * DT

        # Normalizza heading in [-pi, pi]
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # --- Posizione ---
        self.x += math.sin(self.heading) * v_new * DT
        self.z += math.cos(self.heading) * v_new * DT

        # --- G-force ---
        vx_new = v_new * math.sin(self.heading)
        vz_new = v_new * math.cos(self.heading)
        ax = (vx_new - self._prev_vx) / DT
        az = (vz_new - self._prev_vz) / DT
        g_long = (accel) / 9.81
        # Accelerazione centripeta ≈ v² / r
        if abs(steer_angle) > 1e-6 and v_new > 0.5:
            r_cent = WHEELBASE / math.tan(abs(steer_angle))
            g_lat  = (v_new ** 2 / r_cent) / 9.81
            if steer_angle < 0:
                g_lat = -g_lat
        else:
            g_lat = 0.0

        self._prev_vx = vx_new
        self._prev_vz = vz_new
        self.speed_ms = v_new
        self.g_long   = float(np.clip(g_long, -5.0, 5.0))
        self.g_lat    = float(np.clip(g_lat,  -5.0, 5.0))

        # --- Ruote fuori (semplificato: dist > TRACK_HALF_WIDTH) ---
        self.tyres_out = 4 if dist_from_line > TRACK_HALF_WIDTH else 0

        return self._state()

    # ------------------------------------------------------------------
    # Cambio automatico
    # ------------------------------------------------------------------

    def _auto_shift(self, v_ms: float):
        """Aggiorna marcia e RPM in base alla velocità."""
        gear = self.gear

        # Calcola RPM alla marcia corrente
        if gear >= 1 and v_ms > 0.01:
            rpm = (v_ms / WHEEL_RADIUS) * GEAR_RATIOS[gear] * (60 / (2 * math.pi))
        else:
            rpm = RPM_MIN

        # Upshift
        if rpm > RPM_UPSHIFT and gear < 6:
            gear += 1
        # Downshift
        elif rpm < RPM_DOWNSHIFT and gear > 1:
            gear -= 1

        # Ricalcola RPM con la nuova marcia
        if v_ms > 0.01:
            rpm = (v_ms / WHEEL_RADIUS) * GEAR_RATIOS[gear] * (60 / (2 * math.pi))
        else:
            rpm = RPM_MIN

        self.gear = gear
        self.rpm  = float(np.clip(rpm, RPM_MIN, RPM_MAX))

    # ------------------------------------------------------------------
    # Stato come dict
    # ------------------------------------------------------------------

    def _state(self) -> dict:
        return {
            "x"         : self.x,
            "z"         : self.z,
            "heading"   : self.heading,
            "speed"     : self.speed_ms * 3.6,   # km/h (compatibile con main.py)
            "speed_ms"  : self.speed_ms,
            "rpm"       : self.rpm,
            "g_lat"     : self.g_lat,
            "g_long"    : self.g_long,
            "gear"      : self.gear,
            "tyres_out" : self.tyres_out,
        }


# ---------------------------------------------------------------------------
# Self-test rapido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = AcSimulator()
    sim.reset(x=0, z=0, heading=0, speed_ms=0)

    for i in range(200):
        s = sim.step(throttle=0.9, brake=0.0, steer=0.05, dist_from_line=0.5)
        if i % 20 == 0:
            print(f"[{i:3d}] v={s['speed']:6.1f} km/h  rpm={s['rpm']:6.0f}  "
                  f"gear={s['gear']}  x={s['x']:.1f}  z={s['z']:.1f}  "
                  f"g_lat={s['g_lat']:+.2f}g  g_long={s['g_long']:+.2f}g")
