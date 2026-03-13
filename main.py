import mmap
import struct
import time
import numpy as np
import pyvjoy
from scipy.spatial import KDTree
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# --- CONFIGURAZIONE MEMORIA AC ---
AC_SM_PHYSICS = "Local\\AcTools.Physics"


class AssettoCorsaControllerEnv(gym.Env):
    def __init__(self, ai_path):
        super(AssettoCorsaControllerEnv, self).__init__()

        # 1. Init vJoy (Assicurati che il Device 1 sia attivo)
        try:
            self.vj = pyvjoy.VJoyDevice(1)
            print(">>> vJoy Connesso (Modalità Controller)")
        except Exception as e:
            print(f">>> Errore vJoy: {e}")
            exit()

        # 2. Caricamento Traiettoria
        self.ideal_line = self._parse_ai_file(ai_path)
        self.kdtree = KDTree(self.ideal_line[:, [0, 2]])

        # 3. Spazi IA
        # Action: [Sterzo (-1,1), Gas (0,1), Freno (0,1)]
        self.action_space = spaces.Box(low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        # Obs: [Velocità, Errore Lat, G-Lat, G-Long, Buffer_Sterzo]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.prev_steer = 0.0

    def _parse_ai_file(self, path):
        points = []
        with open(path, "rb") as f:
            num_points = struct.unpack('i', f.read(4))[0]
            for _ in range(num_points):
                data = struct.unpack('fffffff', f.read(28))
                points.append(data)
        return np.array(points)

    def _get_telemetry(self):
        try:
            shm = mmap.mmap(0, 800, AC_SM_PHYSICS, access=mmap.ACCESS_READ)
            # Lettura offset fisici
            speed = struct.unpack('f', shm[28:32])[0]
            g_lat = struct.unpack('f', shm[32:36])[0]
            g_long = struct.unpack('f', shm[40:44])[0]
            x = struct.unpack('f', shm[60:64])[0]
            z = struct.unpack('f', shm[68:72])[0]
            shm.close()
            return {"x": x, "z": z, "speed": speed, "g_lat": g_lat, "g_long": g_long}
        except:
            return {"x": 0, "z": 0, "speed": 0, "g_lat": 0, "g_long": 0}

    def step(self, action):
        # --- FILTRO STERZO (Smoothing per Controller) ---
        alpha = 0.3  # Più è basso, più lo sterzo è "morbido"
        current_steer = (alpha * action[0]) + ((1 - alpha) * self.prev_steer)
        self.prev_steer = current_steer

        # --- INVIO A VJOY ---
        # Sterzo -> Asse X (0 a 32767, centro 16384)
        self.vj.set_axis(pyvjoy.HID_USAGE_X, int((current_steer + 1) * 16383.5))
        # Acceleratore -> Asse Z
        self.vj.set_axis(pyvjoy.HID_USAGE_Z, int(action[1] * 32767))
        # Freno -> Asse RX (Rotational X)
        self.vj.set_axis(pyvjoy.HID_USAGE_RX, int(action[2] * 32767))

        time.sleep(0.02)  # Frequenza campionamento ~50Hz

        # --- CALCOLO REWARD E OBS ---
        tel = self._get_telemetry()
        dist, _ = self.kdtree.query([tel['x'], tel['z']])

        # Reward bilanciata: premia la velocità ma punisce severamente l'uscita di traiettoria
        reward = (tel['speed'] / 40.0) - (dist * 2.5)

        # Penalità se l'auto è ferma (per evitare che l'IA non parta mai)
        if tel['speed'] < 5: reward -= 1.0

        obs = np.array([
            tel['speed'] / 250.0,
            dist / 10.0,
            tel['g_lat'] / 3.0,
            tel['g_long'] / 3.0,
            current_steer
        ], dtype=np.float32)

        # Terminazione se l'auto esce troppo dalla linea ideale (es. 7 metri)
        terminated = dist > 7.0
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset comandi
        self.vj.set_axis(pyvjoy.HID_USAGE_X, 16384)
        self.vj.set_axis(pyvjoy.HID_USAGE_Z, 0)
        self.vj.set_axis(pyvjoy.HID_USAGE_RX, 0)
        self.prev_steer = 0.0
        return np.zeros(5, dtype=np.float32), {}


# --- MAIN LOOP ---
if __name__ == "__main__":
    AI_PATH = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks/monza/ai/fast_lane.ai"

    env = AssettoCorsaControllerEnv(AI_PATH)

    # Parametri PPO ottimizzati per stabilità
    model = PPO("MlpPolicy", env,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                ent_coef=0.01)  # ent_coef aiuta l'esplorazione

    print("\n>>> MODALITÀ CONTROLLER ATTIVA")
    print(">>> 1. Vai in AC > Controls > Wheel")
    print(">>> 2. Mappa Steer su X, Throttle su Z, Brake su RX")
    print(">>> 3. Imposta i gradi di rotazione a 200°")

    try:
        model.learn(total_timesteps=1000000)
    except KeyboardInterrupt:
        print("\n[SALVATAGGIO] Modello interrotto...")
    finally:
        model.save("models/ac_controller_ai")
        print(">>> Modello salvato: ac_controller_ai.zip")