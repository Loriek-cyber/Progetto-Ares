import mmap
import struct
import time
import os
import numpy as np
import vgamepad as vg
from scipy.spatial import KDTree
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# --- CONFIGURAZIONE PERCORSI ---
BASE_PATH = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks"
AC_SM_PHYSICS = "Local\\AcTools.Physics"
AC_SM_STATIC = "Local\\AcTools.Static"


class AssettoCorsaMultiTrackEnv(gym.Env):
    def __init__(self):
        super(AssettoCorsaMultiTrackEnv, self).__init__()
        self.gamepad = vg.VX360Gamepad()

        # Identifica la pista attualmente caricata in AC
        self.track_name = self._get_current_track_name()
        print(f">>> Pista rilevata: {self.track_name}")

        # Carica il file AI specifico per la pista rilevata
        ai_path = os.path.join(BASE_PATH, self.track_name, "ai", "fast_lane.ai")
        if not os.path.exists(ai_path):
            raise FileNotFoundError(f"File AI non trovato per {self.track_name} in {ai_path}")

        self.ideal_line = self._parse_ai_file(ai_path)
        self.kdtree = KDTree(self.ideal_line[:, [0, 2]])

        self.action_space = spaces.Box(low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _get_current_track_name(self):
        """ Legge la Shared Memory Static per capire su che pista siamo """
        try:
            # AC_SM_STATIC contiene il nome della pista a partire dall'offset 20 (circa)
            shm_static = mmap.mmap(0, 512, AC_SM_STATIC, access=mmap.ACCESS_READ)
            # Leggiamo i byte e decodifichiamo (stringa Unicode in AC)
            track_bytes = shm_static[20:80].split(b'\x00')[0]
            track_name = track_bytes.decode('utf-16').strip()
            shm_static.close()
            return track_name if track_name else "monza"  # Fallback
        except:
            return "monza"

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
        self.gamepad.left_joystick_float(x_value_float=action[0], y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=action[1])
        self.gamepad.left_trigger_float(value_float=action[2])
        self.gamepad.update()

        time.sleep(0.01)
        tel = self._get_telemetry()
        dist, _ = self.kdtree.query([tel['x'], tel['z']])


        reward = (tel['speed'] / 30.0) - (dist ** 2)
        obs = np.array([tel['speed'] / 250.0, dist / 10.0, tel['g_lat'], tel['g_long'], action[0]], dtype=np.float32)

        return obs, reward, dist > 7.0, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gamepad.reset()
        self.gamepad.update()
        return np.zeros(5, dtype=np.float32), {}


# --- LOGICA DI CARICAMENTO E SALVATAGGIO ---
if __name__ == "__main__":
    env = AssettoCorsaMultiTrackEnv()
    model_name = f"model_{env.track_name}"
    model_path = f"{model_name}.zip"

    if os.path.exists(model_path):
        print(f">>> Trovato modello salvato per {env.track_name}. Caricamento in corso...")
        model = PPO.load(model_path, env=env)
    else:
        print(f">>> Nessun salvataggio per {env.track_name}. Creazione nuovo modello...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    try:
        # Loop di apprendimento
        model.learn(total_timesteps=1000000)
    except KeyboardInterrupt:
        print("\n[INTERRUZIONE] Salvataggio progressi...")
    finally:
        model.save(model_name)
        print(f">>> Modello {model_name} salvato correttamente.")