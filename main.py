import mmap
import struct
import time
import numpy as np
import vgamepad as vg
from scipy.spatial import KDTree
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# --- CONFIGURAZIONE MEMORIA AC ---
AC_SM_PHYSICS = "Local\\AcTools.Physics"


class AssettoCorsaViGEmEnv(gym.Env):
    def __init__(self, ai_path):
        super(AssettoCorsaViGEmEnv, self).__init__()

        # 1. Inizializzazione Controller Xbox Virtuale
        self.gamepad = vg.VX360Gamepad()
        print(">>> Controller Xbox 360 Virtuale (ViGEm) inizializzato.")

        # 2. Caricamento Traiettoria
        self.ideal_line = self._parse_ai_file(ai_path)
        self.kdtree = KDTree(self.ideal_line[:, [0, 2]])

        # 3. Spazi IA
        # Action: [Sterzo (-1 a 1), Gas (0 a 1), Freno (0 a 1)]
        self.action_space = spaces.Box(low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        # Obs: [Velocità, Errore Lat, G-Lat, G-Long, Input_Sterzo_Precedente]
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
        # --- INPUT CONTROLLER XBOX ---
        # Sterzo: Stick Sinistro (X_AXIS: da -1.0 a 1.0)
        self.gamepad.left_joystick_float(x_value_float=action[0], y_value_float=0.0)

        # Acceleratore: Trigger Destro (da 0.0 a 1.0)
        self.gamepad.right_trigger_float(value_float=action[1])

        # Freno: Trigger Sinistro (da 0.0 a 1.0)
        self.gamepad.left_trigger_float(value_float=action[2])

        # Applica i cambiamenti
        self.gamepad.update()

        # Tempo di campionamento
        time.sleep(0.01)

        # --- TELEMETRIA E REWARD ---
        tel = self._get_telemetry()
        dist, _ = self.kdtree.query([tel['x'], tel['z']])

        # Reward: penalizziamo l'errore laterale in modo esponenziale
        reward = (tel['speed'] / 30.0) - (dist ** 2)

        # Bonus per velocità minima (non restare fermi)
        if tel['speed'] < 10: reward -= 2.0

        obs = np.array([
            tel['speed'] / 250.0,
            dist / 10.0,
            tel['g_lat'],
            tel['g_long'],
            action[0]
        ], dtype=np.float32)

        # Reset se l'auto vola fuori pista (7 metri)
        terminated = dist > 7.0
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset controller
        self.gamepad.reset()
        self.gamepad.update()
        return np.zeros(5, dtype=np.float32), {}


# --- TRAINING ---
if __name__ == "__main__":
    AI_PATH = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks/monza/ai/fast_lane.ai"
    env = AssettoCorsaViGEmEnv(AI_PATH)

    # PPO è robusto per il controllo continuo
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=5000)

    print("\n" + "=" * 50)
    print(">>> CONTROLLER XBOX VIRTUALE ATTIVO.")
    print(">>> In Assetto Corsa, vai in 'Controls' e seleziona 'Gamepad'.")
    print(">>> Verifica che gli indicatori si muovano!")
    print("=" * 50 + "\n")

    try:
        model.learn(total_timesteps=1000000)
    except KeyboardInterrupt:
        print("\n[STOP] Salvataggio modello...")
    finally:
        model.save("models/ac_vigem_ai")
        print(">>> Modello salvato come: ac_vige_ai.zip")