import mmap
import struct
import time
import numpy as np
import pyvjoy
from scipy.spatial import KDTree
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# --- COSTANTI MEMORIA ASSETTO CORSA (SPageFilePhysics) ---
AC_SM_PHYSICS = "Local\\AcTools.Physics"


class AssettoCorsaEnv(gym.Env):
    def __init__(self, ai_path):
        super(AssettoCorsaEnv, self).__init__()

        # 1. Inizializzazione Joystick Virtuale (vJoy Device 1)
        try:
            self.joystick = pyvjoy.VJoyDevice(1)
            print(">>> vJoy Device 1 connesso con successo.")
        except Exception as e:
            print(f">>> ERRORE: Impossibile connettersi a vJoy. {e}")
            exit()

        # 2. Caricamento Traiettoria Monza (.ai file)
        print(f">>> Caricamento traiettoria da: {ai_path}")
        self.ideal_line = self._parse_ai_file(ai_path)
        # Usiamo X e Z per il piano della pista (Y è l'altezza)
        self.kdtree = KDTree(self.ideal_line[:, [0, 2]])

        # 3. Definizione Spazi IA
        # Action: [Sterzo (-1,1), Accel (0,1), Freno (0,1)]
        self.action_space = spaces.Box(low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        # Observation: [Velocità, Errore Lat, G-Lat, G-Long, AngoloApprossimativo]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _parse_ai_file(self, path):
        points = []
        with open(path, "rb") as f:
            header = f.read(4)
            num_points = struct.unpack('i', header)[0]
            for _ in range(num_points):
                # Ogni punto della spline AC è 28 byte (7 float)
                data = struct.unpack('fffffff', f.read(28))
                points.append(data)
        return np.array(points)

    def _get_telemetry(self):
        try:
            # Accesso alla Shared Memory Physics
            shm = mmap.mmap(0, 800, AC_SM_PHYSICS, access=mmap.ACCESS_READ)

            # Offset precisi documentazione AC:
            # Speed (km/h) @ offset 28
            speed = struct.unpack('f', shm[28:32])[0]

            # G-Force (Lat, Long) @ offset 32, 40
            g_lat = struct.unpack('f', shm[32:36])[0]
            g_long = struct.unpack('f', shm[40:44])[0]

            # Position X, Y, Z @ offset 60, 64, 68
            x = struct.unpack('f', shm[60:64])[0]
            z = struct.unpack('f', shm[68:72])[0]

            shm.close()
            return {"x": x, "z": z, "speed": speed, "g_lat": g_lat, "g_long": g_long}
        except Exception:
            # Se il gioco è chiuso o la memoria non è pronta, restituisci zeri
            return {"x": 0, "z": 0, "speed": 0, "g_lat": 0, "g_long": 0}

    def step(self, action):
        # --- 1. INVIO COMANDI A VJOY ---
        # Mappatura: vJoy accetta valori interi da 0 a 32767
        vjoy_steer = int((action[0] + 1) * 16383.5)
        vjoy_accel = int(action[1] * 32767)
        vjoy_brake = int(action[2] * 32767)

        self.joystick.set_axis(pyvjoy.HID_USAGE_X, vjoy_steer)
        self.joystick.set_axis(pyvjoy.HID_USAGE_Y, vjoy_accel)
        self.joystick.set_axis(pyvjoy.HID_USAGE_Z, vjoy_brake)

        # Aspettiamo un minimo per la fisica del gioco
        time.sleep(0.01)

        # --- 2. LETTURA RISULTATI ---
        tel = self._get_telemetry()
        dist, _ = self.kdtree.query([tel['x'], tel['z']])

        # --- 3. REWARD DESIGN ---
        # Reward positiva per velocità, negativa per distanza dalla linea ideale
        reward = (tel['speed'] / 50.0) - (dist * 2.0)

        # Penalità per eccessive forze G laterali (opzionale, per guida fluida)
        reward -= abs(tel['g_lat']) * 0.1

        # --- 4. STATO E FINE SESSIONE ---
        # Normalizziamo le osservazioni per la rete neurale
        obs = np.array([
            tel['speed'] / 200.0,
            dist / 5.0,
            tel['g_lat'],
            tel['g_long'],
            0  # Placeholder per angolo
        ], dtype=np.float32)

        # Se l'auto esce di 8 metri dalla linea ideale, consideriamo il tentativo fallito
        terminated = dist > 8.0
        truncated = False

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Riporta i comandi a zero
        self.joystick.set_axis(pyvjoy.HID_USAGE_X, 16384)
        self.joystick.set_axis(pyvjoy.HID_USAGE_Y, 0)
        self.joystick.set_axis(pyvjoy.HID_USAGE_Z, 0)

        return np.zeros(5, dtype=np.float32), {}


# --- AVVIO TRAINING ---
if __name__ == "__main__":
    # Percorso del file AI (assicurati che sia corretto!)
    AI_PATH = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks/monza/ai/fast_lane.ai"

    env = AssettoCorsaEnv(AI_PATH)

    # MLP (Multi-Layer Perceptron) è ottimo per dati vettoriali come questi
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, batch_size=64)

    print("\n" + "=" * 50)
    print(">>> IA PRONTA. CONFIGURA I CONTROLLI IN AC ORA!")
    print(">>> L'IA sta inviando segnali vJoy. Mappa X=Steer, Y=Gas, Z=Brake.")
    print(">>> PREMI CTRL+C PER FERMARE E SALVARE.")
    print("=" * 50 + "\n")

    try:
        model.learn(total_timesteps=1000000)
    except KeyboardInterrupt:
        print("\n[STOP] Salvataggio in corso...")
    finally:
        model.save("monza_ai_driver")
        print(">>> MODELLO SALVATO: monza_ai_driver.zip")