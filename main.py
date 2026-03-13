import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Assumi di avere un modulo 'ac_telemetry' per leggere la Shared Memory
import ac_telemetry 

class AssettoCorsaEnv(gym.Env):
    def __init__(self):
        super(AssettoCorsaEnv, self).__init__()
        
        # Output: Acceleratore, Freno, Sterzo (da -1 a 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Input: Velocità, G-force, Distanza dai bordi (Ray-casting), RPM
        self.observation_space = spaces.Box(low=-inf, high=inf, shape=(10,), dtype=np.float32)

    def step(self, action):
        # 1. Invia l'azione ai pedali/volante tramite vJoy
        self._apply_action(action)
        
        # 2. Leggi i nuovi dati dalla telemetria
        state = ac_telemetry.get_data()
        
        # 3. Calcola il Reward (Il segreto per il tempo sul giro)
        # Premiamo la velocità e penalizziamo se l'auto è troppo angolata rispetto alla pista
        reward = state['speed'] * np.cos(state['track_angle']) 
        
        # Penalità per collisioni o tagli
        if state['off_track']:
            reward -= 100
            done = True
        else:
            done = False
            
        return np.array(list(state.values())), reward, done, False, {}

    def reset(self, seed=None, options=None):
        # Qui dovresti usare un comando per resettare la sessione in AC
        return self.initial_state, {}
