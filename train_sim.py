"""
train_sim.py — Training parallelo PPO sul simulatore AC
=======================================================
Addestra un modello PPO usando N_ENVS istanze in parallelo del simulatore
fisico, senza bisogno di Assetto Corsa in esecuzione.

Il modello risultante è direttamente caricabile in main.py per il fine-tuning
sul gioco reale (stessa obs/action space).

Uso:
    python train_sim.py                   # avvia nuovo training
    python train_sim.py --resume          # riprende dall'ultimo checkpoint
    python train_sim.py --envs 32         # usa 32 ambienti paralleli
"""

import os
import sys
import argparse
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

from sim.sim_env import SimACEnv

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

N_ENVS         = 16          # numero di ambienti paralleli (scalabile)
TOTAL_STEPS    = 10_000_000  # step totali di training
CHECKPOINT_FREQ = 100_000    # salva modello ogni N step (per env)
EVAL_FREQ      = 50_000      # valutazione ogni N step (per env)
EVAL_EPISODES  = 5           # episodi di valutazione

TRACK_NAME     = "monza"
MODEL_DIR      = "models_sim"
LOG_DIR        = "logs_sim"

# TensorBoard opzionale: se non installato, log solo su console
try:
    import tensorboard  # noqa: F401
    _TB_LOG = LOG_DIR
    print("[Train] TensorBoard disponibile — log in:", LOG_DIR)
except ImportError:
    _TB_LOG = None
    print("[Train] TensorBoard non installato — log solo su console.")
    print("        Per installarlo: pip install tensorboard")

# Hyperparametri PPO ottimizzati per questo task
PPO_PARAMS = dict(
    learning_rate   = 3e-4,
    n_steps         = 2048,    # step per env per ogni update
    batch_size      = 512,     # batch size (deve dividere n_steps * n_envs)
    n_epochs        = 10,
    gamma           = 0.995,   # alto per valorizzare reward futuri (giro completo)
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.005,   # bassa entropia → sfruttamento
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
    tensorboard_log = _TB_LOG,
)


# ---------------------------------------------------------------------------
# Funzione factory per SubprocVecEnv
# ---------------------------------------------------------------------------

def make_env(track_name: str, seed: int = 0):
    """Ritorna una funzione lambda che crea un env (richiesto da SubprocVecEnv)."""
    def _init():
        env = SimACEnv(track_name=track_name)
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training PPO su simulatore AC")
    parser.add_argument("--envs",   type=int, default=N_ENVS,    help="Numero di env paralleli")
    parser.add_argument("--steps",  type=int, default=TOTAL_STEPS, help="Step totali di training")
    parser.add_argument("--track",  type=str, default=TRACK_NAME, help="Nome pista")
    parser.add_argument("--resume", action="store_true",          help="Riprendi dall'ultimo checkpoint")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    model_name = f"ppo_sim_{args.track}"
    model_path = os.path.join(MODEL_DIR, model_name)
    best_path  = os.path.join(MODEL_DIR, f"{model_name}_best")

    print(f"[Train] Pista: {args.track}")
    print(f"[Train] Env paralleli: {args.envs}")
    print(f"[Train] Step totali: {args.steps:,}")
    print(f"[Train] Modello: {model_path}.zip")

    # --- Crea ambienti di training ---
    print("[Train] Creazione ambienti di training...")
    train_envs = SubprocVecEnv([
        make_env(args.track, seed=i) for i in range(args.envs)
    ])
    train_envs = VecMonitor(train_envs)

    # --- Crea ambiente di valutazione (singolo processo) ---
    eval_env = VecMonitor(SubprocVecEnv([make_env(args.track, seed=999)]))

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = model_name,
        verbose     = 0,
    )
    eval_cb = EvalCallback(
        eval_env          = eval_env,
        best_model_save_path = best_path,
        log_path          = LOG_DIR,
        eval_freq         = EVAL_FREQ,
        n_eval_episodes   = EVAL_EPISODES,
        deterministic     = True,
        verbose           = 0,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # --- Carica o crea modello ---
    if args.resume and os.path.exists(f"{model_path}.zip"):
        print(f"[Train] Riprendendo da: {model_path}.zip")
        model = PPO.load(model_path, env=train_envs, **{
            k: v for k, v in PPO_PARAMS.items()
            if k not in ("verbose", "tensorboard_log")
        })
        model.set_env(train_envs)
    else:
        print("[Train] Nuovo modello PPO...")
        model = PPO(policy="MlpPolicy", env=train_envs, **PPO_PARAMS)

    # --- Training ---
    print(f"\n[Train] Inizio training — {args.steps:,} step totali")
    print(f"        Effettivi per env: {args.steps // args.envs:,}")
    print(f"        Usa 'tensorboard --logdir {LOG_DIR}' per monitorare\n")

    t0 = time.time()
    try:
        model.learn(
            total_timesteps   = args.steps,
            callback          = callbacks,
            reset_num_timesteps = not args.resume,
            tb_log_name       = model_name,
        )
    except KeyboardInterrupt:
        print("\n[Train] Interruzione manuale — salvataggio...")
    finally:
        model.save(model_path)
        elapsed = time.time() - t0
        print(f"\n[Train] Modello salvato: {model_path}.zip")
        print(f"[Train] Tempo: {elapsed/60:.1f} min")
        print(f"[Train] Step/sec: {args.steps / elapsed:.0f}")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    # Su Windows, SubprocVecEnv richiede questo guard
    import multiprocessing
    multiprocessing.freeze_support()
    main()
