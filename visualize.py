import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from throw_env import ThrowEnv
import time
from sb3_contrib import RecurrentPPO

# load environment and trained model
env = ThrowEnv()
model = RecurrentPPO.load("throw_policy")
lstm_states = None
episode_start = np.ones((1,), dtype=bool)
# reset environment
obs, _ = env.reset(random_reset = False)
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_start = np.array([terminated or truncated])

        sim_time = env.data.time
        print(f"\rSim Time: {env.data.time:.2f}s", end="")

        viewer.sync()
        time.sleep(0.05)
        
        if terminated or truncated:
            ball_pos = env.data.xpos[env.ball_id]
            dist = np.sqrt(
                (ball_pos[0] - env.throw_origin[0])**2 +
                (ball_pos[1] - env.throw_origin[1])**2
            )
            
            print(f"terminated: {terminated}, truncated: {truncated}, released: {env.released}, dist: {dist:.2f}m")
            obs, _ = env.reset(random_reset = False)
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            