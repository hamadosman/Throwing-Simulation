from throw_env import ThrowEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sb3_contrib import RecurrentPPO 

eval_env = ThrowEnv()

# create environment
env = make_vec_env(ThrowEnv, n_envs=4)

model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    learning_rate=1e-5,
    clip_range=0.4,
    ent_coef=0.0,
    policy_kwargs=dict(
        net_arch=dict(pi=[512, 256, 256], vf=[256, 256, 256]),
    ),
)

checkpoint_callback = CheckpointCallback(
    save_freq=250000,
    save_path='./checkpoints/',
    name_prefix='throw_policy'
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/',
    eval_freq=100000,
    n_eval_episodes=10,
    deterministic=True
)

model.learn(
    total_timesteps = 8_000_000,
    callback=[checkpoint_callback, eval_callback]
)
# save the trained model
model.save("throw_policy_002")

print("Training done")
