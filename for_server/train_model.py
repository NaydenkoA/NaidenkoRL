import numpy as np
import pandas as pd
from environment import agentDLVEnvV1
import os
import shutil
import random
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList

df = pd.read_csv('island_data.csv')
timestamp = df['timestamp'].values
nav = df['nav'].values
nect_price = pd.read_csv('nect_honey_price.csv')['price'].values
volatile_part = df['wbtc_frac'].values

SCALE = 6.5*10**7
LEARNING_EPOCHS = 2*1024*300 #default 200
TARGET_LEVERAGE = 3
SEED = 400
nav = nav/SCALE

nav_calm = nav[:3720]
nav_vol = nav[3720:]
volatile_part_calm = volatile_part[:3720]
volatile_part_vol = volatile_part[3720:]
nect_price_calm = nect_price[:3720]
nect_price_vol = nect_price[3720:]

nav_calm_train = nav_calm[:int(0.6*len(nav_calm))]
nav_calm_test = nav_calm[int(0.6*len(nav_calm)):]
nav_vol_train = nav_vol[int(0.4*len(nav_vol)):]
nav_vol_test = nav_vol[:int(0.4*len(nav_vol))]

volatile_part_calm_train = volatile_part_calm[:int(0.6*len(volatile_part_calm))]
volatile_part_calm_test = volatile_part_calm[int(0.6*len(volatile_part_calm)):]
volatile_part_vol_train = volatile_part_vol[int(0.4*len(volatile_part_vol)):]
volatile_part_vol_test = volatile_part_vol[:int(0.4*len(volatile_part_vol))]

nect_price_calm_train = nect_price_calm[:int(0.6*len(nect_price_calm))]
nect_price_calm_test = nect_price_calm[int(0.6*len(nect_price_calm)):]
nect_price_vol_train = nect_price_vol[int(0.4*len(volatile_part_vol)):]
nect_price_vol_test = nect_price_vol[:int(0.4*len(volatile_part_vol))]

log_dir = "loop_model/ppo_mixed"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def make_calm(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_calm_train,
            LP_nav_honey = nav_calm_train,
            volatile_part = volatile_part_calm_train,
            is_training = True,
            target_leverage = TARGET_LEVERAGE
        )
        return Monitor(env, filename=os.path.join(log_dir, f"calm_monitor_{i}.csv"))
    return _init

def make_vol(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_vol_train,
            LP_nav_honey = nav_vol_train,
            volatile_part = volatile_part_vol_train,
            is_training = True,
            target_leverage = TARGET_LEVERAGE
        )
        return Monitor(env, filename=os.path.join(log_dir, f"vol_monitor_{i}.csv"))
    return _init

def make_calm_extrimal(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_calm_train,
            LP_nav_honey = nav_calm_train,
            volatile_part = volatile_part_calm_train,
            is_training = True,
            target_leverage = TARGET_LEVERAGE,
            is_extrimal = False,
            syntetic_part = 0.4
        )
        return Monitor(env, filename=os.path.join(log_dir, f"extrimal_calm_monitor_{i}.csv"))
    return _init

def make_vol_extrimal(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_vol_train,
            LP_nav_honey = nav_vol_train,
            volatile_part = volatile_part_vol_train,
            is_training = True,
            target_leverage = TARGET_LEVERAGE,
            is_extrimal = False,
            syntetic_part = 0.4
        )
        return Monitor(env, filename=os.path.join(log_dir, f"extrimal_vol_monitor_{i}.csv"))
    return _init

"""print(torch.cuda.is_available())   # should be True
print(torch.cuda.device_count())   # should be ≥1
print(torch.cuda.get_device_name(0))"""


#vec_env = SubprocVecEnv([make_calm(0), make_vol(0), make_calm_extrimal(0), make_vol_extrimal(0)])
vec_env = DummyVecEnv([make_calm(0), make_vol(0), make_calm_extrimal(0), make_vol_extrimal(0)])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

eval_env_calm = DummyVecEnv([
    lambda: Monitor(agentDLVEnvV1(
        nect_honey_price=nect_price_calm_train,
        LP_nav_honey=nav_calm_train,
        volatile_part=volatile_part_calm_train,
        is_training=False,
        target_leverage=TARGET_LEVERAGE
    ), filename=os.path.join(log_dir, "eval_calm.csv"))
])  
eval_env_calm = VecNormalize(eval_env_calm, norm_obs=True, norm_reward=True, training=False)
eval_env_calm.training = False
eval_env_calm.norm_reward = False
eval_env_calm.obs_rms = vec_env.obs_rms
eval_env_calm.ret_rms = vec_env.ret_rms

eval_env_vol = DummyVecEnv([
    lambda: Monitor(agentDLVEnvV1(
        nect_honey_price=nect_price_vol_train,
        LP_nav_honey=nav_vol_train,
        volatile_part=volatile_part_vol_train,
        is_training=False,
        target_leverage=TARGET_LEVERAGE
    ), filename=os.path.join(log_dir, "eval_vol.csv"))
])
eval_env_vol = VecNormalize(eval_env_vol, norm_obs=True, norm_reward=True, training=False)
eval_env_vol.training = False
eval_env_vol.norm_reward = False
eval_env_vol.obs_rms = vec_env.obs_rms
eval_env_vol.ret_rms = vec_env.ret_rms

eval_callback_calm = EvalCallback(
    eval_env_calm,
    best_model_save_path=os.path.join(log_dir, "best_model_calm"),
    log_path=os.path.join(log_dir, "eval_logs_calm"),
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

eval_callback_vol = EvalCallback(
    eval_env_vol,
    best_model_save_path=os.path.join(log_dir, "best_model_vol"),
    log_path=os.path.join(log_dir, "eval_logs_vol"),
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

combined_eval_callbacks = CallbackList([eval_callback_calm, eval_callback_vol])

model_raw = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=2e-4,
    verbose=1,
    n_steps=512,
    tensorboard_log=log_dir,
    vf_coef = 2,
    device="cpu"   
)

print("Policy device:", model_raw.policy.device)

pid = os.getpid()
print(f"My training PID is {pid}")

model_raw.learn(total_timesteps=LEARNING_EPOCHS, callback=combined_eval_callbacks)
model_raw.save("loop_model/ppo_model_raw")
vec_env.save("loop_model/vecnormalize_stats.pkl")

model_step_1 = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-4,  
    ent_coef=1e-3,       
    verbose=1,
    n_steps=512,
    tensorboard_log=log_dir,
    device="cpu" 
)

model_step_1.set_parameters("loop_model/ppo_model_raw.zip")

eval_env_calm.obs_rms = vec_env.obs_rms
eval_env_calm.ret_rms = vec_env.ret_rms
eval_env_vol.obs_rms = vec_env.obs_rms
eval_env_vol.ret_rms = vec_env.ret_rms

model_step_1.learn(total_timesteps=LEARNING_EPOCHS, callback=combined_eval_callbacks)
model_step_1.save("loop_model/ppo_model_step_1")
vec_env.save("loop_model/vecnormalize_stats.pkl")

model_1 = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-5,  
    ent_coef=1e-4,       
    verbose=1,
    n_steps=512,
    tensorboard_log=log_dir,
    vf_coef = 1,
    device="cpu" 
)

model_1.set_parameters("loop_model/ppo_model_step_1.zip")

eval_env_calm.obs_rms = vec_env.obs_rms
eval_env_calm.ret_rms = vec_env.ret_rms
eval_env_vol.obs_rms = vec_env.obs_rms
eval_env_vol.ret_rms = vec_env.ret_rms

model_1.learn(total_timesteps=LEARNING_EPOCHS, callback=combined_eval_callbacks)
model_1.save("loop_model/ppo_model_final")
vec_env.save("loop_model/vecnormalize_stats.pkl")