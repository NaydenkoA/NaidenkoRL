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
from ReLuExtractor import ReLUFeatureExtractor
import torch.nn as nn

"""df = pd.read_csv('weth_train_data_expanded.csv')
nav = df['nav'].values
volatile_part = df['weth_frac'].values
nect_price = df['nect_price'].values"""

df = pd.read_csv('weth_island_data.csv')
nav = df['nav'].values
nect_price = pd.read_csv('nect_honey_price.csv')['price'].values
vol_token_price = pd.read_csv('weth_honey_price.csv')['price'].values
volatile_part = df['wbtc_frac'].values

"""nav = nav[::6]
volatile_part = volatile_part[::6]
nect_price = nect_price[::6]"""

SCALE = 100
LEARNING_EPOCHS = 2*1024*300 #default 200
TARGET_LEVERAGE = 3
SEED = 400
THREADS = 4
PRIMORDIAL_ACTION_PENALTY = 1
PENALTY_DAMPING = 0.95
ACTION_PENALTY_SCALE = 0.8

nav_calm_train = nav[:int(0.4*len(nav))]
volatile_part_calm_train = volatile_part[:int(0.4*len(volatile_part))]
nect_price_calm_train = nect_price[:int(0.4*len(nect_price))]
nav_vol_train = nav[int(0.4*len(nav)):]
volatile_part_vol_train = volatile_part[int(0.4*len(volatile_part)):]
nect_price_vol_train = nect_price[int(0.4*len(nect_price)):]
vol_token_price_calm_train = vol_token_price[:int(0.4*len(volatile_part))]
vol_token_price_vol_train = vol_token_price[int(0.4*len(volatile_part)):]

torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
np.random.seed(SEED)
random.seed(SEED)

def make_calm(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_calm_train,
            LP_nav_honey = nav_calm_train,
            volatile_part = volatile_part_calm_train,
            vol_token_price = vol_token_price_calm_train,
            is_training = True,
            target_leverage = TARGET_LEVERAGE,
            primordial_action_penalty=PRIMORDIAL_ACTION_PENALTY,
            action_penalty_scale=ACTION_PENALTY_SCALE,
            penalty_damping=PENALTY_DAMPING
        )
        return Monitor(env, filename=os.path.join(log_dir, f"calm_monitor_{i}.csv"))
    return _init

def make_vol(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_vol_train,
            LP_nav_honey = nav_vol_train,
            volatile_part = volatile_part_vol_train,
            vol_token_price = vol_token_price_vol_train,
            is_training = True,
            target_leverage = TARGET_LEVERAGE,
            primordial_action_penalty=PRIMORDIAL_ACTION_PENALTY,
            action_penalty_scale=ACTION_PENALTY_SCALE,
            penalty_damping=PENALTY_DAMPING
        )
        return Monitor(env, filename=os.path.join(log_dir, f"vol_monitor_{i}.csv"))
    return _init

def make_calm_extrimal(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_calm_train,
            LP_nav_honey = nav_calm_train,
            volatile_part = volatile_part_calm_train,
            vol_token_price = vol_token_price_calm_train,
            is_training = True,
            target_leverage = TARGET_LEVERAGE,
            is_extrimal = False,
            syntetic_part = 0.5,
            primordial_action_penalty=PRIMORDIAL_ACTION_PENALTY,
            action_penalty_scale=ACTION_PENALTY_SCALE,
            penalty_damping=PENALTY_DAMPING
        )
        return Monitor(env, filename=os.path.join(log_dir, f"extrimal_calm_monitor_{i}.csv"))
    return _init

def make_vol_extrimal(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_vol_train,
            LP_nav_honey = nav_vol_train,
            volatile_part = volatile_part_vol_train,
            vol_token_price = vol_token_price_vol_train,
            is_training = True,
            target_leverage = TARGET_LEVERAGE,
            is_extrimal = False,
            syntetic_part = 0.5,
            primordial_action_penalty=PRIMORDIAL_ACTION_PENALTY,
            action_penalty_scale=ACTION_PENALTY_SCALE,
            penalty_damping=PENALTY_DAMPING
        )
        return Monitor(env, filename=os.path.join(log_dir, f"extrimal_vol_monitor_{i}.csv"))
    return _init

"""print(torch.cuda.is_available())   # should be True
print(torch.cuda.device_count())   # should be ≥1
print(torch.cuda.get_device_name(0))"""

if __name__ == "__main__":

    log_dir = "loop_model_weth/ppo_mixed"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    vec_env = SubprocVecEnv([make_calm(0), make_vol(0), make_calm_extrimal(0), make_vol_extrimal(0)])
    #vec_env = DummyVecEnv([make_calm(0), make_vol(0), make_calm_extrimal(0), make_vol_extrimal(0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    eval_env_calm = DummyVecEnv([
        lambda: Monitor(agentDLVEnvV1(
            nect_honey_price=nect_price_calm_train,
            LP_nav_honey=nav_calm_train,
            volatile_part=volatile_part_calm_train,
            vol_token_price = vol_token_price_calm_train,
            is_training=False,
            target_leverage=TARGET_LEVERAGE,
            primordial_action_penalty=PRIMORDIAL_ACTION_PENALTY,
            action_penalty_scale=ACTION_PENALTY_SCALE,
            penalty_damping=PENALTY_DAMPING
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
            vol_token_price = vol_token_price_vol_train,
            is_training=False,
            target_leverage=TARGET_LEVERAGE,
            primordial_action_penalty=PRIMORDIAL_ACTION_PENALTY,
            action_penalty_scale=ACTION_PENALTY_SCALE,
            penalty_damping=PENALTY_DAMPING
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
        deterministic=True
    )

    eval_callback_vol = EvalCallback(
        eval_env_vol,
        best_model_save_path=os.path.join(log_dir, "best_model_vol"),
        log_path=os.path.join(log_dir, "eval_logs_vol"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True
    )

    combined_eval_callbacks = CallbackList([eval_callback_calm, eval_callback_vol])

    policy_kwargs = dict(
        features_extractor_class=ReLUFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64)
    )

    model_raw = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1.8e-4,
        verbose=1,
        n_steps=512,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        vf_coef = 2,
        device="cpu"   
    )

    print("Policy device:", model_raw.policy.device)
    print(model_raw.policy.features_extractor)
    print(model_raw.policy.mlp_extractor.policy_net)
    print(model_raw.policy.mlp_extractor.value_net)

    pid = os.getpid()
    print(f"My training PID is {pid}")

    model_raw.learn(total_timesteps=int(LEARNING_EPOCHS), callback=combined_eval_callbacks)
    model_raw.save("loop_model_weth/ppo_model_raw")
    vec_env.save("loop_model_weth/vecnormalize_stats.pkl")

    model_step_1 = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-5,  
        ent_coef=1e-3,       
        verbose=1,
        n_steps=512,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device="cpu" 
    )

    model_step_1.set_parameters("loop_model_weth/ppo_model_raw.zip")

    eval_env_calm.obs_rms = vec_env.obs_rms
    eval_env_calm.ret_rms = vec_env.ret_rms
    eval_env_vol.obs_rms = vec_env.obs_rms
    eval_env_vol.ret_rms = vec_env.ret_rms

    model_step_1.learn(total_timesteps=int(LEARNING_EPOCHS), callback=combined_eval_callbacks)
    #model_step_1.save("loop_model_weth/ppo_model_step_1")
    vec_env.save("loop_model_weth/vecnormalize_stats.pkl")
    model_step_1.save("loop_model_weth/ppo_model_final")

    """model_1 = PPO(
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

    model_1.set_parameters("loop_model_weth/ppo_model_step_1.zip")

    eval_env_calm.obs_rms = vec_env.obs_rms
    eval_env_calm.ret_rms = vec_env.ret_rms
    eval_env_vol.obs_rms = vec_env.obs_rms
    eval_env_vol.ret_rms = vec_env.ret_rms

    model_1.learn(total_timesteps=LEARNING_EPOCHS, callback=combined_eval_callbacks)
    model_1.save("loop_model_weth/ppo_model_final")
    vec_env.save("loop_model_weth/vecnormalize_stats.pkl")"""