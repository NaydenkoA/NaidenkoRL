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

df_calm = pd.read_csv('wbtc_calm_data_train_expanded.csv')
df_vol = pd.read_csv('wbtc_vol_data_train_expanded.csv')
nav_calm = df_calm['nav'].values
nav_vol = df_vol['nav'].values
vol_part_calm = df_calm['wbtc_frac'].values
vol_part_vol = df_vol['wbtc_frac'].values
nect_price_calm = df_calm['nect_price'].values
nect_price_vol = df_vol['nect_price'].values

SCALE = 6.5*10**7
LEARNING_EPOCHS = 2*1024*300 #default 200
TARGET_LEVERAGE = 3
SEED = 400
THREADS = 6
nav_vol = nav_vol/SCALE
nav_calm = nav_calm/SCALE

torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
np.random.seed(SEED)
random.seed(SEED)

def make_calm(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_calm,
            LP_nav_honey = nav_calm,
            volatile_part = vol_part_calm,
            is_training = True,
            target_leverage = TARGET_LEVERAGE
        )
        return Monitor(env, filename=os.path.join(log_dir, f"calm_monitor_{i}.csv"))
    return _init

def make_vol(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_vol,
            LP_nav_honey = nav_vol,
            volatile_part = vol_part_vol,
            is_training = True,
            target_leverage = TARGET_LEVERAGE
        )
        return Monitor(env, filename=os.path.join(log_dir, f"vol_monitor_{i}.csv"))
    return _init

def make_calm_extrimal(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_calm,
            LP_nav_honey = nav_calm,
            volatile_part = vol_part_calm,
            is_training = True,
            target_leverage = TARGET_LEVERAGE,
            is_extrimal = False,
            syntetic_part = 0.5
        )
        return Monitor(env, filename=os.path.join(log_dir, f"extrimal_calm_monitor_{i}.csv"))
    return _init

def make_vol_extrimal(i):
    def _init():
        env = agentDLVEnvV1(
            nect_honey_price = nect_price_vol,
            LP_nav_honey = nav_vol,
            volatile_part = vol_part_vol,
            is_training = True,
            target_leverage = TARGET_LEVERAGE,
            is_extrimal = False,
            syntetic_part = 0.5
        )
        return Monitor(env, filename=os.path.join(log_dir, f"extrimal_vol_monitor_{i}.csv"))
    return _init

"""print(torch.cuda.is_available())   # should be True
print(torch.cuda.device_count())   # should be ≥1
print(torch.cuda.get_device_name(0))"""


def lr_schedule(frac):
    if frac>0.5:
        return 7e-4
    else:
        return 4e-4 + (7e-4 - 4e-4) * frac * 2  

def ent_schedule(frac):
    if frac>0.5:
        return 0.06 + 0.002
    else:
        return 0.06 * 2 * frac + 0.002

if __name__ == "__main__":

    log_dir = "reccurent_loop_model/ppo_mixed"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    vec_env = SubprocVecEnv([make_calm(0), make_vol(0), make_calm(1), make_vol(1)])
    #vec_env = DummyVecEnv([make_calm(0), make_vol(0), make_calm_extrimal(0), make_vol_extrimal(0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    eval_env_calm = DummyVecEnv([
        lambda: Monitor(agentDLVEnvV1(
            nect_honey_price=nect_price_calm,
            LP_nav_honey=nav_calm,
            volatile_part=vol_part_calm,
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
            nect_honey_price=nect_price_vol,
            LP_nav_honey=nav_vol,
            volatile_part=vol_part_vol,
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

    policy_kwargs = dict(
        features_extractor_class=ReLUFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        lstm_hidden_size=128,      
        n_lstm_layers=1,           
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(
            pi=[128, 128], 
            vf=[256, 256]
        )  
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=lr_schedule,
        verbose=1,
        n_steps=1024,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        vf_coef = 2.1,
        clip_range = 0.15,
        clip_range_vf = 0.15,
        ent_coef = 0.03,
        device = "cuda"   
    )

    print("Policy device:", model.policy.device)
    print(model.policy.features_extractor)
    print(model.policy.mlp_extractor.policy_net)
    print(model.policy.mlp_extractor.value_net)
    print("Entropy coef:", model.ent_coef)
    print("Vf coef:", model.vf_coef)

    pid = os.getpid()
    print(f"My training PID is {pid}")

    model.learn(total_timesteps=int(LEARNING_EPOCHS), callback=combined_eval_callbacks)
    model.save("reccurent_loop_model/ppo_model_final")
    vec_env.save("reccurent_loop_model/vecnormalize_stats.pkl")