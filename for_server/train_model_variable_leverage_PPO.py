import numpy as np
import pandas as pd
from variable_leverage_env_1 import agentDLVEnvV2
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

df = pd.read_csv('island_data.csv')
timestamp = df['timestamp'].values
nav = df['nav'].values
nect_price = pd.read_csv('nect_honey_price.csv')['price'].values
vol_token_price = pd.read_csv('wbtc_honey_price.csv')['price'].values
volatile_part = df['wbtc_frac'].values

SCALE = 6.5*10**7
LEARNING_EPOCHS = 2*1024*400 #default 200
TARGET_LEVERAGE = 3
SEED = 400
THREADS = 5
PRIMORDIAL_ACTION_PENALTY = 0.4
PENALTY_DAMPING = 0.95
ACTION_PENALTY_SCALE = 0.5
LIQUIDATION_PENALTY_SCALE = 5
LOSS_PENALTY_SCALE = 1.5
BOUNDARY_REWARD_SCALE = 1

log_dir = "variable_leverage_model_ppo/ppo_mixed"

torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
np.random.seed(SEED)
#random.seed(SEED)

def make_env(i, b_l, t_l):
    def _init():
        env = agentDLVEnvV2(
            nect_honey_price = nect_price,
            nav_in_honey = nav,
            vol_part = volatile_part,
            vol_token_honey_price = vol_token_price,
            is_training = True,
            is_mixed = True,
            top_leverage = t_l,
            bottom_leverage = b_l,
            primordial_action_penalty = PRIMORDIAL_ACTION_PENALTY,
            penalty_damping = PENALTY_DAMPING,
            action_penalty_scale = ACTION_PENALTY_SCALE,
            liquidation_penalty_scale = LIQUIDATION_PENALTY_SCALE,
            loss_penalty_scale = LOSS_PENALTY_SCALE,
            boundary_reward_scale = BOUNDARY_REWARD_SCALE
        )
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{i}.csv"))
    return _init

def lr_schedule(frac):
    if frac>0.5:
        return 6e-4
    else:
        return 4e-4 + (6e-4 - 4e-4) * frac * 2  
    
if __name__ == "__main__":

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(0, 1.8, 2.2), make_env(1, 2.05, 2.45), make_env(2, 2.3, 2.7), make_env(3, 2.55, 2.95), make_env(3, 2.8, 3.2)])
    #vec_env = DummyVecEnv([make_calm(0), make_vol(0), make_calm_extrimal(0), make_vol_extrimal(0)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    eval_env_1= DummyVecEnv([
        lambda: Monitor(agentDLVEnvV2(
            nect_honey_price = nect_price,
            nav_in_honey = nav,
            vol_part = volatile_part,
            vol_token_honey_price = vol_token_price,
            top_leverage = 2.95,
            bottom_leverage = 2.55,
            primordial_action_penalty = PRIMORDIAL_ACTION_PENALTY,
            penalty_damping = PENALTY_DAMPING,
            action_penalty_scale = ACTION_PENALTY_SCALE,
            liquidation_penalty_scale = LIQUIDATION_PENALTY_SCALE,
            loss_penalty_scale = LOSS_PENALTY_SCALE,
            boundary_reward_scale = BOUNDARY_REWARD_SCALE
        ), filename=os.path.join(log_dir, "eval_1.csv"))
    ])  
    eval_env_1 = VecNormalize(eval_env_1, norm_obs=False, norm_reward=False, training=False)

    eval_env_2 = DummyVecEnv([
        lambda: Monitor(agentDLVEnvV2(
            nect_honey_price = nect_price,
            nav_in_honey = nav,
            vol_part = volatile_part,
            vol_token_honey_price = vol_token_price,
            top_leverage = 2.8,
            bottom_leverage = 3.2,
            primordial_action_penalty = PRIMORDIAL_ACTION_PENALTY,
            penalty_damping = PENALTY_DAMPING,
            action_penalty_scale = ACTION_PENALTY_SCALE,
            liquidation_penalty_scale = LIQUIDATION_PENALTY_SCALE,
            loss_penalty_scale = LOSS_PENALTY_SCALE,
            boundary_reward_scale = BOUNDARY_REWARD_SCALE
        ), filename=os.path.join(log_dir, "eval_2.csv"))
    ])  
    eval_env_2 = VecNormalize(eval_env_2, norm_obs=False, norm_reward=False, training=False)

    eval_callback_1 = EvalCallback(
        eval_env_1,
        best_model_save_path=os.path.join(log_dir, "best_model_1"),
        log_path=os.path.join(log_dir, "eval_logs_1"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    eval_callback_2 = EvalCallback(
        eval_env_2,
        best_model_save_path=os.path.join(log_dir, "best_model_2"),
        log_path=os.path.join(log_dir, "eval_logs_2"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    combined_eval_callbacks = CallbackList([eval_callback_1, eval_callback_2])

    policy_kwargs = dict(
        features_extractor_class=ReLUFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        activation_fn=nn.ReLU,
        net_arch=dict(
            pi=[128, 128, 128],
            vf=[128, 128, 128]
        )  
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr_schedule,
        verbose=1,
        n_steps=1024,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        vf_coef = 1.8,
        clip_range = 0.15,
        clip_range_vf = 0.15,
        ent_coef = 0.01,
        device = "cpu"   
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
    model.save("variable_leverage_model_ppo/ppo_model_final")
    vec_env.save("variable_leverage_model_ppo/vecnormalize_stats.pkl")