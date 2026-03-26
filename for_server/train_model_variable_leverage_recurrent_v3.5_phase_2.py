import numpy as np
import pandas as pd
from variable_leverage_env_3 import agentDLVEnvV3
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
from customLSTM import CustomLayerNormLSTMPolicy, CustomRecurrentPPO
import torch.nn as nn

df = pd.read_csv('wbtc_train_data_expanded.csv')
nav = df['nav'].values
nect_price = df['nect_price'].values
vol_token_price = df['wbtc_price'].values
volatile_part = df['wbtc_frac'].values

nav = nav[::4]
nect_price = nect_price[::4]
vol_token_price = vol_token_price[::4]
volatile_part = volatile_part[::4]

SCALE = 6.5*10**7
TOTAL_TIMESTEPS = 2*1024*500 #default 200
TARGET_LEVERAGE = 3
SEED = 400
THREADS = 10
WIDTH = 0.4
PRIMORDIAL_ACTION_PENALTY = 0.6
PENALTY_DAMPING = 0.998
ACTION_PENALTY_SCALE = 1
REWARD_SCALE = 1
REWARD_BOUND = 2
IDLE_REWARD = 0.2
OPTIMAL_REBALANCING_REWARD = 0.15 #0.08
EXTRAVAGANCE_PENALTY = 0.14
LIQUIDATION_PENALTY_SCALE = 0
LOSS_PENALTY_SCALE = 0
BOUNDARY_REWARD_SCALE = 0

log_dir = "variable_leverage_model_v3.5_phase_2/ppo_mixed"

#torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
#np.random.seed(SEED)
#random.seed(SEED)

def make_env(i, b_l, t_l):
    def _init():
        env = agentDLVEnvV3(
            nect_honey_price = nect_price,
            nav_in_honey = nav,
            vol_part = volatile_part,
            vol_token_honey_price = vol_token_price,
            is_training = True,
            is_mixed = True,
            top_leverage = max(t_l, b_l),
            bottom_leverage = min(t_l, b_l),
            primordial_action_penalty = PRIMORDIAL_ACTION_PENALTY,
            penalty_damping = PENALTY_DAMPING,
            action_penalty_scale = ACTION_PENALTY_SCALE,
            liquidation_penalty_scale = LIQUIDATION_PENALTY_SCALE,
            loss_penalty_scale = LOSS_PENALTY_SCALE,
            boundary_reward_scale = BOUNDARY_REWARD_SCALE,
            reward_scale = REWARD_SCALE,
            bound = REWARD_BOUND,
            idle_reward = IDLE_REWARD,
            optimal_rebalancing_reward = OPTIMAL_REBALANCING_REWARD,
            extravagance_penalty = EXTRAVAGANCE_PENALTY,
            show_setup = False
        )
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{i}.csv"))
    return _init

def make_env_modified(i, b_l, t_l):
    def _init():
        env = agentDLVEnvV3(
            nect_honey_price = nect_price,
            nav_in_honey = nav,
            vol_part = volatile_part,
            vol_token_honey_price = vol_token_price,
            is_training = True,
            is_mixed = True,
            is_reversed = True,
            top_leverage = max(t_l, b_l),
            bottom_leverage = min(t_l, b_l),
            primordial_action_penalty = PRIMORDIAL_ACTION_PENALTY,
            penalty_damping = PENALTY_DAMPING,
            action_penalty_scale = ACTION_PENALTY_SCALE,
            liquidation_penalty_scale = LIQUIDATION_PENALTY_SCALE,
            loss_penalty_scale = LOSS_PENALTY_SCALE,
            boundary_reward_scale = BOUNDARY_REWARD_SCALE,
            reward_scale = REWARD_SCALE,
            bound = REWARD_BOUND,
            idle_reward = IDLE_REWARD,
            optimal_rebalancing_reward = OPTIMAL_REBALANCING_REWARD,
            extravagance_penalty = EXTRAVAGANCE_PENALTY,
            show_setup = False
        )
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{i}.csv"))
    return _init

    
if __name__ == "__main__":

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    vec_env = SubprocVecEnv([
        make_env(0, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env(1, 2.25 - WIDTH/2, 2.25 + WIDTH/2), 
        make_env(2, 2.5 - WIDTH/2, 2.5 + WIDTH/2), 
        make_env(3, 2.75 - WIDTH/2, 2.75 + WIDTH/2), 
        make_env(4, 3 - WIDTH/2, 3 + WIDTH/2), 
        make_env_modified(5, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env_modified(6, 2.25 - WIDTH/2, 2.25 + WIDTH/2), 
        make_env_modified(7, 2.5 - WIDTH/2, 2.5 + WIDTH/2), 
        make_env_modified(8, 2.75 - WIDTH/2, 2.75 + WIDTH/2), 
        make_env_modified(9, 3 - WIDTH/2, 3 + WIDTH/2)
    ])
    vec_env = VecNormalize(vec_env, norm_obs = False, norm_reward = False)
    #vec_env = DummyVecEnv([make_calm(0), make_vol(0), make_calm_extrimal(0), make_vol_extrimal(0)])
    

    eval_env_1= DummyVecEnv([
        lambda: Monitor(agentDLVEnvV3(
            nect_honey_price = nect_price,
            nav_in_honey = nav,
            vol_part = volatile_part,
            vol_token_honey_price = vol_token_price,
            top_leverage = 2.2,
            bottom_leverage = 1.8,
            primordial_action_penalty = PRIMORDIAL_ACTION_PENALTY,
            penalty_damping = PENALTY_DAMPING,
            action_penalty_scale = ACTION_PENALTY_SCALE,
            liquidation_penalty_scale = LIQUIDATION_PENALTY_SCALE,
            loss_penalty_scale = LOSS_PENALTY_SCALE,
            boundary_reward_scale = BOUNDARY_REWARD_SCALE,
            reward_scale = REWARD_SCALE,
            bound = REWARD_BOUND,
            idle_reward = IDLE_REWARD,
            optimal_rebalancing_reward = OPTIMAL_REBALANCING_REWARD,
            show_setup = False
        ), filename=os.path.join(log_dir, "eval_1.csv"))
    ])  
    eval_env_1 = VecNormalize(eval_env_1, norm_obs=False, norm_reward=False, training=False)

    eval_env_2 = DummyVecEnv([
        lambda: Monitor(agentDLVEnvV3(
            nect_honey_price = nect_price,
            nav_in_honey = nav,
            vol_part = volatile_part,
            vol_token_honey_price = vol_token_price,
            top_leverage = 3.2,
            bottom_leverage = 2.8,
            primordial_action_penalty = PRIMORDIAL_ACTION_PENALTY,
            penalty_damping = PENALTY_DAMPING,
            action_penalty_scale = ACTION_PENALTY_SCALE,
            liquidation_penalty_scale = LIQUIDATION_PENALTY_SCALE,
            loss_penalty_scale = LOSS_PENALTY_SCALE,
            boundary_reward_scale = BOUNDARY_REWARD_SCALE,
            reward_scale = REWARD_SCALE,
            bound = REWARD_BOUND,
            idle_reward = IDLE_REWARD,
            optimal_rebalancing_reward = OPTIMAL_REBALANCING_REWARD,
            show_setup = False
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
        features_extractor_class = ReLUFeatureExtractor,
        features_extractor_kwargs = dict(),
        activation_fn = nn.GELU,
        lstm_hidden_size = 256,      
        n_lstm_layers = 1,            
        shared_lstm = False,
        enable_critic_lstm = True,
        net_arch=dict(
            pi=[256, 128],
            vf=[256, 256, 128]
        )  
    )

    def lr_schedule(frac):
        start = 1.8e-4
        end = 0.5e-4
        edge_frac = 0.5
        if frac > edge_frac:
            return start
        else:
            return end + (start - end) * frac / edge_frac

    model = CustomRecurrentPPO(
        vec_env,
        learning_rate = lr_schedule,
        verbose = 1,
        n_steps = 512,
        batch_size = 320,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        vf_coef = 0.9,
        clip_range = 0.2,
        clip_range_vf = None,
        ent_coef = 0.105,
        n_epochs = 12,
        gamma = 0.998,
        device = "cuda"   
    )

    model.set_parameters("variable_leverage_model_v3.5_phase_1/ppo_model_phase_1")

    print("Policy device:", model.policy.device)
    print(model.policy.features_extractor)
    print(model.policy.mlp_extractor.policy_net)
    print(model.policy.mlp_extractor.value_net)
    print("Entropy coef:", model.ent_coef)
    print("Vf coef:", model.vf_coef)

    """print(model.policy.lstm_actor.cell.__class__)
    print(model.policy.lstm_critic.cell.__class__)"""

    model.learn(total_timesteps=int(TOTAL_TIMESTEPS), callback=combined_eval_callbacks)
    model.save("variable_leverage_model_v3.5_phase_2/ppo_model_phase_2")
    vec_env.save("variable_leverage_model_v3.5_phase_2/vecnormalize_stats.pkl")