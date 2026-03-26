import numpy as np
import pandas as pd
from scripts_env_v_4.variable_leverage_env_4 import agentDLVEnvV4
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
from scripts_env_v_4.ReLuExtractor import ReLUFeatureExtractor
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
TOTAL_TIMESTEPS = 2*1024*200 
TARGET_LEVERAGE = 3
SEED = 400
THREADS = 10
WIDTH = 0.1
PRIMORDIAL_ACTION_PENALTY = 0.5
PENALTY_DAMPING = 0.99
ECHO_REWARD_DAMPING = 0.93
ECHO_REWARD_SCALE = 0.3
PLATEAU_FLATNESS = 3.2
ACTION_PENALTY_SCALE = 1
REWARD_SCALE = 1
REWARD_BOUND = 2
IDLE_REWARD = 0.1
OPTIMAL_REBALANCING_REWARD = 0.1 
EXTRAVAGANCE_PENALTY = 0.2
LIQUIDATION_PENALTY_SCALE = 0
LOSS_PENALTY_SCALE = 0
BOUNDARY_REWARD_SCALE = 0
POSITIVE_CONSEQUANCE_REWARD = 0.15
NEGATIVE_CONSEQUANCE_PENALTY = 0.22

log_dir = "variable_leverage_model_v4/ppo_mixed"

#torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
#np.random.seed(SEED)
#random.seed(SEED)

def make_env(i, b_l, t_l, is_reversed = True):
    def _init():
        env = agentDLVEnvV4(
            nect_honey_price = nect_price,
            nav_in_honey = nav,
            vol_part = volatile_part,
            vol_token_honey_price = vol_token_price,
            is_training = True,
            is_mixed = True,
            is_reversed = is_reversed,
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
            echo_penalty_damping = ECHO_REWARD_DAMPING,
            echo_penalty_scale = ECHO_REWARD_SCALE,
            plateau_flatness = PLATEAU_FLATNESS,
            positive_consequence_reward = POSITIVE_CONSEQUANCE_REWARD,
            negative_consequence_penalty = NEGATIVE_CONSEQUANCE_PENALTY
        )
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{i}.csv"))
    return _init

def make_env_modified(i, b_l, t_l):
    def _init():
        env = agentDLVEnvV4(
            nect_honey_price = nect_price,
            nav_in_honey = nav,
            vol_part = volatile_part,
            vol_token_honey_price = vol_token_price,
            is_training = True,
            is_mixed = True,
            is_reversed = True,
            is_syntetic = True,
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
            echo_penalty_damping=ECHO_REWARD_DAMPING,
            echo_penalty_scale=ECHO_REWARD_SCALE,
            plateau_flatness=PLATEAU_FLATNESS,
            positive_consequence_reward = POSITIVE_CONSEQUANCE_REWARD,
            negative_consequence_penalty = NEGATIVE_CONSEQUANCE_PENALTY
        )
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{i}.csv"))
    return _init

    
if __name__ == "__main__":

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    vec_env = SubprocVecEnv([
        make_env(0, 2 - WIDTH/2, 2 + WIDTH/2, False), 
        make_env(1, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env(2, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env(3, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env_modified(4, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env_modified(5, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env_modified(6, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env_modified(7, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env_modified(8, 2 - WIDTH/2, 2 + WIDTH/2), 
        make_env_modified(9, 2 - WIDTH/2, 2 + WIDTH/2)
    ])
    #vec_env = DummyVecEnv([make_calm(0), make_vol(0), make_calm_extrimal(0), make_vol_extrimal(0)])
    vec_env = VecNormalize(vec_env, norm_obs = False, norm_reward = False)

    eval_env_1= DummyVecEnv([
        lambda: Monitor(agentDLVEnvV4(
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
            echo_penalty_damping=ECHO_REWARD_DAMPING,
            echo_penalty_scale=ECHO_REWARD_SCALE,
            plateau_flatness=PLATEAU_FLATNESS,
            positive_consequence_reward = POSITIVE_CONSEQUANCE_REWARD,
            negative_consequence_penalty = NEGATIVE_CONSEQUANCE_PENALTY
        ), filename=os.path.join(log_dir, "eval_1.csv"))
    ])  
    eval_env_1 = VecNormalize(eval_env_1, norm_obs=False, norm_reward=False, training=False)

    eval_env_2 = DummyVecEnv([
        lambda: Monitor(agentDLVEnvV4(
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
            echo_penalty_damping=ECHO_REWARD_DAMPING,
            echo_penalty_scale=ECHO_REWARD_SCALE,
            plateau_flatness=PLATEAU_FLATNESS,
            positive_consequence_reward = POSITIVE_CONSEQUANCE_REWARD,
            negative_consequence_penalty = NEGATIVE_CONSEQUANCE_PENALTY
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
            pi=[256, 256],
            vf=[256, 256, 128]
        )  
    )

    def lr_schedule(frac):
        start = 3.5e-4
        end = 2e-4
        edge_frac = 0.5
        if frac > edge_frac:
            return start
        else:
            return end + (start - end) * frac / edge_frac

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate = lr_schedule,
        verbose = 1,
        n_steps = 512,
        batch_size = 256,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        vf_coef = 0.8,
        clip_range = 0.2,
        clip_range_vf = None,
        ent_coef = 0.2,
        n_epochs = 12,
        gamma = PENALTY_DAMPING,
        device = "cuda"   
    )

    print("Policy device:", model.policy.device)
    print(model.policy.features_extractor)
    print(model.policy.mlp_extractor.policy_net)
    print(model.policy.mlp_extractor.value_net)
    print("Entropy coef:", model.ent_coef)
    print("Vf coef:", model.vf_coef)

    model.learn(total_timesteps=int(TOTAL_TIMESTEPS), callback=combined_eval_callbacks)
    model.save("variable_leverage_model_v4/ppo_model_final")
    model.save("variable_leverage_model_v4/ppo_model_phase_0")
    vec_env.save("variable_leverage_model_v4/vecnormalize_stats.pkl")