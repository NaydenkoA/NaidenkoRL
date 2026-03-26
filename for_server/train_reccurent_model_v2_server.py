import numpy as np
import pandas as pd
from variable_leverage_env import agentDLVEnvV2
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

nav_vol = nav_vol[0::3]
vol_part_vol = vol_part_vol[0::3]
nect_price_vol = nect_price_vol[0::3]

SCALE = 6.5*10**7
LEARNING_EPOCHS = 2*1024*200 #default 200
SEED = 400
THREADS = 2
nav_vol = nav_vol/SCALE
nav_calm = nav_calm/SCALE

torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
np.random.seed(SEED)
random.seed(SEED)

def make_env(i, top_leverage, bottom_leverage):
    def _init():
        env= agentDLVEnvV2(
            nect_honey_price=nect_price_vol,
            nav_in_honey=nav_vol,
            vol_part=vol_part_vol,
            vol_token_honey_price=100_000*np.ones(len(nav_vol)),
            is_training=True,
            is_mixed=True,
            top_leverage=top_leverage,
            bottom_leverage=bottom_leverage
        )
        return Monitor(env, filename=os.path.join(log_dir, f"calm_monitor_{i}.csv"))
    return _init

"""print(torch.cuda.is_available())   # should be True
print(torch.cuda.device_count())   # should be ≥1
print(torch.cuda.get_device_name(0))"""


def lr_schedule(frac):
    if frac>0.5:
        return 2e-4
    else:
        return 2e-4#6e-4 + (9e-4 - 6e-4) * frac * 2  

def ent_schedule(frac):
    if frac>0.5:
        return 0.06 + 0.002
    else:
        return 0.06 * 2 * frac + 0.002

if __name__ == "__main__":

    log_dir = "reccurent_loop_model_v2/ppo_mixed"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(0, 2.2 ,1.8), make_env(1, 2.2 ,1.8)])
    #vec_env = DummyVecEnv([make_env(0, 2.2, 1.8)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)

    eval_env_1 = DummyVecEnv([
        lambda: Monitor(agentDLVEnvV2(
            nect_honey_price=nect_price_vol,
            nav_in_honey=nav_vol,
            vol_part=vol_part_vol,
            vol_token_honey_price=100_000*np.ones(len(nav_vol)),
            top_leverage=2.2,
            bottom_leverage=1.8
        ), filename=os.path.join(log_dir, "eval_calm.csv"))
    ])  
    eval_env_1 = VecNormalize(eval_env_1, norm_obs=False, norm_reward=True, training=False)
    eval_env_1.ret_rms = vec_env.ret_rms

    eval_callback_1= EvalCallback(
        eval_env_1,
        best_model_save_path=os.path.join(log_dir, "best_model_calm"),
        log_path=os.path.join(log_dir, "eval_logs_calm"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    combined_eval_callbacks = CallbackList([eval_callback_1])

    policy_kwargs = dict(
        features_extractor_class=ReLUFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        lstm_hidden_size=64,      
        n_lstm_layers=1,           
        shared_lstm=True,
        enable_critic_lstm=False,
        activation_fn=nn.ReLU,
        net_arch=dict(
            pi=[128, 128], 
            vf=[256, 256, 128]
        )  
    )

    """policy_kwargs = dict(
        features_extractor_class=ReLUFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64)
    )"""

    model = RecurrentPPO(
        "MlpLstmPolicy",
        #"MlpPolicy",
        vec_env,
        learning_rate=lr_schedule,
        verbose=1,
        n_steps=1024,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        vf_coef = 2.4,
        clip_range = 0.15,
        clip_range_vf = 0.25,
        ent_coef = 0.03,
        device = "cpu"   
    )

    print("Policy device:", model.policy.device)
    print(model.policy.features_extractor)
    print(model.policy.mlp_extractor.policy_net)
    print(model.policy.mlp_extractor.value_net)
    print("Entropy coef:", model.ent_coef)
    print("Vf coef:", model.vf_coef)

    """pid = os.getpid()
    print(f"My training PID is {pid}")"""

    model.learn(total_timesteps=int(LEARNING_EPOCHS), callback=combined_eval_callbacks)
    model.save("reccurent_loop_model_v2/ppo_model_final")
    vec_env.save("reccurent_loop_model_v2/vecnormalize_stats.pkl")