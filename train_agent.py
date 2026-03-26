import pandas as pd
import numpy as np
from leverage_env import LeverageDLVEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
import torch
import random

nect_df = pd.read_csv('datas/nect_honey_price.csv')
island_df = pd.read_csv('datas/island_data.csv')

nect_price = nect_df['price'].values
nav = island_df['nav'].values
volatile_part = island_df['wbtc_frac'].values
SCALE = 6.5*10**7
sample = int(0.6*len(nect_price))

nav = nav[:sample]/SCALE
volatile_part = volatile_part[:sample]
nect_price = nect_price[:sample]

def trainPPONaive(seed = None, model_name = "ppo_leverage_model"):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    train_env = LeverageDLVEnv(
        NECT_price_series=nect_price,
        LP_volatile_part_series=volatile_part,
        LP_nav=nav
    )

    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1
    )

    model.learn(total_timesteps=100000)
    model.save(model_name)

    return model

