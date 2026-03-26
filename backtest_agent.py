import pandas as pd
import numpy as np
from leverage_env import LeverageDLVEnv
from stable_baselines3 import PPO
from utils import plot_training_metrics
import torch
import random

nect_df = pd.read_csv('datas/nect_honey_price.csv')
island_df = pd.read_csv('datas/island_data.csv')

nect_price = nect_df['price'].values
nav = island_df['nav'].values
volatile_part = island_df['wbtc_frac'].values
SCALE = 6.5*10**7
sample = len(nect_price) - int(0.6*len(nect_price))

nav = nav[sample:]/SCALE
volatile_part = volatile_part[sample:]
nect_price = nect_price[sample:]

def backtestPPOModel(model):
    eval_env = LeverageDLVEnv(
        NECT_price_series=nect_price,
        LP_volatile_part_series=volatile_part,
        LP_nav=nav
    )

    obs = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = eval_env.step(action)

    # Get and plot the history
    history = eval_env.get_history()
    plot_training_metrics(history)
