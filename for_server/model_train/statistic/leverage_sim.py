from scripts.variable_leverage_env_3 import agentDLVEnvV3
import numpy as np
from data_generator import simulate_correlated_gbm, simulate_gbm, simulate_4series
from scripts.utils import plot_episode_history_v2

def naiveAgent(obs, min_cr=1.2):
    leverage = obs[0]
    cr = 1/obs[1]

    if cr <= min_cr + 0.03:
        return 1
    if leverage >= 0.95 or leverage <= 0.05:
        return 1
    else:
        return 0

def simulate_single_episode(sigma1, sigma2 = 0.085):
    s1, s2, _, _ = simulate_correlated_gbm(
        n = 4000,
        dt = 1/250,
        S10 = 1,
        S20 = 1,
        mu1 = 0,
        mu2 = 0,
        sigma1 = sigma1,
        sigma2 = sigma2,
        rho = 0.9915
    )

    s, _ = simulate_gbm(
        n = 4000,
        dt = 1/250,
        S0 = 1,
        mu = 0,
        sigma = 2.5*sigma1
    )

    s_calm, _ = simulate_gbm(
        n = 4000,
        dt = 1/250,
        S0 = 1,
        mu = 0,
        sigma = 0.001
    )

    vol_token_price = 2*s1
    volatile_part = s1/(s1 + s)
    nav = 10*s2
    nect_price = 0.5*s_calm

    eval_env = agentDLVEnvV3(
        nect_honey_price = nect_price,
        nav_in_honey = nav,
        vol_part = volatile_part,
        vol_token_honey_price = vol_token_price,
        is_training = False,
        is_mixed = False,
        is_reversed = False,
        index = False,
        show_setup = False,
        top_leverage= 2.2,
        bottom_leverage = 1.8
    )

    obs, _ = eval_env.reset()
    done = False

    while not done:
        action = naiveAgent(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

    history = info['episode_history']
    action_1 = info.get('0.5 rebalancing') 
    #plot_episode_history_v2(history, 2.2, 1.8)

    return action_1*100


def simulate_single_episode_v2(sigma = 0.08522254220295641):
    S, _ = simulate_4series(
        5000,
        1/250,
        [85000,20,20*85000,0.15],
        [0,0,0,0],
        [sigma, 0.1764965560805584, 0.17615395829118863, 0.10679331573472989]
    )

    s_calm, _ = simulate_gbm(
        n = 5000,
        dt = 1/250,
        S0 = 1,
        mu = 0,
        sigma = 0.001
    )

    vol_token_price = S[0]
    vol_token_balance = S[1]
    stable_token_balance = S[2]
    ts = S[3]

    volatile_part = vol_token_price*vol_token_balance/(vol_token_price*vol_token_balance + stable_token_balance)
    nav = (vol_token_price*vol_token_balance + stable_token_balance)/ts
    nect_price = s_calm

    eval_env = agentDLVEnvV3(
        nect_honey_price = nect_price,
        nav_in_honey = nav,
        vol_part = volatile_part,
        vol_token_honey_price = vol_token_price,
        is_training = False,
        is_mixed = False,
        is_reversed = False,
        index = False,
        show_setup = False,
        top_leverage= 2.2,
        bottom_leverage = 1.8
    )

    obs, _ = eval_env.reset()
    done = False

    while not done:
        action = naiveAgent(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

    history = info['episode_history']
    action_1 = info.get('0.5 rebalancing') 
    #plot_episode_history_v2(history, 2.2, 1.8)

    return action_1*100
