import matplotlib.pyplot as plt
import numpy as np

def plot(time, series, x, y, name):
    if time is not None:
        plt.figure(figsize=(12, 5))
        plt.plot(time, series)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(name)
        plt.grid(True)
        plt.show()
    else:
        plt.figure(figsize=(12, 5))
        plt.plot(series)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(name)
        plt.grid(True)
        plt.show()

def plotLeverage(time, series, target, label, x, y, name):
    plt.figure(figsize=(12,5))
    plt.plot(time, series, label=label)
    plt.axhline(target, linestyle='--', color='red', label='Target Leverage')
    plt.title(name)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_episode_history_v2(history, top_leverage, bottom_leverage):
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(history['leverage'], label='current leverage')
    axs[0].axhline(top_leverage, color='r', linestyle='--', label='leverage boundary')
    axs[0].axhline(bottom_leverage, color='r', linestyle='--')
    axs[0].set_ylabel('leverage')
    axs[0].legend()
    
    axs[1].plot(history['loss on spread'])
    axs[1].set_ylabel('relative loss on spread')

    axs[2].plot(history['boundary leverage'])
    axs[2].set_ylabel('boundary leverage')
    axs[2].axhline(top_leverage, color='r', linestyle='--', label='leverage boundary')
    axs[2].axhline(bottom_leverage, color='r', linestyle='--')
    axs[2].legend()

    axs[3].plot(history['collaterization ratio'], label='current cr')
    axs[3].axhline(1.2, color='r', linestyle='--', label='min cr')
    axs[3].set_ylabel('collaterization ratio')
    axs[3].legend()

    axs[4].plot(history['reward'])
    axs[4].set_ylabel('reward')

    plt.tight_layout()
    plt.show()

def plot_economic_metrics(history, top_leverage, bottom_leverage):
    fig, axs = plt.subplots(10, 1, figsize=(12, 20), sharex=True)

    axs[0].plot(history['leverage'], label='current leverage')
    axs[0].axhline(top_leverage, color='r', linestyle='--', label='leverage boundary')
    axs[0].axhline(bottom_leverage, color='r', linestyle='--')
    axs[0].set_ylabel('leverage')
    axs[0].legend()
    
    axs[1].plot(history['PNL'])
    axs[1].set_ylabel('PNL')

    axs[2].plot(np.array(history['return'])*100)
    axs[2].set_ylabel('return, %')

    axs[3].plot(history['collaterization ratio'], label='current cr')
    axs[3].axhline(1.2, color='r', linestyle='--', label='min cr')
    axs[3].set_ylabel('collaterization ratio')
    axs[3].legend()

    axs[4].plot(history['vol token amount'])
    axs[4].set_ylabel('vol token amount')

    axs[5].plot(history['nect debt'])
    axs[5].set_ylabel('nect debt')

    axs[6].plot(history['vol token price'])
    axs[6].set_ylabel('vol token price (in honey)')

    axs[7].plot(history['nect price'])
    axs[7].set_ylabel('nect price (in honey)')

    axs[8].plot(history['nav'])
    axs[8].set_ylabel('lp token nav (in honey)')

    axs[9].plot(np.array(history['calm part'])*100)
    axs[9].set_ylabel('calm part, %')

    plt.tight_layout()
    plt.show()

def plot_metrics_v2(history, name):
    n = len(history)
    fig, axs = plt.subplots(10, 1, figsize=(12, 25), sharex=True)

    for i in range(n):
        axs[0].plot(history[i]['leverage'], label = name[i])
    axs[0].set_ylabel('leverage')
    axs[0].legend()
    
    for i in range(n):
        axs[1].plot(history[i]['leverage gain'], label = name[i])
    axs[1].set_ylabel('leverage gain')
    axs[1].legend()

    for i in range(n):
        axs[2].plot(history[i]['leverage mean'], label = name[i])
    axs[2].set_ylabel('leverage mean')
    axs[2].legend()

    for i in range(n):
        axs[3].plot(history[i]['volatile token'], label = name[i])
    axs[3].set_ylabel('volatile token')
    axs[3].legend()

    for i in range(n):
        axs[4].plot(history[i]['volatile token mean'], label = name[i])
    axs[4].set_ylabel('volatile token mean')
    axs[4].legend()

    for i in range(n):
        axs[5].plot(history[i]['volatile token gain'], label = name[i])
    axs[5].set_ylabel('volatile token gain')
    axs[5].legend()

    for i in range(n):
        axs[6].plot(history[i]['calm part'], label = name[i])
    axs[6].set_ylabel('calm part')
    axs[6].legend()

    for i in range(n):
        axs[7].plot(history[i]['calm part mean'], label = name[i])
    axs[7].set_ylabel('calm part mean')
    axs[7].legend()

    for i in range(n):
        axs[8].plot(history[i]['calm part gain'], label = name[i])
    axs[8].set_ylabel('calm part gain')
    axs[8].legend()

    for i in range(n):
        axs[9].plot(history[i]['collaterization ratio'], label = name[i])
    axs[9].set_ylabel('collaterization ratio')
    axs[9].legend()

    """axs[10].plot(history['boundary leverage'])
    axs[10].set_ylabel('boundary leverage')"""

    plt.tight_layout()
    plt.show()

def plot_metrics_v3(history):
    fig, axs = plt.subplots(16, 1, figsize=(16, 25), sharex=True)

    axs[0].plot(history['leverage'])
    axs[0].set_ylabel('leverage')
    
    axs[1].plot(history['leverage gain'])
    axs[1].set_ylabel('leverage gain')

    axs[2].plot(history['leverage mean'])
    axs[2].set_ylabel('leverage mean')

    axs[3].plot(history['volatile token'])
    axs[3].set_ylabel('volatile token')

    axs[4].plot(history['volatile token mean'])
    axs[4].set_ylabel('volatile token mean')

    axs[5].plot(history['volatile token gain'])
    axs[5].set_ylabel('volatile token gain')

    axs[6].plot(history['calm part'])
    axs[6].set_ylabel('calm part')

    axs[7].plot(history['calm part mean'])
    axs[7].set_ylabel('calm part mean')

    axs[8].plot(history['calm part gain'])
    axs[8].set_ylabel('calm part gain')

    axs[9].plot(history['collaterization ratio'])
    axs[9].set_ylabel('collaterization ratio')

    axs[10].plot(history['leverage vol'])
    axs[10].set_ylabel('leverage vol')

    axs[11].plot(history['volatile token vol'])
    axs[11].set_ylabel('volatile token vol')

    axs[12].plot(history['calm part vol'])
    axs[12].set_ylabel('calm part vol')

    axs[13].plot(history['leverage consistency'])
    axs[13].set_ylabel('leverage consistency')

    axs[14].plot(history['volatile token consistency'])
    axs[14].set_ylabel('volatile token consistency')

    axs[15].plot(history['calm part consistency'])
    axs[15].set_ylabel('calm part consistency')

    plt.tight_layout()
    plt.show()

def plot_metrics_v4(history):
    fig, axs = plt.subplots(32, 1, figsize=(16, 60), sharex=True)

    axs[0].plot(history['leverage'])
    axs[0].set_ylabel('leverage')
    
    axs[1].plot(history['leverage gain'])
    axs[1].set_ylabel('leverage gain')

    axs[2].plot(history['leverage mean'])
    axs[2].set_ylabel('leverage mean')

    axs[3].plot(history['volatile token'])
    axs[3].set_ylabel('volatile token')

    axs[4].plot(history['volatile token mean'])
    axs[4].set_ylabel('volatile token mean')

    axs[5].plot(history['volatile token gain'])
    axs[5].set_ylabel('volatile token gain')

    axs[6].plot(history['calm part'])
    axs[6].set_ylabel('calm part')

    axs[7].plot(history['calm part mean'])
    axs[7].set_ylabel('calm part mean')

    axs[8].plot(history['calm part gain'])
    axs[8].set_ylabel('calm part gain')

    axs[9].plot(history['collaterization ratio'])
    axs[9].set_ylabel('collaterization ratio')

    axs[10].plot(history['leverage vol'])
    axs[10].set_ylabel('leverage vol')

    axs[11].plot(history['volatile token vol'])
    axs[11].set_ylabel('volatile token vol')

    axs[12].plot(history['calm part vol'])
    axs[12].set_ylabel('calm part vol')

    axs[13].plot(history['leverage consistency'])
    axs[13].set_ylabel('leverage consistency')

    axs[14].plot(history['volatile token consistency'])
    axs[14].set_ylabel('volatile token consistency')

    axs[15].plot(history['calm part consistency'])
    axs[15].set_ylabel('calm part consistency')

    axs[16].plot(history['p return'])
    axs[16].set_ylabel('p return')

    axs[17].plot(history['p return abs'])
    axs[17].set_ylabel('p return abs')

    axs[18].plot(history['p return square'])
    axs[18].set_ylabel('p return square')

    axs[19].plot(history['p return vol'])
    axs[19].set_ylabel('p return vol')

    axs[20].plot(history['c return'])
    axs[20].set_ylabel('c return')

    axs[21].plot(history['c return abs'])
    axs[21].set_ylabel('c return abs')

    axs[22].plot(history['c return square'])
    axs[22].set_ylabel('c return square')

    axs[23].plot(history['c return vol'])
    axs[23].set_ylabel('c return vol')

    axs[24].plot(history['l return'])
    axs[24].set_ylabel('l return')

    axs[25].plot(history['l return abs'])
    axs[25].set_ylabel('l return abs')

    axs[26].plot(history['l return square'])
    axs[26].set_ylabel('l return square')

    axs[27].plot(history['l return vol'])
    axs[27].set_ylabel('l return vol')

    axs[28].plot(history['nav return'])
    axs[28].set_ylabel('nav return')

    axs[29].plot(history['nav return abs'])
    axs[29].set_ylabel('nav return abs')

    axs[30].plot(history['nav return square'])
    axs[30].set_ylabel('nav return square')

    axs[31].plot(history['nav return vol'])
    axs[31].set_ylabel('nav return vol')

    plt.tight_layout()
    plt.show()

def plot_training_metrics(history, target_leverage = 2, show_loop = False):
    if show_loop:
        fig, axs = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    else:
        fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(history['leverage'], label='current leverage')
    axs[0].axhline(target_leverage, color='r', linestyle='--', label='target leverage')
    axs[0].set_ylabel('leverage')
    axs[0].legend()

    axs[1].plot(history['collaterization ratio'], label='actual CR')
    axs[1].axhline(1.2, color='r', linestyle='--', label='min CR')
    axs[1].set_ylabel('Collateralization ratio')
    axs[1].legend()

    axs[2].plot(history['LP'], label='LP token amount')
    axs[2].set_ylabel('token amount')
    axs[2].set_xlabel('Step')
    axs[2].legend()

    axs[3].plot(history['collateral value'], label='Collateral price')
    axs[3].plot(history['volatile value'], label='Volatile part')
    axs[3].set_ylabel('price (rescaled)')
    axs[3].set_xlabel('Step')
    axs[3].legend()

    axs[4].plot(history['reward'], label='reward')
    axs[4].set_ylabel('reward')
    axs[4].set_xlabel('Step')
    axs[4].legend()

    if show_loop:
        axs[5].plot(history['cumulative debt'])
        axs[5].set_ylabel('looped leverage')
        axs[5].set_xlabel('Step')
        axs[5].legend()

    plt.tight_layout()
    plt.show()

def plot_training_comparison(agent_history, history, name, target_leverage = 2):
    n = len(history)

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)    

    axs[0].plot(agent_history['leverage'], label='mechanical agent')
    for i in range(n):
        axs[0].plot(history[i]['leverage'], label=f'AI agent {i + 1}')
    axs[0].axhline(target_leverage, color='r', linestyle='--', label='target leverage')
    axs[0].set_ylabel('leverage')
    axs[0].legend()

    axs[1].plot(agent_history['LP'], label='mechanical agent')
    for i in range(n):
        axs[1].plot(history[i]['LP'], label=f'AI agent {i + 1}')
    axs[1].set_ylabel('LP token amount')
    axs[1].set_xlabel('Step')
    axs[1].legend()

    axs[2].plot(agent_history['reward'], label='mechanical agent')
    for i in range(n):
        axs[2].plot(history[i]['reward'], label=f'AI agent {i + 1}')
    axs[2].set_ylabel('reward')
    axs[2].set_xlabel('Step')
    axs[2].legend()

    axs[3].plot(agent_history['collaterization ratio'], label='mechanical agent')
    axs[3].axhline(1.2, color='r', linestyle='--', label='min CR')
    for i in range(n):
        axs[3].plot(history[i]['collaterization ratio'], label=f'AI agent {i + 1}')
    axs[3].set_ylabel('CR')
    axs[3].set_xlabel('Step')
    axs[3].legend()

    fig.suptitle(name)
    plt.tight_layout()
    plt.show()

def plot_model_comparison(history, names, target_leverage = 2):
    n = len(history)

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)    

    for i in range(n):
        axs[0].plot(history[i]['leverage'], label=names[i])
    axs[0].axhline(target_leverage, color='r', linestyle='--', label='target leverage')
    axs[0].set_ylabel('leverage')
    axs[0].legend()

    for i in range(n):
        axs[1].plot(history[i]['LP'], label=names[i])
    axs[1].set_ylabel('LP token amount')
    axs[1].set_xlabel('Step')
    axs[1].legend()

    for i in range(n):
        axs[2].plot(history[i]['reward'], label=names[i])
    axs[2].set_ylabel('reward')
    axs[2].set_xlabel('Step')
    axs[2].legend()

    axs[3].axhline(1.2, color='r', linestyle='--', label='min CR')
    for i in range(n):
        axs[3].plot(history[i]['collaterization ratio'], label=names[i])
    axs[3].set_ylabel('CR')
    axs[3].set_xlabel('Step')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

def distribution_comparison(history, names, title, target_leverage = 2):
    n = len(history)

    plt.figure(figsize=(12,5))
    data = []
    for i in range(n):
        leverage = np.array(history[i]['leverage'])
        norm_leverage = np.abs(leverage - target_leverage)/target_leverage
        data.append(norm_leverage)
         
    plt.hist(data, bins=30, density=True, alpha = 0.65, label=names, edgecolor='black') 
    plt.xlabel('Deviation from the target leverage')
    plt.ylabel('Deviation distribution')
    plt.title(f'Deviation from the target leverage comparison ({title})')
    plt.grid(True)
    plt.legend()
    plt.show()

def reward_comparison(history, names):
    n = len(history)

    plt.figure(figsize=(12,5))
    data = []
    for i in range(n):
        reward = np.array(history[i]['reward'])
        data.append(reward)
          
    plt.hist(data, bins=20, alpha = 0.65, density=True, label=names, edgecolor='black')
    plt.xlabel('reward')
    plt.ylabel('Reward distribution')
    plt.title('Reward comparison')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()