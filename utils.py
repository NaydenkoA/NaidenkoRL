import matplotlib.pyplot as plt
import numpy as np

def plot(time, series, x, y, name):
    if time is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(time, series)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(name)
        plt.grid(True)
        plt.show()
    else:
        plt.figure(figsize=(8, 5))
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

def plot_new_metrics(history_1, history_2, target_leverage = 2):
    fig, axs = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    
    axs[0].plot(history_1['leverage'], label='mechanical agent')
    axs[0].plot(history_2['leverage'], label='AI agent')
    axs[0].axhline(target_leverage, color='r', linestyle='--', label='target leverage')
    axs[0].set_ylabel('leverage')
    axs[0].legend()

    axs[1].plot(history_1['collaterization ratio'], label='mechanical agent')
    axs[1].plot(history_2['collaterization ratio'], label='AI agent')
    axs[1].axhline(1.2, color='r', linestyle='--', label='min CR')
    axs[1].set_ylabel('Collateralization ratio')
    axs[1].legend()

    loss_1 = 1 - history_1['LP']
    loss_2 = 1 - history_2['LP']

    axs[2].plot(loss_1, label='mechanical agent')
    axs[2].plot(loss_2, label='AI agent')
    axs[2].set_ylabel('relative loss on spread')
    axs[2].legend()

    volume_1 = history_1['collateral value'] + history_1['volatile value'] - history_1['NECT']
    volume_2 = history_2['collateral value'] + history_2['volatile value'] - history_2['NECT']
    pnl_1 = volume_1 - volume_1[0]
    pnl_2 = volume_2 - volume_2[0]
    return_1 = pnl_1/volume_1[0]*100
    return_2 = pnl_2/volume_2[0]*100

    axs[3].plot(volume_1, label='mechanical agent')
    axs[3].plot(volume_2, label='AI agent')
    axs[3].set_ylabel('portfolio value')
    axs[3].legend()

    axs[4].plot(pnl_1, label='mechanical agent')
    axs[4].plot(pnl_1, label='AI agent')
    axs[4].set_ylabel('PNL')
    axs[4].legend()

    axs[4].plot(return_1, label='mechanical agent')
    axs[4].plot(return_1, label='AI agent')
    axs[4].set_ylabel('return (in %)')
    axs[4].legend()




    


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