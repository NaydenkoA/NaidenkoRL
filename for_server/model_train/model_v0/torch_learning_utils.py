import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from variable_leverage_env_2 import agentDLVEnvV2
from torch_model import reccurentActor
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.optim as optim
from tqdm import trange
import math
from special_functions import sigmoidBarier

def naiveAgent(obs, min_cr=1.2, eps=1e-9):
    leverage = obs[0]
    cr = 1/obs[1]

    if cr <= min_cr + 0.03:
        return 1
    if leverage >= 0.95 or leverage <= 0.05:
        return 1
    else:
        return 0

def generateEpisode(rng, model: reccurentActor, env1: agentDLVEnvV2, env2: agentDLVEnvV2, epsilon = None, confidence_level = 0.99):
    seed = rng.integers(0, 1e6)

    obs, _ = env1.reset(seed = int(seed))
    done = False
    
    while not done:
        action = naiveAgent(obs)
        obs, reward, terminated, truncated, info = env1.step(action)
        done = terminated or truncated
        if done:
            mech_integral_loss = info['integral loss']
            mech_steps = info['steps']
            mech_quantile = info['quantile']

    obs, _ = env2.reset(seed = int(seed))
    done = False
    hidden = model.init_hidden(batch_size=1, device='cpu')

    ep_obs = []
    ep_actions = []
    ep_logs = []
    ep_entropies = []

    while not done:
        action, log_prob, entropy, hidden = model.act(obs=obs, hidden=hidden, epsilon=epsilon, confidence_level=confidence_level)
        obs, reward, terminated, truncated, info = env2.step(action)

        ep_obs.append(obs)
        ep_actions.append(action.item())
        ep_logs.append(log_prob)
        ep_entropies.append(entropy)

        done = terminated or truncated
        if done:
            ep_integral_loss = info['integral loss']
            ep_steps = info['steps']
            ep_quantile = info['quantile']

    reward = mech_integral_loss - ep_integral_loss

    return {
        "reward": reward,
        "obs": np.array(ep_obs),
        "actions": np.array(ep_actions),
        "log_probs": torch.stack(ep_logs),
        "entropy": torch.stack(ep_entropies),
        "mechanical_loss": mech_integral_loss,
        "agent_loss": ep_integral_loss,
        "steps": ep_steps,
        "mech_quantile": mech_quantile,
        "agent_quantile": ep_quantile
    }

def generateEpisodeWorker(model_state, configs, seed, epsilon = None, confidence_level = 0.99, show_setup = True):
    local_rng = np.random.default_rng(seed)
    #index = local_rng.integers(1, 101)

    env1 = agentDLVEnvV2(
        nect_honey_price = configs['nect_price'],
        nav_in_honey = configs['nav'],
        vol_part = configs['vol_part'],
        vol_token_honey_price = configs['vol_token_honey_price'],
        is_training = True,
        is_mixed = True,
        show_setup = show_setup,
        is_reversed = configs['is_reversed'],
        top_leverage = configs['top_leverage'],
        bottom_leverage = configs['bottom_leverage']
    )
    
    env2 = agentDLVEnvV2(
        nect_honey_price = configs['nect_price'],
        nav_in_honey = configs['nav'],
        vol_part = configs['vol_part'],
        vol_token_honey_price = configs['vol_token_honey_price'],
        is_training = True,
        is_mixed = True,
        show_setup = False,
        is_reversed = configs['is_reversed'],
        top_leverage = configs['top_leverage'],
        bottom_leverage = configs['bottom_leverage']
    )

    local_model = reccurentActor()
    local_model.load_state_dict(model_state)
    local_model.eval()

    with torch.no_grad():                   
        result = generateEpisode(
            local_rng, 
            local_model,
            env1, 
            env2, 
            epsilon = epsilon, 
            confidence_level = confidence_level
        )
    
    return result

def generateSeveralEpisodes(model_state, configs, n_envs = 6, epsilon = None, confidence_level = 0.99):
    rng = np.random.default_rng()

    results = []

    show_setup = True
    n_configs = len(configs)
    for env in range(n_envs):
        if env < n_configs:
            config = configs[env]
        else:
            config = configs[-1]

        results.append(generateEpisodeWorker(
            model_state=model_state,
            configs=config, 
            seed=int(rng.integers(0, 1e6)), 
            show_setup=show_setup,
            epsilon = epsilon,
            confidence_level = confidence_level
        ))
        if show_setup:
            show_setup = False

    return results

def generateSeveralEpisodesParallel(model_state, configs, n_envs = 6, epsilon = None, confidence_level = 0.99):
    rng = np.random.default_rng()

    results = []

    n_configs = len(configs)
    with ProcessPoolExecutor(max_workers=n_envs) as executor:
        futures = []
        for env in range(n_envs):
            seed_i = int(rng.integers(0, 1e6))

            if env < n_configs:
                config = configs[env]
            else:
                config = configs[-1]

            futures.append(executor.submit(
                generateEpisodeWorker,
                model_state,
                config,
                seed_i,
                epsilon,
                confidence_level,
                False
            ))
    
    for f in as_completed(futures):
        try:
            results.append(f.result())
        except Exception as e:
            print(f"Worker failed: {e}")
    
    return results

def calculate_reward(reward, quantile):
    reward_normed = 2/math.pi*math.atan(reward) 
    if reward < 0:
        reward_normed += 0.1*reward
    else:
        reward_normed += 0.05*reward
    quantile_penalty = sigmoidBarier(0.5 - quantile)

    total_reward = reward_normed - quantile_penalty

    return total_reward
    """if total_reward < -2:
        return -2
    else:
        if total_reward > 2:
            return 2
        else:
            return total_reward"""
    

def train_model(
        model: reccurentActor, 
        n_episodes, 
        configs, 
        device = 'cpu',
        epsilon = 0.05,
        lr = 3e-4,
        ent_coef = 0,
        ent_coef_alt = 1e-3
    ):

    def resolve_schedule(x, frac):
        if callable(x):
            return x(frac)
        else:
            return x

    reward_baseline = 0.0
    baseline_alpha = 0.05

    n_envs = len(configs)

    optimizer = torch.optim.Adam(model.parameters(), lr=resolve_schedule(lr, 0))
    model.to(device=device)

    training_history = {
        'mean_reward': [],
        'policy_loss': [],
        'entropy_loss': [],
        'effective_entropy_loss': [],
        'average_batch_loss': [],
        'clips': [],
        'epsilon': [],
        'ent_coef': [],
        'ent_coef_alt': [],
        'learning_rate': [],
        'entropy': [],
        'entropy_alt': []
    }

    for episode in trange(n_episodes, desc="Training"):
        frac = episode/(n_episodes - 1) if n_episodes > 1 else 1

        epsilon_t = resolve_schedule(epsilon, frac)
        ent_coef_t = resolve_schedule(ent_coef, frac)
        ent_coef_alt_t = resolve_schedule(ent_coef_alt, frac)
        lr_t = resolve_schedule(lr, frac)

        for g in optimizer.param_groups:
            g['lr'] = lr_t

        ep_data_full = generateSeveralEpisodesParallel(
            model_state = model.state_dict(),
            configs = configs,
            n_envs = n_envs,
            epsilon = epsilon_t
        )

        batch_loss = 0
        batch_reward = 0
        batch_entr_loss = 0
        batch_eff_entr_loss = 0
        batch_policy_loss = 0

        batch_entr = 0
        batch_entr_alt = 0

        for ep in ep_data_full:
            obs_np = ep["obs"]
            actions_np = ep["actions"] 
            reward = ep["reward"]
            quantile = ep["agent_quantile"]

            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            actions_t = torch.tensor(actions_np, dtype=torch.long, device=device) 
            hidden = model.init_hidden(batch_size=1, device=device)

            logits, _ = model.forward(obs_t, hidden)          
            logits = logits[0]                                
            dist = torch.distributions.Categorical(logits=logits)

            probs = torch.softmax(logits, dim=-1)           
            p0 = probs[:, 0]                                
            S = (1 - p0).clamp(min=1e-8)                    
            alt_probs = probs[:, 1:] / S.unsqueeze(-1)      
            effective_entropies = -(alt_probs * torch.log(alt_probs.clamp(min=1e-8))).sum(dim=-1)

            log_probs = dist.log_prob(actions_t)              
            entropies = dist.entropy()

            reward = calculate_reward(reward, quantile)
            batch_reward += reward
            advantage = reward - reward_baseline
            reward_baseline = (1 - baseline_alpha) * reward_baseline + baseline_alpha * reward

            policy_loss = -(advantage * log_probs).mean()
            entropy_loss = -ent_coef_t * entropies.mean()
            effective_entropy_loss = -ent_coef_alt_t * effective_entropies.mean()                     
            total_loss = policy_loss + entropy_loss + effective_entropy_loss

            batch_loss += total_loss
            batch_entr_loss += -entropy_loss.item()
            batch_eff_entr_loss += -effective_entropy_loss.item()
            batch_policy_loss += policy_loss.item()
            batch_entr_alt += effective_entropies
            batch_entr += entropies.mean().item()
    
        batch_loss /= n_envs
        optimizer.zero_grad()
        batch_loss.backward()
        clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        training_history['mean_reward'].append(batch_reward/n_envs)
        training_history['average_batch_loss'].append(batch_loss.item())
        training_history['clips'].append(clip)
        training_history['effective_entropy_loss'].append(batch_eff_entr_loss/n_envs)
        training_history['entropy_loss'].append(batch_entr_loss/n_envs)
        training_history['policy_loss'].append(batch_policy_loss/n_envs)
        training_history['epsilon'].append(epsilon_t)
        training_history['learning_rate'].append(lr_t)
        training_history['ent_coef'].append(ent_coef_t)
        training_history['ent_coef_alt'].append(ent_coef_alt_t)
        training_history['entropy'].append(batch_entr/n_envs)
        training_history['entropy_alt'].append(batch_entr_alt/n_envs)

        if (episode%10==0):
            print('Episode:', episode, ':')
            print('mean reward:', batch_reward/n_envs)

    return training_history