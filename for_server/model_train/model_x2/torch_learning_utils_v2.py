import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from scripts.variable_leverage_env_3 import agentDLVEnvV3
from torch_model_v2 import reccurentActor
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.optim as optim
from tqdm import trange, tqdm
import math
from scripts.special_functions import sigmoidBarier

def naiveAgent(obs, min_cr=1.2, eps=1e-9):
    leverage = obs[0]
    cr = 1/obs[1]

    if cr <= min_cr + 0.03:
        return 1
    if leverage >= 0.95 or leverage <= 0.05:
        return 1
    else:
        return 0

def generateEpisode(
        rng, 
        model: reccurentActor, 
        env1: agentDLVEnvV3, 
        env2: agentDLVEnvV3, 
        epsilon = None, 
        confidence_level = 0.99
    ):

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
            mech_loss_rate = info['loss_rate']

    obs, _ = env2.reset(seed = int(seed))
    done = False
    hidden = model.init_hidden(batch_size=1, device='cpu')

    ep_obs = []
    ep_actions = []
    ep_logs = []
    ep_entropies = []
    ep_rewards = []

    while not done:
        ep_obs.append(obs)

        action, log_prob, entropy, hidden = model.act(obs=obs, hidden=hidden, epsilon=epsilon, confidence_level=confidence_level)
        obs, reward, terminated, truncated, info = env2.step(action)

        ep_actions.append(action.item())
        ep_logs.append(log_prob)
        ep_entropies.append(entropy)
        ep_rewards.append(reward)

        done = terminated or truncated
        if done:
            ep_integral_loss = info['integral loss']
            ep_steps = info['steps']
            ep_quantile = info['quantile']
            ep_loss_rate = info['loss_rate']

    reward = mech_loss_rate - ep_loss_rate
    print(ep_steps)

    return {
        "ep_reward": reward,
        "per_step_reward": np.array(ep_rewards),
        "obs": np.array(ep_obs),
        "actions": np.array(ep_actions),
        "ep_logs": [lp.detach().cpu().item() for lp in ep_logs],
        "ep_entropies": [en.detach().cpu().item() for en in ep_entropies],
        "mechanical_loss": mech_integral_loss,
        "agent_loss": ep_integral_loss,
        "steps": ep_steps,
        "mech_quantile": mech_quantile,
        "agent_quantile": ep_quantile,
        "mech_loss_rate": mech_loss_rate,
        "ep_loss_rate": ep_loss_rate
    }

def generateEpisodeWorker(model_state, configs, seed, epsilon = None, confidence_level = 0.99, show_setup = True):
    local_rng = np.random.default_rng(seed)
    #index = local_rng.integers(1, 101)

    env1 = agentDLVEnvV3(
        nect_honey_price = configs['nect_price'],
        nav_in_honey = configs['nav'],
        vol_part = configs['vol_part'],
        vol_token_honey_price = configs['vol_token_honey_price'],
        is_training = True,
        is_mixed = True,
        show_setup = show_setup,
        is_reversed = configs['is_reversed'],
        top_leverage = configs['top_leverage'],
        bottom_leverage = configs['bottom_leverage'],
        plateau_flatness = configs['plateau_flatness']
    )
    
    env2 = agentDLVEnvV3(
        nect_honey_price = configs['nect_price'],
        nav_in_honey = configs['nav'],
        vol_part = configs['vol_part'],
        vol_token_honey_price = configs['vol_token_honey_price'],
        is_training = True,
        is_mixed = True,
        show_setup = False,
        is_reversed = configs['is_reversed'],
        top_leverage = configs['top_leverage'],
        bottom_leverage = configs['bottom_leverage'],
        plateau_flatness = configs['plateau_flatness']
    )

    local_model = reccurentActor()
    local_model.load_state_dict(model_state)
    local_model.to("cpu")
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

def generateSeveralEpisodes(model_state, configs, seed = None, n_envs = 6, epsilon = None, confidence_level = 0.99):
    rng = np.random.default_rng(seed)

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

def generateSeveralEpisodesParallel(model_state, configs, seed = None, n_envs = 6, epsilon = None, confidence_level = 0.99):
    rng = np.random.default_rng(seed)

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

def normilize_raw_reward(reward):
    return 2/math.pi*math.atan(reward) + 0.15*reward
    
def discounted_returns(rewards, gamma):
    R = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        R[t] = running
    return R

def train_model(
        model: reccurentActor, 
        n_episodes, 
        configs, 
        device = 'cpu',
        n_repeats = 1,
        epsilon = None,
        lr = 3e-5,
        ent_coef = 0,
        ent_coef_alt = 1e-3,
        batch_horizon = 5,
        beta = 1,
        gamma = 0.995
    ):

    def resolve_schedule(x, frac):
        if callable(x):
            return x(frac)
        else:
            return x

    reward_baseline = 0.0
    baseline_alpha = 0.01

    reward_sum_1 = 0
    reward_sum_2 = 0
    N = 0
    rng = np.random.default_rng()

    n_envs = len(configs)
    horizon = n_envs*batch_horizon
    reward_history = [0]*horizon

    model.to(device=device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=resolve_schedule(lr, 0))

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
        'entropy_alt': [],
        'mean_per_step_reward': [],
        'sigma': [],
        'advantage_mean': [],
        'advantage_std': []
    }

    for episode in trange(n_episodes, desc="Training"):
        frac = episode/(n_episodes - 1) if n_episodes > 1 else 1

        if episode % n_repeats == 0:
            seed = int(rng.integers(0, 1e6))
            #print('Step', episode + 1, 'new seed', seed)

        epsilon_t = resolve_schedule(epsilon, frac)
        ent_coef_t = resolve_schedule(ent_coef, frac)
        ent_coef_alt_t = resolve_schedule(ent_coef_alt, frac)
        lr_t = resolve_schedule(lr, frac)

        for g in optimizer.param_groups:
            g['lr'] = lr_t

        ep_data_full = generateSeveralEpisodesParallel(
            model_state = model.state_dict(),
            seed = seed,
            configs = configs,
            n_envs = n_envs,
            epsilon = epsilon_t
        )

        raw_ep_rewards = []
        raw_reward_list = []

        for i, ep in enumerate(ep_data_full):
            rewards_np = ep["per_step_reward"]
            raw_reward = ep["ep_reward"]
            quantile = ep["agent_quantile"]

            T = len(rewards_np)
            delta = normilize_raw_reward(raw_reward)
            c = beta * delta * (1 - gamma) / max(1e-8, (1 - gamma**T))

            rewards_np = rewards_np.astype(np.float32).copy()
            rewards_np += c

            if frac == 0:
                print(raw_reward)

            rewards_t = torch.tensor(rewards_np, dtype=torch.float32, device=device)
            rewards_t = discounted_returns(rewards_t, gamma)
            raw_ep_rewards.append(rewards_t)
            raw_reward_list.append(raw_reward)

            current_reward = rewards_t.mean().item()
            old_reward = reward_history[N % horizon]
            reward_history[N % horizon] = current_reward

            reward_sum_1 += current_reward - old_reward
            reward_sum_2 += current_reward**2 - old_reward**2
            N += 1

        mean_reward = reward_sum_1/min(horizon, N)
        sigma = math.sqrt(max(0, reward_sum_2/min(horizon, N) - mean_reward**2))
        sigma = max(sigma, 0.05)

        training_history['mean_per_step_reward'].append(mean_reward)
        training_history['sigma'].append(sigma)

        batch_loss = 0
        batch_reward = 0
        batch_entr_loss = 0
        batch_eff_entr_loss = 0
        batch_policy_loss = 0

        batch_entr = 0
        batch_entr_alt = 0

        batch_advantage_mean = 0
        batch_advantage_std = 0

        clip = 0

        optimizer.zero_grad()

        for i, ep in enumerate(ep_data_full):
            obs_np = ep["obs"]
            actions_np = ep["actions"] 
            raw_reward = raw_reward_list[i]
            
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            actions_t = torch.tensor(actions_np, dtype=torch.long, device=device) 
            rewards_t = raw_ep_rewards[i]

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

            batch_reward += raw_reward

            advantages_t = (rewards_t - mean_reward)/(sigma + 1e-5)

            policy_loss = - (advantages_t.detach() * log_probs).mean()
            entropy_loss = -ent_coef_t * entropies.mean()
            effective_entropy_loss = -ent_coef_alt_t * effective_entropies.mean()                     
            total_loss = policy_loss + entropy_loss + effective_entropy_loss

            batch_loss += total_loss
            batch_entr_loss += -entropy_loss.item()
            batch_eff_entr_loss += -effective_entropy_loss.item()
            batch_policy_loss += policy_loss.item()
            batch_entr_alt += effective_entropies.mean().item()
            batch_entr += entropies.mean().item()

            batch_advantage_mean += advantages_t.mean().item()
            batch_advantage_std += advantages_t.std().item()
    
            #batch_loss /= n_envs
            #batch_loss.backward()
            total_loss = total_loss/n_envs
            total_loss.backward()
            """current_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if current_clip > 10:
                print(f"Warning: gradient explosion at episode {episode}, norm={clip:.2f}")
            
            if current_clip > clip:
                clip = current_clip

            optimizer.step()"""

        clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if clip > 20:
            print(f"Warning: gradient explosion at episode {episode}, norm={clip:.2f}")
            if clip > 40:
                break
            else:
                continue

        training_history['mean_reward'].append(batch_reward/n_envs)
        training_history['average_batch_loss'].append(batch_loss.item()/n_envs)
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
        training_history['advantage_mean'].append(batch_advantage_mean/n_envs)
        training_history['advantage_std'].append(batch_advantage_std/n_envs)

        if (episode%10==0):
            tqdm.write(f"Episode {episode}:")
            tqdm.write(f"mean reward: {batch_reward / n_envs:.4f}")

    return training_history

def train_model_chunks(
        model: reccurentActor, 
        n_episodes, 
        configs, 
        device = 'cpu',
        n_repeats = 1,
        epsilon = None,
        lr = 3e-5,
        ent_coef = 0,
        ent_coef_alt = 1e-3,
        batch_horizon = 5,
        chunk_size = 256
    ):

    def resolve_schedule(x, frac):
        if callable(x):
            return x(frac)
        else:
            return x

    reward_baseline = 0.0
    baseline_alpha = 0.01

    reward_sum_1 = 0
    reward_sum_2 = 0
    N = 0
    rng = np.random.default_rng()

    n_envs = len(configs)
    horizon = n_envs*batch_horizon
    reward_history = [0]*horizon

    model.to(device=device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=resolve_schedule(lr, 0))

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

        if episode % n_repeats == 0:
            seed = int(rng.integers(0, 1e6))
            #print('Step', episode + 1, 'new seed', seed)

        epsilon_t = resolve_schedule(epsilon, frac)
        ent_coef_t = resolve_schedule(ent_coef, frac)
        ent_coef_alt_t = resolve_schedule(ent_coef_alt, frac)
        lr_t = resolve_schedule(lr, frac)

        for g in optimizer.param_groups:
            g['lr'] = lr_t

        ep_data_full = generateSeveralEpisodesParallel(
            model_state = model.state_dict(),
            seed = seed,
            configs = configs,
            n_envs = n_envs,
            epsilon = epsilon_t
        )

        for ep in ep_data_full:
            raw_reward = ep["reward"]
            quantile = ep["agent_quantile"]

            current_reward = calculate_reward(raw_reward, quantile)
            old_reward = reward_history[N % horizon]
            reward_history[N % horizon] = current_reward

            reward_sum_1 += current_reward - old_reward
            reward_sum_2 += current_reward**2 - old_reward**2
            N += 1
            mean_reward = reward_sum_1/min(horizon, N)
            sigma = math.sqrt(max(0, reward_sum_2/min(horizon, N) - mean_reward**2))

        batch_loss = 0
        batch_reward = 0
        batch_entr_loss = 0
        batch_eff_entr_loss = 0
        batch_policy_loss = 0

        batch_entr = 0
        batch_entr_alt = 0

        max_clip = 0

        for ep in ep_data_full:
            obs_np = ep["obs"]
            actions_np = ep["actions"] 
            reward = ep["reward"]
            quantile = ep["agent_quantile"]
            seq_len = len(obs_np)

            current_reward = calculate_reward(reward, quantile)
            batch_reward += current_reward

            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            actions_t = torch.tensor(actions_np, dtype=torch.long, device=device) 
            hidden = model.init_hidden(batch_size=1, device=device)
            hidden = tuple(h.detach() for h in hidden)

            optimizer.zero_grad()

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_obs = obs_t[:,start:end,:]
                chunk_actions = actions_t[start:end]

                logits, hidden = model.forward(chunk_obs, hidden)
                logits = logits[0]  

                dist = torch.distributions.Categorical(logits=logits)

                probs = torch.softmax(logits, dim=-1)           
                p0 = probs[:, 0]                                
                S = (1 - p0).clamp(min=1e-8)                    
                alt_probs = probs[:, 1:] / S.unsqueeze(-1)      
                effective_entropies = -(alt_probs * torch.log(alt_probs.clamp(min=1e-8))).sum(dim=-1)

                chunk_log_probs = dist.log_prob(chunk_actions)              
                chunk_entropies = dist.entropy()

                advantage = (current_reward - mean_reward)/max(1e-3, sigma)

                policy_loss = - (advantage * chunk_log_probs).mean()
                entropy_loss = - (ent_coef_t * chunk_entropies).mean()
                effective_entropy_loss = -ent_coef_alt_t * effective_entropies.mean()                     
                total_loss = policy_loss + entropy_loss + effective_entropy_loss

                batch_loss += total_loss
                batch_entr_loss += -entropy_loss.item()
                batch_eff_entr_loss += -effective_entropy_loss.item()
                batch_policy_loss += policy_loss.item()
                batch_entr_alt += effective_entropies.mean().item()
                batch_entr += chunk_entropies.mean().item()

                total_loss = total_loss * chunk_size / seq_len
                total_loss.backward()
                hidden = tuple(h.detach() for h in hidden)

            clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)

            if clip > max_clip:
                max_clip = clip

            if clip > 20:
                print(f"Skipping step: grad norm {clip:.2f}")
                optimizer.zero_grad()
                continue
            optimizer.step()

        if max_clip > 10:
            print(f"Warning: gradient explosion at episode {episode}, norm={max_clip:.2f}")
            if max_clip > 30:
                break
            else:
                continue

        training_history['mean_reward'].append(batch_reward/n_envs)
        training_history['average_batch_loss'].append(batch_loss.item()/n_envs)
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
            tqdm.write(f"Episode {episode}:")
            tqdm.write(f"mean reward: {batch_reward / n_envs:.4f}")

    return training_history