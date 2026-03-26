import gymnasium as gym
from gymnasium import spaces
import numpy as np
from special_functions import plateau, sigmoidBarier, plateauFlat
import random
from collections import deque
import math

class agentDLVEnvV2(gym.Env):
    def __init__(
            self,
            nect_honey_price,
            nav_in_honey,
            vol_token_honey_price,
            vol_part,
            top_leverage,
            bottom_leverage,
            is_training = False,
            is_mixed = False,
            mixed_episide_frac = 0.1,
            spread = 0.005,
            impact_coeff = 0.0005,
            min_cr = 1.2,
            boundary_debt_per_loop = 0.9/1.2,
            collateral_LP = 1,
            horizon = 400
        ):
        
        super(agentDLVEnvV2, self).__init__()

        self.pure_nect_honey_price = nect_honey_price.copy()
        self.pure_nav = nav_in_honey.copy()
        self.pure_vol_part = vol_part.copy()
        self.pure_vol_token_honey_price = vol_token_honey_price.copy()
        self.pure_max_episodes = len(self.pure_nav)
        self.top_leverage = top_leverage
        self.bottom_leverage = bottom_leverage
        self.min_cr = min_cr
        self.spread = spread
        self.impact_coeff = impact_coeff
        self.is_training = is_training
        self.is_mixed = is_mixed
        self.mixed_episode_lenght = int(mixed_episide_frac*self.pure_max_episodes)
        self.boundary_debt_per_loop = boundary_debt_per_loop
        self.starting_lp = collateral_LP
        self.max_relative_commulative_debt = 1/(min_cr - 1)
        self.horizon = horizon

        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
    
    def reset(self, seed=None, options=None, top_leverage = None, bottom_leverage = None, interrupt = True):
        super().reset(seed=seed)

        if top_leverage is not None:
            self.top_leverage = top_leverage

        if bottom_leverage is not None:
            self.bottom_leverage = bottom_leverage

        if interrupt:
            self.step_count = 0
            self.lp = self.starting_lp
            self.critical_leverages = []
            self.primordial_leverages = []
            self.relative_loss = 0

            if self.is_mixed:
                N = random.randint(0, self.pure_max_episodes - self.mixed_episode_lenght - 1)

                self.max_episodes = self.mixed_episode_lenght
                self.nav = self.pure_nav[N:N + self.mixed_episode_lenght]
                self.nect_honey_price = self.pure_nect_honey_price[N:N + self.mixed_episode_lenght]
                self.vol_part = self.pure_vol_part[N:N + self.mixed_episode_lenght]
                self.vol_token_honey_price = self.pure_vol_token_honey_price[N:N + self.mixed_episode_lenght]

            else:
                self.max_episodes = self.pure_max_episodes
                self.nav = self.pure_nav
                self.nect_honey_price = self.pure_nect_honey_price
                self.vol_part = self.pure_vol_part
                self.vol_token_honey_price = self.pure_vol_token_honey_price

            self.history = {
                'leverage': [],
                'portfolio value': [],
                'relative loss': [],
                'collaterization ratio': [],
                'leverage trend': [],
                'leverage gain': [],
                'portfolio gain': [],
                'mean loss': [],
                'reward': []
            }
        
        # step 0: initing current market state
        self.current_nav = self.nav[self.step_count]
        self.current_nect_honey_price = self.nect_honey_price[self.step_count]
        self.current_vol_part = self.vol_part[self.step_count]
        self.current_calm_part = 1 - self.current_vol_part
        self.current_vol_token_honey_price = self.vol_token_honey_price[self.step_count]

        self.collateral_value = self.lp*self.current_nav

        # step 1: adjust loop
        loop_num = self._get_loop()
        self.primordial_leverage = self._get_looped_leverage(loop_num) - 1
        self.commulative_debt_value = (self.primordial_leverage)*self.collateral_value
        self.commulative_lp = self.commulative_debt_value/self.current_nav
        self.commulative_nect = self.commulative_debt_value/self.current_nect_honey_price

        # step 2: adjust initial nect state
        self.target_leverage = 1/2*(self.bottom_leverage + self.top_leverage)
        x_0 = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)
        self.debt_value = x_0*self.collateral_value + self.commulative_debt_value
        self.debt_nect = self.debt_value/self.current_nect_honey_price
        
        #step 3: adjust initial vol token state
        self.vol_token_value = x_0*self.collateral_value
        self.vol_token = self.vol_token_value/self.current_vol_token_honey_price

        #step 4: adjust vault state
        x = self.vol_token_value/self.collateral_value
        y = self.commulative_debt_value/self.collateral_value
        self.current_relative_vt = x
        self.current_relative_cd = y
        self.leverage = x/self.current_calm_part + y + 1
        self.portfolio_value = self.collateral_value + self.vol_token_value + self.commulative_debt_value - self.debt_value
        self.current_relative_debt = self.debt_value/self.collateral_value
        self.cr = (self.collateral_value + self.commulative_debt_value)/(self.debt_value)

        self.leverages = np.zeros(self.horizon)
        self.losses = np.zeros(self.horizon)
        self.leverages[self.step_count % self.horizon] = self._normalize_leverage(self.leverage)
        prev_loss = self.losses[self.step_count % self.horizon]
        self.losses[self.step_count % self.horizon] = self.relative_loss
        self.mean_leverage = np.sum(self.leverages)/min(self.step_count + 1, self.horizon)
        self.mean_loss = self.relative_loss - prev_loss

        self.leverage_gain = 0
        self.portfolio_value_gain = 0
        self.relative_portfolio_value_gain = 0

        self.recent_reward = 1

        self._update_history()

        self.step_count += 1

        obs = self._get_obs()
        info = {}
        return obs, info
    
    def _get_obs(self):
        obs = np.array([
            float(self.current_calm_part),
            float(self._normalize_leverage(self.leverage)),
            float(self._normalize_leverage(self.bottom_leverage)),
            float(self.current_relative_vt),
            float(self._normalize_cr(self.cr)),
            float(self.current_relative_cd / self.max_relative_commulative_debt),
            float(self.mean_leverage),
            float(self._normalize_leverage_gain(self.leverage_gain)),
        ], dtype=np.float32)
        obs = np.where(np.isfinite(obs), obs, 0.0)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
        
    def step(self, action):
        a = float(np.asarray(action).squeeze())
        a = float(np.clip(a, self.action_space.low[0], self.action_space.high[0]))
        #step 0: make an action:
        if abs(a - self.current_relative_vt) > 0.001:
            vt_value_new = action*self.collateral_value
            vt_new = vt_value_new/self.current_vol_token_honey_price
            vt_nect_price = self.current_vol_token_honey_price/self.current_nect_honey_price

            if vt_new < self.vol_token:
                vt_to_sell = self.vol_token - vt_new
                effective_selling_price = vt_nect_price*(1 - self.spread - self.impact_coeff)
                nect_to_repay = vt_to_sell*effective_selling_price

                self.debt_nect = self.debt_nect - nect_to_repay
                self.vol_token = vt_new

                recent_loss_nect = vt_to_sell*(vt_nect_price - effective_selling_price)
                self.recent_loss_value = recent_loss_nect*self.current_nect_honey_price

            else:
                vt_to_buy = vt_new - self.vol_token
                nect_to_lend = vt_to_buy*vt_nect_price

                self.debt_nect = self.debt_nect + nect_to_lend

                effective_buying_price = vt_nect_price*(1 + self.spread + self.impact_coeff)
                vt_bought = nect_to_lend/effective_buying_price
                self.vol_token = self.vol_token + vt_bought

                recent_loss_vt = nect_to_lend*(1/vt_nect_price - 1/effective_buying_price)
                self.recent_loss_value = recent_loss_vt*self.current_vol_token_honey_price

        else:
            self.recent_loss_value = 0

        self.relative_loss = self.relative_loss + self.recent_loss_value/self.collateral_value

        #step 0.5: check liquidation threshold
        self.current_cr = self.collateral_value/(self.debt_nect*self.current_nect_honey_price + 1e-6)
        terminated = self.current_cr<=self.min_cr and not self.is_training
        liquidation_penalty = sigmoidBarier(self.cr - self.min_cr)

        #step 0.9: calculate reward
        prev_loss = self.losses[self.step_count % self.horizon]
        self.losses[self.step_count % self.horizon] = self.relative_loss
        self.mean_loss = self.relative_loss - prev_loss

        total_penalty = liquidation_penalty

        #step 1: renew market state
        self.current_nav = self.nav[self.step_count]
        self.current_nect_honey_price = self.nect_honey_price[self.step_count]
        self.current_vol_part = self.vol_part[self.step_count]
        self.current_calm_part = 1 - self.current_vol_part

        self.collateral_value = self.lp*self.current_nav

        #step 1.5: update loop
        critical_leverage = self._get_boundary_leverage(self.primordial_leverage)
        self.critical_leverages.append(critical_leverage)
        if critical_leverage <= 1/3*self.top_leverage + 2/3*self.bottom_leverage:
            self.reset(interrupt = False)
        self.primordial_leverages.append(self.primordial_leverage)

        #step 2: renew vol token state
        self.vol_token_value = self.vol_token*self.current_vol_token_honey_price
        
        #step 3: renew nect state
        self.debt_value = self.debt_nect*self.current_nect_honey_price
        
        #step 4: renew vault state
        self.commulative_debt_value = self.commulative_lp*self.current_nav
        x = self.vol_token_value/self.collateral_value
        y = self.commulative_debt_value/self.collateral_value
        new_portfolio_value = self.collateral_value + self.vol_token_value + self.commulative_debt_value - self.debt_value
        self.portfolio_value_gain = new_portfolio_value - self.portfolio_value
        self.relative_portfolio_value_gain = self.portfolio_value_gain/self.collateral_value
        self.portfolio_value = new_portfolio_value
        self.current_relative_vt = x
        self.current_relative_cd = y
        self.current_relative_debt = self.debt_value/self.collateral_value
        new_leverage = x/self.current_calm_part + 1 + y
        self.leverage_gain = new_leverage - self.leverage
        self.leverage = new_leverage
        self.cr = (self.collateral_value + self.commulative_debt_value)/(self.debt_value + 1e-10)
        self.leverages[self.step_count % self.horizon] = self._normalize_leverage(self.leverage)
        self.mean_leverage = np.sum(self.leverages)/min(self.step_count + 1, self.horizon)

        self.leverage_normed = self._normalize_leverage(self.leverage)
        self.bottom_leverage_normed = self._normalize_leverage(self.bottom_leverage)

        leverage_reward = plateau(10*(self.leverage_normed - 1/2*(1 + self.bottom_leverage_normed )))
        total_reward = leverage_reward
        
        reward = (total_reward - total_penalty)*0.3
        self.recent_reward = reward

        self._update_history()
        terminated = self.cr<=self.min_cr and not self.is_training
        truncated = self.step_count >= self.max_episodes - 1
        obs = self._get_obs()
        self.step_count += 1
        info = {}

        done = terminated or truncated
        if done and not self.is_training:
            saved = {k: v.copy() for k, v in self.history.items()}
            info['episode_history'] = saved
            self.last_episode_history = saved
            info['top leverage'] = self.top_leverage
            info['bottom leverage'] = self.bottom_leverage
            info['critical leverages'] = self.critical_leverages.copy()
            info['primordial leverages'] = self.primordial_leverages.copy()

        return obs, reward, terminated, truncated, info

    def _update_history(self):
        self.history['leverage'].append(self.leverage)
        self.history['portfolio value'].append(self.portfolio_value)
        self.history['relative loss'].append(self.relative_loss)
        self.history['collaterization ratio'].append(self.cr)
        self.history['leverage trend'].append(self.mean_leverage)
        self.history['leverage gain'].append(self._normalize_leverage_gain(self.leverage_gain))
        self.history['portfolio gain'].append(self._normalize_portfolio_gain(self.relative_portfolio_value_gain))
        self.history['mean loss'].append(self.mean_loss)
        self.history['reward'].append(self.recent_reward)
        
    def _normalize_leverage_gain(self, gain):
        sin_gain = gain/math.sqrt(0.1**2 + gain**2)
        return (sin_gain + 1)/2

    def _normalize_portfolio_gain(self, gain):
        sin_gain = gain/math.sqrt(0.01**2 + gain**2)
        return (sin_gain + 1)/2

    def _normalize_leverage(self, l):
        norm = self.top_leverage - self.primordial_leverage
        l_norm = (l - self.primordial_leverage)/norm
        if l_norm > 1:
            return 1
        else:
            if l_norm < 0:
                return 0
            else:
                return l_norm
    
    def _normalize_cr(self, cr):
        if not np.isfinite(cr) or cr <= 0:
            return 1.0  
        return float(np.clip(1.0/cr, 0.0, 1.0))
    
    def _normalize_nect(self, nect_value):
        return nect_value/self.collateral_value
    
    def _normilize_cd(self, cd):
        y = cd/self.collateral_value
        return y/self.max_relative_commulative_debt

    def _compute_transaction_cost(self, action_size):
        return abs(action_size)*self.spread + self.impact_coeff * (action_size ** 2)

    def _get_looped_leverage(self, n):
        q = self.boundary_debt_per_loop
        return (1-q**n)/(1-q)
    
    def _get_loop_naive(self, q, min_leverage):
        n = int(math.log(1-min_leverage*(1-q))/math.log(q))
        return n

    def _get_prev_loop(self, q, current_looped_leverage):
        return 1/q*(current_looped_leverage - 1)

    def _get_next_loop(self, q, current_looped_leverage):
        return q*current_looped_leverage + 1
    
    def _get_boundary_leverage(self, current_min_leverage):
        y = current_min_leverage 
        x = (1 - y*(self.min_cr - 1))/self.min_cr

        return 1 + y + x/self.current_calm_part
    
    def _get_loop(self):
        n = self._get_loop_naive(self.boundary_debt_per_loop, self.bottom_leverage)
        current_lev = self._get_looped_leverage(n)
        boundary_leverage = self._get_boundary_leverage(current_lev - 1)
        if boundary_leverage > 1/2*(self.bottom_leverage + self.top_leverage):
            while True:
                new_current_lev = self._get_prev_loop(self.boundary_debt_per_loop, current_lev)
                new_n = n - 1
                new_boundary_leverage = self._get_boundary_leverage(new_current_lev - 1)
                if new_boundary_leverage < 1/2*(self.bottom_leverage + self.top_leverage):
                    break
                else:
                    current_lev = new_current_lev
                    n = new_n
                    boundary_leverage = new_boundary_leverage
                    if n == 1:
                        break
        return n