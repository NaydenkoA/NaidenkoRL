import numpy as np
import pandas as pd
from uniswap_v2_sim import uniswap_v2_sim
import gymnasium as gym
from gymnasium import spaces
import math

class uniswapV2Backtest(gym.Env):
    def __init__(
            self,
            vol_token_pool: uniswap_v2_sim,
            nav_in_honey,
            vol_part,
            timestamps,
            top_leverage,
            bottom_leverage,
            min_cr = 1.2,
            boundary_debt_per_loop = 0.9/1.2,
            collateral_LP = 1,
            horizon = 600,
        ):
        
        super(uniswapV2Backtest, self).__init__()

        self.nav = nav_in_honey.copy()
        self.vol_part = vol_part.copy()
        self.pool = vol_token_pool
        self.max_episodes = len(self.nav)
        self.agent_timestamps = timestamps.copy()
        self.top_leverage = top_leverage
        self.bottom_leverage = bottom_leverage
        self.min_cr = min_cr
        self.boundary_debt_per_loop = boundary_debt_per_loop
        self.lp = collateral_LP
        self.max_relative_commulative_debt = 1/(min_cr - 1)
        self.horizon = horizon

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self, seed = None, options = None, top_leverage = None, bottom_leverage = None):
        super().reset(seed=seed)

        if top_leverage is not None:
            self.top_leverage = top_leverage

        if bottom_leverage is not None:
            self.bottom_leverage = bottom_leverage

        self.history_economics = {
            'leverage': [],
            'PNL': [],
            'return': [],
            'collaterization ratio': [],
            'vol token amount': [],
            'nect debt': [],
            'vol token price': [],
            'total loss': [],
            'nav': [],
            'calm part': [],
            'timestamp': []
        }

        self.step_count = 0

        self.timestamp = self.agent_timestamps[self.step_count]
        self.pool.set_history_timestamp(self.timestamp)
        self.current_vol_token_price = self.pool.get_price('volatile')
        self.current_nav = self.nav[self.step_count]
        self.current_vol_part = self.vol_part[self.step_count]
        self.current_calm_part = 1 - self.vol_part[self.step_count]

        self.collateral_value = self.lp*self.current_nav

        self.target_leverage = (self.top_leverage + self.bottom_leverage)/2
        self.top_target_leverage = (self.top_leverage + self.target_leverage)/2
        self.bottom_target_leverage = (self.target_leverage + self.bottom_leverage)/2

        self.primordial_leverage = self._get_primordial_leverage_init()
        self.commulative_debt = self.primordial_leverage*self.collateral_value
        self.commulative_lp = self.commulative_debt/self.current_nav

        x = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)
            
        self.vol_token_value = x*self.collateral_value
        self.vol_token = self.vol_token_value/self.current_vol_token_price
        self.noncommulative_debt = self.vol_token_value

        self.leverage = x/self.current_calm_part + self.primordial_leverage + 1

        self.total_lp_value = self.collateral_value + self.commulative_lp*self.current_nav
        self.total_debt_value = self.noncommulative_debt + self.commulative_debt
        self.cr = self.total_lp_value/self.total_debt_value
        self.vol_token_normed = x

        self.leverage_history = np.zeros(self.horizon)
        self.leverage_history[0] = self.leverage
        self.relative_vol_token_value_history = np.zeros(self.horizon)
        self.relative_vol_token_value_history[0] = self.vol_token_normed
        self.calm_part_history = np.zeros(self.horizon)
        self.calm_part_history[0] = self.current_calm_part

        self.leverage_mean = np.sum(self.leverage_history)/min(self.step_count + 1, self.horizon)
        self.leverage_gain_mean = self._get_average_gain(self.leverage_history)
        self.vol_token_mean = np.sum(self.relative_vol_token_value_history)/min(self.step_count + 1, self.horizon)
        self.vol_token_gain_mean = self._get_average_gain(self.relative_vol_token_value_history)
        self.calm_part_mean = np.sum(self.calm_part_history)/min(self.step_count + 1, self.horizon)
        self.calm_part_gain_mean = self._get_average_gain(self.calm_part_history)

        self.norm = self.top_leverage - self.bottom_leverage

        self.leverage_normed = (self.leverage - self.bottom_leverage)/self.norm
        self.leverage_mean_normed = (self.leverage_mean - self.bottom_leverage)/self.norm
        self.leverage_gain_mean_normed = self._normalize_slope(self.leverage_gain_mean)
        self.vol_token_gain_mean_normed = self._normalize_slope(self.vol_token_gain_mean)
        self.calm_part_gain_mean_normed = self._normalize_slope(self.calm_part_gain_mean)

        self.cr_normed = 1/(self.cr)

        self.recent_reward = 1
        self.relative_loss = 0
        self.total_loss = 0
        self.losses_history = np.zeros(self.horizon)
        self.recenet_total_loss = 0

        self.portfolio_value = self.vol_token_value + self.total_lp_value - self.total_debt_value
        self.initial_portfolio_value = self.vol_token_value + self.total_lp_value - self.total_debt_value

        self.current_action_penalty = 0
        self._update_history()

        obs = self._get_obs()
        info = {}
        return obs, info
    
    def _get_obs(self):
        return np.array([
            self.leverage_normed,
            self.cr_normed,
            self.leverage_mean_normed,
            self.leverage_gain_mean_normed,
            self.vol_token_normed,
            self.vol_token_mean,
            self.vol_token_gain_mean_normed,
            self.current_calm_part,
            self.calm_part_mean,
            self.calm_part_gain_mean_normed
        ], dtype=np.float32)

    def step(self, action):

        if action != 0 and action != 4:
            if action == 1:
                x = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)
            else:
                if action == 2:
                    x = self.current_calm_part*(self.top_target_leverage - self.primordial_leverage - 1)
                else:
                    x = self.current_calm_part*(self.bottom_target_leverage - self.primordial_leverage - 1)

            new_vol_token_value = x*self.collateral_value
            new_vol_token = new_vol_token_value/self.current_vol_token_price

            if new_vol_token < self.vol_token:
                vt_to_sell = self.vol_token - new_vol_token
                vt_sold, nect_to_repay = self.pool.make_swap(vt_to_sell, 'volatile', is_pure=True)

                self.noncommulative_debt -= nect_to_repay
                self.vol_token = new_vol_token

                recent_loss_nect = vt_to_sell*self.current_vol_token_price - nect_to_repay
                self.recent_loss_value = recent_loss_nect
            else:
                vt_to_buy = new_vol_token - self.vol_token
                nect_to_borrow = vt_to_buy*self.current_vol_token_price

                self.noncommulative_debt += nect_to_borrow
                
                nect_sold, vt_bought = self.pool.make_swap(nect_to_borrow, 'stable', is_pure=True)
                self.vol_token += vt_bought

                recent_loss_vt = vt_to_buy - vt_bought
                self.recent_loss_value = recent_loss_vt*self.pool.get_price('volatile')
        else:
            self.recent_loss_value = 0

        self.relative_loss += self.recent_loss_value/self.collateral_value
        self.total_loss += self.recent_loss_value
        prev_loss = self.losses_history[self.step_count % self.horizon]
        self.losses_history[self.step_count % self.horizon] = self.relative_loss
        self.recenet_total_loss = self.relative_loss - prev_loss

        self.step_count += 1

        self.timestamp = self.agent_timestamps[self.step_count]
        self.pool.make_history_simulation(self.timestamp)
        self.current_vol_token_price = self.pool.get_price('volatile')
        self.current_nav = self.nav[self.step_count]
        self.current_vol_part = self.vol_part[self.step_count]
        self.current_calm_part = 1 - self.vol_part[self.step_count]

        self.collateral_value = self.lp*self.current_nav
        self.vol_token_value = self.vol_token*self.current_vol_token_price
        self.total_lp_value = self.collateral_value + self.commulative_lp*self.current_nav
        self.total_debt_value = self.noncommulative_debt + self.commulative_debt

        x = self.vol_token_value/self.collateral_value
        leverage = x/self.current_calm_part + self.primordial_leverage + 1
        self.leverage_gain = leverage - self.leverage
        self.leverage = leverage

        self.cr = self.total_lp_value/self.total_debt_value

        self.vol_token_normed = x
        self.boundary_leverage = self._get_boundary_leverage(self.primordial_leverage)

        self.leverage_history[self.step_count % self.horizon] = self.leverage
        self.relative_vol_token_value_history[self.step_count % self.horizon] = self.vol_token_normed
        self.calm_part_history[self.step_count % self.horizon] = self.current_calm_part

        self.leverage_mean = np.sum(self.leverage_history)/min(self.step_count + 1, self.horizon)
        self.vol_token_mean = np.sum(self.relative_vol_token_value_history)/min(self.step_count + 1, self.horizon)
        self.calm_part_mean = np.sum(self.calm_part_history)/min(self.step_count + 1, self.horizon)

        self.leverage_gain_mean = self._get_average_gain(self.leverage_history)
        self.vol_token_gain_mean = self._get_average_gain(self.relative_vol_token_value_history)
        self.calm_part_gain_mean = self._get_average_gain(self.calm_part_history)

        self.boundary_leverage_normed = (self.boundary_leverage - self.bottom_leverage)/self.norm
        self.leverage_normed = (self.leverage - self.bottom_leverage)/self.norm
        self.leverage_mean_normed = (self.leverage_mean - self.bottom_leverage)/self.norm

        self.leverage_gain_mean_normed = self._normalize_slope(self.leverage_gain_mean)
        self.vol_token_gain_mean_normed = self._normalize_slope(self.vol_token_gain_mean)
        self.calm_part_gain_mean_normed = self._normalize_slope(self.calm_part_gain_mean)

        self.cr_normed = min(1/max(self.cr + 1e-6, 1e-10), 1)
        
        if self.boundary_leverage_normed <= 1:
            self._rebalance_loop()

        reward = 0
        self.recent_reward = reward

        self.portfolio_value = self.vol_token_value + self.total_lp_value - self.total_debt_value

        self._update_history()
        terminated = self.cr<=self.min_cr
        truncated = self.step_count >= self.max_episodes - 1
        obs = self._get_obs()
        info = {}

        done = terminated or truncated
        if done:
            saved = {k: v.copy() for k, v in self.history_economics.items()}
            info['episode_economics'] = saved
            info['top leverage'] = self.top_leverage
            info['bottom leverage'] = self.bottom_leverage

        return obs, reward, terminated, truncated, info

    def _update_history(self):
        self.history_economics['leverage'].append(self.leverage)
        self.history_economics['PNL'].append(self.portfolio_value - self.initial_portfolio_value)
        self.history_economics['return'].append(self.portfolio_value/self.initial_portfolio_value - 1)
        self.history_economics['collaterization ratio'].append(self.cr)
        self.history_economics['vol token amount'].append(self.vol_token)
        self.history_economics['nect debt'].append(self.total_debt_value)
        self.history_economics['vol token price'].append(self.current_vol_token_price)
        self.history_economics['total loss'].append(self.total_loss)
        self.history_economics['nav'].append(self.current_nav)
        self.history_economics['calm part'].append(100*self.current_calm_part)
        self.history_economics['timestamp'].append(self.timestamp)

    def _normalize_gain(self, gain, scale = 0.001):
        gain = gain/self.norm
        sin = gain/math.sqrt(scale**2 + gain**2)
        return (sin + 1)/2

    def _get_average_gain(self, series):
        n = min(self.step_count + 1, self.horizon)
        if n < 2:
            return 0
        i_0 = self.step_count % self.horizon
        mu = (n - 1)/2
        sum_1 = 0
        sum_2 = 0
        for i in range(n):
            idx = (i_0 - n + 1 + i) % n  
            sum_1 += (i - mu) * series[idx]
            sum_2 += (i - mu)**2
        return sum_1 / sum_2

    def _normalize_slope(self, slope):
        alpha = math.atan(8000/self.horizon*min(self.step_count + 1, self.horizon)*slope)
        return (2/math.pi*alpha + 1)/2

    def _rebalance_loop(self):
        self.primordial_leverage = self._get_primordial_leverage_init()

        self.commulative_debt_value = self.primordial_leverage*self.collateral_value
        self.commulative_lp = self.commulative_debt_value/self.current_nav
        self.commulative_debt = self.commulative_debt_value

        self.target_leverage = (self.top_leverage + self.bottom_leverage)/2
        x = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)

        self.vol_token_value = x*self.collateral_value
        self.vol_token = self.vol_token_value/self.current_vol_token_price
        self.noncommulative_debt = self.vol_token_value

    def _get_naive_primordial_leverage(self):
        acceptable_boundary_leverage = self.target_leverage + 0.1*(self.top_leverage - self.bottom_leverage)
        x = 1 - acceptable_boundary_leverage*(1 - self.boundary_debt_per_loop)

        if x <= 0:
            return 0, 1

        n = int(math.log(x)/math.log(self.boundary_debt_per_loop))
        n = max(1, n)

        return n, (1 - self.boundary_debt_per_loop**n)/(1 - self.boundary_debt_per_loop) - 1
    
    def _get_prev_primordial_leverage(self, leverage):
        return (leverage - 1)/self.boundary_debt_per_loop
    
    def _get_next_primordial_leverage(self, leverage):
        return 1 + self.boundary_debt_per_loop*leverage
    
    def _get_boundary_leverage_init(self, prim_leverage):
        x_max = (1 + prim_leverage)/self.min_cr - prim_leverage
        return x_max/self.current_calm_part + prim_leverage + 1
    
    def _get_boundary_leverage(self, prim_leverage):
        y2 = self.commulative_debt/self.collateral_value
        x_max = (1 + prim_leverage)/self.min_cr - y2
        max_debt_value = x_max*self.collateral_value
        max_borrow = max_debt_value - self.noncommulative_debt
        x_max = self.vol_token_normed + max_borrow/self.collateral_value
        return x_max/self.current_calm_part + prim_leverage + 1
    
    def _get_primordial_leverage_init(self):
        n, prim_leverage = self._get_naive_primordial_leverage()
        acceptable_boundary_leverage = self.top_target_leverage + 0.1*(self.top_leverage - self.bottom_leverage)

        while True:
            if n <= 1:
                break
            n_new = n - 1
            new_prim_leverage = self._get_prev_primordial_leverage(prim_leverage + 1) - 1
            new_boundary_leverage = self._get_boundary_leverage_init(new_prim_leverage)
            if new_boundary_leverage < acceptable_boundary_leverage:
                break
            else:
                n = n_new
                prim_leverage = new_prim_leverage

        return (1 - self.boundary_debt_per_loop**n)/(1 - self.boundary_debt_per_loop) - 1