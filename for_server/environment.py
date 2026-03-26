import gymnasium as gym
from gymnasium import spaces
import numpy as np
from special_functions import plateau, sigmoidBarier, plateauFlat
import random
from collections import deque
import math

class agentDLVEnvV1(gym.Env):
    def __init__(
            self,
            nect_honey_price,
            LP_nav_honey,
            vol_token_price,
            volatile_part,
            init_LP = 1,
            max_step = 0.15,
            target_leverage = 2,
            loop_threshold = 0.7,
            min_cr = 1.2,
            boundary_debt_per_loop = 0.9/1.2,
            is_training = False,
            is_extrimal = False,
            is_devided = False,
            spread = 0.005,
            impact_coeff = 0.0005,
            primordial_action_penalty = 0.9,
            penalty_damping = 0.95,
            action_penalty_scale = 0.9,
            liquidation_penalty_scale = 5,
            extravagance_penalty_scale = 0,
            horizon = 350,
            syntetic_part = 0.2
        ):
        
        super(agentDLVEnvV1, self).__init__()

        self.nect_price = nect_honey_price.copy()
        self.nav = LP_nav_honey.copy()
        self.volatile_part = volatile_part.copy()
        self.vol_token_honey_price =  vol_token_price.copy()
        self.num_of_liquidations = 0
        self.max_steps = len(self.nav)
        self.target_leverage = target_leverage
        self.loop_threshold = loop_threshold
        self.starting_LP = init_LP
        self.max_action = max_step
        self.min_cr = min_cr
        self.boundary_debt_per_loop = boundary_debt_per_loop
        self.is_training = is_training
        self.spread = spread
        self.impact_coeff = impact_coeff
        self.primordial_action_penalty = primordial_action_penalty
        self.action_penalty_damping = penalty_damping 
        self.liquidation_penalty_scale = liquidation_penalty_scale
        self.action_penalty_scale = action_penalty_scale
        self.extravagance_penalty_scale = extravagance_penalty_scale
        self.history_horizon = horizon
        self.is_extrimal = is_extrimal
        self.syntetic_part = syntetic_part
        if self.is_extrimal:
            self.true_nav = self.nav.copy()
            self.true_volatile_part = self.volatile_part.copy()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        self.current_nav = self.nav[0]
        self.current_nect_honey_price = self.nect_price[0]
        self.current_vol_part = self.volatile_part[0]
        self.current_calm_part = 1 - self.current_vol_part
        self.current_vol_token_honey_price = self.vol_token_honey_price[0]

        self.lp = self.starting_LP
        self.collateral_value = self.lp*self.current_nav

        self.primordial_leverage = self._get_primordial_leverage_init()
        self.commulative_debt_value = (self.primordial_leverage)*self.collateral_value
        self.commulative_lp = self.commulative_debt_value/self.current_nav
        self.commulative_debt_nect = self.commulative_debt_value/self.current_nect_honey_price

        x_0 = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)
        self.noncommulative_debt_value = x_0*self.collateral_value
        self.noncommulative_debt_nect = self.noncommulative_debt_value/self.current_nect_honey_price

        self.vol_token_value = x_0*self.collateral_value
        self.vol_token = self.vol_token_value/self.current_vol_token_honey_price

        x = self.vol_token_value/self.collateral_value
        self.leverage = x/self.current_calm_part + self.primordial_leverage + 1
        self.leverage_margin = self.leverage - self.target_leverage

        self.debt_nect = self.noncommulative_debt_nect + self.commulative_debt_nect
        self.total_lp = self.lp + self.commulative_lp
        self.total_debt_value = self.debt_nect*self.current_nect_honey_price
        self.total_lp_value = self.total_lp*self.current_nav
        self.cr = self.total_lp_value/self.total_debt_value
        self.cr_margin = self.cr - self.min_cr

        self.nect_gain_margin = 0
        self.lp_gain_margin = 0
        self.collateral_value_gain_margin = 0
        self.volatile_part_value_gain_margin = 0
        self.leverage_gain_margin = 0
        self.cr_gain_margin = 0
        self.prev_action_cost = 0
        self.current_action_penalty = 0 

        self.leverage_deviation_history = np.zeros(self.history_horizon)
        self.integral_leverage_deviation = self.leverage_margin
        self.average_leverage_deviation = self.leverage_margin
        self.leverage_gain_history = np.zeros(self.history_horizon)
        self.mean_leverage_gain = 0

        self.nav_history = deque(maxlen=self.history_horizon)
        self.nav_history.append(self.current_nav)
        self.nav_volatility = 0
        self.volatile_tok_history = deque(maxlen=self.history_horizon)
        self.volatile_tok_history.append(self.nav[0]*self.volatile_part[0])
        self.volatile_tok_volatility = 0
        self.calm_part_history = deque(maxlen=self.history_horizon)
        self.calm_part_history.append(self.nav[0]*(1 - self.volatile_part[0]))
        self.calm_part_volatility = 0

        if self.is_extrimal:
            self.nav = self.true_nav.copy()
            self.volatile_part = self.true_volatile_part.copy()
            self._add_extrimal_data()

        self.history = {
            'NECT': [],
            'collateral value': [],
            'volatile value': [],
            'reward': [],
            'leverage': [],
            'collaterization ratio': [],
            'LP': [],
            'cumulative debt': []
        }

        self.history_economics = {
            'leverage': [],
            'PNL': [],
            'return': [],
            'collaterization ratio': [],
            'vol token amount': [],
            'nect debt': [],
            'vol token price': [],
            'nect price': [],
            'nav': [],
            'calm part': []
        }

        self.portfolio_value = self.vol_token_value + self.total_lp_value - self.total_debt_value
        self.initial_portfolio_value = self.vol_token_value + self.total_lp_value - self.total_debt_value

        obs = self._get_obs()
        info = {}
        return obs, info
    
    def _get_obs(self):
        return np.array([
            self.debt_nect,
            self.collateral_value,
            self.vol_token_value,
            self.leverage_margin,
            self.cr_margin,
            self.total_lp,
            self.nect_gain_margin,
            self.lp_gain_margin,
            self.collateral_value_gain_margin,
            self.volatile_part_value_gain_margin,
            self.leverage_gain_margin,
            self.mean_leverage_gain,
            self.cr_gain_margin,
            self.integral_leverage_deviation,
            self.average_leverage_deviation,
            self.nav_volatility,
            self.volatile_tok_volatility,
            self.calm_part_volatility,
            self.current_action_penalty
        ], dtype=np.float32)
    
    def step(self, action):
        if action != 0:
            self.current_action_penalty = self.current_action_penalty + self.primordial_action_penalty

            x = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)
            new_vol_token_value = x*self.collateral_value
            new_vol_token = new_vol_token_value/self.current_vol_token_honey_price
            vt_nect_price = self.current_vol_token_honey_price/self.current_nect_honey_price
            
            if new_vol_token < self.vol_token:
                vt_to_sell = self.vol_token - new_vol_token
                effective_selling_price = vt_nect_price*(1 - self.spread - self.impact_coeff)
                nect_to_repay = vt_to_sell*effective_selling_price

                self.noncommulative_debt_nect = self.noncommulative_debt_nect - nect_to_repay
                self.vol_token = new_vol_token
            else:
                vt_to_buy = new_vol_token - self.vol_token
                nect_to_lend = vt_to_buy*vt_nect_price

                self.noncommulative_debt_nect = self.noncommulative_debt_nect + nect_to_lend

                effective_buying_price = vt_nect_price*(1 + self.spread + self.impact_coeff)
                vt_bought = nect_to_lend/effective_buying_price
                self.vol_token = self.vol_token + vt_bought
        else:
            self.current_action_penalty = self.current_action_penalty*self.action_penalty_damping

        self.step_count += 1

        current_nav = self.nav[self.step_count]
        self.collateral_value_gain_margin = current_nav/self.current_nav - 1
        self.current_nav = current_nav
        self.current_nect_honey_price = self.nect_price[self.step_count]
        self.current_vol_part = self.volatile_part[self.step_count]
        self.current_calm_part = 1 - self.current_vol_part
        current_vol_token_honey_price = self.vol_token_honey_price[self.step_count]
        self.volatile_part_value_gain_margin = current_vol_token_honey_price/self.current_vol_token_honey_price - 1
        self.current_vol_token_honey_price = current_vol_token_honey_price

        critical_leverage = self._get_boundary_leverage(self.primordial_leverage)
        if critical_leverage <= 1.1*self.target_leverage:
            print(1)
            self._rebalance_loop()
            
        self.nav_history.append(current_nav)
        self.volatile_tok_history.append(current_vol_token_honey_price)
        self.calm_part_history.append(current_nav*self.current_calm_part)
        _, self.nav_volatility = self._compute_statistic(self.nav_history)
        _, self.volatile_tok_volatility = self._compute_statistic(self.volatile_tok_history)
        _, self.calm_part_volatility = self._compute_statistic(self.calm_part_history)

        self.collateral_value = self.lp*self.current_nav
        self.vol_token_value = self.vol_token*self.current_vol_token_honey_price
        self.debt_nect = self.commulative_debt_nect + self.noncommulative_debt_nect
        total_debt_value = self.debt_nect*self.current_nect_honey_price
        self.nect_gain_margin = total_debt_value/self.total_debt_value - 1
        self.total_debt_value = total_debt_value
        total_lp_value = (self.lp + self.commulative_lp)*self.current_nav
        self.lp_gain_margin = total_lp_value/self.total_lp_value - 1
        self.total_lp_value = total_lp_value

        self.portfolio_value = self.vol_token_value + self.total_lp_value - self.total_debt_value

        x = self.vol_token_value/self.collateral_value
        leverage = x/self.current_calm_part + self.primordial_leverage + 1
        leverage_gain = leverage - self.leverage
        self.leverage_gain_margin = leverage/self.leverage - 1
        self.leverage = leverage
        self.leverage_margin = leverage - self.target_leverage

        self.leverage_deviation_history[self.step_count % self.history_horizon] = self.leverage_margin
        self.integral_leverage_deviation = np.sum(self.leverage_deviation_history)/min(self.step_count + 1, self.history_horizon)
        self.average_leverage_deviation = np.sum(np.abs(self.leverage_deviation_history))/min(self.step_count + 1, self.history_horizon)
        self.leverage_gain_history[self.step_count % self.history_horizon] = leverage_gain
        self.mean_leverage_gain = np.sum(self.leverage_gain_history)/min(self.step_count + 1, self.history_horizon)

        cr = self.total_lp_value/self.total_debt_value
        self.cr_gain_margin = cr/self.cr - 1
        self.cr = cr
        self.cr_margin = cr - self.min_cr
        if self.cr_margin <= 0:
            self.num_of_liquidations += 1
        terminated = self.cr_margin<=0 and not self.is_training

        liquidation_penalty = sigmoidBarier(self.cr_margin)
        if self.cr_margin<0.03:
            if action == 0:
                action_penalty = self.current_action_penalty
            else:
                action_penalty = 0*self.current_action_penalty
        else:
            action_penalty = self.current_action_penalty
        total_penalty = liquidation_penalty*self.liquidation_penalty_scale + action_penalty*self.action_penalty_scale

        bound_maintain_reward = plateauFlat(self.leverage_margin, width=0.08*self.target_leverage) - 0.1*abs(self.leverage_margin)
        leverage_maintain_reward = 0.75*plateau(10*self.integral_leverage_deviation) + 0.25*plateau(10*self.average_leverage_deviation)
        trend_improving_reward = 2*(abs(self.integral_leverage_deviation) - abs(self.leverage_margin))
        average_deviation_reward = 0*(self.average_leverage_deviation - abs(self.leverage_margin))
        liquidation_recovery_reward = 0#0.1*self.cr_margin if self.cr_margin > 0 else 0
        total_reward = trend_improving_reward + average_deviation_reward + 0.6*leverage_maintain_reward + 0.4*bound_maintain_reward + liquidation_recovery_reward

        reward = total_reward - total_penalty

        self.history['NECT'].append(self.debt_nect)
        self.history['collateral value'].append(self.total_lp_value)
        self.history['volatile value'].append(self.vol_token_value)
        self.history['leverage'].append(self.leverage)
        self.history['reward'].append(reward)
        self.history['collaterization ratio'].append(self.cr)
        self.history['LP'].append(self.total_lp)
        self.history['cumulative debt'].append(self.primordial_leverage)

        self._update_history()

        truncated = self.step_count >= self.max_steps - 1
        obs = self._get_obs()
        info = {}

        done = terminated or truncated
        if done and not self.is_training:
            saved = {k: v.copy() for k, v in self.history.items()}
            info['episode_history'] = saved
            saved = {k: v.copy() for k, v in self.history_economics.items()}
            info['episode_economics'] = saved
            self.last_episode_history = saved

        return obs, reward, terminated, truncated, info

    def _update_history(self):
        self.history_economics['leverage'].append(self.leverage)
        self.history_economics['PNL'].append(self.portfolio_value - self.initial_portfolio_value)
        self.history_economics['return'].append(self.portfolio_value/self.initial_portfolio_value - 1)
        self.history_economics['collaterization ratio'].append(self.cr)
        self.history_economics['vol token amount'].append(self.vol_token)
        self.history_economics['nect debt'].append(self.commulative_debt_nect + self.noncommulative_debt_nect)
        self.history_economics['vol token price'].append(self.current_vol_token_honey_price)
        self.history_economics['nect price'].append(self.current_nect_honey_price)
        self.history_economics['nav'].append(self.current_nav)
        self.history_economics['calm part'].append(self.current_calm_part)

    def _compute_transaction_cost(self, action_size):
        return abs(action_size)*self.spread + self.impact_coeff * (action_size ** 2)

    def _compute_statistic(self, array):
        n = len(array)
        mean = sum(array) / n
        if n>1:
            s = sum((x - mean) ** 2 for x in array) 
            volatility = math.sqrt(s/(n - 1))
        else:
            volatility = 0
        return mean, volatility

    def _rebalance_loop(self):
        self.primordial_leverage = self._get_primordial_leverage_init()

        self.commulative_debt_value = self.primordial_leverage*self.collateral_value
        self.commulative_lp = self.commulative_debt_value/self.current_nav
        self.commulative_debt_nect = self.commulative_debt_value/self.current_nect_honey_price

        x = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)

        self.vol_token_value = x*self.collateral_value
        self.vol_token = self.vol_token_value/self.current_vol_token_honey_price
        self.noncommulative_debt_nect = self.vol_token_value/self.current_nect_honey_price
        self.noncommulative_debt_value = self.vol_token_value

    def _get_naive_primordial_leverage(self):
        acceptable_boundary_leverage = 1.2*self.target_leverage
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
        y2 = self.commulative_debt_nect*self.current_nect_honey_price/self.collateral_value
        x_max = (1 + prim_leverage)/self.min_cr - y2
        max_debt_value = x_max*self.collateral_value
        max_borrow = max_debt_value - self.noncommulative_debt_value
        x_max = self.vol_token_value/self.collateral_value + max_borrow/self.collateral_value
        return x_max/self.current_calm_part + prim_leverage + 1
    
    def _get_primordial_leverage_init(self):
        n, prim_leverage = self._get_naive_primordial_leverage()
        acceptable_boundary_leverage = 1.2*self.target_leverage

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
    
    def get_history(self):
        return self.last_episode_history, self.num_of_liquidations
    
    def _nav_single_shock(self, t0, width):
        width = int(width)
        t1 = min(t0 + width, self.max_steps - 1)
        shock = random.uniform(0.2, 0.4)
        for t in range(t0, t1):
            self.nav[t] = self.nav[t]*(1 - shock*(t-t0)/width + np.random.normal(0, 0.001))
        self.nav[t1:] = self.nav[t1:]*(1 - shock)
        return t1 - t0

    def _vol_part_single_shock(self, t0, width):
        width = int(width)
        t1 = min(t0 + width, self.max_steps - 1)
        shock = random.uniform(0.3, 0.5)
        for t in range(t0, t1):
            self.volatile_part[t] = self.volatile_part[t]*(1 - shock*(t-t0)/width + np.random.normal(0, 0.001))
        self.volatile_part[t1:] = self.volatile_part[t1:]*(1 - shock)
        return t1 - t0

    def _nav_single_recovery(self, t0, width):
        width = int(width)
        t1 = min(t0 + width, self.max_steps - 1)
        shock = random.uniform(0.2, 0.4)
        for t in range(t0, t1):
            self.nav[t] = self.nav[t]*1/(1 - shock*(t-t0)/width + np.random.normal(0, 0.001))
        self.nav[t1:] = self.nav[t1:]*(1 - shock)
        return t1 - t0
    
    def  _vol_part_single_recovery(self, t0, width):
        width = int(width)
        t1 = min(t0 + width, self.max_steps - 1)
        shock = random.uniform(0.3, 0.5)
        for t in range(t0, t1):
            self.volatile_part[t] = self.volatile_part[t]*1/(1 - shock*(t-t0)/width + np.random.normal(0, 0.001))
        self.volatile_part[t1:] = self.volatile_part[t1:]*(1 - shock)
        return t1 - t0

    def _random_shock(self, t, width, with_recovery = False):
        x = random.randint(0,2)
        if x == 0:
            l = self._nav_single_shock(t, width)
            if with_recovery:
                if t + 1 + width*1.1 < self.max_steps:
                    t0 = random.randint(t+1, self.max_steps - 1)
                    l += self._nav_single_recovery(t0, width*random.uniform(0.8, 1.1))
        else:
            l = self._vol_part_single_shock(t, width)
            if with_recovery:
                if t + 1 + width*1.1 < self.max_steps:
                    t0 = random.randint(t+1, self.max_steps - 1)
                    l += self._vol_part_single_recovery(t0, width*random.uniform(0.8, 1.1))
        return l

    def _add_extrimal_data(self):
        width = int(self.syntetic_part*self.max_steps)
        while width > 1:
            current_width = random.randint(1, width)
            x = random.randint(0,1)
            t0 = random.randint(1, self.max_steps - 1)
            if x == 0:
                actual_width = self._random_shock(t0, current_width)
            else:
                actual_width = self._random_shock(t0, current_width, with_recovery = True)
            width = int(width - actual_width)