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
        self.nav = LP_nav_honey.copy()/nect_honey_price.copy() # nav in terms of nect
        self.volatile_part = volatile_part.copy()
        self.vol_token_price =  vol_token_price.copy()# % of the volatile token (wbtc) in the LP token
        self.num_of_liquidations = 0
        self.max_steps = len(self.nav)
        self.target_leverage = target_leverage
        self.loop_threshold = loop_threshold
        self.starting_LP = init_LP
        self.max_action = max_step
        self.min_cr = min_cr
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
        self.LP = self.starting_LP
        self.collateral_value = self.LP*self.nav[0]
        self.volatile_part_value = self.collateral_value*self.volatile_part[0]

        max_leverage = 1/(1 - self.loop_threshold)
        if max_leverage <= self.target_leverage:
            self.target_leverage = 2

        primordial_leverage = 1
        calm_part = 1 - self.volatile_part[0]
        target_value = self.target_leverage*calm_part
        while True:
            calm_value = primordial_leverage*calm_part
            borrow_needed = target_value - calm_value
            cr = primordial_leverage/(borrow_needed + primordial_leverage - 1)
            if cr > self.min_cr:
                self.cumulative_debt = self.collateral_value*(primordial_leverage - 1)
                self.primordial_leverage = primordial_leverage
                self.nect = self.collateral_value*borrow_needed
                if self.is_training:
                    self.nect = random.uniform(0.1, 3*self.nect)
                break
            else:
                primordial_leverage = 1 + self.loop_threshold*primordial_leverage

        self.vol_token = self.nect*self.nect_price[0]/self.vol_token_price[0]
        self.vol_token_value = self.nect*self.nect_price
        self.leverage_margin = (self.nect + self.cumulative_debt*calm_part)/(self.collateral_value - self.volatile_part_value) + 1 - self.target_leverage
        self.cr_margin = (self.collateral_value + self.cumulative_debt)/(self.nect + self.cumulative_debt) - self.min_cr
        self.LP_margin = self.LP*self.primordial_leverage/self.starting_LP

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
        self.nav_history.append(self.nav[0])
        self.nav_volatility = 0
        self.volatile_part_history = deque(maxlen=self.history_horizon)
        self.volatile_part_history.append(self.nav[0]*self.volatile_part[0])
        self.volatile_part_volatility = 0
        self.calm_part_history = deque(maxlen=self.history_horizon)
        self.calm_part_history.append(self.nav[0]*(1 - self.volatile_part[0]))
        self.calm_part_volatility = 0

        if self.is_extrimal:
            self.nav = self.true_nav.copy()
            self.volatile_part = self.true_volatile_part.copy()
            self._add_extrimal_data()
            #print(self._random_shock(random.randint(1, self.max_steps - 1), self.syntetic_part*self.max_steps))

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

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        return np.array([
            self.nect,
            self.collateral_value,
            self.volatile_part_value,
            self.leverage_margin,
            self.cr_margin,
            self.LP_margin,
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
            self.volatile_part_volatility,
            self.calm_part_volatility,
            self.current_action_penalty
        ], dtype=np.float32)
    
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
        

    def step(self, action):
        
        self.current_action_penalty = self.current_action_penalty*self.action_penalty_damping
                
        current_nav = self.nav[self.step_count]
        nect_price = self.nect_price[self.step_count]
        current_vol_token_price = self.vol_token_price[self.step_count]

        if action == 0:
            action_size = 0
            nect = self.nect
            vol_token = self.vol_token
        else:
            self.current_action_penalty = self.current_action_penalty + self.primordial_action_penalty
            target_value = self.target_leverage*(self.collateral_value - self.volatile_part_value)
            calm_value = self.primordial_leverage*(self.collateral_value - self.volatile_part_value)
            if target_value > calm_value:
                primordial_leverage = self.primordial_leverage
                while True:
                    calm_value = primordial_leverage*(self.collateral_value - self.volatile_part_value)
                    borrow_needed = target_value - calm_value
                    cr = primordial_leverage*self.collateral_value/(borrow_needed + (primordial_leverage - 1)*self.collateral_value)
                    if cr > self.min_cr:
                        self.primordial_leverage = primordial_leverage
                        break
                    else:
                        primordial_leverage = 1 + self.loop_threshold*primordial_leverage
            else:
                if target_value < calm_value:
                    primordial_leverage = self.primordial_leverage
                    while True:
                        calm_value = primordial_leverage*(self.collateral_value - self.volatile_part_value)
                        if calm_value < target_value:
                            borrow_needed = target_value - calm_value
                            self.primordial_leverage = primordial_leverage
                            break
                        else:
                            primordial_leverage = (primordial_leverage - 1)/self.loop_threshold
                            if primordial_leverage < 0:
                                self.primordial_leverage = primordial_leverage
                                borrow_needed = target_value - calm_value
                                break
                else:
                    borrow_needed = 0
            action_size = borrow_needed - self.nect
            nect = max(borrow_needed, 0)
            vol_token = nect/current_vol_token_price
            
        cost = self._compute_transaction_cost(action_size)
        self.prev_action_cost = cost

        self.nav_history.append(current_nav)
        self.volatile_part_history.append(current_nav*self.volatile_part[self.step_count])
        self.calm_part_history.append(current_nav*(1 - self.volatile_part[self.step_count]))
        _, self.nav_volatility = self._compute_statistic(self.nav_history)
        _, self.volatile_part_volatility = self._compute_statistic(self.volatile_part_history)
        _, self.calm_part_volatility = self._compute_statistic(self.calm_part_history)

        self.nect_gain_margin = min(nect/(self.nect + 1e-10) - 1, 10)
        self.nect = nect
        self.vol_token = vol_token
        lp = self.LP - cost/current_nav
        self.lp_gain_margin = lp/(self.LP + 1e-10) - 1
        self.LP = lp
        self.LP_margin = (self.LP*self.primordial_leverage)/self.starting_LP

        collateral_value = current_nav*self.LP
        self.collateral_value_gain_margin = collateral_value/(self.collateral_value + 1e-10) - 1
        self.collateral_value = collateral_value

        self.cumulative_debt = self.collateral_value*(self.primordial_leverage - 1)

        volatile_part_value = self.volatile_part[self.step_count]*self.collateral_value
        self.volatile_part_value_gain_margin = volatile_part_value/(self.volatile_part_value + 1e-10) - 1
        self.volatile_part_value = volatile_part_value

        leverage = (self.vol_token*current_vol_token_price + self.cumulative_debt*(1 - self.volatile_part[self.step_count]))/(self.collateral_value - self.volatile_part_value + 1e-10) + 1
        leverage_gain = leverage - self.leverage_margin - self.target_leverage
        self.leverage_gain_margin = leverage/(self.leverage_margin + self.target_leverage + 1e-10) - 1
        self.leverage_margin = leverage - self.target_leverage

        cr = (self.collateral_value + self.cumulative_debt)/(self.nect + self.cumulative_debt + 1e-8)
        self.cr_gain_margin = cr/(self.cr_margin + self.min_cr + 1e-10) - 1
        self.cr_margin = cr - self.min_cr
        if self.cr_margin <= 0:
            self.num_of_liquidations += 1
            if self.is_training:
                terminated = False
                truncated = self.step_count >= self.max_steps - 1
            else:
                terminated = True
                truncated = True
        else:
            terminated = False
            truncated = self.step_count >= self.max_steps - 1

        extravagance_penalty = 100*(self.starting_LP - self.LP)/(self.step_count + 1)
        liquidation_penalty = sigmoidBarier(self.cr_margin)
        if self.cr_margin<0.03:
            if action == 0:
                action_penalty = self.current_action_penalty
            else:
                action_penalty = 0*self.current_action_penalty
        else:
            action_penalty = self.current_action_penalty
        total_penalty = extravagance_penalty*self.extravagance_penalty_scale + liquidation_penalty*self.liquidation_penalty_scale + action_penalty*self.action_penalty_scale

        self.leverage_deviation_history[self.step_count % self.history_horizon] = self.leverage_margin
        self.integral_leverage_deviation = np.sum(self.leverage_deviation_history)/min(self.step_count + 1, self.history_horizon)
        self.average_leverage_deviation = np.sum(np.abs(self.leverage_deviation_history))/min(self.step_count + 1, self.history_horizon)
        self.leverage_gain_history[self.step_count % self.history_horizon] = leverage_gain
        self.mean_leverage_gain = np.sum(self.leverage_gain_history)/min(self.step_count + 1, self.history_horizon)

        bound_maintain_reward = plateauFlat(self.leverage_margin, width=0.08*self.target_leverage) - 0.1*abs(self.leverage_margin)
        leverage_maintain_reward = 0.75*plateau(10*self.integral_leverage_deviation) + 0.25*plateau(10*self.average_leverage_deviation)
        trend_improving_reward = 2*(abs(self.integral_leverage_deviation) - abs(self.leverage_margin))
        average_deviation_reward = 0*(self.average_leverage_deviation - abs(self.leverage_margin))
        liquidation_recovery_reward = 0#0.1*self.cr_margin if self.cr_margin > 0 else 0
        total_reward = trend_improving_reward + average_deviation_reward + 0.6*leverage_maintain_reward + 0.4*bound_maintain_reward + liquidation_recovery_reward

        reward = total_reward - total_penalty

        self.history['NECT'].append(self.nect)
        self.history['collateral value'].append(self.collateral_value*nect_price)
        self.history['volatile value'].append(self.volatile_part_value*nect_price)
        self.history['leverage'].append(self.target_leverage + self.leverage_margin)
        self.history['reward'].append(reward)
        self.history['collaterization ratio'].append(self.cr_margin + self.min_cr)
        self.history['LP'].append(self.LP)
        self.history['cumulative debt'].append(self.primordial_leverage - 1)

        obs = self._get_obs()
        self.step_count += 1
        info = {}

        done = terminated or truncated
        if done and not self.is_training:
            saved = {k: v.copy() for k, v in self.history.items()}
            info['episode_history'] = saved
            self.last_episode_history = saved

        return obs, reward, terminated, truncated, info
    
    def get_history(self):
        return self.history, self.num_of_liquidations