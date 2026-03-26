import gymnasium as gym
from gymnasium import spaces
import numpy as np
from special_functions import plateau, deltaFunctionBarier, plateauFlat
import random
import math

class rawDLVEnv(gym.Env):
    def __init__(
            self,
            nect_honey_price,
            LP_nav_honey,
            volatile_part,
            init_LP = 1,
            max_step = 0.15,
            target_leverage = 3,
            min_cr = 1.2,
            is_training = False,
            spread = 0.005,
            impact_coeff = 0.0005,
            penalty_damping = 0.9,
            action_penalty_scale = 1.2,
            liquidation_penalty_scale = 1/40
        ):
        
        super(rawDLVEnv, self).__init__()

        self.nect_price = nect_honey_price
        self.nav = LP_nav_honey/nect_honey_price # nav in terms of nect
        self.volatile_part = volatile_part # % of the volatile token (wbtc) in the LP token
        self.num_of_liquidations = 0
        self.max_steps = len(self.nav)
        self.target_leverage = target_leverage
        self.starting_LP = init_LP
        self.max_action = max_step
        self.min_cr = min_cr
        self.is_training = is_training
        self.spread = spread
        self.impact_coeff = impact_coeff
        self.liquidation_penalty_scale = liquidation_penalty_scale
        self.action_penalty_damping = penalty_damping 
        self.action_penalty_scale = action_penalty_scale

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.LP = self.starting_LP
        self.collateral_value = self.LP*self.nav[0]
        self.volatile_part_value = self.collateral_value*self.volatile_part[0]
        if self.is_training:
            self.nect = (self.collateral_value - self.volatile_part_value)*(self.target_leverage - 1)
            self.nect = random.uniform(0.1, 3*self.nect)
        else:
            self.nect = (self.collateral_value - self.volatile_part_value)*(self.target_leverage - 1)
        self.leverage_margin = self.nect/(self.collateral_value - self.volatile_part_value) + 1 - self.target_leverage
        self.cr_margin = self.collateral_value/self.nect - self.min_cr
        self.LP_margin = self.LP/self.starting_LP

        self.nect_gain_margin = 0
        self.lp_gain_margin = 0
        self.collateral_value_gain_margin = 0
        self.volatile_part_value_gain_margin = 0
        self.leverage_gain_margin = 0
        self.cr_gain_margin = 0
        self.prev_action_cost = 0
        self.current_action_penalty = 0 

        self.history = {
            'NECT': [],
            'collateral value': [],
            'volatile value': [],
            'reward': [],
            'leverage': [],
            'collaterization ratio': [],
            'LP': []
        }

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        return np.array([
            self.nect,
            self.collateral_value, # in terms of nect
            self.volatile_part_value, # in terms of nect
            self.leverage_margin,
            self.cr_margin,
            self.LP_margin,
            self.nect_gain_margin,
            self.lp_gain_margin,
            self.collateral_value_gain_margin,
            self.volatile_part_value_gain_margin,
            self.leverage_gain_margin,
            self.cr_gain_margin
        ], dtype=np.float32)
    
    def _compute_transaction_cost(self, action_size):
        return abs(action_size)*self.spread + self.impact_coeff * (action_size ** 2)

    def step(self, action):
        action_size = float(action[0]) * self.max_action
        action_size = np.clip(action_size, -self.max_action, self.max_action)
        cost = self._compute_transaction_cost(action_size)
        self.prev_action_cost = cost
        
        current_nav = self.nav[self.step_count]
        nect_price = self.nect_price[self.step_count]

        nect = max(self.nect + action_size, 0)
        self.nect_gain_margin = min(nect/(self.nect + 1e-10) - 1, 10)
        self.nect = nect

        lp = self.LP - cost/current_nav
        self.lp_gain_margin = lp/(self.LP + 1e-10) - 1
        self.LP = lp

        if lp <= 0:
            self.num_of_liquidations += 1
            liquidation_penalty = deltaFunctionBarier(0)
            if self.is_training:
                self.LP = self.starting_LP
                terminated = False
                truncated = self.step_count >= self.max_steps - 1
            else:
                terminated = True
                truncated = True
        else:
            liquidation_penalty = 0

        action_penalty = 150*cost/current_nav + self.current_action_penalty
        self.current_action_penalty=action_penalty*self.action_penalty_damping
        self.LP_margin = self.LP/self.starting_LP

        collateral_value = current_nav*self.LP
        self.collateral_value_gain_margin = collateral_value/(self.collateral_value + 1e-10) - 1
        self.collateral_value = collateral_value

        volatile_part_value = self.volatile_part[self.step_count]*self.collateral_value
        self.volatile_part_value_gain_margin = volatile_part_value/(self.volatile_part_value + 1e-10) - 1
        self.volatile_part_value = volatile_part_value

        leverage = self.nect/(self.collateral_value - self.volatile_part_value + 1e-10) + 1 
        self.leverage_gain_margin = leverage/(self.leverage_margin + self.target_leverage + 1e-10) - 1
        self.leverage_margin = leverage - self.target_leverage

        cr = self.collateral_value/(self.nect + 1e-8)
        self.cr_gain_margin = cr/(self.cr_margin + self.min_cr) - 1
        self.cr_margin = cr - self.min_cr
        if self.cr_margin <= 0:
            self.num_of_liquidations += 1
            liquidation_penalty = liquidation_penalty + deltaFunctionBarier(0) + abs(self.cr_margin)
            if self.is_training:
                terminated = False
                truncated = self.step_count >= self.max_steps - 1
            else:
                terminated = True
                truncated = True
        else:
            liquidation_penalty = liquidation_penalty + deltaFunctionBarier(self.cr_margin)
            terminated = False
            truncated = self.step_count >= self.max_steps - 1

        leverage_reward = plateau(self.leverage_margin) - 0.6*abs(self.leverage_margin)#+ 0.2*plateauFlat(self.leverage_margin)
        reward = leverage_reward - liquidation_penalty*self.liquidation_penalty_scale - action_penalty*self.action_penalty_scale

        self.history['NECT'].append(self.nect)
        self.history['collateral value'].append(self.collateral_value*nect_price)
        self.history['volatile value'].append(self.volatile_part_value*nect_price)
        self.history['leverage'].append(self.target_leverage + self.leverage_margin)
        self.history['reward'].append(reward)
        self.history['collaterization ratio'].append(self.cr_margin + self.min_cr)
        self.history['LP'].append(self.LP)

        obs = self._get_obs()
        self.step_count += 1
        info = {}

        return obs, reward, terminated, truncated, info
    
    def get_history(self):
        return self.history, self.num_of_liquidations