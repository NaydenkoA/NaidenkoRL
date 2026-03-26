import gym
import numpy as np
import pandas as pd
import math
from gym import spaces
H = 0

class LeverageDLVEnv(gym.Env):
    def __init__(self, 
                 NECT_price_series,
                 LP_volatile_part_series,
                 LP_nav,
                 max_position = 0.2,
                 spread = 2, #0.001
                 impact_coeff = 8, #0.005
                 starting_LP = 1,
                 min_cr = 0.2,
                 treshold = 0.7
                 ):

        super(LeverageDLVEnv, self).__init__()
        self.max_position = max_position
        self.spread = spread
        self.impact_coeff = impact_coeff
        self.LP = starting_LP
        self.nav = LP_nav
        self.LP_volatile_part = LP_volatile_part_series
        self.min_cr = min_cr
        self.treshold = treshold

        self.NECT_price_series = NECT_price_series
        self.max_steps = len(NECT_price_series)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.step_count = 0
        self.NECT_price = self.NECT_price_series[0] 
        
        self.collateral_value = self.LP*self.nav[0] 
        self.volatile_collateral_value = self.collateral_value*self.LP_volatile_part[0] 
        self.NECT = (self.collateral_value - self.volatile_collateral_value)/self.NECT_price
        self.cr_margin = 1
        self.leverage_margin = self.NECT*self.NECT_price/(self.collateral_value - self.volatile_collateral_value) - 1
        # For plotting
        self.history = {
            'NECT value': [],
            'NECT': [],
            'LP': [],
            'collateral value': [],
            'volatile value': [],
            'collateral ratio': [],
            'reward': [],
            'leverage': []
        }

        global H
        H = 0

        return self._get_obs()

    def plateau(self,x,width = 0.5):
        return 0.5/math.tanh(width)*(math.tanh(x + width)-math.tanh(x - width))

    def deltaFunction(self, x):
        return 100/math.sqrt(math.pi)*math.exp(-100/0.2*x**2)

    def _get_obs(self):
        return np.array([
            self.NECT_price, #math.log(self.NECT_price)
            self.NECT,       #math.log(NECT)
            self.leverage_margin, #math.log(self.leverage_margin)
            self.collateral_value, #math.log(collateral_value)
            self.volatile_collateral_value #math.log(volatile_collateral_value)
            #math.tanh(self.cr_margin - self.min_cr) 
        ], dtype=np.float32)

    def _compute_transaction_cost(self, action_size):
        return abs(action_size) * self.NECT_price * self.spread + self.impact_coeff * (action_size ** 2)

    def step(self, action):
        action_size = float(action[0]) * self.max_position
        action_size = np.clip(action_size, -self.max_position, self.max_position)

        self.NECT_price = self.NECT_price_series[self.step_count] 
        current_nav = self.nav[self.step_count] 

        cost = self._compute_transaction_cost(action_size)
        self.NECT = max(self.NECT + action_size, 0)
        self.LP -= cost/current_nav

        NECT_value = self.NECT * self.NECT_price
        self.collateral_value = self.LP*current_nav 
        self.volatile_collateral_value = self.collateral_value*self.LP_volatile_part[self.step_count]
        self.leverage_margin = NECT_value/(self.collateral_value - self.volatile_collateral_value) - 1

        self.cr_margin = self.collateral_value/(NECT_value + 1e-8) - 1
        """if self.cr_margin <= self.min_cr:
            reward = -math.tanh(self.collateral_value) - 0.5
            #done = True
            global H
            H += 1
        else:
            reward = 0"""
        penalty = -5*self.plateau(self.leverage_margin) + 2*abs(self.leverage_margin)#+ 0.1*self.deltaFunction(self.cr_margin-self.min_cr)
        reward = -10**(-3)*penalty #+ self.treshold  #- cost / 10_000 
        self.step_count += 1
        done = self.step_count >= self.max_steps - 1
        global H
        H += 1
        if H>15000:
            reward/=10**5

        # Save metrics
        self.history['NECT value'].append(NECT_value)
        self.history['NECT'].append(self.NECT)
        self.history['volatile value'].append(self.volatile_collateral_value)
        self.history['collateral value'].append(self.collateral_value)
        self.history['LP'].append(self.LP)
        self.history['reward'].append(reward)
        self.history['collateral ratio'].append(self.cr_margin + 1)
        self.history['leverage'].append(self.leverage_margin + 2)

        return self._get_obs(), reward, done, {}

    def get_history(self):
        #global H
        #print('liquidation', H)
        return self.history