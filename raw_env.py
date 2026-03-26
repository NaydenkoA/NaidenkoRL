import gymnasium as gym
from gymnasium import spaces
import numpy as np
from special_functions import plateau

class rawDLVEnv(gym.Env):
    def __init__(
            self,
            nect_honey_price,
            LP_nav_honey,
            volatile_part,
            init_LP = 1,
            max_step = 0.15,
            target_leverage = 2
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

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.LP = self.starting_LP
        self.collateral_value = self.LP*self.nav[0]
        self.volatile_part_value = self.collateral_value*self.volatile_part[0]
        self.nect = (self.collateral_value - self.volatile_part_value)*(self.target_leverage - 1)
        self.leverage_margin = self.nect/(self.collateral_value - self.volatile_part_value) + 1 - self.target_leverage

        self.history = {
            'NECT': [],
            'collateral value': [],
            'volatile value': [],
            'reward': [],
            'leverage': []
        }

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        return np.array([
            self.nect,
            self.collateral_value,
            self.volatile_part_value,
            self.leverage_margin
        ], dtype=np.float32)
    
    def step(self, action):
        action_size = float(action[0]) * self.max_action
        action_size = np.clip(action_size, -self.max_action, self.max_action)

        self.nect = max(self.nect + action_size, 0)    

        current_nav = self.nav[self.step_count]
        self.collateral_value = current_nav*self.LP
        self.volatile_part_value = self.volatile_part[self.step_count]*self.collateral_value
        self.leverage_margin = self.nect/(self.collateral_value - self.volatile_part_value) + 1 - self.target_leverage

        leverage_reward = plateau(self.leverage_margin)

        reward = leverage_reward

        self.history['NECT'].append(self.nect)
        self.history['collateral value'].append(self.collateral_value*self.nect_price[self.step_count])
        self.history['volatile value'].append(self.volatile_part_value*self.nect_price[self.step_count])
        self.history['leverage'].append(self.target_leverage + self.leverage_margin)
        self.history['reward'].append(reward)

        obs = self._get_obs()
        terminated = False
        self.step_count += 1
        truncated = self.step_count >= self.max_steps 
        info = {}

        return obs, reward, terminated, truncated, info
    
    def get_history(self):
        return self.history