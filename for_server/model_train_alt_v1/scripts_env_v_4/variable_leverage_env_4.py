import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .special_functions import plateau, sigmoidBarier, plateauFlat
from .data_generator_v2 import simulate_data
from collections import deque
import math

class agentDLVEnvV4(gym.Env):
    def __init__(
            self,
            nect_honey_price,
            nav_in_honey,
            vol_token_honey_price,
            vol_part,
            top_leverage,
            bottom_leverage,
            seed = None,
            is_training = False,
            is_mixed = False,
            is_reversed = False,
            show_setup = True,
            is_syntetic = False,
            mixed_episide_frac = 0.4,
            spread = 0.005,
            min_cr = 1.2,
            boundary_debt_per_loop = 0.9/1.2,
            collateral_LP = 1,
            horizon = 100,
            return_window = 20,
            index = None,
            primordial_action_penalty = 0.25,
            penalty_damping = 0.9,
            action_penalty_scale = 1,
            liquidation_penalty_scale = 0,
            loss_penalty_scale = 0,
            risk_reward_scale = 0,
            boundary_reward_scale = 0,
            reward_scale = 1,
            bound = 5,
            idle_reward = 0.05,
            optimal_rebalancing_reward = 0.2,
            extravagance_penalty = 0.1,
            plateau_flatness = 3.2,
            echo_penalty_scale = 1,
            echo_penalty_damping = 0.97,
            positive_consequence_reward = 0.15,
            negative_consequence_penalty = 0.2
        ):
        
        super(agentDLVEnvV4, self).__init__()

        self.pure_nect_honey_price = nect_honey_price.copy()
        self.pure_nav = nav_in_honey.copy()
        self.pure_vol_part = vol_part.copy()
        self.pure_vol_token_honey_price = vol_token_honey_price.copy()
        self.pure_max_episodes = len(self.pure_nav)
        self.top_leverage = top_leverage
        self.bottom_leverage = bottom_leverage
        self.min_cr = min_cr
        self.spread = spread
        self.is_training = is_training
        self.is_mixed = is_mixed
        self.is_reversed = is_reversed
        self.mixed_episode_lenght = int(mixed_episide_frac*self.pure_max_episodes)
        self.boundary_debt_per_loop = boundary_debt_per_loop
        self.lp = collateral_LP
        self.max_relative_commulative_debt = 1/(min_cr - 1)
        self.horizon = horizon
        self.return_window = return_window
        self.index = index
        self.is_syntetic = is_syntetic

        self.primordial_action_penalty = primordial_action_penalty
        self.penalty_damping = penalty_damping
        self.action_penalty_scale = action_penalty_scale
        self.liquidation_penalty_scale = liquidation_penalty_scale
        self.loss_penalty_scale = loss_penalty_scale
        self.risk_reward_scale = risk_reward_scale
        self.boundary_reward_scale = boundary_reward_scale
        self.reward_scale = reward_scale
        self.bound = bound
        self.idle_reward = idle_reward
        self.optimal_rebalancing_reward = optimal_rebalancing_reward
        self.extravagance_penalty = extravagance_penalty
        self.plateau_flatness = plateau_flatness
        self.echo_penalty_scale = echo_penalty_scale
        self.echo_penalty_damping = echo_penalty_damping
        self.positive_consequence_reward = positive_consequence_reward
        self.negative_consequence_penalty = negative_consequence_penalty

        if show_setup:
            print("==== USING V4 VERSION ====")
            print('action scale', self.action_penalty_scale)
            print('primordial penalty', self.primordial_action_penalty)
            print('penalty damping', self.penalty_damping)
            print('reward scale', self.reward_scale)
            print('reward bound', self.bound)
            print('loss scale', self.loss_penalty_scale)
            print('liquidation penalty', self.liquidation_penalty_scale)
            print('risk coef', self.risk_reward_scale)
            print('boundary reward scale', self.boundary_reward_scale)
            print('idle reward', self.idle_reward)
            print('rebalancing reward', self.optimal_rebalancing_reward)
            print('extravagance penalty', self.extravagance_penalty)

        self.seed = seed
        self.rng = np.random.default_rng(seed = self.seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(37,), dtype=np.float32)

    def reset(self, seed = None, options = None, top_leverage = None, bottom_leverage = None):
        super().reset(seed=seed)

        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            if self.index is not None: 
                print("Env", self.index,"rng test:", self.rng.uniform(0,1))

        if top_leverage is not None:
            self.top_leverage = top_leverage

        if bottom_leverage is not None:
            self.bottom_leverage = bottom_leverage

        if self.is_training:
            lenght = self.top_leverage - self.bottom_leverage
            self.bottom_leverage = self.bottom_leverage*self.rng.uniform(0.93, 1.07)
            self.top_leverage = self.bottom_leverage + lenght*self.rng.uniform(0.8, 1.2)
            if self.top_leverage > 3.95:
                self.top_leverage = 3.95*self.rng.uniform(0.9, 1)
                self.bottom_leverage = self.top_leverage - lenght*self.rng.uniform(0.8, 1.2)

        if self.is_syntetic:
            self._generate_syntetic_data()
        else:
            self._generate_data_from_real_dataset()

        if self.is_training:
            self.history_leverage = []
            self.history_total_losses = []
            self.history_losses_time = []
            self.history_recent_loss = [0] #for testing
        else:
            self.history_metric = {
                'leverage': [],
                'leverage mean': [],
                'leverage gain': [],
                'volatile token': [],
                'volatile token mean': [],
                'volatile token gain': [],
                'calm part': [],
                'calm part mean': [],
                'calm part gain': [],
                'collaterization ratio': [],
                'leverage vol': [],
                'volatile token vol': [],
                'calm part vol': [],
                'leverage consistency': [],
                'volatile token consistency': [],
                'calm part consistency': [],
                'p return': [],
                'p return abs': [],
                'p return square': [],
                'p return vol': [],
                'c return': [],
                'c return abs': [],
                'c return square': [],
                'c return vol': [],
                'l return': [],
                'l return abs': [],
                'l return square': [],
                'l return vol': [],
                'nav return': [],
                'nav return abs': [],
                'nav return square': [],
                'nav return vol': [],
                'leverage slope abs': [],
                'volatile token slope abs': [],
                'calm part slope abs': [],
                'equilibrium distance': [],
                'equilibrium distance slope': [],
                'equilibrium distance mean': []
            }

            self.history = {
                'leverage': [],
                'reward': [],
                'collaterization ratio' : [],
                'loss on spread': [],
                'recent total loss': [],
                'boundary leverage': []
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
                'calm part': [],
                'total loss': []
            }

        self.step_count = 0

        self.current_nav = self.nav[0]
        self.current_nect_honey_price = self.nect_honey_price[0]
        self.current_vol_token_honey_price = self.vol_token_honey_price[0]
        self.current_vol_part = self.vol_part[0]
        self.current_calm_part = 1 - self.vol_part[0]

        self.collateral_value = self.lp*self.current_nav

        self.target_leverage = (self.top_leverage + self.bottom_leverage)/2
        self.top_target_leverage = (self.top_leverage + self.target_leverage)/2
        self.bottom_target_leverage = (self.target_leverage + self.bottom_leverage)/2

        self.primordial_leverage = self._get_primordial_leverage_init()
        #_, self.primordial_leverage = self._get_naive_primordial_leverage()
        self.commulative_debt_value = self.primordial_leverage*self.collateral_value
        self.commulative_lp = self.commulative_debt_value/self.current_nav
        self.commulative_debt_nect = self.commulative_debt_value/self.current_nect_honey_price

        if self.is_training:
            x_max = self.current_calm_part*(self.top_leverage - self.primordial_leverage - 1)
            x_min = self.current_calm_part*(self.bottom_leverage - self.primordial_leverage - 1)
            x = self.rng.uniform(x_min*0.9, x_max*1.1)
            x = min(x, 1)

            self.vol_token_value = x*self.collateral_value
            self.vol_token = self.vol_token_value/self.current_vol_token_honey_price
            self.noncommulative_debt_nect = self.vol_token_value/self.current_nect_honey_price
            self.noncommulative_debt = self.vol_token_value
        else:
            x = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)
            
            self.vol_token_value = x*self.collateral_value
            self.vol_token = self.vol_token_value/self.current_vol_token_honey_price
            self.noncommulative_debt_nect = self.vol_token_value/self.current_nect_honey_price
            self.noncommulative_debt = self.vol_token_value
        #x = 0.742729443038576
        self.leverage = x/self.current_calm_part + self.primordial_leverage + 1

        self.total_lp_value = self.collateral_value + self.commulative_lp*self.current_nav
        self.total_debt_value = (self.noncommulative_debt_nect + self.commulative_debt_nect)*self.current_nect_honey_price
        self.cr = self.total_lp_value/self.total_debt_value
        if self.cr<=0:
            print(x, self.top_leverage, self.bottom_leverage)
        self.vol_token_normed = x

        self.boundary_leverage = self._get_boundary_leverage(self.primordial_leverage)

        self.leverage_history = np.zeros(self.horizon)
        self.relative_vol_token_value_history = np.zeros(self.horizon)
        self.calm_part_history = np.zeros(self.horizon)
        self.p_return_history = np.zeros(self.return_window)
        self.nav_return_history = np.zeros(self.return_window)
        self.l_return_history = np.zeros(self.return_window)
        self.c_return_history = np.zeros(self.return_window)
        self.leverage_slope_dir_history = np.zeros(self.horizon)
        self.vol_token_slope_dir_history = np.zeros(self.horizon)
        self.calm_part_slope_dir_history = np.zeros(self.horizon)
        self.equilibrium_distance_history = np.zeros(self.horizon)
        
        self.p_return = 0
        self.l_return = 0
        self.nav_return = 0
        self.c_return = 0

        self.last_rebalance_step = 0

        self.norm = self.top_leverage - self.bottom_leverage
        self.leverage_normed = (self.leverage - self.bottom_leverage)/self.norm
        self.equilibrium_distance_normed = 2*abs(self.leverage_normed - 0.5)

        self._update_statistic()

        self.boundary_leverage_normed = (self.boundary_leverage - self.bottom_leverage)/self.norm 
        
        self.leverage_mean_normed = (self.leverage_mean - self.bottom_leverage)/self.norm
        self.leverage_gain_mean_normed = self._normalize_slope(self.leverage_gain_mean, 1831.8)
        self.vol_token_gain_mean_normed = self._normalize_slope(self.vol_token_gain_mean, 13290.7)
        self.calm_part_gain_mean_normed = self._normalize_slope(self.calm_part_gain_mean, 3629.6)
        self.leverage_gain_abs_normed = abs(self.leverage_gain_mean_normed)
        self.vol_token_gain_abs_normed = abs(self.vol_token_gain_mean_normed)
        self.calm_part_gain_abs_normed = abs(self.calm_part_gain_mean_normed)
        self.equilibrium_distance_slope_normed = np.sign(self.leverage_normed - 0.5)*self.leverage_gain_mean_normed 

        self.leverage_vol_normed = math.tanh(5*self.leverage_vol/self.norm)
        self.vol_token_vol_normed = math.tanh(100*self.vol_token_vol)
        self.calm_part_vol_normed = math.tanh(10*self.calm_part_vol)

        self._normilize_returns()

        self.cr_normed = 1/(self.cr)

        self.primordial_leverage_normed = self.primordial_leverage/self.max_relative_commulative_debt
        self.recent_reward = 1 + self.idle_reward
        self.relative_loss = 0
        self.losses_history = np.zeros(self.horizon)
        self.recenet_total_loss = 0
        self.integral_loss = 0
        self.last_delta_leverage = 0
        self.last_delta_vol_token = 0
        self.last_delta_eq_dist = 0
        self.last_mechanical_intervention = 0
        self.last_rebalancing_target_leverage = 0

        self.portfolio_value = self.vol_token_value + self.total_lp_value - self.total_debt_value
        self.initial_portfolio_value = self.vol_token_value + self.total_lp_value - self.total_debt_value

        self.current_action_penalty = 0

        if self.is_training:
            self.history_leverage.append(self.leverage_normed)
        else:
            self._update_history()

        if not self.is_training:
            self.action_counter = np.zeros(4)

        obs = self._get_obs()
        info = {}
        return obs, info
    
    def _get_obs(self):
        return np.array([
            self.leverage_normed,
            self.cr_normed,
            self.leverage_vol_normed,
            self.leverage_mean_normed,
            self.leverage_gain_mean_normed,
            self.vol_token_normed,
            self.vol_token_mean,
            self.vol_token_gain_mean_normed,
            self.current_calm_part,
            self.calm_part_mean,
            self.calm_part_gain_mean_normed,
            self.vol_token_vol_normed,
            self.calm_part_vol_normed,
            self.leverage_gain_consistency,
            self.vol_token_gain_consistency,
            self.calm_part_gain_consistency,
            self.p_return_normed,
            self.p_return_abs_normed,
            self.p_return_square_normed,
            self.p_return_sigma_normed,
            self.l_return_normed,
            self.l_return_abs_normed,
            self.l_return_square_normed,
            self.l_return_sigma_normed,
            self.c_return_normed,
            self.c_return_abs_normed,
            self.c_return_square_normed,
            self.c_return_sigma_normed,
            self.nav_return_normed,
            self.nav_return_abs_normed,
            self.nav_return_square_normed,
            self.nav_return_sigma_normed,
            self.leverage_gain_abs_normed,
            self.vol_token_gain_abs_normed,
            self.calm_part_gain_abs_normed,
            self.equilibrium_distance_normed,
            self.equilibrium_distance_slope_normed
        ], dtype=np.float32)
    
    def step(self, action):
        if not self.is_training:
            self.action_counter[action] += 1

        if action != 0:
            self.last_rebalance_step = self.step_count + 1

            if action == 1:
                x = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)
                self.last_delta_leverage = self.leverage_normed - 0.5
                self.last_delta_vol_token = self.vol_token_normed - x
                self.last_rebalancing_target_leverage = 0.5
                self.last_delta_eq_dist = 2*abs(self.leverage_normed - 0.5)
                delta_leverage = abs(self.leverage_normed - 0.5)
                is_tacktical = 0
            else:
                if action == 2:
                    x = self.current_calm_part*(self.top_target_leverage - self.primordial_leverage - 1)
                    self.last_delta_vol_token = self.vol_token_normed - x
                    delta_leverage = abs(self.leverage_normed - 0.75)
                    self.last_delta_leverage = self.leverage_normed - 0.75
                    self.last_rebalancing_target_leverage = 0.75
                    self.last_delta_eq_dist = 2*abs(self.leverage_normed - 0.5) - 0.5
                    is_tacktical = 1
                else:
                    x = self.current_calm_part*(self.bottom_target_leverage - self.primordial_leverage - 1)
                    self.last_delta_vol_token = self.vol_token_normed - x
                    delta_leverage = abs(self.leverage_normed - 0.25)
                    self.last_delta_leverage = self.leverage_normed - 0.25
                    self.last_rebalancing_target_leverage = 0.25
                    self.last_delta_eq_dist = 2*abs(self.leverage_normed - 0.5) - 0.5
                    is_tacktical = 1
            #print(delta_leverage)
            self._reload_running_statistic()

            echo_reward = 0

            if delta_leverage <= 0.15:
                is_extravagance = 1
            else:
                is_extravagance = 0

            self.current_action_penalty = self.primordial_action_penalty - self.optimal_rebalancing_reward*is_tacktical + self.current_action_penalty*self.penalty_damping + self.extravagance_penalty*is_extravagance
            if self.leverage_normed < 0.95 and self.leverage_normed > 0.05:
                action_penalty = self.current_action_penalty
            else:
                action_penalty = self.primordial_action_penalty - self.optimal_rebalancing_reward*is_tacktical + self.extravagance_penalty*is_extravagance

            new_vol_token_value = x*self.collateral_value
            new_vol_token = new_vol_token_value/self.current_vol_token_honey_price
            vt_nect_price = self.current_vol_token_honey_price/self.current_nect_honey_price

            if new_vol_token < self.vol_token:
                vt_to_sell = self.vol_token - new_vol_token
                effective_selling_price = vt_nect_price*(1 - self.spread)
                nect_to_repay = vt_to_sell*effective_selling_price

                self.noncommulative_debt_nect -= nect_to_repay
                self.vol_token = new_vol_token

                recent_loss_nect = vt_to_sell*(vt_nect_price - effective_selling_price)
                self.recent_loss_value = recent_loss_nect*self.current_nect_honey_price
            else:
                vt_to_buy = new_vol_token - self.vol_token
                nect_to_borrow = vt_to_buy*vt_nect_price

                self.noncommulative_debt_nect += nect_to_borrow
                
                effective_buying_price = vt_nect_price*(1 + self.spread)
                vt_bought = nect_to_borrow/effective_buying_price
                self.vol_token += vt_bought

                recent_loss_vt = nect_to_borrow*(1/vt_nect_price - 1/effective_buying_price)
                self.recent_loss_value = recent_loss_vt*self.current_vol_token_honey_price
        else:
            self.current_action_penalty = self.current_action_penalty*self.penalty_damping
            action_penalty = -self.idle_reward if self.leverage_normed < 0.95 and self.leverage_normed > 0.05 else 0
            self.recent_loss_value = 0

            if self.last_rebalance_step != self.last_mechanical_intervention:
                old_leverage_normed = self.leverage_normed + self.last_delta_leverage

                primordial_drift = abs(self.last_rebalancing_target_leverage + self.last_delta_leverage - 0.5) - abs(self.last_rebalancing_target_leverage - 0.5)
                if abs(self.last_delta_leverage) == primordial_drift:
                    if self.last_delta_leverage != 0:
                        norm = abs(self.last_delta_leverage)
                    else:
                        norm = 1
                else:
                    norm = abs(self.last_delta_leverage) - primordial_drift

                drift = (abs(old_leverage_normed - 0.5) - abs(self.leverage_normed - 0.5) - primordial_drift)/(norm)

                if abs(drift*norm) < 0.15 or self.last_delta_leverage == 0:
                    drift = 0
                else:
                    if drift < 0:
                        drift = drift - self.negative_consequence_penalty
                    else:
                        drift = drift + self.positive_consequence_reward

                echo_reward = self.echo_penalty_scale*drift*self.echo_penalty_damping**(self.step_count - self.last_rebalance_step)
            else:
                echo_reward = 0

        if self.recent_loss_value == 0:
            self.relative_loss += self.recent_loss_value/self.collateral_value
            if self.is_training:
                self.history_recent_loss.append(self.relative_loss)
        else:
            self.relative_loss += max(self.recent_loss_value/self.collateral_value, 0.0002) + 0.0002
            if self.is_training:
                self.history_total_losses.append(self.relative_loss)
                self.history_losses_time.append(self.step_count + 1)
                self.history_recent_loss.append(self.relative_loss)
        self.integral_loss += self.relative_loss
        prev_loss = self.losses_history[self.step_count % self.horizon]
        self.losses_history[self.step_count % self.horizon] = self.relative_loss
        self.recenet_total_loss = self.relative_loss - prev_loss

        self.step_count += 1

        self.current_nav = self.nav[self.step_count]
        self.current_nect_honey_price = self.nect_honey_price[self.step_count]
        self.current_vol_token_honey_price = self.vol_token_honey_price[self.step_count]
        self.current_vol_part = self.vol_part[self.step_count]
        self.current_calm_part = 1 - self.vol_part[self.step_count]

        self.collateral_value = self.lp*self.current_nav
        self.vol_token_value = self.vol_token*self.current_vol_token_honey_price
        self.commulative_debt_value = self.commulative_debt_nect*self.current_nect_honey_price
        self.noncommulative_debt = self.noncommulative_debt_nect*self.current_nect_honey_price
        self.total_lp_value = self.collateral_value + self.commulative_lp*self.current_nav
        self.total_debt_value = self.commulative_debt_value + self.noncommulative_debt

        x = self.vol_token_value/self.collateral_value
        leverage = x/self.current_calm_part + self.primordial_leverage + 1

        self.p_return = math.log(self.vol_token_honey_price[self.step_count - 1]/self.vol_token_honey_price[self.step_count])
        if self.last_rebalance_step == self.step_count:
            self.l_return = math.log((self.leverage - self.last_delta_leverage*self.norm)/leverage)
        else:
            self.l_return = math.log(self.leverage/leverage)
        self.nav_return = math.log(self.nav[self.step_count - 1]/self.nav[self.step_count])
        self.c_return = math.log((1 - self.vol_part[self.step_count - 1])/(1 - self.vol_part[self.step_count]))

        self.leverage_gain = leverage - self.leverage
        self.leverage = leverage

        self.cr = self.total_lp_value/self.total_debt_value

        self.vol_token_normed = x
        self.boundary_leverage = self._get_boundary_leverage(self.primordial_leverage)

        self.leverage_normed = (self.leverage - self.bottom_leverage)/self.norm
        self.equilibrium_distance_normed = 2*abs(self.leverage_normed - 0.5)

        self._update_statistic()

        self.boundary_leverage_normed = (self.boundary_leverage - self.bottom_leverage)/self.norm
        self.leverage_mean_normed = (self.leverage_mean - self.bottom_leverage)/self.norm

        self.leverage_gain_mean_normed = self._normalize_slope(self.leverage_gain_mean, 1831.8)
        self.vol_token_gain_mean_normed = self._normalize_slope(self.vol_token_gain_mean, 13290.7)
        self.calm_part_gain_mean_normed = self._normalize_slope(self.calm_part_gain_mean, 3629.6)
        self.leverage_gain_abs_normed = abs(self.leverage_gain_mean_normed)
        self.vol_token_gain_abs_normed = abs(self.vol_token_gain_mean_normed)
        self.calm_part_gain_abs_normed = abs(self.calm_part_gain_mean_normed)

        self.equilibrium_distance_slope_normed = self._normalize_slope(self.equilibrium_distance_slope, 467.2)

        self.leverage_vol_normed = math.tanh(self.leverage_vol/self.norm/0.13)
        self.vol_token_vol_normed = math.tanh(self.vol_token_vol/0.007)
        self.calm_part_vol_normed = math.tanh(self.calm_part_vol/0.019)

        self.cr_normed = min(1/max(self.cr + 1e-6, 1e-10), 1)

        liquidation_penalty = sigmoidBarier(self.cr - self.min_cr)
        #loss_penalty = 1 - plateau(self.recenet_total_loss*2000) + self.recenet_total_loss*1000
        total_penalty = self.action_penalty_scale*action_penalty + self.liquidation_penalty_scale*liquidation_penalty #+ self.loss_penalty_scale*loss_penalty

        leverage_maintain_reward = 2*plateauFlat(self.leverage_normed - 0.5, 0.5, self.plateau_flatness) - 1 + 0.4*(min(self.leverage_normed, 0) + min(1 - self.leverage_normed, 0))
        leverage_improvement_reward = abs(self.leverage_mean_normed - 0.5) - abs(self.leverage_normed - 0.5)
        leverage_reward = plateauFlat(self.leverage_mean_normed - 0.5, 0.45)
        risk_reward = min(self.leverage_normed, 1) + min(self.leverage_mean_normed, 1)
        #boundary_leverage_reward = 5*(min(1, self.bottom_leverage_normed) - 0.8)
        total_reward = 1*leverage_maintain_reward + 1*echo_reward #+ 0*leverage_reward + 0*leverage_improvement_reward + self.risk_reward_scale*risk_reward #+ self.boundary_reward_scale*boundary_leverage_reward

        self._normilize_returns()

        reward = total_reward - total_penalty
        self.recent_reward = reward

        if self.is_training:
            self.history_leverage.append(self.leverage_normed)
        else:
            self._update_history()

        if self.boundary_leverage_normed <= 1:
            self.last_rebalance_step = self.step_count + 1
            self.last_mechanical_intervention = self.step_count + 1
            self._rebalance_loop()
            self._reload_running_statistic()

        self.portfolio_value = self.vol_token_value + self.total_lp_value - self.total_debt_value

        terminated = (self.cr<=self.min_cr and not self.is_training) or self.cr <= 0
        truncated = self.step_count >= self.max_episodes - 1
        obs = self._get_obs()
        info = {}

        done = terminated or truncated
        if done:
            if self.is_training:
                l = np.array(self.history_leverage)
                self.history_losses_time.append(self.step_count)
                self.history_total_losses.append(self.relative_loss)
                loss_times = np.array(self.history_losses_time)
                total_losses = np.array(self.history_total_losses)
                #print(loss_times)
                info = {
                    'integral loss': self.integral_loss,
                    'quantile': np.quantile(np.abs(l - 0.5), 0.98),
                    'mean': np.mean(l),
                    'steps': self.step_count,
                    'loss_rate': 1e7*np.dot(loss_times, total_losses)/np.dot(loss_times, loss_times),
                    'losses': self.history_recent_loss.copy(),
                    'leverage': l.copy()
                }
            else:
                #print(3)
                saved = {k: v.copy() for k, v in self.history.items()}
                info['episode_history'] = saved
                saved = {k: v.copy() for k, v in self.history_metric.items()}
                info['episode_metrics'] = saved
                saved = {k: v.copy() for k, v in self.history_economics.items()}
                info['episode_economics'] = saved
                info['top leverage'] = self.top_leverage
                info['bottom leverage'] = self.bottom_leverage
                s = np.sum(self.action_counter)
                info['0 action'] = self.action_counter[0]/s
                info['0.5 rebalancing'] = self.action_counter[1]/s
                info['0.75 rebalancing'] = self.action_counter[2]/s
                info['0.25 rebalancing'] = self.action_counter[3]/s

        return obs, self._normalize_reward(self.reward_scale*reward), terminated, truncated, info

    def _normalize_reward(self, reward):
        if abs(reward) > self.bound:
            #print(reward)
            if reward > 0:
                return self.bound
            else:
                return -self.bound
        else:
            return reward

    def _generate_data_from_real_dataset(self):

        if self.is_mixed:
            if self.is_reversed:
                self.max_episodes = self.mixed_episode_lenght

                N = self.rng.integers(0, self.pure_max_episodes - self.mixed_episode_lenght)
                self.nav = self.pure_nav[N:N + self.mixed_episode_lenght]
                self.vol_token_honey_price = self.pure_vol_token_honey_price[N:N + self.mixed_episode_lenght]
                #print(N)

                N = self.rng.integers(0, self.pure_max_episodes - self.mixed_episode_lenght)
                self.nect_honey_price = self.pure_nect_honey_price[N:N + self.mixed_episode_lenght]
                #print(N)

                N = self.rng.integers(0, self.pure_max_episodes - self.mixed_episode_lenght)
                self.vol_part = self.pure_vol_part[N:N + self.mixed_episode_lenght]
                #print(N)

            else:
                N = self.rng.integers(0, self.pure_max_episodes - self.mixed_episode_lenght)
                #N = 6650
                #print(N)
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

        if self.is_reversed:
            if self.is_training:
                r = self.rng.integers(0, 2)
                if r == 1:
                    self.nav = self.nav[::-1]
                    self.vol_token_honey_price = self.vol_token_honey_price[::-1]
                #print(r)

                r = self.rng.integers(0, 2)
                if r == 1:
                    self.vol_part = self.vol_part[::-1]
                #print(r)
            else:
                self.nav = self.nav[::-1]
                self.vol_part = self.vol_part[::-1]
                self.vol_token_honey_price = self.vol_token_honey_price[::-1]

    def _generate_syntetic_data(self):
        N = 4*self.mixed_episode_lenght
        self.max_episodes = self.mixed_episode_lenght

        price, nav, vol_part, stable_price = simulate_data(
            n_steps = N,
            rng = self.rng
        )

        self.vol_token_honey_price = price[::4]
        self.nav = nav[::4]
        self.vol_part = vol_part[::4]
        self.nect_honey_price = stable_price[::4]

    def _update_history(self):
        self.history['leverage'].append(self.leverage)
        self.history['reward'].append(self._normalize_reward(self.reward_scale*self.recent_reward))
        self.history['collaterization ratio'].append(self.cr)
        self.history['loss on spread'].append(self.relative_loss)
        self.history['recent total loss'].append(self.recenet_total_loss)
        self.history['boundary leverage'].append(self.boundary_leverage)
        self.history_metric['leverage'].append(self.leverage_normed)
        self.history_metric['leverage gain'].append(self.leverage_gain_mean_normed)
        self.history_metric['leverage mean'].append(self.leverage_mean_normed)
        self.history_metric['collaterization ratio'].append(self.cr_normed)
        self.history_metric['volatile token'].append(self.vol_token_normed)
        self.history_metric['volatile token mean'].append(self.vol_token_mean)
        self.history_metric['volatile token gain'].append(self.vol_token_gain_mean_normed)
        self.history_metric['calm part'].append(self.current_calm_part)
        self.history_metric['calm part mean'].append(self.calm_part_mean)
        self.history_metric['calm part gain'].append(self.calm_part_gain_mean_normed)
        self.history_metric['calm part vol'].append(self.calm_part_vol_normed)
        self.history_metric['volatile token vol'].append(self.vol_token_vol_normed)
        self.history_metric['leverage vol'].append(self.leverage_vol_normed)
        self.history_metric['calm part consistency'].append(self.calm_part_gain_consistency)
        self.history_metric['volatile token consistency'].append(self.vol_token_gain_consistency)
        self.history_metric['leverage consistency'].append(self.leverage_gain_consistency)
        self.history_metric['p return'].append(self.p_return_normed)
        self.history_metric['p return abs'].append(self.p_return_abs_normed)
        self.history_metric['p return square'].append(self.p_return_square_normed)
        self.history_metric['p return vol'].append(self.p_return_sigma_normed)
        self.history_metric['c return'].append(self.c_return_normed)
        self.history_metric['c return abs'].append(self.c_return_abs_normed)
        self.history_metric['c return square'].append(self.c_return_square_normed)
        self.history_metric['c return vol'].append(self.c_return_sigma_normed)
        self.history_metric['l return'].append(self.l_return_normed)
        self.history_metric['l return abs'].append(self.l_return_abs_normed)
        self.history_metric['l return square'].append(self.l_return_square_normed)
        self.history_metric['l return vol'].append(self.l_return_sigma_normed)
        self.history_metric['nav return'].append(self.nav_return_normed)
        self.history_metric['nav return abs'].append(self.nav_return_abs_normed)
        self.history_metric['nav return square'].append(self.nav_return_square_normed)
        self.history_metric['nav return vol'].append(self.nav_return_sigma_normed)
        self.history_metric['leverage slope abs'].append(self.leverage_gain_abs_normed)
        self.history_metric['volatile token slope abs'].append(self.vol_token_gain_abs_normed)
        self.history_metric['calm part slope abs'].append(self.calm_part_gain_abs_normed)
        self.history_metric['equilibrium distance'].append(self.equilibrium_distance_normed)
        self.history_metric['equilibrium distance slope'].append(self.equilibrium_distance_slope_normed)
        self.history_metric['equilibrium distance mean'].append(self.equilibrium_distance_normed_mean)
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
        self.history_economics['total loss'].append(self.relative_loss)

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

    def _get_average_gain_and_consistency(self, series):
        n = min(self.step_count + 1, self.horizon)
        if n < 3:
            return 0.0, 0.5

        i_0 = self.step_count % self.horizon
        mu = (n - 1) / 2

        sum_1 = 0.0
        sum_2 = 0.0
        for i in range(n):
            idx = (i_0 - n + 1 + i) % n
            sum_1 += (i - mu) * series[idx]
            sum_2 += (i - mu)**2
        avg_gain = sum_1 / (sum_2 + 1e-8)

        slope_sign = np.sign(avg_gain)
        if slope_sign == 0:
            return avg_gain, 0.5

        same_sign_count = 0
        total_pairs = n - 1
        for i in range(total_pairs):
            idx1 = (i_0 - n + 1 + i) % n
            idx2 = (idx1 + 1) % n
            local_gain = series[idx2] - series[idx1]
            if np.sign(local_gain) == slope_sign:
                same_sign_count += 1

        consistency = same_sign_count / total_pairs
        return avg_gain, consistency
    
    def _update_return_statistic(self):
        idx = self.step_count % self.return_window
        n = min(self.step_count + 1, self.return_window)

        p_return_old = self.p_return_history[idx]
        l_retutn_old = self.l_return_history[idx]
        c_return_old = self.c_return_history[idx]
        nav_return_old = self.nav_return_history[idx]

        self.p_return_history[idx] = self.p_return
        self.l_return_history[idx] = self.l_return
        self.c_return_history[idx] = self.c_return
        self.nav_return_history[idx] = self.nav_return

        if n == 1:
            self.p_return_sum_2 = self.p_return
            self.l_return_sum_2 = self.l_return
            self.c_return_sum_2 = self.c_return
            self.nav_return_sum_2 = self.nav_return

            self.p_return_sum_3 = self.p_return
            self.l_return_sum_3 = self.l_return
            self.c_return_sum_3 = self.c_return
            self.nav_return_sum_3 = self.nav_return

            self.p_return_vol = 0
            self.l_return_vol = 0
            self.c_return_vol = 0
            self.nav_return_vol = 0

            self.is_return_horizon_reached_first_time = True
        else:
            if n < self.return_window:
                self.p_return_sum_2 += self.p_return
                self.l_return_sum_2 += self.l_return
                self.c_return_sum_2 += self.c_return
                self.nav_return_sum_2 += self.nav_return

                self.p_return_sum_3 += self.p_return**2
                self.l_return_sum_3 += self.l_return**2
                self.c_return_sum_3 += self.c_return**2
                self.nav_return_sum_3 += self.nav_return**2
            else:
                if self.is_return_horizon_reached_first_time:
                    self.p_return_sum_2 += self.p_return
                    self.l_return_sum_2 += self.l_return
                    self.c_return_sum_2 += self.c_return
                    self.nav_return_sum_2 += self.nav_return

                    self.p_return_sum_3 += self.p_return**2
                    self.l_return_sum_3 += self.l_return**2
                    self.c_return_sum_3 += self.c_return**2
                    self.nav_return_sum_3 += self.nav_return**2

                    self.is_return_horizon_reached_first_time = False
                else:
                    self.p_return_sum_2 += self.p_return - p_return_old
                    self.l_return_sum_2 += self.l_return - l_retutn_old
                    self.c_return_sum_2 += self.c_return - c_return_old
                    self.nav_return_sum_2 += self.nav_return - nav_return_old

                    self.p_return_sum_3 += self.p_return**2 - p_return_old**2
                    self.l_return_sum_3 += self.l_return**2 - l_retutn_old**2
                    self.c_return_sum_3 += self.c_return**2 - c_return_old**2
                    self.nav_return_sum_3 += self.nav_return**2 - nav_return_old**2
            
            p_mean = self.p_return_sum_2/n
            l_mean = self.l_return_sum_2/n
            c_mean = self.c_return_sum_2/n
            nav_mean = self.nav_return_sum_2/n

            self.p_return_vol = math.sqrt(max(0, self.p_return_sum_3/n - p_mean**2))
            self.l_return_vol = math.sqrt(max(0, self.l_return_sum_3/n - l_mean**2))
            self.c_return_vol = math.sqrt(max(0, self.c_return_sum_3/n - c_mean**2))
            self.nav_return_vol = math.sqrt(max(0, self.nav_return_sum_3/n - nav_mean**2))

    def _update_statistic(self):
        idx = self.step_count % self.horizon
        prev_idx = (self.step_count - 1) % self.horizon
        n = min(self.step_count + 1, self.horizon)

        leverage_old = self.leverage_history[self.step_count % self.horizon]
        relative_vol_token_value_old = self.relative_vol_token_value_history[self.step_count % self.horizon]
        calm_part_old = self.calm_part_history[self.step_count % self.horizon]

        equilibrium_distance_old = self.equilibrium_distance_history[idx]

        self.leverage_history[idx] = self.leverage
        self.relative_vol_token_value_history[idx] = self.vol_token_normed
        self.calm_part_history[idx] = self.current_calm_part
        self.equilibrium_distance_history[idx] = self.equilibrium_distance_normed

        leverage_slope_dir_old = self.leverage_slope_dir_history[idx]
        vol_token_slope_dir_old = self.vol_token_slope_dir_history[idx]
        calm_part_slope_dir_old = self.calm_part_slope_dir_history[idx]

        self.leverage_slope_dir_history[idx] = np.sign(self.leverage - self.leverage_history[prev_idx]) if n > 1 else 0
        self.vol_token_slope_dir_history[idx] = np.sign(self.vol_token_normed - self.relative_vol_token_value_history[prev_idx]) if n > 1 else 0
        self.calm_part_slope_dir_history[idx] = np.sign(self.current_calm_part - self.calm_part_history[prev_idx]) if n > 1 else 0

        if n == 1: 
            self.leverage_mean = self.leverage
            self.vol_token_mean = self.vol_token_normed
            self.calm_part_mean = self.current_calm_part
            self.equilibrium_distance_normed_mean = self.equilibrium_distance_normed

            self.leverage_gain_mean = 0
            self.vol_token_gain_mean = 0
            self.calm_part_gain_mean = 0

            self.sum_1_leverage = self.leverage
            self.sum_1_vol_token = self.vol_token_normed
            self.sum_1_calm_part = self.current_calm_part
            self.sum_1_equilivrium_distance = self.equilibrium_distance_normed

            self.sum_2_leverage = self.leverage
            self.sum_2_vol_token = self.vol_token_normed
            self.sum_2_calm_part = self.current_calm_part
            self.sum_2_equilivrium_distance = self.equilibrium_distance_normed

            self.is_horizon_reached_first_time = True

            self.sum_3_leverage = self.leverage**2
            self.sum_3_vol_token = self.vol_token_normed**2
            self.sum_3_calm_part = self.current_calm_part**2

            self.leverage_slope_cons_sum = 0
            self.vol_token_slope_cons_sum = 0
            self.calm_part_slope_cons_sum = 0

            self.leverage_vol = 0
            self.vol_token_vol = 0
            self.calm_part_vol = 0

            self.leverage_gain_consistency = 0.5
            self.vol_token_gain_consistency = 0.5
            self.calm_part_gain_consistency = 0.5
        else:
            if n < self.horizon:
                self.sum_1_leverage = self.sum_1_leverage + n*self.leverage
                self.sum_1_vol_token = self.sum_1_vol_token + n*self.vol_token_normed
                self.sum_1_calm_part = self.sum_1_calm_part + n*self.current_calm_part
                self.sum_1_equilivrium_distance = self.sum_1_equilivrium_distance + n*self.equilibrium_distance_normed

                self.sum_2_leverage = self.sum_2_leverage + self.leverage
                self.sum_2_vol_token = self.sum_2_vol_token + self.vol_token_normed
                self.sum_2_calm_part = self.sum_2_calm_part + self.current_calm_part
                self.sum_2_equilivrium_distance = self.sum_2_equilivrium_distance + self.equilibrium_distance_normed
            else:
                if self.is_horizon_reached_first_time:
                    self.sum_1_leverage = self.sum_1_leverage + n*self.leverage
                    self.sum_1_vol_token = self.sum_1_vol_token + n*self.vol_token_normed
                    self.sum_1_calm_part = self.sum_1_calm_part + n*self.current_calm_part
                    self.sum_1_equilivrium_distance = self.sum_1_equilivrium_distance + n*self.equilibrium_distance_normed

                    self.sum_2_leverage = self.sum_2_leverage + self.leverage
                    self.sum_2_vol_token = self.sum_2_vol_token + self.vol_token_normed
                    self.sum_2_calm_part = self.sum_2_calm_part + self.current_calm_part
                    self.sum_2_equilivrium_distance = self.sum_2_equilivrium_distance + self.equilibrium_distance_normed

                    self.is_horizon_reached_first_time = False
                else:
                    self.sum_1_leverage = self.sum_1_leverage - self.sum_2_leverage + n*self.leverage
                    self.sum_1_vol_token = self.sum_1_vol_token - self.sum_2_vol_token + n*self.vol_token_normed
                    self.sum_1_calm_part = self.sum_1_calm_part - self.sum_2_calm_part + n*self.current_calm_part
                    self.sum_1_equilivrium_distance = self.sum_1_equilivrium_distance - self.sum_2_equilivrium_distance + n*self.equilibrium_distance_normed

                    self.sum_2_leverage = self.sum_2_leverage - leverage_old + self.leverage
                    self.sum_2_vol_token = self.sum_2_vol_token - relative_vol_token_value_old + self.vol_token_normed
                    self.sum_2_calm_part = self.sum_2_calm_part - calm_part_old + self.current_calm_part
                    self.sum_2_equilivrium_distance = self.sum_2_equilivrium_distance - equilibrium_distance_old + self.equilibrium_distance_normed

            self.leverage_mean = self.sum_2_leverage/n
            self.vol_token_mean = self.sum_2_vol_token/n
            self.calm_part_mean = self.sum_2_calm_part/n
            self.equilibrium_distance_normed_mean = self.sum_2_equilivrium_distance/n

            mu = (n + 1) / 2
            denom = n*(n**2 - 1) / 12

            self.leverage_gain_mean = (self.sum_1_leverage - mu*self.sum_2_leverage) / denom 
            self.vol_token_gain_mean = (self.sum_1_vol_token - mu*self.sum_2_vol_token) / denom
            self.calm_part_gain_mean = (self.sum_1_calm_part - mu*self.sum_2_calm_part) / denom
            self.equilibrium_distance_slope = (self.sum_1_equilivrium_distance - mu*self.sum_2_equilivrium_distance) / denom

            self.leverage_slope_cons_sum = self.leverage_slope_cons_sum + self.leverage_slope_dir_history[idx] - leverage_slope_dir_old
            self.vol_token_slope_cons_sum = self.vol_token_slope_cons_sum + self.vol_token_slope_dir_history[idx] - vol_token_slope_dir_old
            self.calm_part_slope_cons_sum = self.calm_part_slope_cons_sum + self.calm_part_slope_dir_history[idx] - calm_part_slope_dir_old

            if self.leverage_gain_mean > 0:
                self.leverage_gain_consistency = (self.leverage_slope_cons_sum + n) / (2 * n)
            elif self.leverage_gain_mean < 0:
                self.leverage_gain_consistency = (-self.leverage_slope_cons_sum + n) / (2 * n)
            else:
                self.leverage_gain_consistency = 0.5  

            if self.vol_token_gain_mean > 0:
                self.vol_token_gain_consistency = (self.vol_token_slope_cons_sum + n) / (2 * n)
            elif self.vol_token_gain_mean < 0:
                self.vol_token_gain_consistency = (-self.vol_token_slope_cons_sum + n) / (2 * n)
            else:
                self.vol_token_gain_consistency = 0.5 

            if self.calm_part_gain_mean > 0:
                self.calm_part_gain_consistency = (self.calm_part_slope_cons_sum + n) / (2 * n)
            elif self.calm_part_gain_mean < 0:
                self.calm_part_gain_consistency = (- self.calm_part_slope_cons_sum + n) / (2 * n)
            else:
                self.calm_part_gain_consistency = 0.5   

            self.sum_3_leverage = self.sum_3_leverage - leverage_old**2 + self.leverage**2
            self.sum_3_vol_token = self.sum_3_vol_token - relative_vol_token_value_old**2 + self.vol_token_normed**2
            self.sum_3_calm_part = self.sum_3_calm_part - calm_part_old**2 + self.current_calm_part**2

            self.leverage_vol = math.sqrt(max(0, self.sum_3_leverage / n - self.leverage_mean**2))
            self.vol_token_vol = math.sqrt(max(0, self.sum_3_vol_token / n - self.vol_token_mean**2))
            self.calm_part_vol = math.sqrt(max(0, self.sum_3_calm_part / n - self.calm_part_mean**2))
    
        self._update_return_statistic()


    def _reload_running_statistic(self):
        delta = -self.last_delta_leverage*self.norm
        n = min(self.step_count + 1, self.horizon)
        mu = (n + 1) / 2

        self.leverage_history = self.leverage_history + delta
        self.sum_3_leverage = self.sum_3_leverage + 2*delta*self.sum_2_leverage + n*delta**2
        self.sum_2_leverage = self.sum_2_leverage + n*delta
        self.sum_1_leverage = self.sum_1_leverage + n*mu*delta

        delta = -self.last_delta_vol_token

        self.relative_vol_token_value_history = self.relative_vol_token_value_history + delta
        self.sum_3_vol_token = self.sum_3_vol_token + 2*delta*self.sum_2_vol_token + n*delta**2
        self.sum_2_vol_token = self.sum_2_vol_token + n*delta
        self.sum_1_vol_token = self.sum_1_vol_token + n*mu*delta

        delta = -self.last_delta_eq_dist
        self.equilibrium_distance_history = self.equilibrium_distance_history + delta
        self.sum_2_equilivrium_distance = self.sum_2_equilivrium_distance + n*delta
        self.sum_1_equilivrium_distance = self.sum_1_equilivrium_distance + n*mu*delta
   
    def _normilize_returns(self):
        self.c_return_normed = math.tanh(self.c_return/0.011)
        self.p_return_normed = math.tanh(self.p_return/0.003)
        self.nav_return_normed = math.tanh(self.nav_return/0.005)
        self.l_return_normed = math.tanh(self.l_return/0.0055)

        self.c_return_abs_normed = abs(self.c_return_normed)
        self.p_return_abs_normed = abs(self.p_return_normed)
        self.nav_return_abs_normed = abs(self.nav_return_normed)
        self.l_return_abs_normed = abs(self.l_return_normed)

        self.c_return_square_normed = (self.c_return_normed)**2
        self.p_return_square_normed = (self.p_return_normed)**2
        self.nav_return_square_normed = (self.nav_return_normed)**2
        self.l_return_square_normed = (self.l_return_normed)**2

        self.c_return_sigma_normed = math.tanh(self.c_return_vol/0.007)
        self.p_return_sigma_normed = math.tanh(self.p_return_vol/0.002)
        self.nav_return_sigma_normed = math.tanh(self.nav_return_vol/0.0033)
        self.l_return_sigma_normed = math.tanh(self.l_return_vol/0.0032)

    def _normalize_slope(self, slope, scale = 2000):
        alpha = math.atan(scale/self.horizon*min(self.step_count + 1, self.horizon)*slope)
        return 2/math.pi*alpha

    def _rebalance_loop(self):
        self.primordial_leverage = self._get_primordial_leverage_init()

        self.commulative_debt_value = self.primordial_leverage*self.collateral_value
        self.commulative_lp = self.commulative_debt_value/self.current_nav
        self.commulative_debt_nect = self.commulative_debt_value/self.current_nect_honey_price

        self.target_leverage = (self.top_leverage + self.bottom_leverage)/2
        x = self.current_calm_part*(self.target_leverage - self.primordial_leverage - 1)

        self.last_delta_leverage = self.leverage_normed - 0.5
        self.last_delta_vol_token = self.vol_token_normed - x
        self.last_delta_eq_dist = 2*abs(self.leverage_normed - 0.5)

        self.vol_token_value = x*self.collateral_value
        self.vol_token = self.vol_token_value/self.current_vol_token_honey_price
        self.noncommulative_debt_nect = self.vol_token_value/self.current_nect_honey_price
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
        y2 = self.commulative_debt_nect*self.current_nect_honey_price/self.collateral_value
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




