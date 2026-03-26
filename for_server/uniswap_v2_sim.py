import numpy as np
import math

class uniswap_v2_sim():
    def __init__(
        self,
        initial_token0,
        initial_token1,
        initial_total_supply,
        decimals0,
        decimals1,
        pool_event_history,
        decimals = 18,
        fee_numerator = 3,
        fee_denominator = 1000,
        token0_name = 'volatile',
        token1_name = 'stable',
        minimal_liquidity = 1000,
        starting_point = 0,
        ending_point = None
    ):
        self.initial_total_supply = int(initial_total_supply)
        self.initial_reserve0 = int(initial_token0)
        self.initial_reserve1 = int(initial_token1)
        self.total_supply = self.initial_total_supply
        self.reserve0 = self.initial_reserve0
        self.reserve1 = self.initial_reserve1
        self.decimals0 = decimals0
        self.decimals1 = decimals1
        self.decimals = decimals
        self.token0_name = token0_name
        self.token1_name = token1_name
        self.fee_numerator = fee_numerator
        self.fee_denominator = fee_denominator
        self.effective_multiplier = fee_denominator - fee_numerator
        self.minimal_liquidity = minimal_liquidity

        self.pool_event_history = pool_event_history
        self.step_counter = starting_point
        self.starting_point = starting_point
        if ending_point is not None:
            self.max_steps = ending_point
        else:
            if self.pool_event_history is not None:
                self.max_steps = len(self.pool_event_history['type'])
            else:
                self.max_steps = starting_point

        self.event_history = {
            'type': [],
            'token0': [],
            'token1': [],
            'lp_token': [],
            'price': []
        }

        self.pool_history = {
            'token0': [],
            'token1': [],
            'total_supply': [],
            'price': []
        }

        if initial_token0 != 0 and initial_token1 != 0:
            self._update_pool_history()

    def _update_pool_history(self):
        self.pool_history['token0'].append(self.reserve0)
        self.pool_history['token1'].append(self.reserve1)
        self.pool_history['total_supply'].append(self.total_supply)
        self.pool_history['price'].append(self.get_price())

    def get_amount_out(self, amount, which_token, is_pure = True):
        if amount < 0:
            return 0
        else:
            if which_token == self.token0_name:
                if is_pure:
                    amount = int(amount*10**self.decimals0)
                else:
                    amount = int(amount)

                _, amountOut = self._calculate_swap_token0(amount)

                if is_pure:
                    return amountOut/10**self.decimals1
                else:
                    return amountOut
            else:
                if which_token == self.token1_name:
                    if is_pure:
                        amount = int(amount*10**self.decimals1)
                    else:
                        amount = int(amount)

                    _, amountOut = self._calculate_swap_token1(amount)

                    if is_pure:
                        return amountOut/10**self.decimals0
                    else:
                        return amountOut
                else:
                    print("wrong token name")
                    return 0

    def get_amount_in(self, amount, which_token, is_pure = True):
        if amount < 0:
            return 0
        else:
            if which_token == self.token0_name:
                return self._calculate_amountIn0(amount, is_pure)
            else:
                if which_token == self.token1_name:
                    return self._calculate_amountIn1(amount, is_pure)
                else:
                    print("wrong token name")
                    return 0

    def _calculate_amountIn0(self, amount, is_pure = True):
        if is_pure:
            amountOut = int(amount*10**self.decimals1)
        else:
            amountOut = int(amount)

        if amountOut >= self.reserve1:
            print('AmountIn cant be bigger than pool reserve')
            return 0
        
        amountIn = int(self.reserve0*self.fee_denominator*amountOut // (self.reserve1*self.effective_multiplier - amountOut*self.effective_multiplier))

        if is_pure:
            return amountIn/10**self.decimals0 
        else:
            return amountIn
        
    def _calculate_amountIn1(self, amount, is_pure = True):
        if is_pure:
            amountOut = int(amount*10**self.decimals0)
        else:
            amountOut = int(amount)

        if amountOut >= self.reserve0:
            print('AmountIn cant be bigger than pool reserve')
            return 0

        amountIn = int(self.reserve1*self.fee_denominator*amountOut // (self.reserve0*self.effective_multiplier - amountOut*self.effective_multiplier))

        if is_pure:
            return amountIn/10**self.decimals1 
        else:
            return amountIn

    def make_swap(self, amount, which_token, is_pure = True):
        if amount < 0:
            print("Swap error: negative amount")
            return 0, 0
        else:
            if which_token == self.token0_name:
                return self._make_swap_token0(amount, is_pure)
            else:
                if which_token == self.token1_name:
                    return self._make_swap_token1(amount, is_pure)
                else:
                    print("Swap error: token name mismatch")
                    return 0, 0

    def _make_swap_token0(self, amount, is_pure = True):
        if is_pure:
            amount = int(amount*10**self.decimals0)
        else:
            amount = int(amount)

        amountIn, amountOut = self._calculate_swap_token0(amount)
        self.reserve0 += amountIn
        self.reserve1 -= amountOut

        self._update_pool_history()
        self.event_history['type'].append('swap')
        self.event_history['token0'].append(amountIn)
        self.event_history['token1'].append(0)
        self.event_history['lp_token'].append(0)
        self.event_history['price'].append(self.get_price())

        return amountIn, amountOut

    def _make_swap_token1(self, amount, is_pure = True):
        if is_pure:
            amount = int(amount*10**self.decimals1)
        else:
            amount = int(amount)

        amountIn, amountOut = self._calculate_swap_token1(amount)
        self.reserve1 += amountIn
        self.reserve0 -= amountOut

        self._update_pool_history()
        self.event_history['type'].append('swap')
        self.event_history['token0'].append(0)
        self.event_history['token1'].append(amountIn)
        self.event_history['lp_token'].append(0)
        self.event_history['price'].append(self.get_price())

        return amountIn, amountOut

    def _calculate_swap_token0(self, amount):
        amountIn = int(amount)
        amountInEffective = amountIn*self.effective_multiplier
        amountOut = int(self.reserve1*amountInEffective // (self.reserve0*self.fee_denominator + amountInEffective))

        return amountIn, amountOut

    def _calculate_swap_token1(self, amount):
        amountIn = int(amount)
        amountInEffective = amountIn*self.effective_multiplier
        amountOut = int(self.reserve0*amountInEffective // (self.reserve1*self.fee_denominator + amountInEffective))
        
        return amountIn, amountOut
    
    def get_price(self, which_token = None):
        if self.reserve0 == 0 or self.reserve1 == 0:
            return 0
        if which_token is not None:
            if which_token == self.token0_name:
                return self._get_token0_token1_price()
            else:
                if which_token == self.token1_name:
                    return self._get_token1_token0_price()
                else:
                    print("Get price error: token name mismatch")
                    return 0
        else:
            return self._get_token0_token1_price()

    def _get_token0_token1_price(self):
        pure_reserve0 = self.reserve0/10**self.decimals0
        pure_reserve1 = self.reserve1/10**self.decimals1

        return pure_reserve1/pure_reserve0
    
    def _get_token1_token0_price(self):
        pure_reserve0 = self.reserve0/10**self.decimals0
        pure_reserve1 = self.reserve1/10**self.decimals1

        return pure_reserve0/pure_reserve1

    def add_liquidity(self, amount0, amount1, is_pure = False):
        if amount0 < 0 or amount1 < 0:
            print("Adding liquidity error: negative amount")
            return 0
        else:
            if self.reserve0 == 0 and self.reserve1 == 0:
                return self._add_liquidity_to_empty_pool(amount0, amount1, is_pure)
            else:
                return self._add_liquidity(amount0, amount1, is_pure)
            
    def remove_liquidity(self, lp_token_amount, is_pure = False):
        if lp_token_amount < 0:
            print("Withdrawing liquidity error: negative LP token amount")
            return 0, 0
        else:
            return self._remove_liquidity(lp_token_amount, is_pure)

    def _add_liquidity_to_empty_pool(self, amount0, amount1, is_pure = True):
        if self.reserve0 != 0 or self.reserve1 != 0:
            print('Adding liquidity error: pool is not empty')
            return 0
        else:
            if is_pure:
                amount0 = int(amount0*10**self.decimals0)
                amount1 = int(amount1*10**self.decimals1)
            else:
                amount0 = int(amount0)
                amount1 = int(amount1)
            
            total_supply = int(math.sqrt(amount0*amount1))

            if total_supply < self.minimal_liquidity:
                print('Adding liquidity error: added liquidity is less than MINIMAL_LIQUIDITY =', self.minimal_liquidity)
                return 0
            else:
                self.reserve0 = amount0
                self.reserve1 = amount1
                self.total_supply = total_supply

                liquidity_minted = self.total_supply - self.minimal_liquidity
                self._update_pool_history()
                self.event_history['type'].append('add_liquidity')
                self.event_history['token0'].append(amount0)
                self.event_history['token1'].append(amount1)
                self.event_history['lp_token'].append(liquidity_minted)
                self.event_history['price'].append(self.get_price())

                return liquidity_minted

    def _add_liquidity(self, amount0, amount1, is_pure = True):
        if is_pure:
            amount0 = int(amount0*10**self.decimals0)
            amount1 = int(amount1*10**self.decimals1)
        else:
            amount0 = int(amount0)
            amount1 = int(amount1)

        optimal1 = amount0 * self.reserve1 // self.reserve0
        if optimal1 <= amount1:
            amount1 = optimal1
        else:
            optimal0 = amount1 * self.reserve0 // self.reserve1
            amount0 = optimal0
    
        liquidity_minted = int(min(amount0 * self.total_supply // self.reserve0, amount1 * self.total_supply // self.reserve1))
        self.reserve0 += amount0
        self.reserve1 += amount1

        self.total_supply += liquidity_minted

        self._update_pool_history()
        self.event_history['type'].append('add_liquidity')
        self.event_history['token0'].append(amount0)
        self.event_history['token1'].append(amount1)
        self.event_history['lp_token'].append(liquidity_minted)
        self.event_history['price'].append(self.get_price())

        return liquidity_minted

    def _remove_liquidity(self, lp_token_amount, is_pure = True):
        if is_pure:
            lp_token_amount = int(lp_token_amount * 10**self.decimals)
        else:
            lp_token_amount = int(lp_token_amount)
        
        if lp_token_amount > self.total_supply:
            print("Withdrawing liquidity error: removing more liquidity than in the pool")
            return 0, 0
        else:
            amount0return = int((lp_token_amount * self.reserve0) // self.total_supply)
            amount1return = int((lp_token_amount * self.reserve1) // self.total_supply)

            self.reserve0 -= amount0return
            self.reserve1 -= amount1return

            self.total_supply -= lp_token_amount

            self._update_pool_history()
            self.event_history['type'].append('withdraw_liquidity')
            self.event_history['token0'].append(0)
            self.event_history['token1'].append(0)
            self.event_history['lp_token'].append(lp_token_amount)
            self.event_history['price'].append(self.get_price())

            return amount0return, amount1return
        
    def _make_single_step(self):
        if self.pool_event_history['type'][self.step_counter] == 'swap':
            if self.pool_event_history['token0'][self.step_counter] != 0:
                amountIn = self.pool_event_history['token0'][self.step_counter]
                am_in, am_out = self.make_swap(amountIn, self.token0_name, False)
                if am_in == 0 and am_out == 0:
                    print('Step', self.step_counter)
                    return True
            else:
                amountIn = self.pool_event_history['token1'][self.step_counter]
                am_in, am_out = self.make_swap(amountIn, self.token1_name, False)
                if am_in == 0 and am_out == 0:
                    print('Step', self.step_counter)
                    return True
        else:
            if self.pool_event_history['type'][self.step_counter] == 'add_liquidity':
                amount0 = self.pool_event_history['token0'][self.step_counter]
                amount1 = self.pool_event_history['token1'][self.step_counter]
                liq_minted = self.add_liquidity(amount0, amount1, False)
                if liq_minted == 0:
                    print('Step', self.step_counter)
                    return True
            else:
                if self.pool_event_history['type'][self.step_counter] == 'withdraw_liquidity':
                    lp_token_amount = self.pool_event_history['lp_token'][self.step_counter]
                    am0_r, am1_r = self.remove_liquidity(lp_token_amount, False)
                    if am0_r == 0 and am1_r == 0:
                        print('Step', self.step_counter)
                        return True
                else:
                    print('Step', self.step_counter, '- history: type error')
                    return True
        
        self.step_counter += 1

        if self.step_counter >= self.max_steps:
            self.step_counter = self.starting_point
            return True
        
        return False
    
    def make_simulation(self, simul_to = None):
        if simul_to is None:
            simul_to = self.max_steps
        
        if simul_to > self.max_steps:
            simul_to = self.max_steps
        
        if simul_to <= self.step_counter:
            print("Given target step is behind or equal to current step")
        else:
            while self.step_counter < simul_to:
                done = self._make_single_step()
                if done:
                    break

    def make_history_simulation(self, timestamp):
        if 'timestamp' in self.pool_event_history:
            while self.step_counter < self.max_steps:
                current_timestamp = self.pool_event_history['timestamp'][self.step_counter]
                if current_timestamp > timestamp:
                    break
                done = self._make_single_step()
                if done:
                    break
        else:
            print('no timestamp data')

    def set_history_timestamp(self, timestamp):
        if 'timestamp' in self.pool_event_history:
            self.total_supply = self.initial_total_supply
            self.reserve0 = self.initial_reserve0
            self.reserve1 = self.initial_reserve1   

            self.step_counter = 0

            self.event_history = {
                'type': [],
                'token0': [],
                'token1': [],
                'lp_token': [],
                'price': []
            }

            self.pool_history = {
                'token0': [],
                'token1': [],
                'total_supply': [],
                'price': []
            }

            if self.reserve0 != 0 and self.reserve1 != 0:
                self._update_pool_history()

            self.make_history_simulation(timestamp)
            
        else:
            print('no timestamp data')