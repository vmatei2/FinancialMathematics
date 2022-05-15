import numpy as np
import pandas as pd

from helpers import sn_random_numbers


class simulationClass(object):
    def __init__(self, name, mar_env, corr):
        self.name = name
        self.pricing_date = mar_env.pricing_date
        self.initial_value = mar_env.get_constant('initial_value')
        self.volatility = mar_env.get_constant('volatility')
        self.final_date = mar_env.get_constant('final_date')
        self.currency = mar_env.get_constant('currency')
        self.frequency = mar_env.get_constant('frequency')
        self.paths = mar_env.get_constant('paths')
        self.discount_curve = mar_env.get_curve('discount_curve')

        try:
            self.time_grid = mar_env.get_list('time_grid')
        except:
            self.time_grid = None
        try:
            self.special_dates = mar_env.get_list('special_dates')
        except:
            self.special_dates = []
        self.instrument_values = None
        self.correlated = corr
        if corr:
            # Only needed in the context a portfolio with several variables
            # risk factors are correlated
            self.cholesky_matrix = mar_env.get_list('cholesky_matrix')
            self.rn_set = mar_env.get_list('rn_set')[self.name]
            self.random_numbers = mar_env.get_list('random_numbers')

    def generate_time_grid(self):
        start = self.pricing_date
        end = self.final_date

        # frequenct parameter - B for business days, W - weekly, M - monthly
        time_grid = pd.date_range(start= start, end=end, freq=self.frequency).to_pydatetime()
        time_grid = list(time_grid)


        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)
        if len(self.special_dates) > 0:
            time_grid.extend(self.special_dates)
            time_grid = list(set(time_grid))
            time_grid = time_grid.sort()
        self.time_grid = np.array(time_grid)

    def get_instrument_values(self, fixed_seed=True):
        if self.instrument_values is None:
            # only initiate simulation if there are no instrument values
            self.generate_paths(fixed_seed=fixed_seed, day_count=365)
        elif fixed_seed is False:
            # also initialise resimulation when fixed seed is False
            self.genrate_paths(fixed_seed=fixed_seed, day_count=365)
        return self.instrument_values


class geometricBrownianMotion(simulationClass):
    """
    Class to generate simulated paths based on the Black-Scholes-Merton
    geometric Brownian motion model.
    """

    def __init__(self, name, mar_env, corr=False):
        super(geometricBrownianMotion, self).__init__(name, mar_env, corr)

    def update(self, initial_value=None, volatility=None, final_date=None):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365):
        if self.time_grid is None:
            # method from the generic class above
            self.generate_time_grid()
    # Number of date for the time grid
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        # Initialise first date with initial_value
        paths[0] = self.initial_value
        if not self.correlated:
            rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            rand = self.random_numbers
        short_rate = self.discount_curve.short_rate
        for t in range(1, len(self.time_grid)):
            if not self.correlated:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t-1]).days / day_count
            # difference between two dates as year fraction
            paths[t] = paths[t-1] * np.exp((short_rate - 0.5 * self.volatility ** 3) * dt + self.volatility
                                            * np.sqrt(dt) * ran)
            # generating simulated values for the respective date
        self.instrument_values = paths