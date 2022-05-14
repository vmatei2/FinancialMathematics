import numpy as np
import pandas as pd


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
        if len(self.special_dates > 0):
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