import datetime as dt
import time
import matplotlib.pyplot as plt
import seaborn as sns
from classes.generic_simulation_class import geometricBrownianMotion
from constant_short_rate import constantShortRate

class marketEnvironment(object):
    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant):
        self.constants[key] = constant

    def get_constant(self, key):
        return self.constants[key]

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        return self.lists[key]

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        return self.curves[key]

    def add_environment(self, env):
        # overwrites existing values
        self.constants.update(env.constant)
        self.lists.update(env.lists)
        self.curves.update(env.curves)


if __name__ == '__main__':
    sns.set_style('darkgrid')

    me_gbm = marketEnvironment('me_gbm', dt.datetime(2021, 1, 1))

    me_gbm.add_constant('initial_value', 36)
    low_volatility = 0.2
    me_gbm.add_constant('volatility', low_volatility)
    me_gbm.add_constant('final_date', dt.datetime(2021, 12, 12))
    me_gbm.add_constant('currency', 'USD')
    me_gbm.add_constant('frequency', 'B')
    me_gbm.add_constant('paths', 10000)

    csr = constantShortRate('csr', 0.06)
    me_gbm.add_curve('discount_curve', csr)

    # instantiate a model simulation object to work with
    gbm = geometricBrownianMotion('gbm', me_gbm)
    gbm.generate_time_grid()

    paths_1 = gbm.get_instrument_values()

    higher_volatility = 0.7
    gbm.update(volatility=higher_volatility)
    paths_2 = gbm.get_instrument_values()

    plt.figure(figsize=(12, 10))
    p1 = plt.plot(gbm.time_grid, paths_1[:, :10], 'b')
    p2 = plt.plot(gbm.time_grid, paths_2[:, :10], 'r-.')
    l1 = plt.legend([p1[0], p2[0]],
                    ['low volatility=' +str(low_volatility), 'high volatility='+str(higher_volatility)], loc=2)
    plt.gca().add_artist(l1)
    plt.xticks(rotation=30, fontsize=15)
    plt.title("Simulation with Geometric Brownian Motion paths")
    plt.xlabel("Month", fontsize=16)
    plt.ylabel("Price", fontsize=16)
    plt.show()