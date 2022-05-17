import datetime as dt
import time
import matplotlib.pyplot as plt
import seaborn as sns
from classes.generic_simulation_class import geometricBrownianMotion, jump_diffusion
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
        self.constants.update(env.constants)
        self.lists.update(env.lists)
        self.curves.update(env.curves)


def plot_simulation_results(path1, path2, legend, title=None):
    plt.figure(figsize=(12, 10))
    p1 = plt.plot(gbm.time_grid, path1[:, :10], 'b')
    p2 = plt.plot(gbm.time_grid, path2[:, :10], 'r-.')
    l1 = plt.legend([p1[0], p2[0]],
                    legend, loc=2)
    plt.gca().add_artist(l1)
    plt.xticks(rotation=30, fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title, fontsize=20)
    plt.xlabel("Month", fontsize=16)
    plt.ylabel("Price", fontsize=16)
    plt.show()

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

    legend = ['low volatility=' + str(low_volatility), 'high volatility=' + str(higher_volatility)]
    plot_simulation_results(paths_1, paths_2, legend, "Simulation with Geometric Brownian Motion process")

    me_jd = marketEnvironment('me_jd', dt.datetime(2021, 1, 1))
    me_jd.add_constant('lambda', 0.3)
    me_jd.add_constant('mu', -0.75)
    me_jd.add_constant('delta', 0.1)

    me_jd.add_environment(me_gbm)

    jd = jump_diffusion('jd', me_jd)

    paths_3 = jd.get_instrument_values()

    jd.update(lamb=0.9)

    paths_4 = jd.get_instrument_values()

    legend = ['low intensity', 'high intensity']
    plot_simulation_results(paths_3, paths_4, legend, "Simulation with jump diffusion process")