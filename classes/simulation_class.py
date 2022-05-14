import datetime as dt
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
    me = marketEnvironment('me_gbm', dt.datetime(2021, 1, 1))
    csr = constantShortRate('csr', 0.05)

    me.add_constant('initial_value', 36)
    me.add_constant('volatility', 0.2)
    me.add_constant('final_date', dt.datetime.today())
    me.add_constant('currency', 'EUR')
    me.add_constant('frequency', 'daily')
    me.add_constant('paths', 100)
    me.add_constant("discount_curve", csr)


    # Testing some of the get functions
    print(me.get_constant('volatility'))
    print(me.get_constant('final_date'))