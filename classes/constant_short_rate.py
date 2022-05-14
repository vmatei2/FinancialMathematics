import numpy as np

def get_year_deltas(date_list, day_count=365):
    """
    Function to return a vector of floats with day deltas in year fractions
    :param date_list: collection of date time objects
    :param day_count: number of days for a yer
    :return: delta_list: year fractions
    """
    start = date_list[0]
    delta_list = [(date-start).days /day_count
                  for date in date_list]
    return np.array(delta_list)

class constantShortRate(object):
    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError("Short Rate Negative")

    def get_discount_factors(self, date_list, dtobjects=True):
        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        dflist = np.exp(self.short_rate * np.sort(-dlist))
        return np.array((date_list, dflist)).T

