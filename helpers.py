import math
import numpy as np

import scipy.stats as st


def calculate_normal_distribution(number):
    norm_cdf = st.norm.cdf(number)
    negative_number = -number
    negative_norm_cdf = st.norm.cdf(-number)
    print("N(%f) =  %f" % (number, norm_cdf))
    print("N(%f) =  %f" % (negative_number, negative_norm_cdf))
    print()
    return norm_cdf


def binomial_tree_price_next_step(r, t, qu, payoff_up, qd, payoff_down):
    option_value = math.e ** (-r * t) * (qu * payoff_up + qd * payoff_down)
    print("Option Value is: ", option_value)
    return option_value


def binomial_tree_price_two_steps_calculation(r, t, qu, payoff_up_up, qd, payoff_down_down, payoff_up_down):
    option_value = math.exp(-2 * r * t) * (
            qu ** 2 * payoff_up_up + 2 * qu * qd * payoff_up_down + qd ** 2 * payoff_down_down)
    print("Option Value is: ", option_value)
    return option_value


def black_scholes(r, S, K, T, sigma, option_type="call"):
    """
    Calculate the Black Scholes price of an option
    :param r: interest rate
    :param S: current asset price
    :param K: strike price of option
    :param T: time-to-maturity
    :param sigma: volatility
    :param option_type: call/put
    :return:
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    print(d1)
    d2 = d1 - sigma * np.sqrt(T)
    print(d2)

    try:
        if option_type == "call":
            price = S * st.norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * st.norm.cdf(d2, 0, 1)
        elif option_type == "put":
            price = K * np.exp(-r * T) * st.norm.cdf(-d2, 0, 1) - S * st.norm.cdf(-d1, 0, 1)
        print("Price of the " + option_type + " options is " + str(price))
        return price
    except:
        print("Ensure all parameters supplied are as expected")


def verifiy_put_call_parity(call_price, put_price, stock_price, strike_price, time_to_maturity, interest_rate):
    """
    Function to verify the put call-parity on option price
    :param call_price:
    :param put_price:
    :param stock_price: current price
    :param strike_price: agreed price
    :param time_to_maturity: time until the option expires
    :param interest_rate: risk-free interest rate
    :return:
    """
    return (call_price + strike_price * math.exp(-interest_rate * time_to_maturity)) == (put_price + stock_price)



if __name__ == '__main__':
    binomial_tree_price_next_step(0.05, 1 / 3, 0.505, 0.988, 0, 0.988)
    binomial_tree_price_two_steps_calculation(0.05, 1 / 3, 0.505171, 13.483, 0.494829, 0, 0.988)


    S = 90
    K = 92
    r = 0.04
    sigma = 0.3
    T = 1/3

    c0 = black_scholes(r, S, K, T, sigma, "call")

    p0 = black_scholes(r, S, K, T, sigma, "put")

    print(verifiy_put_call_parity(c0, p0, S, K, T, r))


