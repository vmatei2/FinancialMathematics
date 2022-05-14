import math
import numpy as np
import pandas as pd
import yfinance as yf
import pandas
import scipy.stats as st
import datetime

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


def binomial_tree_calculation(r, S0, K, t, N, up_probability, down_probability, option_type="call", volatility=None):

    binomial_tree_dict = {}
    binomial_tree_dict['0'] = [S0]

    ### Step 1: Let's calculate the prices:
    S1_u = S0 + (S0 * up_probability)
    S1_d = S0 - (S0 * down_probability)
    u = S1_u / S0
    d = S1_d / S0
    if volatility is None:
        print("First step is to calculate the up and down values: ")
        print("U = %f/%f = %f" % (S1_u, S0, u))
        print("D = %f/%f = %f" % (S1_d, S0, d))
    elif volatility is not None:
        print("Calculatung U and D with given volatility")
        u = np.exp(volatility * np.sqrt(t))
        d = np.exp(-volatility * np.sqrt(t))

    disc = np.exp(-(r)*t)
    q = (np.exp(r*t)-d)/(u-d)


    print()
    print("Now let's check the market is arbitrage free: d < e^rt < u")
    print("%f < %f < %f" % (d, np.exp(r*t), u))
    print()
    print("Now let's calculate the risk neutral measure")
    print("Qu = (e^rt-d)/(u-d)")
    print("Qd = 1 - Qu")
    print("%f - %f / (%f-%f) = %f" % (np.exp(r*t), d, u, d, q))
    print("Qd = ", 1-q)
    print()

    print("Now computing the tree branches")

    for i in range(N+1):
        if i == 0:
            binomial_tree_dict[i] = [S0]
            print("At Step %f: price is: %f" % (i, binomial_tree_dict[i][0]))

        elif i == 1:
            previous_price = binomial_tree_dict[0][0]
            binomial_tree_dict[i] = [previous_price * d,
                                        previous_price * u]
            print("At Step %f: prices are: %f, %f" % (i, binomial_tree_dict[i][0], binomial_tree_dict[i][1]))

        elif i == 2:
            previous_price_down = binomial_tree_dict[i-1][0]
            previous_price_up = binomial_tree_dict[i-1][1]
            binomial_tree_dict[i] = [previous_price_down * d, previous_price_down * u,
                                        previous_price_up * u]
            print("At Step %f: prices are: "
                  "%f, %f, %f" % (i, binomial_tree_dict[i][0], binomial_tree_dict[i][1],
                                  binomial_tree_dict[i][2]))

        elif i == 3:
            previous_price_down_down = binomial_tree_dict[i-1][0]
            previous_price_down_up = binomial_tree_dict[i-1][1]
            previous_price_up_up = binomial_tree_dict[i-1][2]

            binomial_tree_dict[i] = [previous_price_down_down * d,
                                     previous_price_down_down * u,
                                     previous_price_down_up * u,
                                     previous_price_up_up * u]

            print("At Step %f: prices are: "
                  "%f, %f, %f % f" % (i, binomial_tree_dict[i][0], binomial_tree_dict[i][1],
                                  binomial_tree_dict[i][2], binomial_tree_dict[i][3]))

    # Initialise option values at maturity
    price_at_maturity = np.array(binomial_tree_dict[N])
    # payoffs at maturity
    if option_type == "call":
        payoffs = np.maximum(price_at_maturity - K, np.zeros(N+1))
    elif option_type == "put":
        payoffs = np.maximum(K - price_at_maturity, np.zeros(N+1))


    C = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N+1, 1))

    if option_type == "call":
        C = np.maximum(C - K, np.zeros(N+1))
    elif option_type == "put":
        C = np.maximum(K-C, np.zeros(N+1))
    elif option_type == "special":
        C = np.maximum((C-30), np.zeros(N+1))**2

    #  Step backwards through tree
    price_list = []
    for i in np.arange(N, 0, -1):

        print("Pay-offs at node %f are:" % i)
        print(C[1:i+1])
        print(C[0:i])
        print()
        C = disc * (q * C[1:i+1] + (1-q) * C[0:i])
        price_list.append(C)
    print("Finaly Pay-off is: ")
    print(C)
    print()





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
    print("d_+= ", d1)
    d2 = d1 - sigma * np.sqrt(T)
    print("d_- = ", d2)

    print("N(d_+= )", st.norm.cdf(d1, 0, 1))
    print("N(d_-= ", st.norm.cdf(d2, 0, 1))

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

    parity_holds = (call_price + strike_price * math.exp(-interest_rate * time_to_maturity)) == (put_price + stock_price)
    if parity_holds:
        print("Put-Call parity value= ", (put_price+stock_price))

    return parity_holds



if __name__ == '__main__':
    binomial_tree_price_next_step(0.05, 1 / 3, 0.505, 0.988, 0, 0.988)
    binomial_tree_price_two_steps_calculation(0.05, 1 / 3, 0.505171, 13.483, 0.494829, 0, 0.988)

    t = 1/3
    N = 2
    S0 = 40
    K = 41
    r = 0.06
    volatility = 0.3
    p_up = 0.25
    p_down = 0.2

    binomial_tree_calculation(r, S0, K, t, N, p_up, p_down, "call")
    binomial_tree_calculation(r, S0, K, t, N, p_up, p_down, "special")


    ### Black Scholes Part
    S = 90
    K = 92
    r = 0.04
    sigma = 0.3
    T = 1/3
    c0 = black_scholes(r, S, K, T, sigma, "call")
    p0 = black_scholes(r, S, K, T, sigma, "put")
    print(verifiy_put_call_parity(c0, p0, S, K, T, r))


    ### stack overflpow questions
    StartDate_T = '2021-12-20'
    EndDate_T = '2022-05-14'

    df = yf.download('CSCO', start=StartDate_T, end=EndDate_T, rounding=True)
    df.sort_values(by=['Date'], inplace=True, ascending=False)

    df.reset_index(inplace=True)  # Make it no longer an Index

    df['Date'] = df['Date'].dt.strftime('%Y/%m/%d')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')

    # df['Date'] = df['Date'].str.replace('-','/')   # Tried this also - but error re str

    stop = 0


