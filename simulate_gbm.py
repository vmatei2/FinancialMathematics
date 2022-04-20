import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    sns.set_style('darkgrid')
    # drift coefficient
    mu = 0.1
    # number of steps
    n = 100
    # time in years
    T = 1
    # number of simis
    M = 100
    # initial stock price
    S0 = 100
    # volatility
    sigma = 0.3


    #  Simulating GBM paths
    # calc each time step
    dt = T/n

    # simulation using numpy arrays

    St = np.exp((mu - sigma **2/2) * dt
                + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T)

    # include array of 1's

    St = np.vstack([np.ones(M), St])

    #  mutliply through by S0 and return the cunulative product of elements along a given simluation path (axis =0)
    St = S0 * St.cumprod(axis=0)

    print(St)


    #  Consider time intervals in years

    # Definte time interval correctly
    time = np.linspace(0, T, n+1)

    # Require numpy array that is the same shape as St

    tt = np.full(shape=(M, n+1), fill_value=time).T

    plt.figure(figsize=(12,12))
    plt.plot(tt, St)
    plt.xlabel("Years")
    plt.ylabel("Stock Price")

    plt.title("Realizations of Geometric Brownian Motion \n  $dS_t = \mu S_tdt + \sigma S_tdW_t$ \n"
              "$S_0 = 100, \mu = 0.1, \sigma=0.3 $",
              fontsize=18)
    plt.show()