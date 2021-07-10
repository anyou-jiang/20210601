import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import seaborn as sns
import matplotlib.pyplot as plt

prob_mtx = pd.read_csv("prob_mtx_weib_single_threshld.csv").iloc[:, 1:]
zc_bond = pd.read_csv("cir_prices.csv").iloc[:, 1:]
K = 100.  # face value
A = 0.5  # amount lost if bond is triggered
payoff = K * prob_mtx + (K * A * (1 - prob_mtx));
payoff = payoff.to_numpy()

scaler = 1000  # a scaler to V_o to avoid overflow in np.exp; after scaling, the unit of price is one thousand dollar.
cir_price = pd.read_csv("cir_prices.csv").iloc[:, 1:].to_numpy() / scaler


def ann_to_inst(r):
    """
    Converts annualized to short rate
    """
    return np.log1p(r)


def mkt_price(payoff, cir_price, risk_p, T, t):
    pi_ = [1 / payoff.shape[0]] * payoff.shape[0]
    market_price = np.zeros(payoff.shape)

    for i in range(1, payoff.shape[1] + 1):
        market_price[i - 1, :] = (
                cir_price[i - 1, :] * payoff[i - 1, :] * np.exp(ann_to_inst(risk_p) * (T - t[i - 1])) * pi_[i - 1])

    return sum(market_price.sum(0))


t_time = np.linspace(0., 2., 100)
market_price = mkt_price(payoff, cir_price, risk_p=4.35 / 100, T=2., t=t_time)
print("market price V_0 = ", market_price)

# Finding the risk neutral probabilities
T, N = cir_price.shape
v_0 = market_price

def lagr_func_scale(lam):
    Rsum = [0] * N
    alpha = [0] * N
    for i in range(1, N + 1):
        for t in range(1, T + 1):
            alpha[i - 1] += cir_price[t - 1, i - 1] * payoff[t - 1, i - 1]
        Rsum[i - 1] = np.exp(lam * (alpha[i - 1] - v_0))
    return np.sum(Rsum)

print("optimizing with T = " + str(T) + " N = " + str(N))
res = minimize_scalar(fun=lagr_func_scale, method='brent')
opt_lam = res['x']
print("opt_lam = ", opt_lam)

alpha = [0] * N
for i in range(1, N + 1):
    for t in range(1, T + 1):
        alpha[i - 1] += cir_price[t - 1, i - 1] * payoff[t - 1, i - 1]

for i in range(1, N + 1):
    print("alpha[", i, "]=", alpha[i - 1], ", alpha-v_0=", alpha[i - 1] - v_0)

lam_alpha = [0] * N
for i in range(1, N + 1):
    lam_alpha[i - 1] = opt_lam * alpha[i - 1]

max_lam_alpha = np.max(lam_alpha)

lam_alpha_normalized = [0] * N
for i in range(1, N + 1):
    lam_alpha_normalized[i - 1] = lam_alpha[i - 1] - max_lam_alpha

pi_ = [1 / N] * N  # equal probability of all N scenarios generated/simulated
Denominator = 0
for i in range(1, N + 1):
    Denominator += pi_[i - 1] * np.exp(lam_alpha_normalized[i - 1])

pi_star = [0] * N
for i in range(1, N + 1):
    pi_star[i - 1] = pi_[i - 1] * np.exp(lam_alpha_normalized[i - 1]) / Denominator

print("sum of pi_start = ", np.sum(pi_star))

v_start = 0
for i in range(1, N + 1):
    v_start += pi_star[i - 1] * alpha[i - 1]

print('V_0=', v_0, ", V_start=", v_start)

print("pi_star = ", pi_star)

rskn_pv = np.zeros((T, N))
for i in range(1, T + 1):
    rskn_pv[i - 1, :] = cir_price[i - 1, :] * payoff[i - 1, :] * pi_star[i - 1]

rskn_pv = rskn_pv.sum(0)

# physical present value
pi_ = [1 / N] * N
phy_pv = np.zeros((T, N))
for i in range(1, T + 1):
    phy_pv[i - 1, :] = cir_price[i - 1, :] * payoff[i - 1, :] * pi_[i - 1]

phy_pv = phy_pv.sum(0)

pv = pd.DataFrame({'Physical': phy_pv, 'Risk Neutral': rskn_pv})
pv.sum(0)

# expected risk premium per annum ( risk neutral - physical )
print("pv sum diff = ", pv.sum(0)[1] - pv.sum(0)[0])

sns.kdeplot(pv.iloc[:, 0], color='r')
sns.kdeplot(pv.iloc[:, 1], color='b')
plt.show()
