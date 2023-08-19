import math
import numpy as np
import pandas as pd
from sympy import bell
from scipy.stats import expon

def adco_exponential(mu, lmd=3, c=3):
        return 1 / mu - lmd / c

def lagrange_inversion_exponential(mu, lmd=3, c=3, iter=100):
    a1 = lmd * expon.moment(2, scale=mu) / 2
    series = (c - lmd * mu) / a1
    A_hat = []
    for n in range(2, iter+1):
        series_old = series
        a_n1 = lmd * expon.moment(n + 1, scale=mu) / (n + 1)
        a_n_hat = a_n1 / (n * a1)
        A_hat.append(a_n_hat)
        bn = 0
        for k in range(1, n):
            bn += (-1)**k * math.factorial(n + k - 1) / math.factorial(n - 1) * bell(n - 1, k, A_hat[:(n - k)])
        series += bn * (c - lmd * mu)**n / math.factorial(n) / a1**n
        if abs(series - series_old) <= 1e-4:
            break
    return series

Mu = np.arange(0.75, 0.95, 0.01)
adco_explicit = [adco_exponential(mu) for mu in Mu]
adco_approx = [lagrange_inversion_exponential(mu) for mu in Mu]

Error = np.round(np.array((np.array(adco_approx) - np.array(adco_explicit)) / adco_explicit * 100).astype(float), 4)
adco_approx = np.round(np.array(adco_approx).astype(float), 4)
adco_explicit = np.round(np.array(adco_explicit).astype(float), 4)
df = pd.DataFrame({
    "\mu": Mu, 
    "Approximate value": adco_approx, 
    "Explicit value": adco_explicit, 
    "Approximation %error": Error
    })
print(df)