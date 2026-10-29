
from market import Market
from options import VanillaOption

import numpy as np
from math import erfc, sqrt
from datetime import datetime as dt

def norm_cdf(x:float) -> float:
    return 0.5 * erfc(-x / sqrt(2))

class BlackScholes:

    def __init__(self, market:Market, option:VanillaOption):
        self.market = market
        self.option = option

    def vanilla_pricing(self, pricing_date: dt = None, spot: float = None) -> float:
        S = self.market.S0 if spot is None else spot
        K = self.option.strike
        r = self.market.rf_rate
        sigma = self.market.volatility

        # pricing_date is useful to price option for nodes N-1
        if pricing_date is None:
            pricing_date = dt.now()

        T = (self.option.mty_date-pricing_date).total_seconds()/(365*24*3600)

        d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1-sigma*np.sqrt(T)

        if self.option.option_type == "call":
            return S*norm_cdf(d1)-K*np.exp(-r*T)*norm_cdf(d2)

        elif self.option.option_type == "put":
            return K*np.exp(-r*T)*norm_cdf(-d2)-S*norm_cdf(-d1)
