
from market import Market
from options import VanillaOption
from datetime import datetime as dt
from math import erfc, sqrt
import numpy as np


def norm_cdf(x):
    return 0.5 * erfc(-x / sqrt(2))


class BlackScholes:

    def __init__(self, market: Market, option: VanillaOption):
        self.market = market
        self.option = option

    def vanilla_pricing(self, pricing_date=None, spot=None):
        S = self.market.S0 if spot is None else spot
        r = self.market.rf_rate
        sigma = self.market.volatility

        K = self.option.strike

        if pricing_date:
            T = (self.option.mty_date-pricing_date).total_seconds()/(365*24*3600)
        else:
            T = (self.option.mty_date-dt.now()).total_seconds()/(365*24*3600)

        d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1-sigma*np.sqrt(T)

        if self.option.option_type == "call":
            return S*norm_cdf(d1)-K*np.exp(-r*T)*norm_cdf(d2)

        elif self.option.option_type == "put":
            return K*np.exp(-r*T)*norm_cdf(-d2)-S*norm_cdf(-d1)
