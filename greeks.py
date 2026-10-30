from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from options import VanillaOption

class OneDimDerivative:
    def __init__(self, function, shift):
        self.function = function
        self.shift = shift

    def first(self, x):
        return (self.function(x+self.shift)-self.function(x-self.shift))/(2*self.shift)

    def second(self, x):
        return (self.function(x+self.shift)-2*self.function(x)+self.function(x-self.shift))/(self.shift**2)


class Greeks:
    def __init__(self, option, market, model="trinomial", steps=1000, pricing_date=None):
        self.option = option
        self.market = market
        self.model_name = model.lower()
        self.steps = steps
        self.pricing_date = pricing_date or datetime.today()

    def set_model(self, model):
        self.model_name = model.lower()

    def _build_model(self, market):
        if self.model_name == "binomial":
            from model.binomial import BinomialModel
            return BinomialModel(market, self.option, self.steps, self.pricing_date)

        if self.model_name == "trinomial":
            from model.trinomial import TrinomialModel
            return TrinomialModel(market, self.option, self.steps, self.pricing_date, pruning_threshold = 1e-7)

        raise ValueError(f"Unknown model '{self.model_name}'. Use 'binomial' or 'trinomial'.")

    def _price(self, S0=None, volatility=None, rf_rate=None):
        market_copy = deepcopy(self.market)

        if S0 is not None:
            market_copy.S0 = S0
        if volatility is not None:
            market_copy.volatility = volatility
        if rf_rate is not None:
            market_copy.rf_rate = rf_rate

        model = self._build_model(market_copy)
        return model.price_from_tree()

    def delta(self, h=None):
        h = max(1e-5, self.market.S0 * 0.01) if h is None else h
        derivative = OneDimDerivative(lambda s: self._price(S0=s), shift=h)
        return derivative.first(self.market.S0)

    def gamma(self, h=None):
        h = max(1e-5, self.market.S0 * 0.01) if h is None else h
        derivative = OneDimDerivative(lambda s: self._price(S0=s), shift=h)
        return derivative.second(self.market.S0)

    def vega(self, dv=None):
        dv = max(1e-5, 0.01) if dv is None else dv
        derivative = OneDimDerivative(lambda sigma: self._price(volatility=sigma), shift=dv)
        return derivative.first(self.market.volatility)/100

    def rho(self, dr=None):
        dr = max(1e-5, abs(self.market.rf_rate) * 1e-4) if dr is None else dr
        derivative = OneDimDerivative(lambda r: self._price(rf_rate=r), shift=dr)
        return derivative.first(self.market.rf_rate)

    def theta(self, dt_days=1):

        def price_with_shift(days):
            option_copy = deepcopy(self.option)
            option_copy.mty_date = self.option.mty_date + timedelta(days=days)

            market_copy = deepcopy(self.market)

            model = self._build_model(market_copy)
            model.option = option_copy
            return model.price_from_tree()

        derivative = OneDimDerivative(price_with_shift, shift=dt_days)
        return derivative.first(0) / 365.0

    def vanna(self, h=None, dv=None):
        h = max(1e-5, self.market.S0 * 0.01) if h is None else h
        dv = max(1e-5, 0.01) if dv is None else dv

        def delta_for_sigma(sigma):
            derivative = OneDimDerivative(
                lambda s: self._price(S0=s, volatility=sigma),
                shift=h,
            )
            return derivative.first(self.market.S0)

        derivative = OneDimDerivative(delta_for_sigma, shift=dv)
        return derivative.first(self.market.volatility)/100

    def volga(self, dv=0.01):
        derivative = OneDimDerivative(
            lambda sigma: self._price(volatility=sigma),
            shift=dv
        )
        return derivative.second(self.market.volatility)/10000


from datetime import datetime

from market import Market, Dividend
from options import VanillaOption
from greeks import Greeks


if __name__ == "__main__":

    market = Market(
        S0=101.15,
        rf_rate=0.038,
        volatility=0.26,
        # dividends=[
        #     Dividend(date=datetime(2027, 6, 9), amount=2.9)
        # ]
    )

    option = VanillaOption(
        strike=116.2,
        option_type="put",
        exercise_type="american",
        mty_date=datetime(2027, 8, 27)
    )

    pricing_date = datetime(2026, 10, 29)

    greeks = Greeks(
        option=option,
        market=market,
        model="trinomial",
        steps=300,
        pricing_date=pricing_date
    )

    price = greeks._price(S0=101.15, volatility=0.26, rf_rate=0.038)
    print(f"prix = {price:.3f}")

    #delta = greeks.delta()
    #print(delta)
    # gamma = greeks.gamma()
    vega = greeks.vega()
    # vanna = greeks.vanna()
    # volga = greeks.volga()
    
    # print(gamma)
    print(vega)
    # print(vanna)
    # print(volga)
