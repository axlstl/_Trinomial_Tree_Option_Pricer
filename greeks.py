# from datetime import datetime, timedelta
# from copy import deepcopy

# from market import Market, Dividend
# from model.binomial import BinomialModel
# from model.trinomial import TrinomialModel


# class Greeks:
#     def __init__(self, option, market: Market, model="trinomial", steps=1000, pricing_date=None):
#         self.option = option
#         self.market = market
#         self.model_name = model.lower()
#         self.steps = steps
#         self.pricing_date = pricing_date or datetime.today()

#         self.model = self._build_model(market, self.pricing_date)

#     def _build_model(self, market: Market, pricing_date=None):
#         pricing_date = pricing_date or self.pricing_date

#         if self.model_name == "binomial":
#             return BinomialModel(market, self.option, self.steps, pricing_date)

#         if self.model_name == "trinomial":
#             return TrinomialModel(market, self.option, self.steps, pricing_date)

#     def _clone_market(self, *, spot=None, volatility=None, rf_rate=None):
#         dividends = [Dividend(d.date, d.amount) for d in self.market.dividends]

#         return Market(
#             S0=self.market.S0 if spot is None else spot,
#             rf_rate=self.market.rf_rate if rf_rate is None else rf_rate,
#             volatility=self.market.volatility if volatility is None else volatility,
#             dividends=dividends,
#         )

#     def _price(self, *, spot=None, volatility=None, rf_rate=None, pricing_date=None):
#         market = self._clone_market(
#             spot=self.market.S0 if spot is None else spot,
#             volatility=self.market.volatility if volatility is None else volatility,
#             rf_rate=self.market.rf_rate if rf_rate is None else rf_rate,
#         )

#         model = self._build_model(market, pricing_date)
#         return model.price_from_tree()

#     def delta(self, h=None):
#         h = max(1e-5, self.market.S0 * 1e-4) if h is None else h
#         s = self.market.S0

#         spot_up = self._price(spot=s + h)
#         spot_down = self._price(spot=s - h)

#         return (spot_up - spot_down) / (2 * h)

#     def gamma(self, h=None):
#         h = max(1e-5, self.market.S0 * 1e-4) if h is None else h
#         s = self.market.S0

#         spot_up = self._price(spot=s + h)
#         spot_0 = self._price(spot=s)
#         spot_down = self._price(spot=s - h)

#         return (spot_up - 2 * spot_0 + spot_down) / (h ** 2)

#     def vega(self, dv=1e-4):
#         sigma = self.market.volatility

#         v_up = self._price(volatility=sigma + dv)
#         v_down = self._price(volatility=sigma - dv)

#         return (v_up - v_down) / (2 * dv)

#     def theta(self, dt_days=1):
#         dt = dt_days / 365

#         t_minus = self._price(pricing_date=self.pricing_date - timedelta(days=dt_days))
#         t_plus = self._price(pricing_date=self.pricing_date + timedelta(days=dt_days))

#         return (t_minus - t_plus) / (2 * dt)

#     def rho(self, dr=1e-4):
#         r = self.market.rf_rate

#         r_up = self._price(rf_rate=r + dr)
#         r_down = self._price(rf_rate=r - dr)

#         return (r_up - r_down) / (2 * dr)

#     def vanna(self, h=None, dv=1e-4):
#         h = max(1e-5, self.market.S0 * 1e-4) if h is None else h
#         sigma = self.market.volatility

#         def delta_at_vol(vol):
#             s = self.market.S0
#             v_up = self._price(spot=s + h, volatility=vol)
#             v_down = self._price(spot=s - h, volatility=vol)
#             return (v_up - v_down) / (2 * h)

#         delta_up = delta_at_vol(sigma + dv)
#         delta_down = delta_at_vol(sigma - dv)

#         return (delta_up - delta_down) / (2 * dv)

#     def volga(self, h=None, dv=1e-4):
#         h = max(1e-5, self.market.S0 * 1e-4) if h is None else h
#         sigma = self.market.volatility

#         def gamma_at_vol(vol):
#             s = self.market.S0
#             v_up = self._price(spot=s + h, volatility=vol)
#             v_0 = self._price(spot=s, volatility=vol)
#             v_down = self._price(spot=s - h, volatility=vol)
#             return (v_up - 2 * v_0 + v_down) / (h ** 2)

#         gamma_up = gamma_at_vol(sigma + dv)
#         gamma_down = gamma_at_vol(sigma - dv)

#         return (gamma_up - gamma_down) / (2 * dv)

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta


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
        h = max(1e-5, self.market.S0 * 0.0025) if h is None else h
        derivative = OneDimDerivative(lambda s: self._price(S0=s), shift=h)
        return derivative.first(self.market.S0)

    def gamma(self, h=None):
        h = max(1e-5, self.market.S0 * 0.01) if h is None else h
        derivative = OneDimDerivative(lambda s: self._price(S0=s), shift=h)
        return derivative.second(self.market.S0)

    def vega(self, dv=None):
        dv = max(1e-5, abs(self.market.volatility) * 1e-4) if dv is None else dv
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
        h = max(1e-5, self.market.S0 * 1e-4) if h is None else h
        dv = max(1e-5, abs(self.market.volatility) * 1e-4) if dv is None else dv

        def delta_for_sigma(sigma):
            derivative = OneDimDerivative(
                lambda s: self._price(S0=s, volatility=sigma),
                shift=h,
            )
            return derivative.first(self.market.S0)

        derivative = OneDimDerivative(delta_for_sigma, shift=dv)
        return derivative.first(self.market.volatility)

def volga(self, dv=0.001):
    derivative = OneDimDerivative(
        lambda sigma: self._price(volatility=sigma),
        shift=dv
    )
    return derivative.second(self.market.volatility)



from datetime import datetime

from market import Market, Dividend
from options import VanillaOption
from greeks import Greeks


if __name__ == "__main__":

    # Données du prof
    market = Market(
        S0=100.0,
        rf_rate=0.03,
        volatility=0.30,
        dividends=[
            Dividend(date=datetime(2027, 4, 21), amount=4.0)
        ]
    )

    option = VanillaOption(
        strike=100.0,
        option_type="call",
        exercise_type="american",
        mty_date=datetime(2027, 6, 28)
    )

    pricing_date = datetime(2026, 9, 1)

    greeks = Greeks(
        option=option,
        market=market,
        model="trinomial",
        steps=300,
        pricing_date=pricing_date
    )

    price = greeks._price(S0=100.0, volatility=0.30, rf_rate=0.03)
    print(f"prix = {price:.3f}")

    delta = greeks.delta()
    gamma = greeks.gamma()
