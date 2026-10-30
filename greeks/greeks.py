from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from calibrate_h import HCalibration
from model.lattice import LatticeModel
from options import VanillaOption

class OneDimDerivative:
    def __init__(self, function):
        self.function = function


    def first(self, x, h):
        up_price = self.function(x+h)
        down_price = self.function(x-h)
        return (up_price - down_price) / (2 * h)

    def second(self, x, h):

        up_price = self.function(x+h)
        down_price = self.function(x-h)
        mid_price = self.function(x)

        return (up_price - 2 * mid_price + down_price) / (h**2)

class TwoDimDerivative:
    def __init__(self, function):
        self.function = function

    def second(self, x, y, h, k = None):
        if k is None:
            k = h
        
        up_up = self.function(x + h, y + k)
        up_down = self.function(x + h, y - k)
        down_up = self.function(x - h, y + k)
        down_down = self.function(x - h, y - k)

        return (up_up - up_down - down_up + down_down) / (4 * h * k)


class Greeks:
    def __init__(self, model: LatticeModel) -> None:
        self.model = model

        self.one_dim = OneDimDerivative(self._price)
        self.two_dim = TwoDimDerivative(self._price)
        self.h_calibration = HCalibration(self)


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



    def _price(self, dict_pricing_inputs: dict) -> float:
        """Function used to get the price variation after shifting the parameters by h or 1+h
            First, it iterates thourgh the dictionary of parameters and updates it
            Then it updates the tree and computes the new price
            The typical size of the dictionary is 1 (for OneDimDerivatives) or 2 (for TwoDimDerivatives such as vomma)
        Args:
            dict_pricing_inputs (dict): Dictionary containing the parameters to update.
                Nested dictionaries specify attributes of sub-objects.
                Other entries specify attributes of the model directly.

                ```python
                dict_pricing_inputs = {
                    "market": {"spot": 105}, #Delta or gamma
                    "market": {"volatility": 0.31}, #Vega
                    "start_date": new_date #Theta
                }
                ```

        Returns:
            float: option price after shifting the parameters' value
        """
        model_copy = deepcopy(self.model)

        
        for attribute, value in dict_pricing_inputs.items():
            if isinstance(value, dict):
                #If the value is a dictionary, it means we have to update an attribute of a sub-object
                #The only normal case in which we dont have a dictionary is for theta (start_date)
                #But we could imagine a case where we want to compute the derivative with respect to the strike or the number of steps of the model
                target = deepcopy(getattr(model_copy, attribute))

                for var_to_change, new_val in value.items():
                    setattr(target, var_to_change, new_val)

                setattr(model_copy, attribute, target)
            else:
                setattr(model_copy, attribute, value)

        model_copy.update_tree()
        return model_copy.price_from_tree()
    

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
from greeks.greeks import Greeks


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
