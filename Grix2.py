from collections.abc import Callable
from copy import deepcopy
from datetime import date, datetime, timedelta

from model.lattice import LatticeModel


# Each Greek specifies its market attribute, derivative order and unit conversion.
GREEKS = {
    "delta": ("S0", 1, 1),
    "gamma": ("S0", 2, 1),
    "vega": ("volatility", 1, 100),
    "rho": ("rf_rate", 1, 1),
    "volga": ("volatility", 2, 10000),
}


class OneDimDerivative:
    def __init__(self, function: Callable[[dict[str, float]], float]) -> None:
        self.function = function

    def first(self, attribute: str, x: float, h: float) -> float:
        # Shift only the selected attribute to calculate the first derivative.
        up = self.function({attribute: x + h})
        down = self.function({attribute: x - h})
        return (up - down) / (2 * h)

    def second(self, attribute: str, x: float, h: float) -> float:
        up = self.function({attribute: x + h})
        down = self.function({attribute: x - h})
        center = self.function({attribute: x})
        return (up - 2 * center + down) / h**2


class TwoDimDerivative:
    def __init__(self, function: Callable[[dict[str, float]], float]) -> None:
        self.function = function

    def mixed(self, attr_x: str, attr_y: str, x: float, y: float, h: float, k: float) -> float:
        # Shift both attributes to get the four prices used by the mixed derivative.
        up_up = self.function({attr_x: x + h, attr_y: y + k})
        up_down = self.function({attr_x: x + h, attr_y: y - k})
        down_up = self.function({attr_x: x - h, attr_y: y + k})
        down_down = self.function({attr_x: x - h, attr_y: y - k})
        return (up_up - up_down - down_up + down_down) / (4 * h * k)


class HCalibration:
    def __init__(self, greeks: "Greeks") -> None:
        self.greeks = greeks
        # Start with default steps; the future calibration will replace these values.
        for name, (attribute, order, divisor) in GREEKS.items():
            value = getattr(greeks.model.market, attribute)
            h = 0.01 if attribute == "volatility" else abs(value) * 0.01
            setattr(self, name, h)
        self.theta = 1
        self.vanna_h = greeks.model.market.S0 * 0.01
        self.vanna_dv = 0.01


class Greeks:
    def __init__(self, model: LatticeModel) -> None:
        self.model = model
        # Create the derivative objects once; all Greeks reuse them.
        self.one_dim = OneDimDerivative(self._price)
        self.two_dim = TwoDimDerivative(self._price)
        self.h_calibration = HCalibration(self)

    def _price(self, changes: dict[str, float] | None = None, pricing_date: date | datetime | None = None) -> float:
        market = deepcopy(self.model.market)
        
        # Change the requested attributes on the copy before building the tree.
        if changes:
            for attribute, value in changes.items():
                setattr(market, attribute, value)
        pricing_date = self.model.start_date if pricing_date is None else pricing_date
        # Reuse the original model's type and settings for each shifted price.
        model = type(self.model)(market, self.model.option, self.model.steps, pricing_date,
                                 bs_smoothing=self.model.bs_smoothing,
                                 pruning_threshold=self.model.pruning_threshold)
        return model.price_from_tree()

    def compute_greek(self, name: str, h: float | None = None) -> float:
        attribute, order, divisor = GREEKS[name]
        x = getattr(self.model.market, attribute)
        # Use the stored step unless a step is supplied for this calculation.
        h = getattr(self.h_calibration, name) if h is None else h
        if order == 1:
            result = self.one_dim.first(attribute, x, h)
        else:
            result = self.one_dim.second(attribute, x, h)
        return result / divisor

    def theta(self, dt_days: int | None = None) -> float:
        dt_days = self.h_calibration.theta if dt_days is None else dt_days
        # Move the pricing date to calculate the change in price per calendar day.
        later = self._price(pricing_date=self.model.start_date + timedelta(days=dt_days))
        earlier = self._price(pricing_date=self.model.start_date - timedelta(days=dt_days))
        return (later - earlier) / (2 * dt_days)

    def vanna(self, h: float | None = None, dv: float | None = None) -> float:
        # Vanna needs a separate step for spot and volatility.
        h = self.h_calibration.vanna_h if h is None else h
        dv = self.h_calibration.vanna_dv if dv is None else dv
        return self.two_dim.mixed("S0", "volatility", self.model.market.S0, self.model.market.volatility, h, dv) / 100
