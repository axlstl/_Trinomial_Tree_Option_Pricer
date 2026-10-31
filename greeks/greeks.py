from __future__ import annotations

from collections.abc import Callable
from copy import copy
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from src.constants import (
    DEFAULT_RELATIVE_GREEK_STEP,
    DEFAULT_THETA_DAYS,
    DEFAULT_VOLATILITY_STEP,
    GREEKS,
)
from .calibrate_h import HCalibration
from .derivatives import OneDimDerivative, TwoDimDerivative

if TYPE_CHECKING:
    from model.lattice import LatticeModel

PricingArg = int | float | date | datetime

class Greeks:
    def __init__(self, model: LatticeModel) -> None:
        self.model = model
        self.h_calibration = HCalibration(self)

    def _build_dict_inputs(self, name: str, *values: PricingArg) -> dict:
        dict_pricing_inputs = {}

        #Here we build a dictionary of the inputs to be changed in the model
        #Usually, there should be only one input to change, but for vanna we have two inputs to change
        #We could also imagine an nD greek
        for path, value in zip(GREEKS[name]["inputs"], values):
            keys = path.split(".")
            target = dict_pricing_inputs

            for key in keys[:-1]:
                target = target.setdefault(key, {})

            target[keys[-1]] = value

        return dict_pricing_inputs

    def _price(self, dict_pricing_inputs: dict) -> float:
        """Returns the option price after modifying the model's inputs.
        Args:
            dict_pricing_inputs (dict): Sub-object attributes or model attributes to update.
                ```python
                {"market": {"S0": 105, "volatility": 0.31}}  # Vanna
                ```
        Returns:
            float: Option price with the updated inputs.
        """
        model_copy = copy(self.model)
        # Detach the shared tree so update_tree() only clears the copy's tree.
        model_copy.root = model_copy.root_node = model_copy.last_trunk = None
        for attribute, value in dict_pricing_inputs.items():
            if isinstance(value, dict):
                target = copy(getattr(model_copy, attribute))
                for var_to_change, new_val in value.items():
                    setattr(target, var_to_change, new_val)
                setattr(model_copy, attribute, target)
            else:
                setattr(model_copy, attribute, value)
        model_copy.update_tree()
        price = model_copy.price_from_tree()
        model_copy.clear_tree()
        return price

    def _build_price_function(self, name: str) -> Callable[..., float]:
        """generalistic pricing function for any greek"""
        def price(*values: PricingArg) -> float:
            dict_pricing_inputs = self._build_dict_inputs(name, *values)
            return self._price(dict_pricing_inputs)

        return price

    def delta(self, h: float | None = None) -> float:
        if h is None:
            h = self.h_calibration.calibrated_h["delta"]
        if h is None:
            h = DEFAULT_RELATIVE_GREEK_STEP * self.model.S0

        result = OneDimDerivative(self._build_price_function("delta")).first(self.model.market.S0, h)
        return result / GREEKS["delta"]["divisor"]

    def gamma(self, h: float | None = None) -> float:
        if h is None:
            h = self.h_calibration.calibrated_h["gamma"]
        if h is None:
            h = DEFAULT_RELATIVE_GREEK_STEP * self.model.S0

        result = OneDimDerivative(self._build_price_function("gamma")).second(self.model.market.S0, h)
        return result / GREEKS["gamma"]["divisor"]

    def vega(self, h: float | None = None) -> float:
        if h is None:
            h = self.h_calibration.calibrated_h["vega"]
        if h is None:
            h = DEFAULT_VOLATILITY_STEP
        #
        result = OneDimDerivative(self._build_price_function("vega")).first(self.model.market.volatility, h)
        return result / GREEKS["vega"]["divisor"]

    def rho(self, h: float | None = None) -> float:
        if h is None:
            h = self.h_calibration.calibrated_h["rho"]
        if h is None:
            h = DEFAULT_RELATIVE_GREEK_STEP * abs(self.model.rf_rate)

        result = OneDimDerivative(self._build_price_function("rho")).first(self.model.market.rf_rate, h)
        return result / GREEKS["rho"]["divisor"]

    def theta(self, dt_days: int | None = None) -> float:
        if dt_days is None:
            dt_days = self.h_calibration.calibrated_h["theta"]
        if dt_days is None:
            dt_days = DEFAULT_THETA_DAYS
        #as we are using the time derivative the function we use has to be tweaked
        def price_after_days(days: float) -> float:
            new_date = self.model.start_date + timedelta(days=days)
            return self._price(self._build_dict_inputs("theta", new_date))

        result = OneDimDerivative(price_after_days).first(0, dt_days)
        return result / GREEKS["theta"]["divisor"]

    def vanna(self, h: float | None = None, dv: float | None = None) -> float:
        if h is None:
            h = self.h_calibration.calibrated_h["vanna"]["h"]
        if dv is None:
            dv = self.h_calibration.calibrated_h["vanna"]["dv"]
        if h is None:
            h = DEFAULT_RELATIVE_GREEK_STEP * self.model.S0
        if dv is None:
            dv = DEFAULT_VOLATILITY_STEP

        result = TwoDimDerivative(self._build_price_function("vanna")).second(
            self.model.market.S0, self.model.market.volatility, h, dv)
        return result / GREEKS["vanna"]["divisor"]

    def volga(self, h: float | None = None) -> float:
        #
        if h is None:
            h = self.h_calibration.calibrated_h["volga"]
        if h is None:
            h = DEFAULT_VOLATILITY_STEP

        result = OneDimDerivative(self._build_price_function("volga")).second(self.model.market.volatility, h)
        return result / GREEKS["volga"]["divisor"]
