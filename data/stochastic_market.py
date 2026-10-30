from dataclasses import dataclass
from .market import Market


@dataclass
class StochasticMarket(Market):
    """Market with lognormal stochastic volatility and a continuous dividend yield."""
    vol_of_vol: float = 0.5
    correlation: float = -0.5
    dividend_yield: float = 0.0
