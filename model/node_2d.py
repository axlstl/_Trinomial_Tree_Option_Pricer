from __future__ import annotations


class Node2D:
    # Fixed fields keep each node small when the two-dimensional tree grows.
    __slots__ = (
        "underlying_price", "step", "volatility_column", "up_count", "up_node", "down_node",
        "next_up_up", "next_up_mid", "next_up_down", "next_down_up", "next_down_mid", "next_down_down",
        "p_up_up", "p_up_mid", "p_up_down", "p_down_up", "p_down_mid", "p_down_down",
        "opt_price", "cum_proba", "exercise_now",
    )

    def __init__(self, underlying_price: float, column: VolatilityColumn, up_count: int = 0) -> None:
        self.underlying_price = underlying_price
        self.step = column.step
        self.volatility_column = column
        self.up_count = up_count
        self.up_node = self.down_node = None
        # Three spot branches for each of the two next volatility states.
        self.next_up_up = self.next_up_mid = self.next_up_down = None
        self.next_down_up = self.next_down_mid = self.next_down_down = None
        self.p_up_up = self.p_up_mid = self.p_up_down = 0.0
        self.p_down_up = self.p_down_mid = self.p_down_down = 0.0
        self.opt_price = None
        self.cum_proba = 0.0
        self.exercise_now = False

    @property
    def volatility(self) -> float:
        return self.volatility_column.volatility


class VolatilityColumn:
    # A date contains linked volatility columns; each column contains linked spot nodes.
    __slots__ = (
        "volatility", "step", "up_count", "forward", "alpha", "log_adjustment",
        "up_node", "mid_node", "down_node", "up_vol", "down_vol",
        "next_up", "next_down", "prev_trunk", "next_trunk",
    )

    def __init__(self, volatility: float, step: int, up_count: int = 0) -> None:
        self.volatility = volatility
        self.step = step
        self.up_count = up_count
        self.forward = 0.0
        self.alpha = 1.0
        self.log_adjustment = 0.0
        self.up_node = self.mid_node = self.down_node = None
        self.up_vol = self.down_vol = None
        self.next_up = self.next_down = None
        # Only the lowest volatility column carries the links between dates.
        self.prev_trunk = self.next_trunk = None
