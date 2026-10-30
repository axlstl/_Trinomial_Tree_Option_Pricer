from collections.abc import Callable

class OneDimDerivative:
    def __init__(self, function: Callable[[float], float]) -> None:
        self.function = function #should correspond to the _price function of the Greeks class

    def first(self, x: float, h: float) -> float:
        # central first order finite difference.
        up_price = self.function(x+h)
        down_price = self.function(x-h)
        return (up_price - down_price) / (2 * h)

    def second(self, x: float, h: float) -> float:
        # central second order finite difference
        up_price = self.function(x+h)
        down_price = self.function(x-h)
        mid_price = self.function(x)

        return (up_price - 2 * mid_price + down_price) / (h**2)

class TwoDimDerivative:
    def __init__(self, function: Callable[[float, float], float]) -> None:
        self.function = function

    def second(self, x: float, y: float, h: float, k: float | None = None) -> float:
        if k is None:
            k = h

        up_up = self.function(x + h, y + k)
        up_down = self.function(x + h, y - k)
        down_up = self.function(x - h, y + k)
        down_down = self.function(x - h, y - k)

        return (up_up - up_down - down_up + down_down) / (4 * h * k)
