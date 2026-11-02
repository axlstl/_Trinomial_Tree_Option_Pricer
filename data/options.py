from abc import ABC, abstractmethod
from datetime import datetime as dt

class Option(ABC):

    @abstractmethod
    def payoff(self, spot:float) -> float:
        """ each option must implement its own payoff function """
        pass


class VanillaOption(Option):

    def __init__(self, strike:float, option_type:str, exercise_type:str, mty_date:dt):
        
        self.strike = strike

        self.option_type = option_type
        self.exercise_type = exercise_type

        self.mty_date = mty_date

    def payoff(self, spot:float) -> float:
        if self.option_type == "call":
            return max(spot - self.strike, 0)
        elif self.option_type == "put":
            return max(self.strike - spot, 0)

class DigitalOption(Option):

    def __init__(self, strike:float, option_type:str, exercise_type:str, mty_date:dt, coupon:float = 1):

        self.strike = strike
        self.option_type = option_type
        self.exercise_type = exercise_type
        self.mty_date = mty_date
        self.coupon = coupon

    def payoff(self, spot:float) -> float:
        if self.option_type == "call":
            return self.coupon if spot > self.strike else 0
        elif self.option_type == "put":
            return self.coupon if spot < self.strike else 0