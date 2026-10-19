from abc import ABC, abstractmethod


class Option(ABC):

    @abstractmethod
    def payoff(self, spot):
        pass


class VanillaOption(Option):

    def __init__(self, strike, option_type, exercise_type,mty_date):
        
        self.strike = strike

        self.option_type = option_type
        self.exercise_type = exercise_type

        self.mty_date = mty_date

    def payoff(self, spot):
        if self.option_type == "call":
            return max(spot - self.strike, 0)
        elif self.option_type == "put":
            return max(self.strike - spot, 0)
