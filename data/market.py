from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Dividend:
    date: datetime
    amount: float

@dataclass
class Market:
    S0 : float
    rf_rate : float
    volatility: float
    dividends: list[Dividend] = field(default_factory=list)

    def __post_init__(self):
        self.dividends.sort(key=lambda d: d.date)

#ajouter verification de type
