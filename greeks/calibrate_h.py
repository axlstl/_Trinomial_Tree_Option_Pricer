from __future__ import annotations

from typing import TYPE_CHECKING
from src.constants import GREEKS

if TYPE_CHECKING:
    from .greeks import Greeks

class HCalibration:
    def __init__(self, greeks: Greeks) -> None:
        self.greeks = greeks
        # Leave each step unset until calibration supplies a value.
        self.calibrated_h: dict = {name: None for name in GREEKS}
        self.calibrated_h["vanna"] = {"h": None, "dv": None}
