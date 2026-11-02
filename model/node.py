from __future__ import annotations

class Node:
    __slots__ = (
        "underlying_price", "up_count",
        "up_node", "down_node",
        "next_up", "next_mid", "next_down",
        "p_up", "p_mid", "p_down",
        "trunk_node", "opt_price", "exercise_now", "cum_proba",
    )

    def __init__(self, underlying_price: float, trunk_node: TrunkNode) -> None:
        self.underlying_price = underlying_price
        self.up_count = None

        # horizontal connexion
        self.up_node = None
        self.down_node = None

        # vertical connexion 
        self.next_up = None
        self.next_mid = None
        self.next_down = None

        # transition propabilities
        self.p_up = None
        self.p_mid = None
        self.p_down = None
        
        self.trunk_node = trunk_node

        self.opt_price = None
        self.exercise_now = False
        self.cum_proba = 0.0

class TrunkNode:
    __slots__ = ("forward", "step","up_node", "mid_node", "down_node","prev_trunk", "next_trunk")

    def __init__(self, forward:float, step:int):
        self.forward = forward
        self.step = step

        # vertical connexion 
        self.up_node = None
        self.mid_node = None
        self.down_node = None

        # horizontal connexion
        self.prev_trunk = None
        self.next_trunk = None
