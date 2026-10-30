from __future__ import annotations

class Node:

    def __init__(self, underlying_price:float, step:int):
        
        self.underlying_price = underlying_price
        self.step = step

        self.up_count = None

        # horizontal connexion
        self.up_node = None
        self.mid_node = None
        self.down_node = None

        # vertical connexion 
        self.next_up = None
        self.next_mid = None
        self.next_down = None

        # transition propabilities
        self.p_up = None
        self.p_mid = None
        self.p_down = None
        
        self.trunk_node = None

        self.opt_price = None
        self.exercise_now = False
        self.cum_proba = 0.0

class TrunkNode:

    def __init__(self, forward:float, step:int):
        self.forward = forward
        self.step = step

        # vertical connexion 
        self.up_node = None
        self.down_node = None

        # horizontal connexion
        self.prev_trunk = None
        self.next_trunk = None

