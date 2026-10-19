from __future__ import annotations

class Node:

    def __init__(self, underlying_price, step):
        
        self.underlying_price = underlying_price
        self.step = step

        self.up_count = None

        self.up_node = None
        self.mid_node = None
        self.down_node = None

        self.next_up = None
        self.next_mid = None
        self.next_down = None

        self.p_up = None
        self.p_mid = None
        self.p_down = None
        
        self.trunk_node = None

        self.opt_price = None
        self.cum_proba = 0.0
class TrunkNode:

    def __init__(self, forward, step):
        self.forward = forward
        self.step = step

        self.up_node = None
        self.down_node = None

        self.prev_trunk = None
        self.next_trunk = None

