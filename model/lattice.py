from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from options import Option
from node import Node, TrunkNode
from constants import TIME_BASIS
from black_scholes import BlackScholes


class LatticeModel(ABC):

    def __init__(self, market, option, steps, pricing_date, bs_smoothing = False, pruning_threshold = None):
        self.market = market
        self.option = option
        self.steps = steps

        self.start_date = pricing_date
        self.bs_smoothing = bs_smoothing
        self.pruning_threshold = pruning_threshold

        self.dt = ((self.mty_date - self.start_date).total_seconds()/ (TIME_BASIS)/ self.steps)
        self.index_dt = (self.mty_date - self.start_date) / self.steps
        self.root = TrunkNode(self.S0, 0)

        self.alpha = None
        self.p_up = None
        self.p_mid = None
        self.p_down = None

    def _initialize_parameters(self):
        self.alpha = self._compute_alpha()
        self.p_up, self.p_mid, self.p_down = self._compute_probabilities()

    def _build_trunk(self):
        curr_trunk = self.root

        if self.bs_smoothing:
            range_trunk = range(1,self.steps)
        else:
            range_trunk = range(1,self.steps+1)

        for step in range_trunk:
            next_trunk = TrunkNode(None,step)

            next_trunk.prev_trunk = curr_trunk
            curr_trunk.next_trunk = next_trunk

            curr_trunk = next_trunk

        self.last_trunk = curr_trunk

    def _build_tree(self):
        self._initialize_root()
        curr_trunk = self.root
        while curr_trunk.next_trunk is not None:
            self._build_next_layer(curr_trunk)
            curr_trunk = curr_trunk.next_trunk


    def price_from_tree(self):

        discount = np.exp(-self.rf_rate * self.dt)

        trunk = self.last_trunk
        node = self._get_bottom_node(trunk)

        if self.bs_smoothing:
            pricing_date = self.start_date + (self.steps-1)*self.index_dt
            bs = BlackScholes(self.market, self.option)

        while node is not None:

            if self.bs_smoothing:
                continuation = bs.vanilla_pricing(pricing_date, node.underlying_price)

                if self.option.exercise_type == "american":
                    node.opt_price = max(continuation, self.option.payoff(node.underlying_price))
                else:
                    node.opt_price = continuation

            else:
                node.opt_price = self.option.payoff(node.underlying_price)
            node = node.up_node

        trunk = trunk.prev_trunk
        while trunk is not None:
            node = self._get_bottom_node(trunk)
            while node is not None:
                continuation = discount * (self._get_expected_value(node))
                
                if self.option.exercise_type == "american":
                    node.opt_price = max(continuation,self.option.payoff(node.underlying_price))
                else:
                    node.opt_price = continuation
                node = node.up_node
            trunk = trunk.prev_trunk

        return self.root_node.opt_price

    def _get_bottom_node(self, trunk):
        node = trunk.down_node

        while node.down_node is not None:
            node = node.down_node

        return node

    @abstractmethod
    def _compute_alpha(self):
        pass

    @abstractmethod
    def _compute_probabilities(self):
        pass

    @abstractmethod
    def _build_trunk_forwards(self):
        pass


    def _initialize_root(self):
        self.root_node = Node(self.S0, 0)
        self.root_node.trunk_node = self.root
        self.root_node.up_count = 0

        self.root.up_node = self.root_node
        self.root.mid_node = self.root_node
        self.root.down_node = self.root_node

        self.root_node.cum_proba = 1.0

    @abstractmethod
    def _build_next_layer(self,trunk):
        pass

    def _should_prune(self, node):
        return self.pruning_threshold is not None and node.cum_proba < self.pruning_threshold

    def _set_transitions(self, node, *, up=None, mid=None, down=None,
                         p_up=0.0, p_mid=0.0, p_down=0.0):
        node.next_up, node.next_mid, node.next_down = up, mid, down
        node.p_up, node.p_mid, node.p_down = p_up, p_mid, p_down

        for child, probability in ((up, p_up), (mid, p_mid), (down, p_down)):
            if child is not None:
                child.cum_proba += node.cum_proba * probability

    def _get_expected_value(self, node):
        return sum(
            probability * child.opt_price
            for child, probability in (
                (node.next_up, node.p_up),
                (node.next_mid, node.p_mid),
                (node.next_down, node.p_down),
            )
            if child is not None
        )

    @property
    def mty_date(self):
        return self.option.mty_date
    @property
    def volatility(self):
        return self.market.volatility

    @property
    def rf_rate(self):
        return self.market.rf_rate

    @property
    def S0(self):
        return self.market.S0
