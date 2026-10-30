from abc import ABC, abstractmethod
from datetime import datetime as dt
from math import isclose

import numpy as np

from greeks.greeks import Greeks
from data.market import Market
from data.options import Option
from model.node import Node, TrunkNode
from src.constants import TIME_BASIS
from convergence.black_scholes import BlackScholes


class LatticeModel(ABC):

    def __init__(self, market:Market, option:Option, steps:int, pricing_date:dt, 
                 bs_smoothing:bool = False, pruning_threshold:float = None):
        
        self.market = market
        self.option = option
        self.steps = steps

        self.start_date = pricing_date
        self.bs_smoothing = bs_smoothing
        self.pruning_threshold = pruning_threshold

        self.dt = ((self.mty_date - self.start_date).total_seconds()/ (TIME_BASIS)/ self.steps)
        self.index_dt = (self.mty_date - self.start_date) / self.steps

        # initialize the root of the tree
        self.root = TrunkNode(self.S0, 0)

        self.alpha = None
        self.p_up = None
        self.p_mid = None
        self.p_down = None

        self.greeks = Greeks(self)

    @property
    def mty_date(self) -> dt:
        return self.option.mty_date

    @property
    def volatility(self) -> float:
        return self.market.volatility

    @property
    def rf_rate(self) -> float:
        return self.market.rf_rate

    @property
    def S0(self) -> float:
        return self.market.S0

    def _initialize_parameters(self) -> None:
        self.alpha = self._compute_alpha()
        self.p_up, self.p_mid, self.p_down = self._compute_probabilities()

    def _build_trunk(self) -> None:
        curr_trunk = self.root

        # with BS smoothing, build N-1 layers instead of N
        for step in range(1, self.steps + 1 - self.bs_smoothing):
            next_trunk = TrunkNode(None, step)
            next_trunk.prev_trunk = curr_trunk
            curr_trunk.next_trunk = next_trunk
            curr_trunk = next_trunk

        self.last_trunk = curr_trunk

    def set_greeks(self, method : str = "ClassicGreeks") -> None:
        """This function is made just in case we would like to implement other methods for computing greeks."""
        if method == "ClassicGreeks":
            self.greeks = Greeks(self)

    def update_tree(self) -> None:

        self.dt = ((self.mty_date - self.start_date).total_seconds() / TIME_BASIS) / self.steps
        self.index_dt = (self.mty_date - self.start_date) / self.steps

        self.root = TrunkNode(self.S0, 0)
        self._initialize_parameters()
        self._build_trunk()
        self._build_trunk_forwards()
        self._build_tree()

    def _build_tree(self) -> None:
        self._initialize_root()
        curr_trunk = self.root
        while curr_trunk.next_trunk is not None:
            self._build_next_layer(curr_trunk)
            curr_trunk = curr_trunk.next_trunk

    def price_from_tree(self) -> float:
        discount = np.exp(-self.rf_rate * self.dt)
        bs = BlackScholes(self.market, self.option) if self.bs_smoothing else None
        trunk = self.last_trunk
        # iterate backward through the trunk, pricing each layer from bottom to top
        while trunk is not None:
            node = self._get_bottom_node(trunk)
            while node is not None:
                continuation = self._get_continuation_value(node, discount, bs)
                self._price_node(node, continuation)
                node = node.up_node
            trunk = trunk.prev_trunk
        return self.root_node.opt_price

    def _get_continuation_value(self, node: Node, discount: float, bs: BlackScholes | None) -> float:
        # BS smoothing replaces only the last time interval.
        if node.trunk_node is not self.last_trunk:
            return discount * self._get_expected_value(node)
        if bs is None:
            return self.option.payoff(node.underlying_price)
        pricing_date = self.start_date + (self.steps - 1) * self.index_dt
        return bs.vanilla_pricing(pricing_date, node.underlying_price)

    def _price_node(self, node: Node, continuation: float) -> None:
        node.exercise_now = False
        node.opt_price = continuation
        if self.option.exercise_type != "american":
            return
        payoff = self.option.payoff(node.underlying_price)
        # Calculating continuation value for svg
        node.exercise_now = (node.step < self.steps and payoff > 0 and payoff > continuation)
        node.opt_price = max(continuation, payoff)

    def _get_bottom_node(self, trunk:TrunkNode) -> Node:
        node = trunk.down_node
        # get the last node of the layer to price from bottom to top 
        while node.down_node is not None:
            node = node.down_node
        return node

    @abstractmethod
    def _compute_alpha(self) -> float:
        pass

    @abstractmethod
    def _compute_probabilities(self) -> float:
        pass

    @abstractmethod
    def _build_trunk_forwards(self) -> None:
        pass

    @abstractmethod
    def _build_next_layer(self, trunk: TrunkNode) -> None:
        pass

    def _initialize_root(self) -> None:
        self.root_node = Node(self.S0, 0)
        self.root_node.trunk_node = self.root
        self.root_node.up_count = 0

        self.root.up_node = self.root_node
        self.root.mid_node = self.root_node
        self.root.down_node = self.root_node

        self.root_node.cum_proba = 1.0

    def _should_prune(self, node:Node) -> bool:
        # check if the pruning threshold is reached using cumulative probabilities
        return self.pruning_threshold is not None and node.cum_proba < self.pruning_threshold

    def _set_transitions(self, node:Node, *, up:float = None, mid:float = None, down:float = None,
                         p_up:float = 0.0, p_mid:float = 0.0, p_down:float = 0.0) -> None:
        node.next_up, node.next_mid, node.next_down = up, mid, down
        node.p_up, node.p_mid, node.p_down = p_up, p_mid, p_down

        for child, probability in ((up, p_up), (mid, p_mid), (down, p_down)):
            if child is not None:
                child.cum_proba += node.cum_proba * probability

    def _get_expected_value(self, node:Node) -> None:
        return sum(
            probability * child.opt_price
            for child, probability in (
                (node.next_up, node.p_up),
                (node.next_mid, node.p_mid),
                (node.next_down, node.p_down),
            )
            if child is not None
        )
    
