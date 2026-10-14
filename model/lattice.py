from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from node import Node, TrunkNode
from constants import TIME_BASIS


class LatticeModel(ABC):

    def __init__(self, market, option, steps):
        self.market = market
        self.option = option
        self.steps = steps

        self.start_date = datetime.now()
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

        for step in range(1,self.steps+1):
            next_trunk = TrunkNode(None,step)

            next_trunk.prev_trunk = curr_trunk
            curr_trunk.next_trunk = next_trunk

            curr_trunk = next_trunk

        self.last_trunk = curr_trunk

    def _build_tree(self):
        curr_trunk = self.root
        self._initialize_root()

        parents = [self.root_node]

        while curr_trunk.next_trunk is not None:
            next_trunk = curr_trunk.next_trunk
            step = next_trunk.step
            children = []
            prev_node = None

            for i in range(self._node_count(step)):
                price = self._node_price(next_trunk,i,step)

                node = Node(price,step)
                node.up_count = i
                node.trunk_node = next_trunk

                if prev_node is not None:
                    prev_node.up_node = node
                    node.down_node = prev_node

                children.append(node)
                prev_node = node

            self._attach_trunk(next_trunk,children,step)

            for i,parent in enumerate(parents):
                self._link_next_nodes(parent,children,i)

            parents = children
            curr_trunk = next_trunk


    def price_from_tree(self):

        discount = np.exp(-self.rf_rate * self.dt)

        trunk = self.last_trunk
        node = self._get_bottom_node(trunk)

        while node is not None:
            node.opt_price = self.option.payoff(node.underlying_price)
            node = node.up_node

        # backward propagation
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



    @abstractmethod
    def _initialize_root(self):
        pass

    @abstractmethod
    def _node_count(self,step):
        pass

    @abstractmethod
    def _node_price(self,trunk,nb_up,step):
        pass

    @abstractmethod
    def _attach_trunk(self,trunk,children,step):
        pass

    @abstractmethod
    def _link_next_nodes(self,parent,children,nb_up):
        pass

    @abstractmethod
    def _get_expected_value(self,node):
        pass

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