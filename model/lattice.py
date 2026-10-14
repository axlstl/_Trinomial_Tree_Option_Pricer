from abc import ABC, abstractmethod
from datetime import datetime as dt

import numpy as np

from node import Node, TrunkNode

TIME_BASIS = 365 * 24 * 3600

class LatticeModel(ABC):

    def __init__(self, market, option, steps):
        self.market = market
        self.option = option
        self.steps = steps

        self.start_date = dt.now()
        self.dt = ((self.mty_date - self.start_date).total_seconds()/ (TIME_BASIS)/ self.steps)

        self.root = TrunkNode(self.S0, 0)

        self.alpha = None
        self.p_up = None
        self.p_mid = None
        self.p_down = None

    def initialize_parameters(self):
        self.alpha = self._compute_alpha()
        self.p_up, self.p_mid, self.p_down = self._compute_probabilities()

    def build_trunk(self):
        curr_node = self.root
        growth_factor = np.exp(self.rf_rate*self.dt)

        for step in range(self.steps):

            price = curr_node.forward*growth_factor

            next_node = TrunkNode(price,step+1)
            next_node.prev_trunk = curr_node
            curr_node.next_trunk = next_node

            curr_node = next_node

        self.last_trunk = curr_node


    def build_tree(self):
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
    def _initialize_root(self):
        pass

    @abstractmethod
    def _node_count(self,step):
        pass

    @abstractmethod
    def _node_price(self,trunk,i,step):
        pass

    @abstractmethod
    def _attach_trunk(self,trunk,children,step):
        pass

    @abstractmethod
    def _link_next_nodes(self,parent,children,nb_up):
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