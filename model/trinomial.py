from .lattice import LatticeModel
from node import Node

import numpy as np



class TrinomialModel(LatticeModel):

    def __init__(self, market, option, steps):
        super().__init__(market, option, steps)

        self.initialize_parameters()

        self.build_trunk()
        self.build_tree()


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
                continuation = discount * (self.p_up * node.next_up.opt_price+ self.p_mid * node.next_mid.opt_price+ self.p_down * node.next_down.opt_price)
                
                if self.option.exercise_type == "american":
                    node.opt_price = max(continuation,self.option.payoff(node.underlying_price))
                else:
                    node.opt_price = continuation
                node = node.up_node
            trunk = trunk.prev_trunk

        return self.root_node.opt_price
        



    def _compute_alpha(self):
        return np.exp(self.volatility * np.sqrt(3 * self.dt))

    def _compute_probabilities(self):
        m = 1
        q = np.exp(self.volatility**2*self.dt)

        p_down = (q-1-(self.alpha+1)*(m-1))/((1-self.alpha)*((self.alpha**-2)-1))
        p_up = (m-1-((1/self.alpha)-1)*p_down)/(self.alpha-1)
        p_mid = 1-p_up-p_down

        return p_up,p_mid,p_down

    def _initialize_root(self):
        self.root_node = Node(self.S0,0)

        self.root.up_node = self.root_node
        self.root.mid_node = self.root_node
        self.root.down_node = self.root_node

        self.root_node.up_count = 0
        self.root_node.trunk_node = self.root

    def _node_count(self,step):
        return 2*step + 1

    def _node_price(self,next_trunk,nb_up,step):
        return next_trunk.forward*self.alpha**(nb_up-step)

    def _attach_trunk(self,next_trunk,children,step):
        next_trunk.up_node = children[step]
        next_trunk.mid_node = children[step]
        next_trunk.down_node = children[step]

    def _link_next_nodes(self,parent,children,nb_up):
        parent.next_down = children[nb_up]
        parent.next_mid = children[nb_up + 1]
        parent.next_up = children[nb_up + 2]