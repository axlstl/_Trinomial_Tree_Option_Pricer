from .lattice import LatticeModel
from constants import TIME_BASIS
from node import Node
from collections import defaultdict
import numpy as np



class TrinomialModel(LatticeModel):

    def __init__(self, market, option, steps):
        super().__init__(market, option, steps)

        self._initialize_parameters()

        self._map_dividends()

        self._build_trunk()
        self._build_trunk_forwards()

        self._build_tree()



    def _build_trunk_forwards(self):
        curr_trunk=self.root
        curr_trunk.forward=self.S0

        while curr_trunk.next_trunk is not None:
            next_trunk=curr_trunk.next_trunk
            next_date=self.start_date+next_trunk.step*self.index_dt

            forward=curr_trunk.forward*np.exp(self.rf_rate*self.dt)

            for dividend in self.dividends_by_step[next_trunk.step]:
                forward-=dividend.amount*np.exp(self.rf_rate*((next_date-dividend.date).total_seconds()/TIME_BASIS))

            next_trunk.forward=forward
            curr_trunk=next_trunk

    def _map_dividends(self):
        dict_div_index = defaultdict(list)

        for dividend in self.market.dividends:
            ratio = (dividend.date-self.start_date)/self.index_dt
            curr_idx = int(ratio)

            if ratio == curr_idx:
                step = curr_idx
            else:
                step = curr_idx + 1

            if 1 <= step <= self.steps:
                dict_div_index[step].append(dividend)

        self.dividends_by_step = dict_div_index



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

    def _get_expected_value(self,node):
        return self.p_up * node.next_up.opt_price+ self.p_mid * node.next_mid.opt_price+ self.p_down * node.next_down.opt_price