from collections import defaultdict

import numpy as np

from constants import TIME_BASIS
from node import Node
from .lattice import LatticeModel


class TrinomialModel(LatticeModel):
    def __init__(self, market, option, steps, pricing_date, pruning_threshold=None):
        super().__init__(market, option, steps, pricing_date)

        self.pruning_threshold = pruning_threshold

        self._initialize_parameters()
        self._map_dividends()

        self._build_trunk()
        self._build_trunk_forwards()

        self._build_tree()

    def _map_dividends(self):
        dict_div_index = defaultdict(list)
        total_days = (self.mty_date-self.start_date).days

        for dividend in self.market.dividends:
            if dividend.date > self.start_date and dividend.date <= self.mty_date:
                days = (dividend.date - self.start_date).days
                step = (days*self.steps+total_days-1)//total_days
                dict_div_index[max(1,step)].append(dividend)

        self.dividends_by_step = dict_div_index

    def _build_trunk_forwards(self):
        curr_trunk = self.root
        while curr_trunk.next_trunk is not None:
            next_trunk = curr_trunk.next_trunk
            next_trunk.forward, _ = self._compute_mean_and_variance(curr_trunk.forward, next_trunk.step)
            curr_trunk = next_trunk

    def _compute_alpha(self):
        return np.exp(self.volatility * np.sqrt(3 * self.dt))

    def _compute_probabilities(self, expected_value=1, variance=None, next_mid=1, alpha=None):

        alpha = self.alpha if alpha is None else alpha
        variance = (np.exp(self.volatility **2 * self.dt) - 1) if variance is None else variance

        m = expected_value / next_mid
        q = (variance + (expected_value + next_mid) * (expected_value - next_mid)) / next_mid**2

        p_down = (q - (alpha + 1) * (m - 1)) / ((1 - alpha) * (alpha**-2 - 1))
        p_up = (m - 1 - ((alpha ** - 1) - 1) * p_down) / (alpha - 1)
        p_mid = 1 - p_up - p_down

        return p_up, p_mid, p_down

    def _compute_mean_and_variance(self, spot, next_step):
        """Computes mean & variance for next steps :
            -This function can be called while building the trunk forwards, in which case we only need the expected value (to get the forward value)
            -While building the tree, we need both the expected value (to get mid node) and the variance (to get the spacing of up_node and down_node + their corresponding probabilities)
        """
        next_date = self.start_date + (self.mty_date - self.start_date) * next_step / self.steps
        expected_value = spot * np.exp(self.rf_rate * self.dt)
        adjusted_spot = spot #if theres no dividend, then the spot in the variance is just the spot

        for dividend in self.dividends_by_step[next_step]:
            time_spread = max(0, (next_date - dividend.date).total_seconds() / TIME_BASIS) #if dividend date = last_step_date then no spread -> leads to 0 capitalization

            capitalization_factor = np.exp(self.rf_rate * time_spread)
            expected_value -= dividend.amount * capitalization_factor
            adjusted_spot -= dividend.amount * time_spread / self.dt #in the case there are dividends, the spot is adjusted by D * (t+dt-t_div)/dt


        variance = adjusted_spot ** 2 * np.exp(2 * self.rf_rate * self.dt) * (np.exp(self.volatility**2 * self.dt)-1)
        return expected_value, variance

    def _create_node(self, node, nb_up):

        if nb_up == 1:
            next_node = node.up_node
        else:
            next_node = node.down_node

        if next_node is None:

            next_node = Node(node.underlying_price * self.alpha**nb_up, node.step)
            next_node.trunk_node = node.trunk_node
            next_node.up_count = node.up_count + nb_up

            if nb_up == 1:
                node.up_node = next_node
                next_node.down_node = node
            else:
                node.down_node = next_node
                next_node.up_node = node
        
        return next_node

    def _build_next_layer(self, curr_trunk):
        """this function is building next layers, taking into account the next dividends:
        as the cash dividends unaligns the expected values compared with the intial lattice grid,
        we link all the current nodes (parents) to the node tha tis closest to the next forward price (child)
        as this mid is not exactly equal to the expected value, we need to update the probabilities to have : sum(p_i*S_i) = E
        and sum(p_i*S_i^2) = V + E^2"""
        next_trunk = curr_trunk.next_trunk

        next_mid_node = Node(next_trunk.forward, next_trunk.step)

        next_mid_node.trunk_node=  next_trunk
        next_mid_node.up_count = 0

        next_trunk.up_node = next_trunk.mid_node = next_trunk.down_node = next_mid_node

        curr_node = self._get_bottom_node(curr_trunk)
        while curr_node is not None:
            expected_value, variance = self._compute_mean_and_variance(curr_node.underlying_price, next_trunk.step)

            #we look for the next mid_node that is closest to the expected value
            #then we build nodes until the price is in the interval between expected_value*(1+1/alpha)/2 and expected_value*(1+alpha)/2
            while expected_value < next_mid_node.underlying_price * (1 + 1 / self.alpha) / 2:
                next_mid_node = self._create_node(next_mid_node, -1)
            while expected_value > next_mid_node.underlying_price * (1 + self.alpha) / 2:
                next_mid_node = self._create_node(next_mid_node, 1)


            #if the cumulative probability is below the pruning threshold, we only keep the next mid node 
            if self.pruning_threshold is not None and curr_node.cum_proba < self.pruning_threshold:
                curr_node.next_mid = next_mid_node

                curr_node.p_up= 0
                curr_node.p_mid = 1
                curr_node.p_down =0

                next_mid_node.cum_proba += curr_node.cum_proba
                curr_node = curr_node.up_node
                continue
            #we go from base probabilities and variance values to get the updated probabilities
            width = 1 #1 variance = alpha**1
            p_up = self.p_up
            p_mid = self.p_mid
            p_down = self.p_down

            if self.dividends_by_step[next_trunk.step]:
                p_up, p_mid, p_down = self._compute_probabilities(expected_value, variance, next_mid_node.underlying_price)
                while p_mid < 0:
                    width += 1
                    p_up, p_mid, p_down = self._compute_probabilities(expected_value, variance, next_mid_node.underlying_price, self.alpha**width)

            curr_node.next_up = next_mid_node
            curr_node.next_mid = next_mid_node
            curr_node.next_down = next_mid_node

            for i in range(width):
                curr_node.next_down = self._create_node(curr_node.next_down, -1)
                curr_node.next_up = self._create_node(curr_node.next_up, 1)
            curr_node.p_up, curr_node.p_mid, curr_node.p_down = p_up, p_mid, p_down
            curr_node.next_up.cum_proba += curr_node.cum_proba * p_up
            curr_node.next_mid.cum_proba += curr_node.cum_proba * p_mid
            curr_node.next_down.cum_proba += curr_node.cum_proba * p_down
            curr_node = curr_node.up_node

    def _get_expected_value(self, node):
        #here next_up or next_down is equivalent as pruning removes both lateral branches
        if node.next_up is None:
            return node.next_mid.opt_price
        return node.p_up * node.next_up.opt_price + node.p_mid * node.next_mid.opt_price + node.p_down * node.next_down.opt_price
