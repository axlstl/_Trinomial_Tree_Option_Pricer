from __future__ import annotations

import xml.etree.ElementTree as ET

from .lattice import LatticeModel
from node import Node
import numpy as np

class BinomialModel(LatticeModel):

    def __init__(self, market, option, steps, pricing_date):
        super().__init__(market, option, steps, pricing_date)

        if self.market.dividends:
            raise ValueError("no dividends allowed for binomial model")
        self._initialize_parameters()

        self._build_trunk()
        self._build_trunk_forwards()

        self._build_tree()


    def _get_expected_value(self, node):
        return self.p_up * node.next_up.opt_price + self.p_down * node.next_down.opt_price
    
    def _compute_alpha(self):
        exp_sigma_dt = (1 + np.exp((self.volatility ** 2) * self.dt)) / 2
        return exp_sigma_dt + np.sqrt((exp_sigma_dt ** 2) - 1)

    def _compute_probabilities(self):
        p_up = 1 / (self.alpha + 1)
        p_down = self.alpha / (self.alpha + 1)

        return p_up, None, p_down

    def _build_trunk_forwards(self):
        curr_trunk = self.root
        curr_trunk.forward = self.S0

        while curr_trunk.next_trunk is not None:
            next_trunk = curr_trunk.next_trunk
            next_trunk.forward = curr_trunk.forward * np.exp(self.rf_rate * self.dt)
            curr_trunk = next_trunk
            
    def _build_next_layer(self, trunk):
        next_trunk = trunk.next_trunk

        step = next_trunk.step

        child = Node(next_trunk.forward * self.alpha**(-step), step)
        child.up_count = 0
        child.trunk_node = next_trunk

        bottom_node = child
        parent = self._get_bottom_node(trunk)

        while parent is not None:
            parent.next_down = child

            upper = Node(child.underlying_price * self.alpha**2, step)
            upper.up_count = child.up_count + 1
            upper.trunk_node = next_trunk

            child.up_node = upper

            upper.down_node = child
            parent.next_up = upper
            child = upper
            
            parent = parent.up_node

        mid_node = bottom_node
        for i in range(step // 2):
            mid_node = mid_node.up_node
        next_trunk.down_node = mid_node

        if step % 2 == 0:
            next_trunk.mid_node = mid_node
        else:
            next_trunk.mid_node = mid_node.up_node


    def print_trunk(self):

        curr = self.root

        while curr is not None:

            prev_trunk = (curr.prev_trunk.step if curr.prev_trunk is not None else None)
            next_trunk = (curr.next_trunk.step if curr.next_trunk is not None else None)

            up_node = (f"Node(up_count={curr.up_node.up_count}, "f"S={curr.up_node.underlying_price:.4f})"if curr.up_node is not None else None)
            down_node = (f"Node(up_count={curr.down_node.up_count}, "f"S={curr.down_node.underlying_price:.4f})"if curr.down_node is not None else None)
            if curr.step %2 == 0:
                print(
                    f"\nTRUNK STEP {curr.step}\n"
                    f"  Forward     = {curr.forward:.4f}\n"
                    f"  prev_trunk  = {prev_trunk}\n"
                    f"  next_trunk  = {next_trunk}\n"
                    f"  up_node     = {up_node}\n"
                    f"  down_node   = {down_node}"
                )

            curr = curr.next_trunk

    def print_tree(self):

        nodes = [self.root_node]
        step = 0

        while nodes:
            print(f"\nSTEP {step}")

            for node in nodes:
                up = node.up_node.up_count if node.up_node is not None else None
                down = node.down_node.up_count if node.down_node is not None else None
                next_up = node.next_up.up_count if node.next_up is not None else None
                next_down = node.next_down.up_count if node.next_down is not None else None

                print(
                    f"Node {node.up_count} | "
                    f"S={node.underlying_price:.4f} | "
                    f"up_node={up} | "
                    f"down_node={down} | "
                    f"next_up={next_up} | "
                    f"next_down={next_down}"
                )

            children = []

            for node in nodes:
                if node.next_down is not None and node.next_down not in children:
                    children.append(node.next_down)

                if node.next_up is not None and node.next_up not in children:
                    children.append(node.next_up)

            nodes = children
            step += 1

    def export_svg(
        self,
        filename="tree.svg",
        x_spacing=220,
        graph_height=1400,
        node_radius=6,
    ):


        def get_bottom_node(trunk):

            if trunk.step == 0:
                return self.root_node

            if hasattr(trunk, "bottom_node") and trunk.bottom_node is not None:
                return trunk.bottom_node

            node = trunk.down_node

            while node.down_node is not None:
                node = node.down_node

            return node

        columns = []

        trunk = self.root

        while trunk is not None:

            col_nodes = []

            node = get_bottom_node(trunk)

            while node is not None:
                col_nodes.append(node)
                node = node.up_node

            columns.append(col_nodes)

            trunk = trunk.next_trunk


        all_spots = []

        for col in columns:
            for node in col:
                all_spots.append(node.underlying_price)


        trunk = self.root

        while trunk is not None:
            all_spots.append(trunk.forward)
            trunk = trunk.next_trunk


        s_min = min(all_spots)
        s_max = max(all_spots)


        all_option_prices = []

        for col in columns:
            for node in col:

                if node.opt_price is not None:
                    all_option_prices.append(node.opt_price)

        if all_option_prices:
            v_max = max(all_option_prices)
        else:
            v_max = 0


        margin_left = 180
        margin_right = 180

        margin_top = 150
        margin_bottom = 150

        width = (
            margin_left
            + margin_right
            + (len(columns) - 1) * x_spacing
        )

        height = graph_height + margin_top + margin_bottom


        def price_to_y(price):

            if abs(s_max - s_min) < 1e-14:
                return height / 2

            relative_position = (
                (price - s_min)
                / (s_max - s_min)
            )


            return (
                margin_top
                + (1 - relative_position) * graph_height
            )


        coords = {}

        for step, col in enumerate(columns):

            x = margin_left + step * x_spacing

            for node in col:

                y = price_to_y(
                    node.underlying_price
                )

                coords[node] = (x, y)


    def option_price_to_color(option_price):

        if option_price is None:
            return "rgb(255,255,255)"

        if v_max <= 0:
            return "rgb(255,255,255)"

        t = option_price / v_max
        t = max(0.0, min(1.0, t))


        if t <= 0.5:

            u = t / 0.5

            r = int(0   + u * (255 - 0))
            g = int(0   + u * (255 - 0))
            b = int(255 + u * (255 - 255))

        else:

            u = (t - 0.5) / 0.5

            r = 255
            g = int(255 + u * (0 - 255))
            b = int(255 + u * (0 - 255))

        return f"rgb({r},{g},{b})"


