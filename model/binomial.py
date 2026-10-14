from __future__ import annotations

import xml.etree.ElementTree as ET

from .lattice import LatticeModel
from node import Node
import numpy as np

class BinomialModel(LatticeModel):

    def __init__(self, market, option, steps):
        super().__init__(market, option, steps)
        if self.market.dividends:
            raise ValueError("ATTENTION  DIV INTERDIT EN BINOMIAL")

        
        self._initialize_parameters()

        self._build_trunk()
        self._build_trunk_forwards()

        self._build_tree()


    def _get_expected_value(self, node):
        return self.p_up * node.next_up.opt_price + self.p_down * node.next_down.opt_price
    def _compute_alpha(self):
        exp_sigma_dt = (1 + np.exp(self.volatility**2 * self.dt)) / 2
        return exp_sigma_dt + np.sqrt(exp_sigma_dt**2 - 1)

    def _compute_probabilities(self):
        p_up = 1 / (self.alpha + 1)
        p_down = self.alpha / (self.alpha + 1)

        return p_up, None, p_down

    def _build_trunk_forwards(self):
        curr_trunk = self.root
        curr_trunk.forward = self.S0

        while curr_trunk.next_trunk is not None:
            next_trunk = curr_trunk.next_trunk
            next_trunk.forward = curr_trunk.forward* np.exp(self.rf_rate*self.dt)
            curr_trunk = next_trunk
            
    def _initialize_root(self):
        curr_trunk = self.root

        self.root_node = Node(self.S0, 0)

        self.root.up_node = self.root_node
        self.root.down_node = self.root_node

        self.root_node.up_count = 0
        self.root_node.trunk_node = curr_trunk
 
    def _node_count(self,step):
        return step + 1

    def _node_price(self,next_trunk,nb_up,step):
        return next_trunk.forward*self.alpha**(2*nb_up-step)

    def _attach_trunk(self,next_trunk,children,step):
        middle = step // 2

        if step % 2 == 0:
            next_trunk.down_node = children[middle]
            next_trunk.up_node = children[middle]

        else:
            next_trunk.down_node = children[middle]
            next_trunk.up_node = children[middle + 1]

    def _link_next_nodes(self,parent,children,nb_up):

        parent.next_down = children[nb_up]
        parent.next_up = children[nb_up + 1]




    def print_trunk(self):
        #a modifier pour exporter dans un csv
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
        # a modifier pour exporter dans un txt
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

        # =========================================================
        # 1. Retrieve all nodes
        # =========================================================

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


        # =========================================================
        # 2. Retrieve all underlying prices
        # =========================================================

        all_spots = []

        for col in columns:
            for node in col:
                all_spots.append(node.underlying_price)

        # also include the forward prices
        trunk = self.root

        while trunk is not None:
            all_spots.append(trunk.forward)
            trunk = trunk.next_trunk


        s_min = min(all_spots)
        s_max = max(all_spots)


        # =========================================================
        # 3. Retrieve option prices
        # =========================================================

        all_option_prices = []

        for col in columns:
            for node in col:

                if node.opt_price is not None:
                    all_option_prices.append(node.opt_price)

        if all_option_prices:
            v_max = max(all_option_prices)
        else:
            v_max = 0


        # =========================================================
        # 4. SVG dimensions
        # =========================================================

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


        # =========================================================
        # 5. Convert a financial price into a vertical coordinate
        # =========================================================

        def price_to_y(price):

            if abs(s_max - s_min) < 1e-14:
                return height / 2

            relative_position = (
                (price - s_min)
                / (s_max - s_min)
            )

            # SVG y increases DOWNWARD,
            # so high financial prices must have low y coordinates
            return (
                margin_top
                + (1 - relative_position) * graph_height
            )


        # =========================================================
        # 6. Coordinates of tree nodes
        # =========================================================

        coords = {}

        for step, col in enumerate(columns):

            x = margin_left + step * x_spacing

            for node in col:

                y = price_to_y(
                    node.underlying_price
                )

                coords[node] = (x, y)


        # =========================================================
        # 7. Option value -> white/red color
        # =========================================================

    def option_price_to_color(option_price):

        if option_price is None:
            return "rgb(255,255,255)"

        if v_max <= 0:
            return "rgb(255,255,255)"

        t = option_price / v_max
        t = max(0.0, min(1.0, t))

        # 0   -> bleu
        # 0.5 -> blanc
        # 1   -> rouge

        if t <= 0.5:
            # interpolation bleu -> blanc
            u = t / 0.5   # u in [0,1]

            r = int(0   + u * (255 - 0))
            g = int(0   + u * (255 - 0))
            b = int(255 + u * (255 - 255))

        else:
            # interpolation blanc -> rouge
            u = (t - 0.5) / 0.5   # u in [0,1]

            r = 255
            g = int(255 + u * (0 - 255))
            b = int(255 + u * (0 - 255))

        return f"rgb({r},{g},{b})"


        # =========================================================
        # 8. SVG header
        # =========================================================

        # svg = []

        # svg.append(
        #     f'<svg xmlns="http://www.w3.org/2000/svg" '
        #     f'width="{width}" '
        #     f'height="{height}" '
        #     f'viewBox="0 0 {width} {height}">'
        # )

        # svg.append(
        #     '<rect width="100%" height="100%" fill="white"/>'
        # )


        # # =========================================================
        # # 9. Draw tree branches
        # # =========================================================

        # for col in columns:

        #     for node in col:

        #         x1, y1 = coords[node]

        #         if node.next_up is not None:

        #             x2, y2 = coords[node.next_up]

        #             svg.append(
        #                 f'<line '
        #                 f'x1="{x1}" y1="{y1}" '
        #                 f'x2="{x2}" y2="{y2}" '
        #                 f'stroke="#AAAAAA" '
        #                 f'stroke-width="2"/>'
        #             )

        #         if node.next_down is not None:

        #             x2, y2 = coords[node.next_down]

        #             svg.append(
        #                 f'<line '
        #                 f'x1="{x1}" y1="{y1}" '
        #                 f'x2="{x2}" y2="{y2}" '
        #                 f'stroke="#AAAAAA" '
        #                 f'stroke-width="2"/>'
        #             )


        # # =========================================================
        # # 10. Draw trunk according to REAL forward price
        # # =========================================================

        # trunk_coords = []

        # trunk = self.root

        # while trunk is not None:

        #     x = (
        #         margin_left
        #         + trunk.step * x_spacing
        #     )

        #     y = price_to_y(
        #         trunk.forward
        #     )

        #     trunk_coords.append(
        #         (x, y)
        #     )

        #     trunk = trunk.next_trunk


        # # black trunk lines
        # for i in range(
        #     len(trunk_coords) - 1
        # ):

        #     x1, y1 = trunk_coords[i]
        #     x2, y2 = trunk_coords[i + 1]

        #     svg.append(
        #         f'<line '
        #         f'x1="{x1}" y1="{y1}" '
        #         f'x2="{x2}" y2="{y2}" '
        #         f'stroke="black" '
        #         f'stroke-width="5"/>'
        #     )


        # # black trunk circles
        # for x, y in trunk_coords:

        #     svg.append(
        #         f'<circle '
        #         f'cx="{x}" cy="{y}" '
        #         f'r="8" '
        #         f'fill="black"/>'
        #     )


        # # =========================================================
        # # 11. Draw tree nodes
        # # =========================================================

        # for col in columns:

        #     for node in col:

        #         x, y = coords[node]

        #         color = option_price_to_color(
        #             node.opt_price
        #         )

        #         if node.opt_price is None:
        #             v_text = "None"
        #         else:
        #             v_text = f"{node.opt_price:.2f}"


        #         # node circle
        #         svg.append(
        #             f'<circle '
        #             f'cx="{x}" cy="{y}" '
        #             f'r="{node_radius}" '
        #             f'fill="{color}" '
        #             f'stroke="black" '
        #             f'stroke-width="1"/>'
        #         )



         

        # # =========================================================
        # # 12. Color legend : WHITE -> RED
        # # =========================================================

        # legend_x = 50
        # legend_y = height - 70

        # legend_width = 500
        # legend_height = 25

        # n_colors = 100

        # for i in range(n_colors):

        #     t = i / (n_colors - 1)

        #     fake_price = t * v_max

        #     color = option_price_to_color(
        #         fake_price
        #     )

        #     x = (
        #         legend_x
        #         + t * legend_width
        #     )

        #     svg.append(
        #         f'<rect '
        #         f'x="{x}" '
        #         f'y="{legend_y}" '
        #         f'width="{legend_width / n_colors + 1}" '
        #         f'height="{legend_height}" '
        #         f'fill="{color}"/>'
        #     )


        # svg.append(
        #     f'<text '
        #     f'x="{legend_x}" '
        #     f'y="{legend_y - 10}" '
        #     f'font-family="Arial" '
        #     f'font-size="16">'
        #     f'Option price'
        #     f'</text>'
        # )


        # svg.append(
        #     f'<text '
        #     f'x="{legend_x}" '
        #     f'y="{legend_y + 50}" '
        #     f'font-family="Arial" '
        #     f'font-size="16">'
        #     f'0'
        #     f'</text>'
        # )


        # svg.append(
        #     f'<text '
        #     f'x="{legend_x + legend_width}" '
        #     f'y="{legend_y + 50}" '
        #     f'text-anchor="end" '
        #     f'font-family="Arial" '
        #     f'font-size="16">'
        #     f'{v_max:.2f}'
        #     f'</text>'
        # )


        # # =========================================================
        # # 13. End SVG
        # # =========================================================

        # svg.append("</svg>")

        # with open(
        #     filename,
        #     "w",
        #     encoding="utf-8"
        # ) as f:

        #     f.write("\n".join(svg))

