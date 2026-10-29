from __future__ import annotations

from market import Market
from options import Option
from node import Node, TrunkNode
from .lattice import LatticeModel

import numpy as np
from datetime import datetime as dt

class BinomialModel(LatticeModel):

    def __init__(self, market:Market, option:Option, steps:int, pricing_date:dt, bs_smoothing:bool = False, 
                 pruning_threshold:float = None):
        
        super().__init__(market, option, steps, pricing_date, bs_smoothing, pruning_threshold)

        if self.market.dividends:
            raise ValueError("no dividends allowed for binomial model")

        # build the binomial tree during initialization
        self._initialize_parameters()
        self._build_trunk()
        self._build_trunk_forwards()
        self._build_tree()

    def _compute_alpha(self) -> float:
        exp_sigma_dt = (1 + np.exp((self.volatility ** 2) * self.dt)) / 2
        return exp_sigma_dt + np.sqrt((exp_sigma_dt ** 2) - 1)

    def _compute_probabilities(self) -> float:
        p_up = 1 / (self.alpha + 1)
        p_down = self.alpha / (self.alpha + 1)
        return p_up, None, p_down

    def _build_trunk_forwards(self) -> None:
        curr_trunk = self.root
        curr_trunk.forward = self.S0

        # computing forwards by iterating through the trunk 
        while curr_trunk.next_trunk is not None:
            next_trunk = curr_trunk.next_trunk
            next_trunk.forward = curr_trunk.forward * np.exp(self.rf_rate * self.dt)
            curr_trunk = next_trunk

    def get_child(self, up_count:int, children:dict, step:int, next_trunk:TrunkNode) -> dict:
        #create a node or reuse existing children for recombination
        if up_count not in children:
            child = Node(next_trunk.forward * self.alpha**(2 * up_count - step), step)
            child.up_count = up_count
            child.trunk_node = next_trunk
            children[up_count] = child
        return children[up_count]
    
    def _build_next_layer(self, trunk:TrunkNode) -> None:
        next_trunk = trunk.next_trunk
        step = next_trunk.step
        children = {}

        parent = self._get_bottom_node(trunk)
        while parent is not None:
            p_up, p_down = self.p_up, self.p_down

            # prune unlikely branches by redirecting transitions toward the trunk
            if self._should_prune(parent):
                p_up = float(2 * parent.up_count < trunk.step)
                p_down = 1.0 - p_up

            up = self.get_child(parent.up_count + 1, children, step, next_trunk) if p_up > 0 else None
            down = self.get_child(parent.up_count, children, step, next_trunk) if p_down > 0 else None
            self._set_transitions(parent, up=up, down=down, p_up=p_up, p_down=p_down)
            parent = parent.up_node

        # creates horizontal connections between nodes
        nodes = [children[k] for k in sorted(children)]
        for lower, upper in zip(nodes, nodes[1:]):
            lower.up_node = upper
            upper.down_node = lower

        next_trunk.down_node = nodes[0]
        next_trunk.up_node = nodes[-1]
        next_trunk.mid_node = min(nodes, key=lambda node: abs(node.underlying_price - next_trunk.forward))

    def print_trunk(self) -> None:
        curr = self.root

        # display trunk nodes and their connections for debugging
        while curr is not None:

            prev_trunk = (curr.prev_trunk.step if curr.prev_trunk is not None else None)
            next_trunk = (curr.next_trunk.step if curr.next_trunk is not None else None)

            up_node = (f"Node(up_count={curr.up_node.up_count}, "f"S={curr.up_node.underlying_price:.4f})"
                       if curr.up_node is not None else None)
            down_node = (f"Node(up_count={curr.down_node.up_count}, "f"S={curr.down_node.underlying_price:.4f})"
                         if curr.down_node is not None else None)
            
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

    def print_tree(self) -> None:
        nodes = [self.root_node]
        step = 0

        # display tree nodes and their connections layer by layer for debugging
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