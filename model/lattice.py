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

    def __init__(self, market: Market, option: Option, steps: int, pricing_date: dt,
                 bs_smoothing: bool = False, pruning_threshold: float | None = None,
                 keep_tree: bool = True) -> None:
        self.market = market
        self.option = option
        self.steps = steps

        self.start_date = pricing_date
        self.bs_smoothing = bs_smoothing
        self.pruning_threshold = pruning_threshold
        self.keep_tree = keep_tree

        self.dt = (self.mty_date - self.start_date).total_seconds() / TIME_BASIS / self.steps
        self.index_dt = (self.mty_date - self.start_date) / self.steps

        # initialize the root of the tree
        self.root = TrunkNode(self.S0, 0)

        self.alpha = self.p_up = self.p_mid = self.p_down = None
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
        self.clear_tree()
        self.dt = ((self.mty_date - self.start_date).total_seconds() / TIME_BASIS) / self.steps
        self.index_dt = (self.mty_date - self.start_date) / self.steps

        self.root = TrunkNode(self.S0, 0)
        self._initialize_parameters()
        self._build_trunk()
        self._build_trunk_forwards()
        self._build_tree()

    def _clear_column(self, trunk: TrunkNode) -> None:
        node = self._get_bottom_node(trunk)
        # here we iterate through the nodes and free all the memory
        while node is not None:
            next_node = node.up_node
            node.up_node = node.down_node = None
            node.next_up = node.next_mid = node.next_down = None
            node.trunk_node = None
            node = next_node
        trunk.up_node = None
        trunk.mid_node = None
        trunk.down_node = None

    def clear_tree(self) -> None:
        #Disconnects the tree, update_tree() can be used to rebuild it
        trunk = self.root
        while trunk is not None:
            self._clear_column(trunk)
            next_trunk = trunk.next_trunk
            trunk.prev_trunk = trunk.next_trunk = None
            trunk = next_trunk
        self.root = None
        self.root_node = None
        self.last_trunk = None


    def _build_tree(self) -> None:
        self._block_size = int(np.sqrt(self.steps))
        self._initialize_root()
        curr_trunk = self.root
        while curr_trunk.next_trunk is not None:
            self._build_next_layer(curr_trunk)
            next_trunk = curr_trunk.next_trunk
            if not self.keep_tree:
                if self.option.exercise_type == "american" and curr_trunk.step % self._block_size == 0:
                    #we keep the current column in memory until the end of the block
                    self._detach_children(curr_trunk)
                else:
                    self._clear_column(curr_trunk)
            curr_trunk = next_trunk

    def _detach_children(self, trunk: TrunkNode) -> None:
        cur_bottom_node = self._get_bottom_node(trunk)
        while cur_bottom_node is not None:
            
            cur_bottom_node.next_up = None
            cur_bottom_node.next_mid = None
            cur_bottom_node.next_down = None

            cur_bottom_node = cur_bottom_node.up_node

    def price_from_tree(self) -> float:
        """Computes the option price from the model configuration
        There are 3 different cases :
        1. keep_tree = True : This is the default situation, the whole tree is buildt and kept in memory
            price_from_tree() calls _price_block() on the whole tree

        2. keep_tree = False and option.exercise_type = "european" : The tree is built layer by layer, 
                In this case, _build_tree() builds successive columns and frees parent columns once the children are fully built
                Then price_from_tree() computes the probability weighted expected value of the option (discounted to get the price)

        3. keep_tree = False and option.exercise_type = "american" : As the american option value depends on the continuation value,
                we need to have both the children & parents columns in memory to compute it. In this case, we are keeping parents in memory until
                they are no longer needed (the number of columns kept in memory is a block)

                For the pricing, price_from_tree() will call price_by_blocks(), which repeats two steps until the whole tree is priced:
                    1. _rebuild_block() first, which rebuilds the missing columns of the current block and then gets back the previous known option prices at its right boundary
                    2. _price_block() then prices backward until reaching the first column of the current block, and then free all unused columns of the block.
                Then the first columns of the block is used as the new boundary for the previous block. 
                Then it iterates until we reach the root and have the final option price

        If bs_smoothing is enabled, the last layer is priced using the BS formula.
        Returns:
            float: final option price
        """
        discount_factor = np.exp(-self.rf_rate * self.dt)
        bs = BlackScholes(self.market, self.option) if self.bs_smoothing else None
        #experimental features for memory management
        if not self.keep_tree and self.option.exercise_type == "american":
            return self._price_by_blocks(discount_factor, bs)
        if not self.keep_tree:
            expected_value = 0
            curr_node = self._get_bottom_node(self.last_trunk)
            #we iterate backward through the trunk, pricing each layer from bottom to top
            while curr_node is not None:
                continuation_value = self._get_continuation_value(curr_node, discount_factor, bs)
                expected_value += curr_node.cum_proba * continuation_value
                curr_node = curr_node.up_node
            self.root_node.opt_price = discount_factor**self.last_trunk.step * expected_value
        else:
            self._price_block(self.root, self.last_trunk, discount_factor, bs)
        return self.root_node.opt_price
    
    def _price_by_blocks(self, discount_factor: float, bs: BlackScholes | None) -> float:
        """This function computes the price for american options when keep_tree=False
        
        It starts from the last column and iterates backward through the tree following 3 steps:
            1. IT finds the previously saved column (which is now the beginning of the block)
            2. _rebuild_block() rebuilds the missing columns of the current block and then gets back the previous known option prices at its right boundary
            3. _price_block() then prices backward until reaching the first column of the current block, and then free all unused columns of the block.
        Once the root is priced, the function returns the price

        This process is longer but it reduces a lot the memory used while pricing

        Args:
            discount_factor (float): discount factor for a time step
            bs (BlackScholes | None): smoothing if needed

        Returns:
            float: option price
        """
        if self.root_node.opt_price is not None:
            return self.root_node.opt_price

        block_end_trunk = self.last_trunk
        #now we iterate through the block
        while True:
            block_start_trunk = block_end_trunk.prev_trunk
            #we start by finding the first trunk of the block
            if block_start_trunk is None:
                block_start_trunk = block_end_trunk
            # then we iterate backward until we reach the first trunk of the block 
            while block_start_trunk.step % self._block_size != 0:
                block_start_trunk = block_start_trunk.prev_trunk

            if block_start_trunk is not block_end_trunk:
                self._rebuild_block(block_start_trunk, block_end_trunk)
            self._price_block(block_start_trunk, block_end_trunk, discount_factor, bs)

            if block_start_trunk is self.root:
                break
            block_end_trunk = block_start_trunk

        return self.root_node.opt_price
        
    def _rebuild_block(self, block_start_trunk: TrunkNode, block_end_trunk: TrunkNode) -> None:
        """Rebuilds the columns of the block from its first saved column.

        1. Keeps the old nodes of the last column to preserve their known option prices.
        2. Calls _build_next_layer() successively until reaching block_end_trunk.
        3. Transfers the option prices to the rebuilt last column, in the same up_count order.
        4. Disconnects the old nodes of the last column.

        The first saved column is reused without being rebuilt.

        Args:
            block_start_trunk (TrunkNode): Trunk node of the first saved column of the block.
            block_end_trunk (TrunkNode): Trunk node of the last column of the block,
                whose nodes are replaced during reconstruction.
        """
        saved_boundary_node = self._get_bottom_node(block_end_trunk)
        curr_trunk = block_start_trunk

        while curr_trunk is not block_end_trunk:
            self._build_next_layer(curr_trunk)
            curr_trunk = curr_trunk.next_trunk

        #we iterate through the rebuilt last column and the saved last column to transfer the option prices
        rebuilt_boundary_node = self._get_bottom_node(block_end_trunk)
        while rebuilt_boundary_node is not None:
            rebuilt_boundary_node.opt_price = saved_boundary_node.opt_price

            next_saved_boundary_node = saved_boundary_node.up_node
            saved_boundary_node.up_node = saved_boundary_node.down_node = None
            saved_boundary_node.next_up = saved_boundary_node.next_mid = saved_boundary_node.next_down = None
            saved_boundary_node.trunk_node = None

            saved_boundary_node = next_saved_boundary_node
            rebuilt_boundary_node = rebuilt_boundary_node.up_node

    def _price_block(self, block_start_trunk: TrunkNode, block_end_trunk: TrunkNode,
                    discount: float, bs: BlackScholes | None) -> None:
        """Prices the option backward within the current block.

        For the last block, pricing starts from the last column.
        Otherwise, pricing starts from the previous column, as the option prices
        at block_end_trunk are already known.

        Each column is priced from bottom to top using _price_node().
        For American options, it compares immediate exercise and continuation.
        If keep_tree=False, child branches are disconnected and the following
        column is freed once it is no longer needed.

        Pricing stops after the first column of the block has been priced.

        Args:
            block_start_trunk (TrunkNode): Trunk node of the first column to price.
            block_end_trunk (TrunkNode): Trunk node of the last column of the block.
            discount (float): Discount factor for one time step.
            bs (BlackScholes | None): Black-Scholes model used for BS smoothing,
                or None if smoothing is disabled.
        """
        curr_trunk = block_end_trunk if block_end_trunk is self.last_trunk else block_end_trunk.prev_trunk

        # we iterate through the block, pricing each layer from bottom to top
        while curr_trunk is not block_start_trunk.prev_trunk:
            curr_node = self._get_bottom_node(curr_trunk)
            while curr_node is not None:
                continuation_value = self._get_continuation_value(curr_node, discount, bs)
                self._price_node(curr_node, continuation_value)
                curr_node = curr_node.up_node

            # if keep_tree=False, we free the memory of the children
            if not self.keep_tree:
                self._detach_children(curr_trunk)
                if curr_trunk.next_trunk is not None:
                    self._clear_column(curr_trunk.next_trunk)

            curr_trunk = curr_trunk.prev_trunk          

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
        #exercise_now is used for svg (to display a stroke on the node if it early exercise is optimal)
        node.exercise_now = (node.trunk_node.step < self.steps and payoff > 0 and payoff > continuation)
        node.opt_price = max(continuation, payoff)

    def _get_bottom_node(self, trunk: TrunkNode) -> Node | None:
        node = trunk.down_node
        # get the last node of the layer to price from bottom to top 
        while node is not None and node.down_node is not None:
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
        self.root_node = Node(self.S0, self.root)
        self.root_node.up_count = 0

        self.root.up_node = self.root_node
        self.root.mid_node = self.root_node
        self.root.down_node = self.root_node

        self.root_node.cum_proba = 1.0

    def _should_prune(self, node:Node) -> bool:
        #checking if the pruning threshold is reached )
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
    
