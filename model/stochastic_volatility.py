from datetime import date, datetime
from math import exp, expm1, floor, isfinite, log, sqrt

from data.options import Option
from data.stochastic_market import StochasticMarket
from greeks.greeks import Greeks
from src.constants import TIME_BASIS
from .node_2d import Node2D, VolatilityColumn


class StochasticVolModel:
    """Two volatility branches and three conditional spot branches, following the course's simple model.

    Reference: Fruchard, '2-dimensional lattice for American options', simple model, pp. 12-15.
    Volatility is lognormal without mean reversion. Only a continuous dividend yield is supported.
    Grids producing invalid probabilities are rejected, rather than used to price the option.
    """

    def __init__(self, market: StochasticMarket, option: Option, steps: int, pricing_date: date | datetime,
                 pruning_threshold: float | None = None, max_nodes: int = 1_000_000) -> None:
        self.market = market
        self.option = option
        self.steps = steps
        self.start_date = pricing_date
        self.pruning_threshold = pruning_threshold
        self.max_nodes = max_nodes
        # Build the tree before attaching the Greek calculator.
        self.update_tree()
        self.greeks = Greeks(self)

    @property
    def S0(self) -> float:
        return self.market.S0

    @property
    def volatility(self) -> float:
        return self.market.volatility

    @property
    def rf_rate(self) -> float:
        return self.market.rf_rate

    @property
    def mty_date(self) -> date | datetime:
        return self.option.mty_date

    def update_tree(self) -> None:
        self._check_inputs()
        self.dt = (self.mty_date - self.start_date).total_seconds() / TIME_BASIS / self.steps
        self.index_dt = (self.mty_date - self.start_date) / self.steps
        self._initialize_parameters()
        self.node_count = 0
        # Rebuild all columns and nodes so copied models do not modify the original tree.
        self._build_tree()

    def _check_inputs(self) -> None:
        values = (self.S0, self.volatility, self.rf_rate, self.market.vol_of_vol,
                  self.market.correlation, self.market.dividend_yield)
        if not all(isfinite(value) for value in values):
            raise ValueError("The stochastic market inputs must be finite.")
        if self.steps < 1 or self.mty_date <= self.start_date:
            raise ValueError("Use positive steps and a maturity after the pricing date.")
        if self.S0 <= 0 or self.volatility <= 0 or self.market.vol_of_vol < 0:
            raise ValueError("Spot and volatility must be positive; vol-of-vol must be nonnegative.")
        if abs(self.market.correlation) >= 1:
            raise ValueError("This scheme requires correlation strictly between -1 and 1.")
        if self.market.dividends:
            raise ValueError("The simple 2D model supports dividend_yield, not cash dividends.")
        if self.max_nodes < 1 or (self.pruning_threshold is not None and self.pruning_threshold < 0):
            raise ValueError("Use a positive node limit and a nonnegative pruning threshold.")

    def _initialize_parameters(self) -> None:
        # Match the first two moments of the lognormal volatility with a binomial step.
        half_variance = 0.5 * expm1(self.market.vol_of_vol**2 * self.dt)
        self.vol_step = 1 + half_variance + sqrt(half_variance * (half_variance + 2))
        if self.market.vol_of_vol == 0:
            self.p_up, self.p_down = 0.0, 1.0
        else:
            self.p_up = 1 / (self.vol_step + 1)
            self.p_down = 1 - self.p_up

    def _correlation_term(self, volatility: float, next_volatility: float) -> float:
        if self.market.vol_of_vol == 0:
            return 0.0
        brownian = 0.5 * self.market.vol_of_vol * self.dt
        brownian += log(next_volatility / volatility) / self.market.vol_of_vol
        return 0.5 * (volatility + next_volatility) * self.market.correlation * brownian

    def _log_adjustment(self, volatility: float) -> float:
        if self.market.vol_of_vol == 0:
            return 0.0
        # Align the weighted conditional means with the risk-neutral forward.
        up = self._correlation_term(volatility, volatility * self.vol_step)
        down = self._correlation_term(volatility, volatility / self.vol_step)
        largest = max(up, down)
        return largest + log(self.p_up * exp(up - largest) + self.p_down * exp(down - largest))

    def _compute_alpha(self, volatility: float) -> float:
        if self.market.vol_of_vol == 0:
            return exp(volatility * sqrt(3 * self.dt))
        variance = volatility**2 * (1 - self.market.correlation**2) * self.dt
        half = 1 + 1.5 * expm1(variance)
        return half + sqrt((half - 1) * (half + 1))

    def _create_column(self, volatility: float, step: int, up_count: int) -> VolatilityColumn:
        column = VolatilityColumn(volatility, step, up_count)
        column.alpha = self._compute_alpha(volatility)
        if column.alpha <= 1:
            raise ValueError("The spot grid spacing is too small for floating-point precision.")
        column.log_adjustment = self._log_adjustment(volatility)
        log_forward = log(self.S0) + (self.rf_rate - self.market.dividend_yield) * step * self.dt
        if self.market.vol_of_vol > 0 and step > 0:
            brownian = 0.5 * self.market.vol_of_vol * self.dt
            brownian += log(volatility / self.volatility) / self.market.vol_of_vol
            log_forward += volatility * self.market.correlation * brownian - step * column.log_adjustment
        column.forward = exp(log_forward) if step > 0 else self.S0
        # Each volatility column starts with one central spot node.
        self._check_node_limit()
        node = Node2D(column.forward, column)
        column.up_node = column.mid_node = column.down_node = node
        self.node_count += 1
        return column

    def _check_node_limit(self) -> None:
        if self.node_count >= self.max_nodes:
            raise MemoryError("The 2D tree reached max_nodes; reduce steps or use pruning.")

    def _build_tree(self) -> None:
        self.root = self._create_column(self.volatility, 0, 0)
        self.root_node = self.root.mid_node
        self.root_node.cum_proba = 1.0
        trunk = self.root
        for step in range(self.steps):
            next_trunk = self._build_volatility_layer(trunk)
            column = trunk
            while column is not None:
                self._connect_column(column)
                column = column.up_vol
            trunk = next_trunk
        self.last_trunk = trunk

    def _build_volatility_layer(self, trunk: VolatilityColumn) -> VolatilityColumn:
        # Adjacent parent columns share the same next volatility column.
        next_trunk = self._create_column(trunk.volatility / self.vol_step, trunk.step + 1, 0)
        trunk.next_trunk = next_trunk
        next_trunk.prev_trunk = trunk
        next_down = next_trunk
        column = trunk
        while column is not None:
            column.next_down = next_down
            if self.market.vol_of_vol > 0:
                next_up = self._create_column(column.volatility * self.vol_step, column.step + 1, column.up_count + 1)
                next_down.up_vol, next_up.down_vol = next_up, next_down
                column.next_up = next_up
                next_down = next_up
            column = column.up_vol
        return next_trunk

    def _connect_column(self, column: VolatilityColumn) -> None:
        node = column.down_node
        while node is not None:
            # Intermediate spot nodes with zero probability do not need outgoing branches.
            if node.cum_proba > 0:
                if column.next_up is not None:
                    self._connect_branch(node, column.next_up, "up", self.p_up)
                self._connect_branch(node, column.next_down, "down", self.p_down)
            node = node.up_node

    def _conditional_moments(self, node: Node2D, column: VolatilityColumn) -> tuple[float, float]:
        """Return the conditional mean and the variance divided by the mean squared."""
        current = node.volatility_column
        correction = self._correlation_term(current.volatility, column.volatility) - current.log_adjustment
        mean = node.underlying_price * exp((self.rf_rate - self.market.dividend_yield) * self.dt + correction)
        average_volatility = 0.5 * (current.volatility + column.volatility)
        independent_part = 1 - self.market.correlation**2 if self.market.vol_of_vol > 0 else 1
        # Return variance divided by mean squared to avoid squaring large prices.
        relative_variance = expm1(average_volatility**2 * independent_part * self.dt)
        return mean, relative_variance

    def _middle_position(self, column: VolatilityColumn, mean: float) -> tuple[int, float]:
        index = floor((log(mean) - log(column.forward)) / log(column.alpha))
        price = column.forward * column.alpha**index
        if mean > price * (1 + column.alpha) / 2:
            index += 1
            price *= column.alpha
        return index, price

    def _compute_probabilities(self, mean: float, variance: float, mid: float, alpha: float) -> tuple[float, float, float]:
        m = mean / mid
        q = variance * m**2 + (m + 1) * (m - 1)
        p_down = (q - (alpha + 1) * (m - 1)) / ((1 - alpha) * (alpha**-2 - 1))
        p_up = (m - 1 - (1 / alpha - 1) * p_down) / (alpha - 1)
        p_mid = 1 - p_up - p_down
        # Remove only rounding errors; genuinely negative probabilities invalidate the grid.
        if not all(isfinite(p) for p in (p_up, p_mid, p_down)) or min(p_up, p_mid, p_down) < -1e-12:
            raise ValueError("Invalid 2D transition probabilities; change steps, vol-of-vol or correlation.")
        p_up, p_mid, p_down = max(0.0, p_up), max(0.0, p_mid), max(0.0, p_down)
        total = p_up + p_mid + p_down
        return p_up / total, p_mid / total, p_down / total

    def _connect_branch(self, node: Node2D, column: VolatilityColumn, direction: str, probability: float) -> None:
        mean, variance = self._conditional_moments(node, column)
        index, mid_price = self._middle_position(column, mean)
        pruned = self.pruning_threshold is not None and node.cum_proba < self.pruning_threshold
        if pruned:
            p_up, p_mid, p_down = 0.0, 1.0, 0.0
        else:
            p_up, p_mid, p_down = self._compute_probabilities(mean, variance, mid_price, column.alpha)
        # Check probabilities before allocating additional spot nodes.
        mid = self._get_node(column, index)
        up = self._create_node(mid, 1) if p_up > 0 else None
        down = self._create_node(mid, -1) if p_down > 0 else None
        self._set_transitions(node, direction, up, mid, down,
                              probability * p_up, probability * p_mid, probability * p_down)

    def _get_node(self, column: VolatilityColumn, index: int) -> Node2D:
        node = column.mid_node
        while node.up_count < index:
            node = self._create_node(node, 1)
        while node.up_count > index:
            node = self._create_node(node, -1)
        return node

    def _create_node(self, node: Node2D, direction: int) -> Node2D:
        column = node.volatility_column
        next_node = node.up_node if direction == 1 else node.down_node
        if next_node is None:
            self._check_node_limit()
            next_node = Node2D(node.underlying_price * column.alpha**direction, column, node.up_count + direction)
            self.node_count += 1
            # Link the new spot node to its neighbour, as in the one-dimensional tree.
            if direction == 1:
                node.up_node, next_node.down_node, column.up_node = next_node, node, next_node
            else:
                node.down_node, next_node.up_node, column.down_node = next_node, node, next_node
        return next_node

    def _set_transitions(self, node: Node2D, direction: str, up: Node2D | None, mid: Node2D, down: Node2D | None,
                         p_up: float, p_mid: float, p_down: float) -> None:
        if direction == "up":
            node.next_up_up, node.next_up_mid, node.next_up_down = up, mid, down
            node.p_up_up, node.p_up_mid, node.p_up_down = p_up, p_mid, p_down
        else:
            node.next_down_up, node.next_down_mid, node.next_down_down = up, mid, down
            node.p_down_up, node.p_down_mid, node.p_down_down = p_up, p_mid, p_down
        # Pass the cumulative probability from this parent to its children.
        for child, probability in ((up, p_up), (mid, p_mid), (down, p_down)):
            if child is not None:
                child.cum_proba += node.cum_proba * probability

    def price_from_tree(self) -> float:
        # Start at maturity and move backwards through the linked dates.
        trunk = self.last_trunk
        while trunk is not None:
            column = trunk
            while column is not None:
                self._price_column(column)
                column = column.up_vol
            trunk = trunk.prev_trunk
        return self.root_node.opt_price

    def _price_column(self, column: VolatilityColumn) -> None:
        node = column.down_node
        while node is not None:
            payoff = self.option.payoff(node.underlying_price)
            continuation = payoff if column.step == self.steps else exp(-self.rf_rate * self.dt) * self._expected_value(node)
            self._price_node(node, payoff, continuation)
            node = node.up_node

    def _expected_value(self, node: Node2D) -> float:
        # Sum the six direct branches, including their volatility probabilities.
        return sum(probability * child.opt_price for child, probability in (
            (node.next_up_up, node.p_up_up), (node.next_up_mid, node.p_up_mid),
            (node.next_up_down, node.p_up_down), (node.next_down_up, node.p_down_up),
            (node.next_down_mid, node.p_down_mid), (node.next_down_down, node.p_down_down),
        ) if child is not None)

    def _price_node(self, node: Node2D, payoff: float, continuation: float) -> None:
        node.exercise_now = False
        node.opt_price = continuation
        if self.option.exercise_type == "american":
            node.exercise_now = node.step < self.steps and payoff > 0 and payoff > continuation
            node.opt_price = max(payoff, continuation)
