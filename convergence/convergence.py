import copy
import time
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime as dt, date, timedelta

from market import Market
from model.lattice import LatticeModel
from black_scholes import BlackScholes
from model.binomial import BinomialModel
from model.trinomial import TrinomialModel
from options import Option, VanillaOption, DigitalOption


class Convergence:
    # set colors for graphs
    COLORS = {
        "Binomial":"#2563eb",
        "Trinomial": "#7c3aed",
        "Binomial with smoothing": "#93b4f5",
        "Trinomial with smoothing": "#c4a1fa",
    }

    def __init__(self, market: Market,
                 option: VanillaOption, pricing_date:dt, pruning_threshold:float = 1e-7):

        self.market = market
        self.option = option
        self.pruning_threshold = pruning_threshold
        self.pricing_date = pricing_date

    @staticmethod
    def _methods() -> list[tuple[str, type[LatticeModel], bool]]:
        # list every configurations (model + w/o smoothing)
        return [
            ("Binomial", BinomialModel, False),
            ("Trinomial", TrinomialModel, False),
            ("Binomial with smoothing", BinomialModel, True),
            ("Trinomial with smoothing", TrinomialModel, True)
        ]
    
    def _tree_price(self, option:VanillaOption, steps:int, model:type[LatticeModel], 
                    get_times:bool = False, bs_smoothing:bool = False) -> float | tuple[float, float, float]:
        
        params = {"steps": int(steps), "pricing_date": self.pricing_date, "bs_smoothing": bs_smoothing, 
                  "pruning_threshold": self.pruning_threshold}

        start = time.perf_counter()
        tree = model(self.market, option, **params)
        pricing_start = time.perf_counter()
        tree_price = tree.price_from_tree()

        # time for pricing
        pricing_time = time.perf_counter() - pricing_start

        # time for pricing + bulding tree
        running_time = time.perf_counter() - start

        return (tree_price, pricing_time, running_time) if get_times else tree_price

    def _has_bs_reference(self) -> bool:
        # check if it is an european pricing without dividends
        return self.option.exercise_type.lower() == "european" and not getattr(self.market, "dividends", None)

    def _bs_price(self, option:VanillaOption) -> float:
        return BlackScholes(self.market, option).vanilla_pricing(self.pricing_date)

    def theoretical_gap(self, N: int, method: str = "Binomial") -> float:
        S0 = self.market.S0
        r = self.market.rf_rate
        sigma = self.market.volatility
        T = (self.option.mty_date-self.pricing_date).total_seconds()/(365*24*3600)
        dt_step = T/N
        # gap_trinomial = 3/4*gap_binomial
        factor = 0.75 if method == "Trinomial" else 1.0
        return factor*S0*np.exp(r*T)/(2*np.sqrt(2*np.pi))*sigma**2*dt_step/np.sqrt(np.exp(sigma**2*T)-1)

    def _theoretical_steps(self, error: float, model: str) -> int:
        return int(np.ceil(abs(self.theoretical_gap(1, model)) / error))

    def _pricing_error(self, steps: int, model: type[LatticeModel],
                    smoothing: bool, ref_price: float) -> float:
        price = self._tree_price(self.option, steps, model, bs_smoothing=smoothing)
        gap = abs(price - ref_price)
        return gap
    
    def _search_steps(self, n: int, target: float, model: type[LatticeModel],
                    bs_smoothing: bool, ref_price: float, max_steps: int,
                    initial_error: float) -> int:

        # pricing with defined nb of steps
        pricing_error = lambda steps: self._pricing_error(steps, model, bs_smoothing, ref_price)

        # 2 cases : error <= target or error > target
        if initial_error <= target:
            upper_steps = n
            while upper_steps > 10 and pricing_error(max(10, upper_steps // 2)) <= target:
                upper_steps = max(10, upper_steps // 2)
            lower_steps = max(10, upper_steps // 2) if upper_steps > 10 else 9
        else:
            lower_steps, upper_steps = n, min(2*n, max_steps)
            while pricing_error(upper_steps) > target:
                if upper_steps == max_steps:
                    raise ValueError("Target error not reached within max_steps")
                lower_steps, upper_steps = upper_steps, min(2*upper_steps, max_steps)

        while upper_steps - lower_steps > max(10, 0.01 * upper_steps):
            # use a binary search approach 
            mid_steps = (lower_steps + upper_steps) // 2
            if pricing_error(mid_steps) <= target:
                upper_steps = mid_steps
            else:
                lower_steps = mid_steps

        return upper_steps

    def get_nb_steps(self, error: float, model_name: str, bs_smoothing: bool,
                    max_steps: int = 10000) -> int:

        if not isinstance(self.option, VanillaOption) or not self._has_bs_reference():
            raise ValueError("Requires a European vanilla option without dividends")

        model_name = model_name.capitalize()
        model_label = f"{model_name} with smoothing" if bs_smoothing else model_name
        model = next((cls for name, cls, smooth in self._methods()
                            if name == model_label and smooth == bs_smoothing))

        ref_price = self._bs_price(self.option)
        n = min(max(10, self._theoretical_steps(error, model_name)), max_steps)
        # gap between Black-Scholes price and tree price with theoretical nb of steps
        gap = self._pricing_error(n, model, bs_smoothing, ref_price)

        # case where theoretical nb_steps is quite accurate
        if 0.95 * error <= gap <= error:
            return n

        return self._search_steps(n, error, model, bs_smoothing, ref_price, max_steps, gap)

    def pricing_by_steps(self, steps_range: list[int]) -> pd.DataFrame:
        # ref price
        bs_price = self._bs_price(self.option)
        steps_range = sorted({int(n) for n in steps_range if int(n) >= 1})

        rows = []
        for step in steps_range:
            # iterating on each model type + w/o smoothing
            for method, model, smoothing in self._methods():
                tree_price, pricing_time, running_time = self._tree_price(
                    self.option, step, model, get_times=True, bs_smoothing=smoothing
                )
                gap = tree_price - bs_price
                rows.append({
                    "N": step, "Method": method, "BS_smoothing": smoothing,
                    "BS_price": bs_price, "Tree_price": tree_price,
                    "Gap_abs": abs(gap), "Gap": gap,
                    "Theoretical_gap": self.theoretical_gap(step, method) 
                        if method in ("Binomial", "Trinomial") and self._has_bs_reference() else np.nan,
                    "Running_time": running_time, "Pricing_time": pricing_time,
                    "Tunnel": gap * step,
                })
        return pd.DataFrame(rows)

    def pricing_by_strike(self, strike_range:list[int], steps:int = 10) ->  pd.DataFrame:
        strikes = np.unique(np.asarray(strike_range, dtype=float))

        rows = []
        for strike in strikes:
            option = copy.copy(self.option)
            option.strike = float(strike)

            # ref price
            bs_price = self._bs_price(option)

            # iterating on each model type + w/o smoothing
            for method, model, smoothing in self._methods():
                tree_price = self._tree_price(option, steps, model, bs_smoothing=smoothing)
                rows.append({
                    "K": strike, "Method": method, "BS_smoothing": smoothing,
                    "BS": bs_price, "Tree": tree_price, "Gap": tree_price-bs_price,
                })

        prices = pd.DataFrame(rows)
        for method, data in prices.groupby("Method", sort=False):
            indices = data.index
            prices.loc[indices, "Slope_BS"] = np.gradient(data["BS"].to_numpy(), data["K"].to_numpy())
            prices.loc[indices, "Slope_Tree"] = np.gradient(data["Tree"].to_numpy(), data["K"].to_numpy())
        return prices

    def plot_steps(self, prices: pd.DataFrame) -> None:
        """Creates 2 plots: running time and pricing gap"""
        fig_time = go.Figure()
        fig_error = go.Figure()

        for method, data in prices.groupby("Method", sort=False):
            data = data.sort_values("N")
            # graph of the running time for each nb of step
            fig_time.add_trace(go.Scatter(
                x=data["N"], y=data["Running_time"], mode="lines+markers", name=method,
                line=dict(color=self.COLORS[method]),
            ))
            # graph of the gaps for each step
            fig_error.add_trace(go.Scatter(
                x=data["N"], y=data["Gap_abs"], mode="lines+markers", name=method,
                line=dict(color=self.COLORS[method]),
            ))

        # adding theoretical gap
        theory = prices[prices["Theoretical_gap"].notna()]
        for method, data in theory.groupby("Method", sort=False):
            data = data.sort_values("N")
            fig_error.add_trace(go.Scatter(
                x=data["N"], y=data["Theoretical_gap"], mode="lines",
                name=f"{method} theoretical gap",
                line=dict(color=self.COLORS[method], dash="dash"),
            ))

        fig_time.update_layout(title="Running time by number of steps", xaxis_title="Number of steps N", yaxis_title="Seconds")
        fig_error.update_layout(title="Absolute pricing errors", xaxis_title="Number of steps N", yaxis_title="Absolute error")
        self._figure_background(fig_time)
        self._figure_background(fig_error)
        self._save(fig_time, "running_time.png")
        self._save(fig_error, "gap_N.png")


    def plot_convergence_order(self, prices: pd.DataFrame) -> None:
        """Creates two plots: (tree_price-bs_price)*N pricing errors and convergence orders across steps."""
        fig_tunnel, fig_order = go.Figure(), go.Figure()

        # plot (tree_price-bs_price)*N by number of steps
        self._add_traces(fig_tunnel, prices, "N", "Tunnel", mode="lines+markers")

        # plot convergence orders and linear fits in log-log scale
        for method, data in prices.groupby("Method", sort=False):
            data = data[(data["Gap_abs"] > 0) & (data["N"] > 1)].sort_values("N")
            if len(data) < 2: continue
            x, y = np.log(data["N"]), np.log(data["Gap_abs"])
            slope, intercept = np.polyfit(x, y, 1)
            fig_order.add_traces([
                go.Scatter(x=x, y=y, mode="lines+markers",
                        name=f"{method} (slope = {slope:.3f})", line=dict(color=self.COLORS[method])),
                go.Scatter(x=x, y=intercept + slope*x, mode="lines",
                        showlegend=False, line=dict(color=self.COLORS[method], dash="dash"))
            ])

        fig_tunnel.add_hline(y=0, line_color="black", line_width=1)
        fig_tunnel.update_layout(title="Signed scaled error: (Tree - BS) × N",
                                xaxis_title="Number of steps N",
                                yaxis_title="(Tree price - BS price) × N")

        fig_order.update_layout(title="Convergence order (log-log)",
                                xaxis_title="log(N)",
                                yaxis_title="log(|Tree - BS|)")

        for fig, filename in ((fig_tunnel, "tunnel_N.png"), (fig_order, "convergence_order.png")):
            self._figure_background(fig)
            self._save(fig, filename)

    def plot_strikes(self, prices: pd.DataFrame) -> None:
        """Creates three plots: option prices, pricing errors, and price slopes across strikes."""
        fig_price, fig_gap, fig_slope = go.Figure(), go.Figure(), go.Figure()
        bs = prices.drop_duplicates(subset="K")

        # plot price and slope of price by strike
        for fig, bs_col, tree_col in ((fig_price, "BS", "Tree"), (fig_slope, "Slope_BS", "Slope_Tree")):
            fig.add_trace(go.Scatter(x=bs["K"], y=bs[bs_col], name="Black-Scholes",
                                    mode="lines", line=dict(color="black", width=2)))
            self._add_traces(fig, prices, "K", tree_col)

        # plot gaps by strike
        self._add_traces(fig_gap, prices, "K", "Gap")
        fig_gap.add_hline(y=0, line_color="black", line_width=1)

        # plot theoretical error without dividends
        if not getattr(self.market, "dividends", None):
            forward = self.market.S0*np.exp(self.market.rf_rate
                                            *(self.option.mty_date-self.pricing_date).total_seconds()
                                            /(365*24*3600))
            if bs["K"].min() <= forward <= bs["K"].max():
                fig_price.add_vline(x=forward, line_dash="dot", line_color="gray", 
                                    annotation_text="Forward price", annotation_position="top right")
        
        for fig, title, ylabel, filename in (
            (fig_price, "Option price by strike", "Price", "price_strike.png"),
            (fig_gap, "Tree price - BS price", "Pricing error", "gap_strike.png"),
            (fig_slope, "Price slope by strike", "dPrice/dK", "slope_strike.png")):
            fig.update_layout(title=title, xaxis_title="Strike K", yaxis_title=ylabel)
            self._figure_background(fig)
            self._save(fig, filename)

    def _add_traces(self, fig:go.Figure, prices:pd.DataFrame, x:str, y:str, mode:str = "lines", showlegend:bool = True) -> None:
        for method, data in prices.groupby("Method", sort=False):
            data = data.sort_values(x)
            fig.add_trace(go.Scatter(
                x=data[x], y=data[y], name=method, mode=mode,
                legendgroup=method, showlegend=showlegend,
                line=dict(color=self.COLORS[method]),
            ))

    @staticmethod
    def _figure_background(fig:go.Figure) -> None:
        # update layout with white background + lightgray grid
        fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
                          legend=dict(orientation="h", y=-0.2), margin=dict(l=70, r=30, t=70, b=95))
        fig.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)

    @staticmethod
    def _save(fig:go.Figure, filename:str) -> None:
        # save graphs in the folder "graphs"
        folder = Path(__file__).resolve().parent / "graphs"
        folder.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(folder / filename), width=1600, height=900, scale=2)

if __name__ == "__main__":

    pricing_date = dt.now()
    maturity_date = pricing_date + timedelta(days=365)
    market = Market(S0=100, rf_rate=0.05, volatility=0.2)
    option = VanillaOption(strike=110, mty_date=maturity_date,
                           option_type="call", exercise_type="european")

    convergence = Convergence(market, option, pricing_date)

    steps_range = np.arange(100, 3000,10)
    prices_N = convergence.pricing_by_steps(steps_range=steps_range)
    convergence.plot_steps(prices_N)
    convergence.plot_convergence_order(prices_N)

    strike_range = np.linspace(80, 140, 2)
    prices_K = convergence.pricing_by_strike(strike_range, steps=100)
    convergence.plot_strikes(prices_K)

    # Find the required number of steps for a 0.01 pricing error
    print("Required steps:", convergence.get_nb_steps(0.001, "trinomial", False))    
