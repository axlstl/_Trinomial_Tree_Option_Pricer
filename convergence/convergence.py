from datetime import datetime as dt, date, timedelta
import copy
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data.market import Market
from data.options import Option, VanillaOption, DigitalOption
from model.binomial import BinomialModel
from model.trinomial import TrinomialModel
from model.lattice import LatticeModel
from convergence.black_scholes import BlackScholes


class Convergence:
    # set colors for graphs
    COLORS = {
        "Binomial":"#2563eb",
        "Trinomial": "#7c3aed",
        "Binomial with smoothing": "#93b4f5",
        "Trinomial with smoothing": "#c4a1fa",
    }

    def __init__(self, market: Market,
                 option: Option, pricing_date:dt, pruning_threshold:float = 1e-7):

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
    
    def _tree_price(self, option:Option, steps:int, model:type[LatticeModel], 
                    get_times:bool = False, bs_smoothing:bool = False) -> float | tuple[float, float, float]:
        
        params = {"steps": int(steps), "pricing_date": self.pricing_date, "bs_smoothing": bs_smoothing, 
                  "pruning_threshold": self.pruning_threshold}

        start = time.perf_counter()
        tree = model(self.market, option, **params)
        pricing_start = time.perf_counter()
        tree_price = tree.price_from_tree()

        # time for pricing
        pricing_time = time.perf_counter() - pricing_start

        tree.clear_tree()
        # Total time includes building, pricing and clearing the tree.
        running_time = time.perf_counter() - start

        return (tree_price, pricing_time, running_time) if get_times else tree_price

    def _bs_price(self, option:Option) -> float:
        return BlackScholes(self.market, option).vanilla_pricing(self.pricing_date)

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
                    "Theoretical_gap": self.theoretical_gap(step, method) if method in ("Binomial", "Trinomial") and self._has_bs_reference() else np.nan,
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

    def _has_bs_reference(self) -> bool:
        # check if it is an european pricing without dividends
        return self.option.exercise_type.lower() == "european" and not getattr(self.market, "dividends", None)

    def theoretical_gap(self, N: int, method: str = "Binomial") -> float:
        S0 = self.market.S0
        r = self.market.rf_rate
        sigma = self.market.volatility
        T = (self.option.mty_date-self.pricing_date).total_seconds()/(365*24*3600)
        dt_step = T/N
        factor = 0.75 if method == "Trinomial" else 1.0
        return factor*S0*np.exp(r*T)/(2*np.sqrt(2*np.pi))*sigma**2*dt_step/np.sqrt(np.exp(sigma**2*T)-1)

    def _add_traces(self, fig:go.Figure, prices:pd.DataFrame, x:str, y:str, mode:str = "lines", showlegend:bool = True) -> None:
        for method, data in prices.groupby("Method", sort=False):
            data = data.sort_values(x)
            fig.add_trace(go.Scatter(
                x=data[x], y=data[y], name=method, mode=mode,
                legendgroup=method, showlegend=showlegend,
                line=dict(color=self.COLORS[method]),
            ))

    @staticmethod
    def _save(fig:go.Figure, filename:str) -> None:
        fig.write_image(filename, width=1600, height=900, scale=2)

    def plot_steps(self, prices: pd.DataFrame) -> None:
        fig_time = go.Figure()
        fig_error = go.Figure()

        for method, data in prices.groupby("Method", sort=False):
            data = data.sort_values("N")
            fig_time.add_trace(go.Scatter(
                x=data["N"], y=data["Running_time"], mode="lines+markers", name=method,
                line=dict(color=self.COLORS[method]),
            ))
            fig_error.add_trace(go.Scatter(
                x=data["N"], y=data["Gap_abs"], mode="lines+markers", name=method,
                line=dict(color=self.COLORS[method]),
            ))

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
        self.figure_background(fig_time)
        self.figure_background(fig_error)
        self._save(fig_time, "running_time.png")
        self._save(fig_error, "gap_N.png")

    def plot_convergence_order(self, prices:pd.DataFrame) -> None:
        fig_tunnel = go.Figure()
        fig_order = go.Figure()
        for method, data in prices.groupby("Method", sort=False):
            data = data.sort_values("N")
            fig_tunnel.add_trace(go.Scatter(
                x=data["N"], y=data["Tunnel"], mode="lines+markers", name=method,
                line=dict(color=self.COLORS[method]),
            ))

            data = data[(data["Gap_abs"] > 0) & (data["N"] > 1)].copy()
            if len(data) < 2:
                continue
            data["log_N"] = np.log(data["N"])
            data["log_Gap"] = np.log(data["Gap_abs"])
            slope, intercept = np.polyfit(data["log_N"], data["log_Gap"], 1)
            fig_order.add_trace(go.Scatter(
                x=data["log_N"], y=data["log_Gap"], mode="lines+markers",
                name=f"{method} (slope = {slope:.3f})", line=dict(color=self.COLORS[method]),
            ))
            fig_order.add_trace(go.Scatter(
                x=data["log_N"], y=intercept+slope*data["log_N"], mode="lines",
                showlegend=False, line=dict(color=self.COLORS[method], dash="dash"),
            ))

        fig_tunnel.add_hline(y=0, line_color="black", line_width=1)
        fig_tunnel.update_xaxes(title_text="Number of steps N")
        fig_tunnel.update_yaxes(title_text="(Tree price - BS price) × N")
        fig_tunnel.update_layout(title="Signed scaled error: (Tree - BS) × N")
        fig_order.update_layout(title="Convergence order (log-log)", xaxis_title="log(N)", yaxis_title="log(|Tree - BS|)")
        self.figure_background(fig_tunnel)
        self.figure_background(fig_order)
        self._save(fig_tunnel, "tunnel_N.png")
        self._save(fig_order, "convergence_order.png")

    def plot_strikes(self, prices:pd.DataFrame):
        fig_price = go.Figure()
        fig_gap = go.Figure()
        fig_slope = go.Figure()

        bs = prices.drop_duplicates(subset="K").sort_values("K")
        fig_price.add_trace(go.Scatter(x=bs["K"], y=bs["BS"], name="Black-Scholes",
                                       mode="lines", line=dict(color="black", width=2)))
        self._add_traces(fig_price, prices, "K", "Tree")
        self._add_traces(fig_gap, prices, "K", "Gap")
        fig_gap.add_hline(y=0, line_color="black", line_width=1)

        fig_slope.add_trace(go.Scatter(x=bs["K"], y=bs["Slope_BS"], name="Black-Scholes",
                                       mode="lines", line=dict(color="black", width=2)))
        self._add_traces(fig_slope, prices, "K", "Slope_Tree")

        if not getattr(self.market, "dividends", None):
            T = (self.option.mty_date-self.pricing_date).total_seconds()/(365*24*3600)
            forward = self.market.S0*np.exp(self.market.rf_rate*T)
            if bs["K"].min() <= forward <= bs["K"].max():
                fig_price.add_vline(x=forward, line_dash="dot", line_color="gray",
                                    annotation_text="Forward price", annotation_position="top right")

        fig_price.update_layout(title="Option price by strike", xaxis_title="Strike K", yaxis_title="Price")
        fig_gap.update_layout(title="Tree price - BS price", xaxis_title="Strike K", yaxis_title="Pricing error")
        fig_slope.update_layout(title="Price slope by strike", xaxis_title="Strike K", yaxis_title="dPrice/dK")

        for fig, filename in ((fig_price, "price_strike.png"), (fig_gap, "gap_strike.png"),
                              (fig_slope, "slope_strike.png")):
            self.figure_background(fig)
            self._save(fig, filename)

    @staticmethod
    def figure_background(fig:go.Figure) -> None:
        # update layout with white background + lightgray grid
        fig.update_layout(template="plotly_white", plot_bgcolor="white", paper_bgcolor="white",
                          legend=dict(orientation="h", y=-0.2), margin=dict(l=70, r=30, t=70, b=95))
        fig.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)


if __name__ == "__main__":

    pricing_date = dt.now()
    maturity_date = pricing_date + timedelta(days=365)
    market = Market(S0=100, rf_rate=0.05, volatility=0.2)
    option = VanillaOption(strike=110, mty_date=maturity_date,
                           option_type="call", exercise_type="european")

    convergence = Convergence(market, option, pricing_date)

    steps_range = np.arange(10, 200,20)
    prices_N = convergence.pricing_by_steps(steps_range=steps_range)
    convergence.plot_steps(prices_N)
    convergence.plot_convergence_order(prices_N)

    strike_range = np.linspace(100, 120, 2)
    prices_K = convergence.pricing_by_strike(strike_range, steps=50)
    convergence.plot_strikes(prices_K)
