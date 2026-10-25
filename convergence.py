# import classes
from market import Market
from options import VanillaOption
from model.binomial import BinomialModel
from black_scholes import BlackScholes

# import libraries
import time
import copy
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta

# import plotly for graphs
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 


class Convergence:

    def __init__(self, steps:int, model:BinomialModel, market:Market, option:VanillaOption):

        self.steps = steps
        self.model = model
        self.market = market
        self.option = option

    def _tree_price(self, option,steps , get_times=False):
        # binomial pricing with time counter
        start = time.perf_counter()
        binomial_price,pricing_time = self.model(self.market,option,steps).price_from_tree()
        running_time = time.perf_counter()-start

        if get_times:
            return binomial_price,pricing_time,running_time
        else:
            return binomial_price

    def pricing_by_steps(self,nb_pricing=1):
        bs_price = BlackScholes(self.market,self.option).black_scholes_pricing()
        rows = []

        # pricing for each nb of steps, saving results in a DataFrame
        for i in range(1,nb_pricing+1):
            step = self.steps*i//nb_pricing
            binomial_price,pricing_time,running_time = self._tree_price(self.option,step,True)

            gap = binomial_price-bs_price

            rows.append({
                "N":step,
                "BS_price":bs_price,
                "Tree_price":binomial_price,
                "Gap_abs":abs(gap),
                "Gap":gap,
                "Theoretical_gap":self.theoretical_gap(step),
                "Running_time":running_time,
                "Pricing_time":pricing_time,
                "Tunnel":gap*step
            })

        return pd.DataFrame(rows)

    def pricing_by_strike(self,strike_range,steps=10):
        rows = []

        # pricing for each strike, saving results in a DataFram
        for strike in strike_range:
            # create a copy to not change the initial parameters
            option = copy.copy(self.option)
            option.strike = strike

            bs_price = BlackScholes(self.market,option).black_scholes_pricing()
            binomial_price = self._tree_price(option,steps)

            rows.append({
                "K":strike,
                "BS":bs_price,
                "Tree":binomial_price,
                "Gap":binomial_price-bs_price
            })

        prices = pd.DataFrame(rows)

        # slope of price with respect to the strike
        if len(prices)>1:
            prices["Slope_BS"] = np.gradient(prices["BS"],prices["K"])
            prices["Slope_Tree"] = np.gradient(prices["Tree"],prices["K"])

        return prices

    def theoretical_gap(self,N):
        S0 = self.market.S0
        r = self.market.rf_rate
        sigma = self.market.volatility
        T = (self.option.mty_date-self.option.strike_date).total_seconds()/(365*24*3600)
        dt = T/N

        return S0*np.exp(r*T)/(2*np.sqrt(2*np.pi))*sigma**2*dt/np.sqrt(np.exp(sigma**2*T)-1)

    def plot_steps(self,prices):
        prices = prices.copy().apply(pd.to_numeric)

        # graph of total running time and pricing time
        figure_1 = px.line(prices,x="N",y=["Running_time","Pricing_time"],markers=True,title="Running time et pricing time")
        self.figure_background(figure_1)
        figure_1.write_image("running_time.png",width=1600,height=900,scale=2)

        # graph of absolute gaps and theoretical gap
        figure_2 = px.line(prices,x="N",y=["Gap_abs","Theoretical_gap"],markers=True,title="Observed vs theoretical error")
        self.figure_background(figure_2)
        figure_2.write_image("gap_N.png",width=1600,height=900,scale=2)

    def plot_convergence_order(self,prices):
        prices = prices.copy().apply(pd.to_numeric)
        data = prices[prices["Gap_abs"]>0].copy()

        # log-log regression to find the speed of convergence
        data["log_N"] = np.log(data["N"])
        data["log_Gap"] = np.log(data["Gap_abs"])
        slope,intercept = np.polyfit(data["log_N"],data["log_Gap"],1)
        data["Regression"] = intercept+slope*data["log_N"]

        # graph of gap * N
        fig = make_subplots(rows=2,cols=1,subplot_titles=("(Tree - BS) * N","Log-log convergence"))
        fig.add_trace(go.Scatter(x=prices["N"],y=prices["Tunnel"],name="gap * N"),row=1,col=1)
        fig.update_xaxes(title_text="N",row=1,col=1)
        fig.update_yaxes(title_text="Gap × N",row=1,col=1)

        # graph of log-log regression
        fig.add_trace(go.Scatter(x=data["log_N"],y=data["log_Gap"],mode="markers",name="Observed error"),row=2,col=1)
        fig.add_trace(go.Scatter(x=data["log_N"],y=data["Regression"],mode="lines",name=f"Regression slope = {slope:.3f}"),row=2,col=1)
        fig.update_xaxes(title_text="log(N)",row=2,col=1)
        fig.update_yaxes(title_text="log(|Tree - BS|)",row=2,col=1)

        fig.update_layout(title=f"Study of convergence order",height=900)
        self.figure_background(fig)

        fig.write_image("convergence_order.png",width=1600,height=1000,scale=2)

    def plot_strikes(self,prices):
        prices = prices.copy().apply(pd.to_numeric)

        fig = make_subplots(specs=[[{"secondary_y":True}]])

        # graphs of slopes and errors of prices
        fig.add_trace(go.Scatter(x=prices["K"],y=prices["Gap"],name="Tree - BS"),secondary_y=False)
        fig.add_trace(go.Scatter(x=prices["K"],y=prices["Slope_BS"],name="Slope BS",line=dict(dash="dash")),secondary_y=True)
        fig.add_trace(go.Scatter(x=prices["K"],y=prices["Slope_Tree"],name="Slope Tree",line=dict(dash="dash")),secondary_y=True)

        fig.update_layout(title="Errors and slopes as a function of strike")
        fig.update_xaxes(title_text="Strike K")
        fig.update_yaxes(title_text="Tree - BS",secondary_y=False)
        fig.update_yaxes(title_text="Slope",secondary_y=True)

        fig.update_layout(title="Price convergence",height=900)
        self.figure_background(fig)

        fig.write_image("gap_strike.png",width=1600,height=900,scale=2)
        

    @staticmethod
    def figure_background(fig):
        # update background colors of graphs
        fig.update_layout(template = "plotly_white", plot_bgcolor = "white", paper_bgcolor = "white")
        fig.update_xaxes(showgrid = True, gridcolor = "lightgray", zeroline = False)
        fig.update_yaxes(showgrid = True, gridcolor = "lightgray", zeroline = False)


if __name__=="__main__":
    # market data
    spot = 100
    rf_rate = 0.05
    volatility = 0.2

    # option characteristics
    strike = 110
    strike_date = dt.now()
    option_type = "call"
    exercise_type = "european"
    maturity = dt.now()+timedelta(days=365)

    market = Market(spot,rf_rate,volatility)
    option = VanillaOption(strike, strike_date, option_type, exercise_type, maturity)

    convergence = Convergence(1000,BinomialModel,market,option)

    # convergence by step
    prices_N = convergence.pricing_by_steps(100)
    convergence.plot_steps(prices_N)
    order = convergence.plot_convergence_order(prices_N)

    # convergence by strike
    strike_range = np.linspace(80,140,100)
    prices_K = convergence.pricing_by_strike(strike_range,steps=10)
    convergence.plot_strikes(prices_K)
