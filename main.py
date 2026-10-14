import xml.etree.ElementTree as ET

from datetime import datetime as dt, timedelta

from model.trinomial import TrinomialModel
from options import VanillaOption
from market import Market
from model.binomial import BinomialModel


if __name__=="__main__":
    from datetime import datetime as dt,timedelta
    from market import Market
    from options import VanillaOption


    mty_date=dt.now()+timedelta(days=365)
    option=VanillaOption(strike=110,strike_date=dt.now(),option_type="call",exercise_type="european",mty_date=mty_date)
    market=Market(S0=100,rf_rate=0.05,volatility=0.2)
    


    model_binomial=BinomialModel(market,option,steps=1000)    
    model_tri=TrinomialModel(market,option,steps=1000)


    print(model_binomial.price_from_tree())
    print(model_tri.price_from_tree())
