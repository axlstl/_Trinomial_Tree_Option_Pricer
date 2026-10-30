from pathlib import Path



from model.trinomial import TrinomialModel
from options import VanillaOption
from market import Market
from model.binomial import BinomialModel
from svg.export_svg import export_svg


if __name__=="__main__":
    from datetime import datetime,timedelta

    from market import Market,Dividend
    from options import VanillaOption
    from model.trinomial import TrinomialModel
    #all params
    pricing_date = datetime.now().date()
    maturity_date = pricing_date + timedelta(days=365)
    dividend_date = pricing_date + timedelta(days=180)

    option_params = {"strike":100,"mty_date":maturity_date,"option_type":"call","exercise_type":"american"}
    market_params = {"S0":100,"rf_rate":0.05,"volatility":0.2}
    dividend = Dividend(date=dividend_date,amount=5)

    option = VanillaOption(**option_params)

    market_no_div = Market(**market_params)
    market_div = Market(**market_params,dividends=[dividend])

    pruning_threshold = 1e-7
    nb_steps = 50

    model_div = TrinomialModel(market_div,option,steps=nb_steps,pricing_date=pricing_date,pruning_threshold=pruning_threshold)

    print("Prix de l'option avec dividende :",model_div.price_from_tree())

    path_svg = Path(__file__).parent / "tree_div.svg"
    print("SVG avec dividende :",export_svg(model_div,path_svg))
