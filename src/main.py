from pathlib import Path



from model.trinomial import TrinomialModel
from data.options import VanillaOption
from data.market import Market
from src.constants import DEFAULT_PRUNING_THRESHOLD, DEFAULT_STEPS
from svg.export_svg import export_svg


if __name__=="__main__":
    from datetime import datetime,timedelta

    from data.market import Market,Dividend
    from data.options import VanillaOption
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

    pruning_threshold = DEFAULT_PRUNING_THRESHOLD
    nb_steps = DEFAULT_STEPS

    model_div = TrinomialModel(market_div,option,steps=nb_steps,pricing_date=pricing_date,pruning_threshold=pruning_threshold)

    export_svg(model_div,"outputs/svg/model_div.svg")