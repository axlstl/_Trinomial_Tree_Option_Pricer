from pathlib import Path



from model.trinomial import TrinomialModel
from options import VanillaOption
from market import Market
from model.binomial import BinomialModel
from export_svg import export_svg


if __name__=="__main__":
    from datetime import datetime,timedelta

    from market import Market,Dividend
    from options import VanillaOption
    from model.trinomial import TrinomialModel


    pricing_date = datetime.now().date()
    maturity_date = pricing_date + timedelta(days=365)
    dividend_date = pricing_date + timedelta(days=180)

    option_params = {"strike":100,"mty_date":maturity_date,"option_type":"call","exercise_type":"european"}
    market_params = {"S0":100,"rf_rate":0.05,"volatility":0.2}
    dividend = Dividend(date=dividend_date,amount=5)

    option = VanillaOption(**option_params)

    market_no_div = Market(**market_params)
    market_div = Market(**market_params,dividends=[dividend])


    pruning_threshold = 1e-7
    nb_steps = 50
    model_no_div = TrinomialModel(market_no_div,option,steps=nb_steps,pricing_date=pricing_date,pruning_threshold=pruning_threshold)
    model_div = TrinomialModel(market_div,option,steps=nb_steps,pricing_date=pricing_date,pruning_threshold=pruning_threshold)

    print("Sans dividende :",model_no_div.price_from_tree())
    print("Avec dividende :",model_div.price_from_tree())

    print("SVG sans dividende :",export_svg(model_no_div,Path(__file__).with_name("tree_no_div.svg")))
    print("SVG avec dividende :",export_svg(model_div,Path(__file__).with_name("tree_div.svg")))


    trunk = model_div.root

    while trunk is not None:
        node = model_div._get_bottom_node(trunk)

        while node is not None:
            if node.p_up is not None:
                total = node.p_up+node.p_mid+node.p_down

                if node.p_up < 0 or node.p_mid < 0 or node.p_down < 0 or abs(total-1) > 1e-10:
                    print("PROBLEME")
                    print("step :",trunk.step)
                    print("spot :",node.underlying_price)
                    print("p :",node.p_up,node.p_mid,node.p_down)
                    print("sum :",total)

            node = node.up_node

        trunk = trunk.next_trunk


        
    # from black_scholes import BlackScholes

    # # Test avec un seul pas
    # model_classic = TrinomialModel(market_no_div, option, steps=1, pricing_date=pricing_date, pruning_threshold=pruning_threshold, bs_smoothing=False)
    # model_bs = TrinomialModel(market_no_div, option, steps=1, pricing_date=pricing_date, pruning_threshold=pruning_threshold, bs_smoothing=True)

    # price_classic = model_classic.price_from_tree()
    # price_bs_tree = model_bs.price_from_tree()
    # price_bs = BlackScholes(market_no_div, option).vanilla_pricing(pricing_date)

    # print("\n--- TEST STEPS = 1 ---")
    # print("Trinomial classique :", price_classic)
    # print("Trinomial avec BS   :", price_bs_tree)
    # print("Black-Scholes       :", price_bs)
    # print("Erreur classique    :", abs(price_classic-price_bs))
    # print("Erreur avec BS      :", abs(price_bs_tree-price_bs))

    # assert abs(price_bs_tree-price_bs) < 1e-10
