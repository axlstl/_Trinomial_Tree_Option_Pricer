TIME_BASIS = 365*24*60*60
TOL_THRESHOLD = 1e-4

#Model base values
DEFAULT_PRUNING_THRESHOLD = 1e-7
DEFAULT_STEPS = 50

# SVG layout settings, in pixels except for the node-count frequency.
SVG_SIDE_MARGIN = 80
SVG_TOP_MARGIN = 20
SVG_BOTTOM_MARGIN = 60
SVG_X_SPACING = 50
SVG_Y_SPACING = 10
SVG_NODE_RADIUS = 5
SVG_NODE_COUNT_EVERY = 5
SVG_NODE_COUNT_OFFSET = 35
SVG_DIVIDEND_OFFSET = 15
SVG_EXERCISE_RING_OFFSET = 3

# Default calculation settings for greeks.
DEFAULT_RELATIVE_GREEK_STEP = 0.01
DEFAULT_VOLATILITY_STEP = 0.01
DEFAULT_THETA_DAYS = 1

#class attributes, derivative order and unit conversions for every greek
GREEKS = {
    "delta": {
        "inputs": ["market.S0"],
        "order": 1,
        "divisor": 1,
    },
    "gamma": {
        "inputs": ["market.S0"],
        "order": 2,
        "divisor": 1,
    },
    "vega": {
        "inputs": ["market.volatility"],
        "order": 1,
        "divisor": 100,
    },
    "rho": {
        "inputs": ["market.rf_rate"],
        "order": 1,
        "divisor": 1,
    },
    "volga": {
        "inputs": ["market.volatility"],
        "order": 2,
        "divisor": 10000,
    },
    "vanna": {
        "inputs": ["market.S0", "market.volatility"],
        "order": 2,
        "divisor": 100,
    },
    "theta": {
        "inputs": ["start_date"],
        "order": 1,
        "divisor": 1,
    },
}
