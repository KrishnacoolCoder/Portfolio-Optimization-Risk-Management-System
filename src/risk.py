import numpy as np


def portfolio_volatility(weights, covariance):
    weights = np.asarray(weights)
    return float(np.sqrt(weights @ covariance @ weights))


def portfolio_return(weights, expected_returns):
    return float(np.asarray(weights) @ np.asarray(expected_returns))


def downside_deviation(weights, returns, target=0.0):
    portfolio = returns @ np.asarray(weights)
    downside = np.minimum(portfolio - target, 0)
    return float(np.sqrt(np.mean(downside ** 2)))


def historical_cvar(weights, returns, confidence=0.95):
    portfolio = returns @ np.asarray(weights)
    cutoff = np.quantile(portfolio, 1 - confidence)
    tail = portfolio[portfolio <= cutoff]
    return float(-tail.mean()) if len(tail) else 0.0


def stress_loss(weights, returns, quantile=0.05):
    portfolio = returns @ np.asarray(weights)
    return float(np.quantile(portfolio, quantile))
