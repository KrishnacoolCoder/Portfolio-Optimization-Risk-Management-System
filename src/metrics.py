import numpy as np
import pandas as pd


def portfolio_returns(asset_returns, weights):
    return asset_returns @ np.asarray(weights)


def annualized_return(returns, periods=252):
    if len(returns) == 0:
        return np.nan
    return (1 + returns).prod() ** (periods / len(returns)) - 1


def annualized_volatility(returns, periods=252):
    return returns.std() * np.sqrt(periods)


def sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    ann_ret = annualized_return(returns, periods)
    ann_vol = annualized_volatility(returns, periods)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(returns, risk_free_rate=0.0, periods=252):
    ann_ret = annualized_return(returns, periods)
    daily_rf = (1 + risk_free_rate) ** (1 / periods) - 1
    downside = returns[returns < daily_rf] - daily_rf

    if len(downside) == 0:
        return np.inf

    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods)
    if downside_dev == 0:
        return np.inf

    return (ann_ret - risk_free_rate) / downside_dev


def max_drawdown(returns):
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return drawdown.min()


def var_historical(returns, confidence=0.95):
    return -np.quantile(returns, 1 - confidence)


def cvar_historical(returns, confidence=0.95):
    var = np.quantile(returns, 1 - confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return 0.0
    return -tail.mean()


def calmar_ratio(returns):
    cagr = annualized_return(returns)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return cagr / mdd


def metrics_table(strategy_returns):
    rows = []

    for name, ret in strategy_returns.items():
        rows.append({
            "Strategy": name,
            "CAGR": annualized_return(ret),
            "Volatility": annualized_volatility(ret),
            "Sharpe": sharpe_ratio(ret),
            "Sortino": sortino_ratio(ret),
            "Max Drawdown": max_drawdown(ret),
            "VaR 95%": var_historical(ret),
            "CVaR 95%": cvar_historical(ret),
            "Calmar": calmar_ratio(ret),
        })

    return pd.DataFrame(rows).set_index("Strategy")
