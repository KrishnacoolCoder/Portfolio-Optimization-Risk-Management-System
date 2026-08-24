import numpy as np
import pandas as pd

from src.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)


def test_annualized_return_positive():
    returns = pd.Series([0.01] * 252)
    assert annualized_return(returns) > 0


def test_volatility_nonnegative():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 252))
    assert annualized_volatility(returns) >= 0


def test_drawdown_is_nonpositive():
    returns = pd.Series([0.10, -0.05, -0.20, 0.05])
    assert max_drawdown(returns) <= 0


def test_sharpe_exists():
    returns = pd.Series([0.001] * 252)
    assert np.isfinite(sharpe_ratio(returns))
