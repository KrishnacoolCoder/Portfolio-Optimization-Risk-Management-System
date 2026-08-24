import numpy as np
import pandas as pd

from src.backtest import (
    calculate_transaction_cost,
    walk_forward_backtest,
)


def make_returns(rows=700, assets=4):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=rows)

    return pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(rows, assets)),
        index=dates,
        columns=[f"A{i}" for i in range(assets)],
    )


def test_transaction_cost_zero_when_weights_do_not_change():
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    assert calculate_transaction_cost(
        weights,
        weights,
        10,
    ) == 0.0


def test_transaction_cost_is_nonnegative():
    old = np.array([0.50, 0.25, 0.15, 0.10])
    new = np.array([0.25, 0.25, 0.25, 0.25])

    assert calculate_transaction_cost(
        old,
        new,
        10,
    ) >= 0.0


def test_walk_forward_has_only_out_of_sample_returns():
    returns = make_returns()

    result = walk_forward_backtest(
        returns,
        "Equal Weight",
        lookback_days=252,
        rebalance_frequency=21,
        max_weight=0.50,
        transaction_cost_bps=10,
    )

    assert not result.empty
    assert result.index.min() >= returns.index[252]


def test_infeasible_weight_constraint_fails():
    returns = make_returns(700, 4)

    try:
        walk_forward_backtest(
            returns,
            "Equal Weight",
            lookback_days=252,
            max_weight=0.20,
        )
    except ValueError as exc:
        assert "infeasible" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected infeasible max-weight constraint to fail."
        )
