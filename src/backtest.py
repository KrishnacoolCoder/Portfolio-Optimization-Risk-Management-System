import numpy as np
import pandas as pd

from .optimization import (
    minimum_variance,
    maximum_sharpe,
    maximum_sortino,
    risk_parity,
)
from .metrics import portfolio_returns


def equal_weight(n):
    if n <= 0:
        raise ValueError("Number of assets must be positive.")

    return np.ones(n) / n


def calculate_transaction_cost(old_weights, new_weights, cost_bps):
    """
    One-way turnover cost.

    turnover = sum(|new target - current holdings|)
    cost = turnover * bps / 10,000
    """
    old_weights = np.asarray(old_weights, dtype=float)
    new_weights = np.asarray(new_weights, dtype=float)

    if old_weights.shape != new_weights.shape:
        raise ValueError("Old and new weight vectors must have the same shape.")

    if cost_bps < 0:
        raise ValueError("Transaction cost cannot be negative.")

    turnover = np.sum(np.abs(new_weights - old_weights))
    return float(turnover * cost_bps / 10000.0)


def _drift_weights(weights, asset_returns):
    """
    Calculate the portfolio weights immediately before the next rebalance
    after the assets have moved during the current test period.
    """
    weights = np.asarray(weights, dtype=float)
    asset_returns = asset_returns.dropna()

    if len(asset_returns) == 0:
        return weights.copy()

    growth = (1.0 + asset_returns).prod().to_numpy()
    holdings = weights * growth
    total = holdings.sum()

    if not np.isfinite(total) or total <= 0:
        return weights.copy()

    return holdings / total


def _validate_inputs(
    returns,
    lookback_days,
    rebalance_frequency,
    max_weight,
    transaction_cost_bps,
):
    if returns is None or returns.empty:
        raise ValueError("Backtest returns data is empty.")

    if returns.isna().any().any():
        raise ValueError("Backtest returns contain missing values.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Backtest returns contain invalid values.")

    if lookback_days <= 0:
        raise ValueError("Lookback window must be positive.")

    if rebalance_frequency <= 0:
        raise ValueError("Rebalance frequency must be positive.")

    if max_weight <= 0 or max_weight > 1:
        raise ValueError("Maximum weight must be between 0 and 1.")

    if max_weight * returns.shape[1] < 1:
        raise ValueError(
            f"Maximum weight {max_weight:.2%} is infeasible for "
            f"{returns.shape[1]} assets."
        )

    if transaction_cost_bps < 0:
        raise ValueError("Transaction cost cannot be negative.")

    if len(returns) <= lookback_days:
        raise ValueError(
            "Not enough observations for the requested walk-forward "
            "lookback window."
        )


def _optimize_weights(
    train,
    optimizer_name,
    risk_free_rate,
    max_weight,
):
    n = train.shape[1]

    mu = train.mean().values * 252
    cov = train.cov().values * 252

    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError(
            f"Training window contains invalid statistics for "
            f"{optimizer_name}."
        )

    if optimizer_name == "Equal Weight":
        return equal_weight(n)

    if optimizer_name == "Minimum Volatility":
        return minimum_variance(
            mu,
            cov,
            max_weight,
        )

    if optimizer_name == "Maximum Sharpe":
        return maximum_sharpe(
            mu,
            cov,
            risk_free_rate,
            max_weight,
        )

    if optimizer_name == "Maximum Sortino":
        return maximum_sortino(
            train,
            risk_free_rate,
            max_weight,
        )

    if optimizer_name == "Risk Parity":
        return risk_parity(
            cov,
            max_weight,
        )

    raise ValueError(
        f"Unknown strategy: {optimizer_name}"
    )


def walk_forward_backtest(
    returns,
    optimizer_name,
    lookback_days=504,
    rebalance_frequency=21,
    risk_free_rate=0.0,
    max_weight=1.0,
    transaction_cost_bps=10.0,
):
    """
    Walk-forward out-of-sample backtest.

    At each rebalance:
      1. Train only on observations before the rebalance date.
      2. Optimize the portfolio using that training window.
      3. Charge transaction costs based on the actual drifted holdings.
      4. Apply the target weights to the following unseen test period.

    This prevents future test-period returns from entering the optimizer.
    """
    returns = returns.sort_index().dropna()

    _validate_inputs(
        returns,
        lookback_days,
        rebalance_frequency,
        max_weight,
        transaction_cost_bps,
    )

    n = returns.shape[1]

    strategy_returns = []
    dates = []

    # Before the first rebalance, assume an equal-weight starting portfolio.
    current_weights = equal_weight(n)

    for start in range(
        lookback_days,
        len(returns),
        rebalance_frequency,
    ):
        train = returns.iloc[
            start - lookback_days:start
        ]

        end = min(
            start + rebalance_frequency,
            len(returns),
        )

        test = returns.iloc[
            start:end
        ]

        if test.empty:
            continue

        # IMPORTANT:
        # train ends strictly before test starts.
        weights = _optimize_weights(
            train,
            optimizer_name,
            risk_free_rate,
            max_weight,
        )

        weights = np.asarray(
            weights,
            dtype=float,
        )

        if (
            len(weights) != n
            or not np.isfinite(weights).all()
            or abs(weights.sum() - 1.0) > 1e-5
            or np.any(weights < -1e-8)
            or np.any(weights > max_weight + 1e-6)
        ):
            raise RuntimeError(
                f"{optimizer_name} produced an invalid portfolio."
            )

        # Transaction cost is based on actual holdings immediately
        # before the rebalance, not simply the previous target weights.
        cost = calculate_transaction_cost(
            current_weights,
            weights,
            transaction_cost_bps,
        )

        period_returns = portfolio_returns(
            test,
            weights,
        ).copy()

        if len(period_returns):
            # Cost is paid at the rebalance at the beginning of
            # the out-of-sample period.
            period_returns.iloc[0] -= cost

        strategy_returns.append(period_returns)
        dates.extend(period_returns.index.tolist())

        # The target portfolio drifts during the unseen test period.
        # Those drifted weights become the holdings used for turnover
        # at the next rebalance.
        current_weights = _drift_weights(
            weights,
            test,
        )

    if not strategy_returns:
        return pd.Series(
            dtype=float,
            name=optimizer_name,
        )

    result = pd.concat(strategy_returns)

    result = result[~result.index.duplicated(keep="first")]
    result = result.sort_index()

    result.name = optimizer_name

    return result
