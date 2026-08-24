import numpy as np
from scipy.optimize import minimize


def _constraints(n):
    return {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }


def _bounds(n, max_weight):
    if max_weight * n < 1:
        raise ValueError(
            f"Maximum weight {max_weight:.2f} is too small "
            f"for {n} assets. It must be at least {1/n:.2%}."
        )

    return [(0.0, max_weight) for _ in range(n)]


def _initial_weights(n, max_weight):
    """
    Creates a feasible starting portfolio.
    """
    w = np.ones(n) / n

    if np.all(w <= max_weight):
        return w

    # Fallback allocation if equal weights are not feasible
    w = np.zeros(n)
    remaining = 1.0

    for i in range(n):
        amount = min(max_weight, remaining)
        w[i] = amount
        remaining -= amount

        if remaining <= 1e-12:
            break

    return w


def portfolio_return(weights, mu):
    return float(weights @ mu)


def portfolio_volatility(weights, cov):
    value = weights @ cov @ weights

    # Numerical safety
    value = max(value, 0.0)

    return float(np.sqrt(value))


def sharpe_objective(weights, mu, cov, risk_free_rate):
    vol = portfolio_volatility(weights, cov)

    if vol <= 1e-12:
        return 1e6

    return -(
        portfolio_return(weights, mu) - risk_free_rate
    ) / vol


def _run_slsqp(objective, n, max_weight, extra_constraints=None):
    """
    Robust wrapper around SLSQP.
    """

    bounds = _bounds(n, max_weight)
    x0 = _initial_weights(n, max_weight)

    constraints = [_constraints(n)]

    if extra_constraints:
        constraints.extend(extra_constraints)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-9,
            "disp": False
        }
    )

    if result.success:
        return result.x

    # Retry with a slightly different initial point
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 5000,
            "ftol": 1e-7,
            "disp": False
        }
    )

    if result.success:
        return result.x

    raise RuntimeError(
        f"Optimization failed: {result.message}"
    )


def minimum_variance(mu, cov, max_weight=1.0):
    n = len(mu)

    return _run_slsqp(
        lambda w: portfolio_volatility(w, cov),
        n,
        max_weight
    )


def maximum_sharpe(mu, cov, risk_free_rate=0.0, max_weight=1.0):
    n = len(mu)

    return _run_slsqp(
        lambda w: sharpe_objective(
            w,
            mu,
            cov,
            risk_free_rate
        ),
        n,
        max_weight
    )


def maximum_sortino(
    returns,
    risk_free_rate=0.0,
    max_weight=1.0
):
    """
    Maximum Sortino portfolio.

    Uses SLSQP with a stable objective and multiple
    optimization attempts.
    """

    n = returns.shape[1]

    # Convert returns to numpy for faster calculations
    r = returns.values

    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    def objective(w):

        portfolio = r @ w

        # Annualized return
        if len(portfolio) == 0:
            return 1e6

        annual_return = (
            (1 + portfolio).prod()
            ** (252 / len(portfolio))
            - 1
        )

        # Downside returns
        downside = np.minimum(
            portfolio - daily_rf,
            0
        )

        downside_deviation = (
            np.sqrt(np.mean(downside ** 2))
            * np.sqrt(252)
        )

        if downside_deviation < 1e-10:
            return -100.0

        sortino = (
            annual_return - risk_free_rate
        ) / downside_deviation

        return -sortino

    return _run_slsqp(
        objective,
        n,
        max_weight
    )


def risk_parity(cov, max_weight=1.0):
    n = cov.shape[0]

    def objective(w):

        portfolio_vol = portfolio_volatility(
            w,
            cov
        )

        if portfolio_vol <= 1e-12:
            return 1e6

        marginal_contribution = cov @ w

        risk_contribution = (
            w
            * marginal_contribution
            / portfolio_vol
        )

        target = portfolio_vol / n

        return np.sum(
            (risk_contribution - target) ** 2
        )

    return _run_slsqp(
        objective,
        n,
        max_weight
    )


def efficient_frontier(
    mu,
    cov,
    points=50,
    max_weight=1.0
):
    """
    Generates a constrained efficient frontier.

    The old implementation tried target returns
    outside the feasible region. This version first
    calculates the true feasible minimum and maximum
    return portfolios.
    """

    n = len(mu)

    # Find minimum-return feasible portfolio
    min_result = minimize(
        lambda w: w @ mu,
        _initial_weights(n, max_weight),
        method="SLSQP",
        bounds=_bounds(n, max_weight),
        constraints=_constraints(n),
        options={
            "maxiter": 2000,
            "ftol": 1e-9
        }
    )

    # Find maximum-return feasible portfolio
    max_result = minimize(
        lambda w: -(w @ mu),
        _initial_weights(n, max_weight),
        method="SLSQP",
        bounds=_bounds(n, max_weight),
        constraints=_constraints(n),
        options={
            "maxiter": 2000,
            "ftol": 1e-9
        }
    )

    if not min_result.success or not max_result.success:
        return []

    min_return = float(min_result.x @ mu)
    max_return = float(max_result.x @ mu)

    target_returns = np.linspace(
        min_return,
        max_return,
        points
    )

    frontier = []

    for target in target_returns:

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            },
            {
                "type": "eq",
                "fun": lambda w, t=target:
                    w @ mu - t
            }
        ]

        result = minimize(
            lambda w: portfolio_volatility(w, cov),
            _initial_weights(n, max_weight),
            method="SLSQP",
            bounds=_bounds(n, max_weight),
            constraints=constraints,
            options={
                "maxiter": 3000,
                "ftol": 1e-8
            }
        )

        if result.success:

            frontier.append({
                "Return": target,
                "Volatility": portfolio_volatility(
                    result.x,
                    cov
                ),
                "Weights": result.x
            })

    return frontier