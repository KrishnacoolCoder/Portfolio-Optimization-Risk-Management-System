import numpy as np

from src.optimization import minimum_variance, maximum_sharpe, risk_parity


def sample_problem():
    mu = np.array([0.08, 0.10, 0.12])
    cov = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.05, 0.008],
        [0.005, 0.008, 0.09],
    ])
    return mu, cov


def test_minimum_variance_weights_sum_to_one():
    mu, cov = sample_problem()
    w = minimum_variance(mu, cov, 0.8)
    assert abs(w.sum() - 1) < 1e-6
    assert np.all(w >= -1e-8)


def test_max_sharpe_weights_sum_to_one():
    mu, cov = sample_problem()
    w = maximum_sharpe(mu, cov, 0.03, 0.8)
    assert abs(w.sum() - 1) < 1e-6


def test_risk_parity_weights_sum_to_one():
    _, cov = sample_problem()
    w = risk_parity(cov, 0.8)
    assert abs(w.sum() - 1) < 1e-6
