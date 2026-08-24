import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end):
    """
    Download adjusted historical prices and validate the requested universe.

    Raises a clear error when dates are invalid, no data is returned,
    or one or more requested tickers have no usable price history.
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    if not tickers:
        raise ValueError("At least one ticker is required.")

    if len(set(tickers)) != len(tickers):
        raise ValueError("Duplicate ticker symbols were provided.")

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if start >= end:
        raise ValueError("Start date must be earlier than end date.")

    data = yf.download(
        tickers,
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError(
            "No market data was downloaded. Check the ticker symbols, "
            "dates, and internet connection."
        )

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)

        if "Close" in level0:
            prices = data["Close"].copy()
        elif "Adj Close" in level0:
            prices = data["Adj Close"].copy()
        else:
            raise ValueError(
                "Downloaded data does not contain a usable Close price field."
            )
    else:
        if "Close" in data.columns:
            prices = data[["Close"]].copy()
            prices.columns = [tickers[0]]
        elif "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()
            prices.columns = [tickers[0]]
        else:
            raise ValueError(
                "Downloaded data does not contain a usable Close price field."
            )

    prices = prices.reindex(columns=tickers)

    missing = [
        ticker
        for ticker in tickers
        if ticker not in prices.columns
        or prices[ticker].dropna().empty
    ]

    if missing:
        raise ValueError(
            "No usable price history was found for: "
            + ", ".join(missing)
            + ". Check the ticker symbols."
        )

    # Forward-fill only short internal gaps, then remove remaining rows
    # with missing observations so optimization never receives NaNs.
    prices = (
        prices
        .sort_index()
        .ffill()
        .dropna()
    )

    if len(prices) < 2:
        raise ValueError("Not enough valid price observations were downloaded.")

    return prices


def returns_from_prices(prices):
    if prices is None or prices.empty:
        raise ValueError("Price data is empty.")

    returns = prices.pct_change().replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if returns.empty:
        raise ValueError("Unable to calculate valid asset returns.")

    return returns


def annualized_statistics(returns, trading_days=252):
    if returns is None or returns.empty:
        raise ValueError("Returns data is empty.")

    mu = returns.mean() * trading_days
    cov = returns.cov() * trading_days

    if not np.isfinite(mu.to_numpy()).all():
        raise ValueError("Expected returns contain invalid values.")

    if not np.isfinite(cov.to_numpy()).all():
        raise ValueError("Covariance matrix contains invalid values.")

    return mu, cov


def annualized_return_from_daily(returns, trading_days=252):
    if len(returns) == 0:
        return np.nan

    return (
        (1 + returns).prod()
        ** (trading_days / len(returns))
        - 1
    )
