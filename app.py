import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data import download_prices, returns_from_prices
from src.optimization import (
    minimum_variance,
    maximum_sharpe,
    maximum_sortino,
    risk_parity,
    efficient_frontier,
)
from src.backtest import walk_forward_backtest
from src.metrics import (
    metrics_table,
    portfolio_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    var_historical,
    cvar_historical,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Portfolio Optimization & Risk Management",
    page_icon="📈",
    layout="wide",
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

STRATEGIES = [
    "Equal Weight",
    "Minimum Volatility",
    "Maximum Sharpe",
    "Maximum Sortino",
    "Risk Parity",
]


def classify_candidate_regime(volatility, drawdown):
    """Classify a day using rolling volatility and drawdown."""
    if volatility >= 0.25 or drawdown <= -0.15:
        return "High Risk"

    if volatility >= 0.15 or drawdown <= -0.05:
        return "Medium Risk"

    return "Low Risk"


def build_confirmed_regimes(
    rolling_volatility,
    drawdown,
    confirmation_days=5,
):
    """
    Convert noisy daily risk signals into persistent regimes.

    A new regime must remain the candidate regime for
    `confirmation_days` consecutive observations before
    the confirmed regime changes.
    """
    frame = pd.DataFrame({
        "Rolling Volatility": rolling_volatility,
        "Drawdown": drawdown,
    }).dropna()

    if frame.empty:
        return frame

    frame["Candidate Regime"] = [
        classify_candidate_regime(vol, dd)
        for vol, dd in zip(
            frame["Rolling Volatility"],
            frame["Drawdown"],
        )
    ]

    confirmed = []
    current = frame["Candidate Regime"].iloc[0]
    candidate = current
    candidate_count = 0

    for regime in frame["Candidate Regime"]:
        if regime == current:
            candidate = current
            candidate_count = 0
            confirmed.append(current)
            continue

        if regime == candidate:
            candidate_count += 1
        else:
            candidate = regime
            candidate_count = 1

        if candidate_count >= confirmation_days:
            current = candidate
            candidate_count = 0

        confirmed.append(current)

    frame["Risk Regime"] = confirmed
    return frame


def regime_recommendation(regime):
    """
    Map the detected risk regime to one of the already
    optimized portfolios. This is a decision-support layer,
    not an automatic trading signal.
    """
    mapping = {
        "Low Risk": (
            "Maximum Sharpe",
            "Normal risk conditions: prioritize risk-adjusted return.",
        ),
        "Medium Risk": (
            "Risk Parity",
            "Elevated risk: favor balanced exposure across risk sources.",
        ),
        "High Risk": (
            "Minimum Volatility",
            "High risk: favor the most defensive available allocation.",
        ),
    }

    return mapping[regime]


def rolling_cvar(series, window=63, confidence=0.95):
    """Historical rolling CVaR / Expected Shortfall."""
    values = []

    for i in range(len(series)):
        if i < window:
            values.append(np.nan)
            continue

        window_data = series.iloc[i - window:i]
        cutoff = np.quantile(
            window_data,
            1 - confidence,
        )
        tail = window_data[
            window_data <= cutoff
        ]

        values.append(
            -tail.mean()
            if len(tail)
            else np.nan
        )

    return pd.Series(
        values,
        index=series.index,
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "📈 Portfolio Optimization & Risk Management System"
)

st.caption(
    "Constrained optimization, downside-risk analysis "
    "and walk-forward out-of-sample backtesting"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Market Setup")

    ticker_text = st.text_input(
        "Tickers",
        "AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,JNJ,XOM",
    )

    start = st.date_input(
        "Start date",
        pd.Timestamp("2018-01-01"),
    )

    end = st.date_input(
        "End date",
        pd.Timestamp.today(),
    )

    max_weight = st.slider(
        "Maximum weight per asset",
        min_value=0.10,
        max_value=1.00,
        value=0.35,
        step=0.05,
    )

    risk_free_rate = st.number_input(
        "Annual risk-free rate",
        min_value=0.0,
        max_value=0.20,
        value=0.04,
        step=0.005,
        format="%.3f",
    )

    transaction_cost = st.number_input(
        "Transaction cost (bps)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )

    run = st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True,
    )


if not run:
    st.info(
        "Choose your portfolio universe and click "
        "**Run Analysis**."
    )
    st.stop()


# =========================================================
# MAIN ANALYSIS
# =========================================================

try:

    # -----------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------

    if pd.Timestamp(start) >= pd.Timestamp(end):
        st.error("Start date must be earlier than end date.")
        st.stop()

    tickers = [
        ticker.strip().upper()
        for ticker in ticker_text.split(",")
        if ticker.strip()
    ]

    if len(tickers) < 2:
        st.error("Please enter at least two ticker symbols.")
        st.stop()

    if len(set(tickers)) != len(tickers):
        st.error(
            "Duplicate ticker symbols detected. "
            "Please enter each ticker only once."
        )
        st.stop()

    minimum_possible_weight = 1.0 / len(tickers)

    if max_weight < minimum_possible_weight:
        st.error(
            f"Maximum weight of {max_weight:.2%} is infeasible "
            f"for {len(tickers)} assets. It must be at least "
            f"{minimum_possible_weight:.2%}."
        )
        st.stop()


    # -----------------------------------------------------
    # Download and validate market data
    # -----------------------------------------------------

    with st.spinner("Downloading historical market data..."):
        prices = download_prices(
            tickers,
            start,
            end,
        )

    returns = returns_from_prices(prices)

    if len(returns) < 260:
        st.error(
            "The selected period contains fewer than 260 "
            "valid trading observations. Please select a "
            "longer history for reliable optimization and backtesting."
        )
        st.stop()

    if len(returns) < 600:
        st.warning(
            "The selected period has limited observations. "
            "Results may be less stable than a multi-year study."
        )

    if not np.isfinite(returns.to_numpy()).all():
        st.error(
            "The downloaded return matrix contains invalid values."
        )
        st.stop()


    # -----------------------------------------------------
    # Full-sample statistics
    # -----------------------------------------------------

    mu = returns.mean().values * 252
    cov = returns.cov().values * 252

    if (
        not np.isfinite(mu).all()
        or not np.isfinite(cov).all()
    ):
        st.error(
            "Expected returns or covariance estimates are invalid. "
            "Try a longer date range or a different asset universe."
        )
        st.stop()


    # =====================================================
    # PORTFOLIO OPTIMIZATION
    # =====================================================

    with st.spinner("Optimizing portfolios..."):

        equal_weight = (
            np.ones(len(tickers))
            / len(tickers)
        )

        min_volatility = minimum_variance(
            mu,
            cov,
            max_weight,
        )

        max_sharpe = maximum_sharpe(
            mu,
            cov,
            risk_free_rate,
            max_weight,
        )

        max_sortino = maximum_sortino(
            returns,
            risk_free_rate,
            max_weight,
        )

        risk_parity_weights = risk_parity(
            cov,
            max_weight,
        )

    weights = {
        "Equal Weight": equal_weight,
        "Minimum Volatility": min_volatility,
        "Maximum Sharpe": max_sharpe,
        "Maximum Sortino": max_sortino,
        "Risk Parity": risk_parity_weights,
    }


    # =====================================================
    # PORTFOLIO ALLOCATIONS
    # =====================================================

    st.subheader("Optimized Portfolio Allocations")

    weight_df = pd.DataFrame(
        weights,
        index=tickers,
    ).T

    st.dataframe(
        weight_df.style.format("{:.2%}"),
        use_container_width=True,
    )


    # =====================================================
    # PORTFOLIO INSPECTION
    # =====================================================

    selected = st.selectbox(
        "Inspect portfolio",
        STRATEGIES,
    )

    selected_weights = weights[selected]

    selected_returns = portfolio_returns(
        returns,
        selected_weights,
    )

    selected_annual_return = annualized_return(
        selected_returns
    )

    selected_annual_volatility = annualized_volatility(
        selected_returns
    )

    selected_sharpe = sharpe_ratio(
        selected_returns,
        risk_free_rate,
    )

    selected_drawdown = max_drawdown(
        selected_returns
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Annual Return",
        f"{selected_annual_return:.2%}",
    )

    c2.metric(
        "Volatility",
        f"{selected_annual_volatility:.2%}",
    )

    c3.metric(
        "Sharpe",
        f"{selected_sharpe:.2f}",
    )

    c4.metric(
        "Max Drawdown",
        f"{selected_drawdown:.2%}",
    )


    # =====================================================
    # EFFICIENT FRONTIER
    # =====================================================

    st.subheader("Efficient Frontier")

    with st.spinner("Calculating efficient frontier..."):
        frontier = efficient_frontier(
            mu,
            cov,
            points=60,
            max_weight=max_weight,
        )

    if frontier:

        frontier_df = pd.DataFrame({
            "Return": [
                item["Return"]
                for item in frontier
            ],
            "Volatility": [
                item["Volatility"]
                for item in frontier
            ],
        })

        fig_frontier = px.scatter(
            frontier_df,
            x="Volatility",
            y="Return",
            title="Constrained Efficient Frontier",
        )

        for name, w in weights.items():

            portfolio_return_value = float(
                w @ mu
            )

            portfolio_volatility_value = float(
                np.sqrt(
                    max(
                        w @ cov @ w,
                        0.0,
                    )
                )
            )

            fig_frontier.add_trace(
                go.Scatter(
                    x=[portfolio_volatility_value],
                    y=[portfolio_return_value],
                    mode="markers+text",
                    text=[name],
                    textposition="top center",
                    name=name,
                )
            )

        fig_frontier.update_xaxes(
            tickformat=".0%"
        )

        fig_frontier.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            fig_frontier,
            use_container_width=True,
        )

    else:
        st.warning(
            "The efficient frontier could not be calculated "
            "for the selected constraints."
        )


    # =====================================================
    # WALK-FORWARD BACKTEST
    # =====================================================

    st.subheader(
        "Walk-Forward Out-of-Sample Backtest"
    )

    lookback_days = min(
        504,
        max(
            252,
            len(returns) // 3,
        ),
    )

    rebalance_frequency = 21

    st.caption(
        f"Training window: {lookback_days} trading days | "
        f"Rebalance frequency: every {rebalance_frequency} trading days | "
        f"Transaction cost: {transaction_cost:.1f} bps"
    )

    all_strategy_returns = {}

    with st.spinner("Running walk-forward backtests..."):

        for strategy in STRATEGIES:

            result = walk_forward_backtest(
                returns,
                strategy,
                lookback_days=lookback_days,
                rebalance_frequency=rebalance_frequency,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                transaction_cost_bps=transaction_cost,
            )

            if result.empty:
                raise RuntimeError(
                    f"{strategy} produced no out-of-sample results."
                )

            all_strategy_returns[strategy] = result


    # =====================================================
    # BACKTEST PERFORMANCE
    # =====================================================

    table = metrics_table(
        all_strategy_returns
    )

    st.dataframe(
        table.style
        .format(
            "{:.2%}",
            subset=[
                "CAGR",
                "Volatility",
                "Max Drawdown",
                "VaR 95%",
                "CVaR 95%",
            ],
        )
        .format(
            "{:.2f}",
            subset=[
                "Sharpe",
                "Sortino",
                "Calmar",
            ],
        ),
        use_container_width=True,
    )


    # -----------------------------------------------------
    # Automatically identify strongest risk-adjusted result
    # -----------------------------------------------------

    best_strategy = table["Sharpe"].idxmax()
    best_sharpe = table.loc[
        best_strategy,
        "Sharpe",
    ]

    st.info(
        f"Highest out-of-sample Sharpe in this run: "
        f"**{best_strategy} ({best_sharpe:.2f})**. "
        "This is a historical comparison, not a guarantee of future performance."
    )


    # =====================================================
    # CUMULATIVE WEALTH
    # =====================================================

    st.subheader("Cumulative Wealth")

    equity = pd.DataFrame({
        name: (series + 1).cumprod()
        for name, series
        in all_strategy_returns.items()
    }).dropna(
        how="all"
    )

    fig_equity = px.line(
        equity,
        title="Out-of-Sample Cumulative Wealth",
        labels={
            "value": "Portfolio Value",
            "index": "Date",
        },
    )

    st.plotly_chart(
        fig_equity,
        use_container_width=True,
    )


    # =====================================================
    # RISK ANALYSIS
    # =====================================================

    st.subheader("Risk Analysis")

    selected_bt = st.selectbox(
        "Risk analysis strategy",
        STRATEGIES,
        key="risk_strategy",
    )

    risk_series = (
        all_strategy_returns[selected_bt]
        .dropna()
    )

    risk_volatility = annualized_volatility(
        risk_series
    )

    risk_drawdown = max_drawdown(
        risk_series
    )

    risk_var = var_historical(
        risk_series,
        confidence=0.95,
    )

    risk_cvar = cvar_historical(
        risk_series,
        confidence=0.95,
    )

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Annualized Volatility",
        f"{risk_volatility:.2%}",
    )

    r2.metric(
        "Maximum Drawdown",
        f"{risk_drawdown:.2%}",
    )

    r3.metric(
        "VaR 95%",
        f"{risk_var:.2%}",
    )

    r4.metric(
        "CVaR 95%",
        f"{risk_cvar:.2%}",
    )


    # =====================================================
    # ROLLING VOLATILITY
    # =====================================================

    st.markdown("### Rolling Volatility")

    rolling_volatility = (
        risk_series
        .rolling(63)
        .std()
        * np.sqrt(252)
    )

    volatility_df = pd.DataFrame({
        "Rolling Volatility":
            rolling_volatility
    }).dropna()

    fig_volatility = px.line(
        volatility_df,
        y="Rolling Volatility",
        title="63-Day Rolling Annualized Volatility",
        labels={
            "value": "Rolling Volatility",
            "index": "Date",
        },
    )

    fig_volatility.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig_volatility,
        use_container_width=True,
    )


    # =====================================================
    # DRAWDOWN
    # =====================================================

    st.markdown("### Drawdown")

    wealth = (
        1 + risk_series
    ).cumprod()

    running_peak = wealth.cummax()

    drawdown = (
        wealth
        / running_peak
        - 1
    )

    drawdown_df = pd.DataFrame({
        "Drawdown": drawdown
    })

    fig_drawdown = px.area(
        drawdown_df,
        y="Drawdown",
        title="Portfolio Drawdown",
        labels={
            "value": "Drawdown",
            "index": "Date",
        },
    )

    fig_drawdown.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig_drawdown,
        use_container_width=True,
    )


    # =====================================================
    # TAIL RISK
    # =====================================================

    st.markdown("### Tail Risk")

    rolling_cvar_values = rolling_cvar(
        risk_series,
        window=63,
        confidence=0.95,
    )

    cvar_df = pd.DataFrame({
        "Rolling CVaR 95%":
            rolling_cvar_values
    }).dropna()

    fig_cvar = px.line(
        cvar_df,
        y="Rolling CVaR 95%",
        title="63-Day Rolling CVaR at 95% Confidence",
        labels={
            "value": "Expected Tail Loss",
            "index": "Date",
        },
    )

    fig_cvar.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig_cvar,
        use_container_width=True,
    )


    # =====================================================
    # RISK REGIME DETECTION
    # =====================================================

    st.markdown("### Risk Regime Detection")

    CONFIRMATION_DAYS = 5

    regime_df = build_confirmed_regimes(
        rolling_volatility,
        drawdown,
        confirmation_days=CONFIRMATION_DAYS,
    )

    if regime_df.empty:
        st.warning(
            "Not enough observations to determine risk regimes."
        )
        st.stop()

    current_regime = regime_df[
        "Risk Regime"
    ].iloc[-1]

    current_volatility = regime_df[
        "Rolling Volatility"
    ].iloc[-1]

    current_drawdown = regime_df[
        "Drawdown"
    ].iloc[-1]

    st.markdown(
        f"#### Current Risk Regime — {current_regime}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current Regime",
        current_regime,
    )

    c2.metric(
        "Rolling Volatility",
        f"{current_volatility:.2%}",
    )

    c3.metric(
        "Current Drawdown",
        f"{current_drawdown:.2%}",
    )


    # -----------------------------------------------------
    # Historical regime chart
    # -----------------------------------------------------

    st.markdown("#### Historical Risk Regimes")

    regime_numeric = regime_df[
        "Risk Regime"
    ].map({
        "Low Risk": 1,
        "Medium Risk": 2,
        "High Risk": 3,
    })

    regime_plot_df = pd.DataFrame({
        "Risk Regime": regime_numeric
    })

    fig_regime = px.line(
        regime_plot_df,
        y="Risk Regime",
        title="Historical Risk Regime",
        labels={
            "value": "Risk Level",
            "index": "Date",
        },
    )

    fig_regime.update_yaxes(
        tickmode="array",
        tickvals=[1, 2, 3],
        ticktext=[
            "Low Risk",
            "Medium Risk",
            "High Risk",
        ],
    )

    st.plotly_chart(
        fig_regime,
        use_container_width=True,
    )


    # -----------------------------------------------------
    # Regime distribution
    # -----------------------------------------------------

    regime_counts = (
        regime_df["Risk Regime"]
        .value_counts()
        .reindex(
            [
                "Low Risk",
                "Medium Risk",
                "High Risk",
            ],
            fill_value=0,
        )
    )

    regime_percentages = (
        regime_counts
        / regime_counts.sum()
        * 100
    )

    regime_distribution = pd.DataFrame({
        "Regime": regime_counts.index,
        "Percentage": regime_percentages.values,
    })

    fig_distribution = px.bar(
        regime_distribution,
        x="Regime",
        y="Percentage",
        title="Historical Regime Distribution",
        labels={
            "Percentage":
                "Percentage of Trading Days",
            "Regime":
                "Risk Regime",
        },
    )

    fig_distribution.update_yaxes(
        ticksuffix="%"
    )

    st.plotly_chart(
        fig_distribution,
        use_container_width=True,
    )


    # =====================================================
    # REGIME-AWARE RECOMMENDATION
    # =====================================================

    st.markdown("### Regime-Aware Portfolio Recommendation")

    recommended_strategy, recommendation_reason = (
        regime_recommendation(current_regime)
    )

    recommended_weights = weights[
        recommended_strategy
    ]

    rc1, rc2 = st.columns([1, 2])

    with rc1:
        st.metric(
            "Recommended Strategy",
            recommended_strategy,
        )

    with rc2:
        st.info(
            recommendation_reason
            + " The system reuses the constrained optimizer "
            "results above rather than creating an untested "
            "allocation rule."
        )

    recommendation_df = pd.DataFrame({
        "Asset": tickers,
        "Recommended Weight": recommended_weights,
    })

    st.dataframe(
        recommendation_df.style.format(
            {"Recommended Weight": "{:.2%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


    # -----------------------------------------------------
    # Regime message
    # -----------------------------------------------------

    if current_regime == "High Risk":
        st.error(
            "Current conditions are classified as HIGH RISK. "
            "The system therefore recommends the Minimum "
            "Volatility portfolio as the defensive choice."
        )

    elif current_regime == "Medium Risk":
        st.warning(
            "Current conditions are classified as MEDIUM RISK. "
            "The system therefore recommends Risk Parity to "
            "balance exposure across risk sources."
        )

    else:
        st.success(
            "Current conditions are classified as LOW RISK. "
            "The system therefore recommends Maximum Sharpe "
            "to prioritize risk-adjusted return."
        )

    st.caption(
        f"Regime changes require {CONFIRMATION_DAYS} "
        "consecutive trading days of confirmation."
    )


    # =====================================================
    # RISK INTERPRETATION
    # =====================================================

    st.markdown("### Risk Interpretation")

    st.info(
        f"""
**{selected_bt}** currently has:

- **Annualized volatility:** {risk_volatility:.2%}
- **Maximum drawdown:** {risk_drawdown:.2%}
- **95% VaR:** {risk_var:.2%}
- **95% CVaR:** {risk_cvar:.2%}

CVaR measures the average loss in the worst 5% of
historical daily returns, making it useful for understanding
tail losses beyond the VaR threshold.
"""
    )


    # =====================================================
    # FINAL STATUS
    # =====================================================

    st.success(
        "Analysis complete. Portfolio weights are optimized "
        "using historical training data, while walk-forward "
        "performance is evaluated on subsequent unseen data."
    )


except Exception as exc:

    st.error(
        f"Analysis failed: {exc}"
    )

    st.info(
        "Check ticker symbols, dates, internet connectivity, "
        "data availability, and whether the maximum-weight "
        "constraint is feasible."
    )
