import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 1️ Load datasets
btc = pd.read_csv("btc_full_dataset_with_indicators.csv")
eth = pd.read_csv("eth_full_dataset_with_indicators.csv")

btc["Date"] = pd.to_datetime(btc["Date"])
eth["Date"] = pd.to_datetime(eth["Date"])

# Ensure numeric columns
for col in ["Close", "Volume", "AdrActCnt", "Inflation", "USD_LBP"]:
    btc[col] = pd.to_numeric(btc[col], errors="coerce")
    eth[col] = pd.to_numeric(eth[col], errors="coerce")


# 2️ Monthly smoothing for Inflation & USD_LBP
btc_monthly = btc.set_index("Date").resample("M").mean(numeric_only=True).reset_index()
eth_monthly = eth.set_index("Date").resample("M").mean(numeric_only=True).reset_index()

indicators = ["Volume", "AdrActCnt", "Inflation", "USD_LBP"]


# 3️ Create subplots (2x2 grid, each with secondary Y-axis)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=indicators,
    specs=[[{"secondary_y": True}, {"secondary_y": True}],
           [{"secondary_y": True}, {"secondary_y": True}]],
    vertical_spacing=0.18,
    horizontal_spacing=0.10
)

for i, ind in enumerate(indicators):
    row = i // 2 + 1
    col = i % 2 + 1

    btc_df = btc_monthly if ind in ["Inflation", "USD_LBP"] else btc
    eth_df = eth_monthly if ind in ["Inflation", "USD_LBP"] else eth

    # BTC Close
    fig.add_trace(
        go.Scatter(
            x=btc_df["Date"], y=btc_df["Close"],
            mode="lines",
            name="BTC Close",
            line=dict(color="orange", width=1.8),
            hovertemplate="Date: %{x}<br>BTC Close: %{y:.2f}<extra></extra>"
        ),
        row=row, col=col, secondary_y=False
    )

    # ETH Close
    fig.add_trace(
        go.Scatter(
            x=eth_df["Date"], y=eth_df["Close"],
            mode="lines",
            name="ETH Close",
            line=dict(color="red", width=1.8),
            hovertemplate="Date: %{x}<br>ETH Close: %{y:.2f}<extra></extra>"
        ),
        row=row, col=col, secondary_y=False
    )

    # BTC Indicator
    fig.add_trace(
        go.Scatter(
            x=btc_df["Date"], y=btc_df[ind],
            mode="lines",
            name=f"BTC {ind}",
            line=dict(color="blue", width=1, dash="dot"),
            opacity=0.7,
            hovertemplate=f"Date: %{{x}}<br>BTC {ind}: %{{y:.2f}}<extra></extra>"
        ),
        row=row, col=col, secondary_y=True
    )

    # ETH Indicator
    fig.add_trace(
        go.Scatter(
            x=eth_df["Date"], y=eth_df[ind],
            mode="lines",
            name=f"ETH {ind}",
            line=dict(color="green", width=1, dash="dot"),
            opacity=0.7,
            hovertemplate=f"Date: %{{x}}<br>ETH {ind}: %{{y:.2f}}<extra></extra>"
        ),
        row=row, col=col, secondary_y=True
    )


# 4️ Layout & Styling
fig.update_layout(
    title=dict(
        text="BTC vs ETH — Relationship with Economic and Demand Indicators (2018–2025)",
        x=0.5, xanchor="center", font=dict(size=20)
    ),
    width=1500, height=1100,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.05,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="lightgray",
        borderwidth=1
    ),
    margin=dict(l=80, r=200, t=100, b=100)
)

fig.update_xaxes(title_text="Date", showgrid=True, gridcolor="lightgray")
fig.update_yaxes(title_text="Close Price (USD)", secondary_y=False)
fig.update_yaxes(title_text="Indicator Value", secondary_y=True)


# 5️ Show figure
fig.show()

# 6️ Save figure as HTML (portable path)
fig.write_html("btc_eth_indicators1_plot.html")