# 1 Import libraries
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 2️ Load datasets
btc = pd.read_csv("btc_full_dataset_with_indicators.csv", parse_dates=["Date"])
eth = pd.read_csv("eth_full_dataset_with_indicators.csv", parse_dates=["Date"])

# Ensure numeric Close column
btc["Close"] = pd.to_numeric(btc["Close"], errors="coerce")
eth["Close"] = pd.to_numeric(eth["Close"], errors="coerce")


# 3️ Prepare data for Prophet (must be ds & y)
btc_df = btc[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
eth_df = eth[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})


# 4️ Train Prophet models
btc_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
eth_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)

btc_model.fit(btc_df)
eth_model.fit(eth_df)


# 5️ Create future dataframe — until end of 2027
# Prophet forecasts in days, so for ~2 years → 730 days
# You can extend this (e.g., 1000 days) for more.

future_periods = 730  # ≈ 2 years ahead

btc_future = btc_model.make_future_dataframe(periods=future_periods)
eth_future = eth_model.make_future_dataframe(periods=future_periods)

btc_forecast = btc_model.predict(btc_future)
eth_forecast = eth_model.predict(eth_future)


# 6️ Plot BTC and ETH Forecasts

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Bitcoin (BTC) Forecast until 2027", "Ethereum (ETH) Forecast until 2027"),
    shared_xaxes=True,
    vertical_spacing=0.12
)

# BTC
fig.add_trace(go.Scatter(
    x=btc_df["ds"], y=btc_df["y"],
    mode="lines", name="BTC Actual",
    line=dict(color="orange", width=2)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=btc_forecast["ds"], y=btc_forecast["yhat"],
    mode="lines", name="BTC Forecast",
    line=dict(color="blue", width=2, dash="dot")
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=btc_forecast["ds"], y=btc_forecast["yhat_upper"],
    mode="lines", name="BTC Upper CI",
    line=dict(color="lightblue", width=0.8),
    showlegend=False
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=btc_forecast["ds"], y=btc_forecast["yhat_lower"],
    mode="lines", name="BTC Lower CI",
    line=dict(color="lightblue", width=0.8),
    fill="tonexty", fillcolor="rgba(173,216,230,0.2)",
    showlegend=False
), row=1, col=1)

# ETH
fig.add_trace(go.Scatter(
    x=eth_df["ds"], y=eth_df["y"],
    mode="lines", name="ETH Actual",
    line=dict(color="red", width=2)
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=eth_forecast["ds"], y=eth_forecast["yhat"],
    mode="lines", name="ETH Forecast",
    line=dict(color="green", width=2, dash="dot")
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=eth_forecast["ds"], y=eth_forecast["yhat_upper"],
    mode="lines", name="ETH Upper CI",
    line=dict(color="lightgreen", width=0.8),
    showlegend=False
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=eth_forecast["ds"], y=eth_forecast["yhat_lower"],
    mode="lines", name="ETH Lower CI",
    line=dict(color="lightgreen", width=0.8),
    fill="tonexty", fillcolor="rgba(144,238,144,0.2)",
    showlegend=False
), row=2, col=1)


# 7️ Layout & Display
fig.update_layout(
    title=dict(
        text="Bitcoin & Ethereum Price Forecast (Prophet Model) — 2026 to 2027",
        x=0.5, font=dict(size=20)
    ),
    width=1300, height=900,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=-0.15,
        xanchor="center", x=0.5
    )
)

fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Price (USD)", row=2, col=1)




# 8️ Save as HTML
# Save as HTML
fig.write_html("forecast_btc_eth_using_prophet_model.html")

# Show plot
fig.show()