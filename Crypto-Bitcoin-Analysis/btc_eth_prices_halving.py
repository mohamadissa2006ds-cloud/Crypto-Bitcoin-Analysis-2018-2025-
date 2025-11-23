import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load datasets
btc = pd.read_csv(r"btc_full_dataset_with_indicators.csv", parse_dates=['Date'])
eth = pd.read_csv(r"eth_full_dataset_with_indicators.csv", parse_dates=['Date'])

# Define BTC Halving Dates
halving_dates = ['2020-05-11', '2024-04-01']
halving_dates = pd.to_datetime(halving_dates)

# Create Figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Plot BTC Price (primary y-axis)
fig.add_trace(go.Scatter(
    x=btc['Date'], y=btc['Close'],
    mode='lines', name='BTC Price',
    line=dict(color='gold', width=2)
), secondary_y=False)

# Plot ETH Price (secondary y-axis)
fig.add_trace(go.Scatter(
    x=eth['Date'], y=eth['Close'],
    mode='lines', name='ETH Price',
    line=dict(color='blue', width=2)
), secondary_y=True)

# Add shaded regions for Halving
for date in halving_dates:
    fig.add_vrect(
        x0=date - pd.Timedelta(days=15),
        x1=date + pd.Timedelta(days=15),
        fillcolor="red",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="BTC Halving" if date == halving_dates[0] else "",
        annotation_position="top left"
    )

# Customize layout
fig.update_layout(
    title="Bitcoin and Ethereum Prices (2018-2025) with Halving Events",
    xaxis_title="Date",
    template="plotly_white",
    legend=dict(font=dict(size=12)),
    hovermode="x unified"
)

# Set y-axis titles
fig.update_yaxes(title_text="BTC Price (USD)", secondary_y=False)
fig.update_yaxes(title_text="ETH Price (USD)", secondary_y=True)

# Save as HTML
fig.write_html("btc_eth_prices_halving_secondary_y.html")

# Show plot
fig.show()
