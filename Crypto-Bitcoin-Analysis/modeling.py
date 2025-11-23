
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression

#Load datasets
btc = pd.read_csv("btc_full_dataset_with_indicators.csv", parse_dates=['Date'])
eth = pd.read_csv("eth_full_dataset_with_indicators.csv", parse_dates=['Date'])

btc = btc.sort_values('Date').reset_index(drop=True)
eth = eth.sort_values('Date').reset_index(drop=True)

btc[['Close', 'SplyCur', 'AdrActCnt']] = btc[['Close', 'SplyCur', 'AdrActCnt']].ffill()
eth[['Close', 'SplyCur', 'AdrActCnt']] = eth[['Close', 'SplyCur', 'AdrActCnt']].ffill()

#Compute Elasticity
def compute_elasticity(df):
    X = df[['SplyCur', 'AdrActCnt']]
    y = df['Close']
    model = LinearRegression().fit(X, y)
    return model.coef_

btc_coef = compute_elasticity(btc)
eth_coef = compute_elasticity(eth)
print("BTC Elasticity:", btc_coef)
print("ETH Elasticity:", eth_coef)

#Prophet Forecast
def forecast(df):
    df_prophet = df[['Date', 'Close', 'SplyCur', 'AdrActCnt']].rename(
        columns={'Date': 'ds', 'Close': 'y', 'SplyCur': 'Supply', 'AdrActCnt': 'Demand'})
    model = Prophet()
    model.add_regressor('Supply')
    model.add_regressor('Demand')
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    future['Supply'] = df_prophet['Supply'].iloc[-1]
    future['Demand'] = df_prophet['Demand'].iloc[-1]
    forecast = model.predict(future)
    return forecast

btc_forecast = forecast(btc)
eth_forecast = forecast(eth)

#Define key events
events = [
    {"date": "2020-03-12", "label": "COVID-19 Crash", "details": "Global market crash due to pandemic"},
    {"date": "2020-05-11", "label": "BTC Halving 3", "details": "Bitcoin reward halved from 12.5 to 6.25 BTC"},
    {"date": "2022-11-10", "label": "Bear Market", "details": "Major cryptocurrency market downturn"}
]

# Plotly Figure with dual y-axis
fig = go.Figure()

# BTC Actual
fig.add_trace(go.Scatter(
    x=btc['Date'], y=btc['Close'], mode='lines+markers', name='BTC Actual',
    line=dict(color='gold'), marker=dict(size=6, opacity=0.7),
    hovertemplate='Date: %{x}<br>BTC Price: %{y:.2f}<br>Supply: %{customdata[0]:.2f}<br>Demand: %{customdata[1]:.0f}',
    customdata=btc[['SplyCur', 'AdrActCnt']],
    yaxis='y1'
))

# BTC Forecast
fig.add_trace(go.Scatter(
    x=btc_forecast['ds'], y=btc_forecast['yhat'], mode='lines+markers', name='BTC Forecast',
    line=dict(color='orange', dash='dot'), marker=dict(size=4, opacity=0.5),
    hovertemplate='Date: %{x}<br>Forecast Price: %{y:.2f}',
    yaxis='y1'
))

# BTC Confidence Interval
fig.add_trace(go.Scatter(
    x=btc_forecast['ds'], y=btc_forecast['yhat_upper'], mode='lines',
    line=dict(width=0), showlegend=False, yaxis='y1'
))
fig.add_trace(go.Scatter(
    x=btc_forecast['ds'], y=btc_forecast['yhat_lower'], mode='lines',
    fill='tonexty', fillcolor='rgba(255,165,0,0.2)',
    line=dict(width=0), showlegend=True, name='BTC Forecast CI',
    yaxis='y1'
))

# ETH Actual (secondary y-axis)
fig.add_trace(go.Scatter(
    x=eth['Date'], y=eth['Close'], mode='lines+markers', name='ETH Actual',
    line=dict(color='blue'), marker=dict(size=6, opacity=0.7),
    hovertemplate='Date: %{x}<br>ETH Price: %{y:.2f}<br>Supply: %{customdata[0]:.2f}<br>Demand: %{customdata[1]:.0f}',
    customdata=eth[['SplyCur', 'AdrActCnt']],
    yaxis='y2'
))

# ETH Forecast (secondary y-axis)
fig.add_trace(go.Scatter(
    x=eth_forecast['ds'], y=eth_forecast['yhat'], mode='lines+markers', name='ETH Forecast',
    line=dict(color='lightblue', dash='dot'), marker=dict(size=4, opacity=0.5),
    hovertemplate='Date: %{x}<br>Forecast Price: %{y:.2f}',
    yaxis='y2'
))

# ETH Confidence Interval
fig.add_trace(go.Scatter(
    x=eth_forecast['ds'], y=eth_forecast['yhat_upper'], mode='lines',
    line=dict(width=0), showlegend=False, yaxis='y2'
))
fig.add_trace(go.Scatter(
    x=eth_forecast['ds'], y=eth_forecast['yhat_lower'], mode='lines',
    fill='tonexty', fillcolor='rgba(173,216,230,0.2)',
    line=dict(width=0), showlegend=True, name='ETH Forecast CI',
    yaxis='y2'
))

# Events for BTC and ETH with corresponding axes
for event in events:
    event_date = pd.to_datetime(event["date"])
    closest_btc = btc.iloc[(btc['Date'] - event_date).abs().argsort()[:1]]['Close'].values[0]
    closest_eth = eth.iloc[(eth['Date'] - event_date).abs().argsort()[:1]]['Close'].values[0]

    # BTC marker and guideline
    fig.add_trace(go.Scatter(
        x=[event_date], y=[closest_btc], mode="markers",
        marker=dict(color='rgba(255,0,0,0.2)', size=20),
        showlegend=False, hoverinfo="skip", yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=[event_date], y=[closest_btc], mode="markers+text",
        marker=dict(color='red', size=14, line=dict(color="black", width=2)),
        text=[f"{event['label']} (BTC)"], textposition="top center",
        hovertext=[f"{event['details']}<br>BTC Price: {closest_btc:.2f} USD"],
        hoverinfo="text+name", name=f"{event['label']} (BTC)", yaxis='y1'
    ))
    fig.add_shape(type="line", x0=event_date, y0=0, x1=event_date, y1=closest_btc,
                  line=dict(color='red', width=1, dash="dot"), yref='y1')

    # ETH marker and guideline
    fig.add_trace(go.Scatter(
        x=[event_date], y=[closest_eth], mode="markers",
        marker=dict(color='rgba(128,0,255,0.2)', size=20),
        showlegend=False, hoverinfo="skip", yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=[event_date], y=[closest_eth], mode="markers+text",
        marker=dict(color='purple', size=14, line=dict(color="black", width=2)),
        text=[f"{event['label']} (ETH)"], textposition="top center",
        hovertext=[f"{event['details']}<br>ETH Price: {closest_eth:.2f} USD"],
        hoverinfo="text+name", name=f"{event['label']} (ETH)", yaxis='y2'
    ))
    fig.add_shape(type="line", x0=event_date, y0=0, x1=event_date, y1=closest_eth,
                  line=dict(color='purple', width=1, dash="dot"), yref='y2')

# Layout with dual y-axis
fig.update_layout(
    title="BTC & ETH Price: Interactive Forecast with Selected Events + Confidence Interval",
    xaxis=dict(title="Date", rangeslider=dict(visible=True), type="date"),
    yaxis=dict(title="BTC Price (USD)", side='left', showgrid=True, zeroline=False),
    yaxis2=dict(title="ETH Price (USD)", overlaying='y', side='right', showgrid=False, zeroline=False),
    template="plotly_dark",
    legend=dict(x=0.01, y=0.99)
)

fig.show()
fig.write_html("btc_eth_forecast_interactive_selected_events_CI_dual_y.html")