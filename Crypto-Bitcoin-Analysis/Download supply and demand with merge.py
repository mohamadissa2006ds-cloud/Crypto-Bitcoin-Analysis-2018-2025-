import pandas as pd
import requests
import time


# 1️ Coin Metrics data fetching function with retries

def fetch_coinmetrics(asset, metrics, filename, start="2018-01-01", end="2025-10-31", frequency="1d"):
    base_url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": asset,
        "metrics": ",".join(metrics),
        "start_time": start,
        "end_time": end,
        "frequency": frequency,
        "page_size": 10000,
        "format": "json"
    }
    print(f"Requesting {asset} metrics {metrics} from {start} to {end}")

    max_retries = 3
    for attempt in range(max_retries):
        resp = requests.get(base_url, params=params)
        if resp.status_code == 200:
            break
        print(f"Retry {attempt + 1}/{max_retries} for {asset} {metrics}")
        time.sleep(2)
    else:
        print(f"Failed to fetch {asset} {metrics} after {max_retries} attempts")
        return None

    j = resp.json()
    if "data" not in j:
        print(f"Warning: no 'data' field returned for {asset} {metrics}")
        return None

    df = pd.DataFrame(j["data"])
    df.rename(columns={"time": "Date"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df.to_csv(filename, index=False, float_format="%.8f")
    print(f"Saved {filename}")
    time.sleep(1)
    return df


# 2️ Fetch Supply-side and Demand-side data

btc_supply = fetch_coinmetrics("btc", ["SplyCur"], "btc_total_supply.csv")
eth_supply = fetch_coinmetrics("eth", ["SplyCur"], "eth_total_supply.csv")

# ETH burn data is not available for free
# eth_burn = fetch_coinmetrics("eth", ["BurnedValue"], "eth_burn.csv")

btc_active = fetch_coinmetrics("btc", ["AdrActCnt"], "btc_active_addresses.csv")
eth_active = fetch_coinmetrics("eth", ["AdrActCnt"], "eth_active_addresses.csv")


# 3️ Load price data from CSV, skipping the first 3 rows

columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

btc_price = pd.read_csv("bitcoin_dataset.csv", skiprows=3, names=columns)
eth_price = pd.read_csv("ethereum_dataset.csv", skiprows=3, names=columns)

# Convert Date column to datetime
btc_price['Date'] = pd.to_datetime(btc_price['Date']).dt.tz_localize(None)
eth_price['Date'] = pd.to_datetime(eth_price['Date']).dt.tz_localize(None)


# 4️ Merge data

btc = btc_price.merge(btc_supply, on='Date', how='outer').merge(btc_active, on='Date', how='outer')
eth = eth_price.merge(eth_supply, on='Date', how='outer').merge(eth_active, on='Date', how='outer')

# 5️ Convert numeric columns to avoid TypeError

for col in ['Close', 'High', 'Low', 'Open', 'Volume', 'SplyCur']:
    btc[col] = pd.to_numeric(btc[col], errors='coerce')
    eth[col] = pd.to_numeric(eth[col], errors='coerce')


# 6️ Calculate Market Cap and daily returns

btc['MarketCap'] = btc['Close'] * btc['SplyCur']
eth['MarketCap'] = eth['Close'] * eth['SplyCur']

btc['Return'] = btc['Close'].pct_change()
eth['Return'] = eth['Close'].pct_change()


# 7️ Save final datasets

btc.to_csv("btc_full_dataset.csv", index=False, float_format="%.8f")
eth.to_csv("eth_full_dataset.csv", index=False, float_format="%.8f")
print("Final files saved: btc_full_dataset.csv and eth_full_dataset.csv")
