import pandas as pd

# 1. Load the original CSV files
btc = pd.read_csv("bitcoin_dataset.csv", skiprows=3)
eth = pd.read_csv("ethereum_dataset.csv", skiprows=3)

#  2. Rename columns
for df in [btc, eth]:
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])

#  3. Calculate Daily Returns
btc['Returns'] = btc['Close'].pct_change()
eth['Returns'] = eth['Close'].pct_change()

#4. Save to new CSV files
btc.to_csv("bitcoin_dataset_with_returns.csv", index=False)
eth.to_csv("ethereum_dataset_with_returns.csv", index=False)

print("✅ Returns column added and new CSV files saved successfully!")
