import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

#Load CSV files
btc = pd.read_csv("bitcoin_dataset.csv", skiprows=3)
eth = pd.read_csv("ethereum_dataset.csv", skiprows=3)
gold = pd.read_csv("gold_dataset.csv", skiprows=3)

# 2. Prepare datasets
for df in [btc, eth, gold]:
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop_duplicates(subset='Date', inplace=True)
    df.set_index('Date', inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Return'] = df['Close'].pct_change()

# Restrict to 2018–2025
btc = btc.loc["2018":"2025"].copy()
eth = eth.loc["2018":"2025"].copy()
gold = gold.loc["2018":"2025"].copy()

# 3. Key prices
btc_start, btc_max, btc_end = btc['Close'].iloc[0], btc['Close'].max(), btc['Close'].iloc[-1]
eth_start, eth_max, eth_end = eth['Close'].iloc[0], eth['Close'].max(), eth['Close'].iloc[-1]

# 4. Detect peaks
btc_peaks, _ = find_peaks(btc['Close'], distance=90, prominence=3000)
eth_peaks, _ = find_peaks(eth['Close'], distance=120, prominence=400)
eth_top_peaks = eth_peaks[np.argsort(eth['Close'].iloc[eth_peaks])[-3:]]

# 5. Plot Price Evolution
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(13,6))

plt.plot(btc.index, btc['Close'], label='Bitcoin', color='#F7931A', linewidth=2)
plt.plot(eth.index, eth['Close'], label='Ethereum', color='#3C3C3D', linewidth=2)

def annotate(text, x, y, color):
    plt.annotate(text, xy=(x, y), xytext=(0, 10),
                 textcoords="offset points", ha='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85, edgecolor=color),
                 color=color)

# Annotate BTC
annotate(f"${btc_start:,.0f}", btc.index[0], btc_start, '#F7931A')
annotate(f"MAX\n${btc_max:,.0f}", btc['Close'].idxmax(), btc_max, '#F7931A')
annotate(f"${btc_end:,.0f}", btc.index[-1], btc_end, '#F7931A')
for p in btc_peaks:
    annotate(f"${btc['Close'].iloc[p]:,.0f}", btc.index[p], btc['Close'].iloc[p], '#F7931A')

# Annotate ETH
annotate(f"${eth_start:,.0f}", eth.index[0], eth_start, '#3C3C3D')
annotate(f"MAX\n${eth_max:,.0f}", eth['Close'].idxmax(), eth_max, '#3C3C3D')
annotate(f"${eth_end:,.0f}", eth.index[-1], eth_end, '#3C3C3D')
for p in eth_top_peaks:
    annotate(f"${eth['Close'].iloc[p]:,.0f}", eth.index[p], eth['Close'].iloc[p], '#3C3C3D')

plt.title('Bitcoin vs Ethereum Price Evolution (2018–2025)', fontsize=14, weight='bold')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# 6. 30-Day Rolling Volatility with Shaded Events
btc['Volatility'] = btc['Return'].rolling(window=30).std() * np.sqrt(365)
eth['Volatility'] = eth['Return'].rolling(window=30).std() * np.sqrt(365)

plt.figure(figsize=(13,6))
plt.plot(btc.index, btc['Volatility'], label='BTC Volatility', color='#F7931A', linewidth=2)
plt.plot(eth.index, eth['Volatility'], label='ETH Volatility', color='#3C6EFA', linewidth=2)

#Shaded regions for major events with labels for legend
plt.axvspan(pd.to_datetime('2020-02-01'), pd.to_datetime('2021-06-30'), color='red', alpha=0.2, label='COVID Crash')
plt.axvspan(pd.to_datetime('2021-05-01'), pd.to_datetime('2022-06-30'), color='orange', alpha=0.15, label='Bear Market')
plt.axvspan(pd.to_datetime('2022-11-01'), pd.to_datetime('2022-11-30'), color='purple', alpha=0.2, label='FTX Collapse')

# Add text annotations
plt.text(pd.to_datetime('2020-07-01'), max(btc['Volatility'])*0.9, 'COVID Crash', color='red', fontsize=10)
plt.text(pd.to_datetime('2021-06-01'), max(btc['Volatility'])*0.8, 'Bear Market', color='orange', fontsize=10)
plt.text(pd.to_datetime('2022-11-05'), max(btc['Volatility'])*0.85, 'FTX Collapse', color='purple', fontsize=10)

plt.title('Bitcoin vs Ethereum Volatility (30-Day Rolling, 2018–2025)', fontsize=14, weight='bold')
plt.xlabel('Date')
plt.ylabel('Volatility (Annualized)')
plt.legend()
plt.xlim([pd.to_datetime('2018-01-01'), pd.to_datetime('2025-12-31')])
plt.tight_layout()
plt.show()

# 7. Correlation Heatmap (include Gold)
combined = pd.DataFrame({
    'Bitcoin_Return': btc['Return'],
    'Ethereum_Return': eth['Return'],
    'Gold_Return': gold['Return']
}).dropna()

plt.figure(figsize=(6,5))
corr = combined.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Between Daily Returns (BTC, ETH, Gold)', fontsize=13)
plt.show()

print("\n📊 Correlation matrix:\n", corr.round(2))
