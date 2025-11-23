

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytrends.request import TrendReq
import time

# 1️ Google Trends Setup

pytrends = TrendReq(hl='en-US', tz=360, retries=3, backoff_factor=0.5)
kw_list = ["Bitcoin", "Ethereum"]  # Keywords

trends_data_list = []

for keyword in kw_list:
    pytrends.build_payload([keyword], cat=0, timeframe='today 1-m', geo='', gprop='')
    data = pytrends.interest_over_time()
    if not data.empty and 'isPartial' in data.columns:
        data = data.drop(columns=['isPartial'])
    data.rename(columns={keyword: f"{keyword}"}, inplace=True)
    trends_data_list.append(data)
    time.sleep(1)

trends_data = pd.concat(trends_data_list, axis=1)


# 2️ Scale Data 0 → 1

for col in trends_data.columns:
    trends_data[col] = (trends_data[col] - trends_data[col].min()) / (trends_data[col].max() - trends_data[col].min())


# 3️ Set Colors
# Bitcoin → Yellow, Ethereum → Blue
colors = ['#f1c40f', '#3498db']


# 4️ Bar Chart (Average Interest)

mean_values = trends_data.mean()
plt.figure(figsize=(8,5))
sns.barplot(x=mean_values.index, y=mean_values.values, palette=colors)
plt.title("Average Google Trends Interest Last Month", fontsize=16, weight='bold')
plt.ylabel("Scaled Interest (0 → 1)", fontsize=12)
plt.xlabel("Keyword", fontsize=12)
plt.ylim(0,1)
plt.tight_layout()
plt.show()

# 5️ Area Plot

plt.figure(figsize=(14,7))
trends_data.plot(kind='area', stacked=False, alpha=0.4, color=colors)
plt.title("Google Trends Area Plot Last Month", fontsize=18, weight='bold')
plt.xlabel("Date", fontsize=14)
plt.ylabel("Scaled Interest (0 → 1)", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="Keyword", fontsize=12, title_fontsize=13, loc="upper left", frameon=True)
plt.tight_layout()
plt.show()

# 6️ Display Data

print("Google Trends Data (last 5 rows):")
print(trends_data.tail())
