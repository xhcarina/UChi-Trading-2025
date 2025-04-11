import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 1) Load Data ==========
data = pd.read_csv("/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv")
print("Columns:", data.columns)

#selected_assets = ['Asset_1', 'Asset_2', 'Asset_3']
#data = data[selected_assets]
#selected_assets = ['Asset_4', 'Asset_5', 'Asset_6']
#data = data[selected_assets]

selected_assets = ['Asset_1', 'Asset_3']
data = data[selected_assets]

# ========== 2) Compute Daily Returns ==========
returns = data.pct_change().dropna()  # Daily returns, dropping the first NaN

# ========== 3) Rolling Window Parameters ==========
window = 900

# ========== 4) Calculate Rolling (Cumulative) Return for Each Asset ==========
cumulative_returns = (1 + returns).cumprod()

# ========== 5) Calculate Rolling Volatility (Std Dev) ==========
rolling_vol = returns.rolling(window=window).std()

# ========== 6) Calculate Rolling Sharpe Ratio ==========
rolling_mean = returns.rolling(window=window).mean()
rolling_sharpe = rolling_mean / rolling_vol

# ========== 7) Plot Rolling (Cumulative) Return over Time ==========
plt.figure(figsize=(10, 6))
for col in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[col], label=col)
plt.title("Cumulative Returns Over Time (All Assets)")
plt.xlabel("Time (Index)")
plt.ylabel("Cumulative Return (Growth Factor)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 8) Plot Rolling Volatility over Time ==========
plt.figure(figsize=(10, 6))
for col in rolling_vol.columns:
    plt.plot(rolling_vol.index, rolling_vol[col], label=col)
plt.title(f"{window}-Day Rolling Volatility")
plt.xlabel("Time (Index)")
plt.ylabel("Volatility (Standard Deviation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 9) Plot Rolling Sharpe Ratio over Time ==========
plt.figure(figsize=(10, 6))
for col in rolling_sharpe.columns:
    plt.plot(rolling_sharpe.index, rolling_sharpe[col], label=col)
plt.title(f"{window}-Day Rolling Sharpe Ratio")
plt.xlabel("Time (Index)")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 10) Plot Daily Returns as Percentages ==========
daily_returns_percent = returns * 100  # Convert to percent
plt.figure(figsize=(10, 6))
for col in daily_returns_percent.columns:
    plt.plot(daily_returns_percent.index, daily_returns_percent[col], label=col)
plt.title("Daily Returns (% Change)")
plt.xlabel("Time (Index)")
plt.ylabel("Daily Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
