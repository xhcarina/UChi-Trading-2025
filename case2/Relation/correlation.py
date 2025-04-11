import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========= 1) Load Data ==========
# Remove index_col if you want all asset columns; otherwise, ensure Asset_1 is not used as index.
data = pd.read_csv("/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv")
print("Columns:", data.columns)

# ========= 2) Compute Returns ==========
# Daily returns
daily_returns = data.pct_change().dropna()
daily_log_returns = np.log(data / data.shift(1)).dropna()

# Monthly returns (21 trading days)
monthly_returns = data.pct_change(periods=21).dropna()
monthly_log_returns = np.log(data / data.shift(21)).dropna()

# ========= 3) Analyze All Possible Pairs ==========
def analyze_all_pairs(data):
    """Analyze all possible pairs of assets."""
    assets = data.columns
    n_assets = len(assets)
    results = []
    
    print("\n=== Detailed Pair Analysis ===")
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            asset1 = assets[i]
            asset2 = assets[j]
            
            # Calculate basic statistics
            returns1 = daily_returns[asset1]
            returns2 = daily_returns[asset2]
            
            # Correlation
            corr, pvalue = stats.pearsonr(returns1, returns2)
            
            # Cointegration test
            score, pvalue_coint, _ = coint(data[asset1], data[asset2])
            
            # Spread analysis
            spread = data[asset1] - data[asset2]
            zscore = (spread - spread.mean()) / spread.std()
            
            # Rolling statistics
            rolling_corr = returns1.rolling(60).corr(returns2)
            rolling_spread_std = spread.rolling(60).std()
            
            results.append({
                'pair': f"{asset1}_{asset2}",
                'correlation': corr,
                'correlation_pvalue': pvalue,
                'cointegration_pvalue': pvalue_coint,
                'cointegration_score': score,
                'mean_spread': spread.mean(),
                'std_spread': spread.std(),
                'zscore_mean': zscore.mean(),
                'zscore_std': zscore.std(),
                'rolling_corr_mean': rolling_corr.mean(),
                'rolling_corr_std': rolling_corr.std(),
                'rolling_spread_std_mean': rolling_spread_std.mean()
            })
            
            # Print detailed analysis
            print(f"\nPair: {asset1} - {asset2}")
            print(f"Correlation: {corr:.4f} (p-value: {pvalue:.4f})")
            print(f"Cointegration p-value: {pvalue_coint:.4f}")
            print(f"Mean Spread: {spread.mean():.6f}")
            print(f"Spread Std: {spread.std():.6f}")
            print(f"Rolling Correlation Mean: {rolling_corr.mean():.4f}")
            print(f"Rolling Correlation Std: {rolling_corr.std():.4f}")
            
            # Plot the analysis
            plt.figure(figsize=(15, 10))
            
            # Price ratio
            plt.subplot(2, 2, 1)
            price_ratio = data[asset1] / data[asset2]
            plt.plot(price_ratio, label='Price Ratio')
            plt.title(f"Price Ratio: {asset1}/{asset2}")
            plt.legend()
            
            # Rolling correlation
            plt.subplot(2, 2, 2)
            plt.plot(rolling_corr, label='60-day Rolling Correlation')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f"Rolling Correlation: {asset1} - {asset2}")
            plt.legend()
            
            # Spread and z-score
            plt.subplot(2, 2, 3)
            plt.plot(spread, label='Spread')
            plt.title(f"Spread: {asset1} - {asset2}")
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.plot(zscore, label='Z-Score')
            plt.axhline(y=2, color='r', linestyle='--')
            plt.axhline(y=-2, color='r', linestyle='--')
            plt.title(f"Z-Score: {asset1} - {asset2}")
            plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    return pd.DataFrame(results)

# Run the analysis
all_pairs_analysis = analyze_all_pairs(data)

# ========= 4) Correlation Matrix Visualization ==========
plt.figure(figsize=(12, 10))
corr_matrix = daily_returns.corr()
sns.heatmap(corr_matrix, annot=True, cmap="RdBu", center=0)
plt.title("Daily Returns Correlation Matrix")
plt.show()

# ========= 5) Print Summary Statistics ==========
print("\n=== Summary Statistics ===")
print("Most Correlated Pairs:")
print(all_pairs_analysis.nlargest(3, 'correlation')[['pair', 'correlation']])

print("\nMost Cointegrated Pairs:")
print(all_pairs_analysis.nsmallest(3, 'cointegration_pvalue')[['pair', 'cointegration_pvalue']])

print("\nPairs with Most Stable Spreads:")
print(all_pairs_analysis.nsmallest(3, 'std_spread')[['pair', 'std_spread']])
