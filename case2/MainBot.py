import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize  # Allowed dependency
import time
from tqdm import tqdm  # For progress bars

print("Starting script...")
start_time = time.time()

# Read the CSV file.
print("Reading CSV file...")
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv', index_col=0)
print(f"Data loaded. Shape: {data.shape}")

# If your CSV has an extra column (e.g., a "Case2" header), remove it or adjust the reading parameters.
# For instance, if the CSV's first row is not useful, you might need to set header=1.

'''
We recommend that you change your train and test split.
Here, we use an 80/20 split without shuffling.
'''
print("Splitting data into train and test sets...")
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)
print(f"Train set shape: {TRAIN.shape}, Test set shape: {TEST.shape}")

class Allocator():
    def __init__(self, train_data):
        """
        Initialize the allocator with historical data.
        The running_price_paths field stores the entire price history.
        """
        self.train_data = train_data.copy()
        self.running_price_paths = train_data.copy()
        self.last_optimization_day = -1
        self.optimization_cache = {}
        
    def equal_weight_allocation(self):
        num_assets = self.train_data.shape[1]
        return np.array([1/num_assets] * num_assets)
    
    def markowitz_allocation(self, returns):
        # Check cache first
        cache_key = ('markowitz', len(returns))
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
            
        num_assets = returns.shape[1]
        mu = returns.mean().values
        sigma = returns.cov().values

        def neg_sharpe(w):
            port_return = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, sigma.dot(w)))
            if port_vol == 0:
                return 1e6
            return -port_return / port_vol
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(-1, 1)] * num_assets
        init_guess = np.array([1/num_assets] * num_assets)
        result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x if result.success else init_guess
        
        # Cache the result
        self.optimization_cache[cache_key] = weights
        return weights
    
    def risk_parity_allocation(self, returns):
        # Check cache first
        cache_key = ('risk_parity', len(returns))
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
            
        num_assets = returns.shape[1]
        sigma = returns.cov().values
        n = num_assets
        
        def risk_parity_obj(w):
            port_var = np.dot(w, sigma.dot(w))
            risk_contrib = w * (sigma.dot(w))
            target = port_var / n
            return np.sum((risk_contrib - target)**2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(-1, 1)] * num_assets
        init_guess = np.array([1/num_assets] * num_assets)
        result = minimize(risk_parity_obj, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x if result.success else init_guess
        
        # Cache the result
        self.optimization_cache[cache_key] = weights
        return weights
    
    def allocate_portfolio(self, asset_prices, day):
        """
        asset_prices: np.array or Series of length equal to the number of assets,
                      representing the prices on a particular day.
        day: current day index
        Returns:
            weights: np.array of portfolio allocation for the next day.
        """
        # Reindex the new row to ensure columns match the training data.
        new_row = pd.DataFrame([asset_prices], columns=self.train_data.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)
    
        # If there's not enough history to compute returns, use equal weights.
        if len(self.running_price_paths) < 2:
            return self.equal_weight_allocation()
    
        # Compute daily returns.
        historical_returns = self.running_price_paths.pct_change().dropna()
        
        # Only run optimization every 10 days to save time
        if day - self.last_optimization_day >= 10 or self.last_optimization_day == -1:
            self.last_optimization_day = day
            # Compute allocations from the three strategies.
            ew = self.equal_weight_allocation()
            mw = self.markowitz_allocation(historical_returns)
            rp = self.risk_parity_allocation(historical_returns)
            
            # Combine strategies by averaging.
            combined_weights = (ew + mw + rp) / 3
        else:
            # Use the last computed weights
            combined_weights = self.optimization_cache.get(('last_weights',), self.equal_weight_allocation())
        
        # Cache the last used weights
        self.optimization_cache[('last_weights',)] = combined_weights
        
        # Ensure weights are in the allowed range [-1, 1].
        return np.clip(combined_weights, -1, 1)

def grading(train_data, test_data): 
    """
    Grading script that simulates daily portfolio allocation and computes the Sharpe ratio.
    """
    print("\nStarting portfolio allocation process...")
    num_assets = train_data.shape[1]
    weights = np.full(shape=(len(test_data.index), num_assets), fill_value=0.0)
    alloc = Allocator(train_data)
    
    # Generate portfolio allocations for each day in the test set.
    total_days = len(test_data)
    print(f"Processing {total_days} days of test data...")
    for i in tqdm(range(len(test_data)), desc="Allocating portfolio"):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :], i)
        if np.any(weights[i, :] < -1) or np.any(weights[i, :] > 1):
            raise Exception("Weights Outside of Bounds")
    
    print("\nSimulating capital evolution...")
    # Simulate capital evolution over the test period.
    capital = [1]
    for i in tqdm(range(len(test_data) - 1), desc="Simulating returns"):
        current_prices = np.array(test_data.iloc[i, :])
        next_prices = np.array(test_data.iloc[i+1, :])
        shares = capital[-1] * weights[i, :] / current_prices
        balance = capital[-1] - np.dot(shares, current_prices)
        net_change = np.dot(shares, next_prices)
        capital.append(balance + net_change)
    capital = np.array(capital)
    
    # Compute daily returns and the Sharpe ratio.
    print("\nComputing performance metrics...")
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        
    return sharpe, capital, weights

def main():
    print("\nStarting main execution...")
    sharpe, capital, weights = grading(TRAIN, TEST)
    print(f"\nSharpe Ratio: {sharpe:.4f}")
    
    print("\nGenerating plots...")
    # Plot capital evolution.
    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("Capital Evolution")
    plt.plot(np.arange(len(TEST)), capital, label="Capital")
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.legend()
    plt.show()
    
    # Plot portfolio weights over time.
    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("Portfolio Weights")
    plt.plot(np.arange(len(TEST)), weights)
    plt.legend(TEST.columns)
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()
