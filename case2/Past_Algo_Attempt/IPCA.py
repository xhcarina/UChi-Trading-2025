import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import time
from tqdm import tqdm

# Additional imports for the IPCA methods
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize

start_time = time.time()
print("Starting script...")

# Read the CSV file (make sure the path is correct)
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')
print(f"Data loaded. Shape: {data.shape}")

'''
We recommend that you change your train and test split
'''
# Using original structure: split with test_size=0.2 and no shuffling.
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)
print(f"Train set shape: {TRAIN.shape}, Test set shape: {TEST.shape}")


# -----------------------------
# IPCA Strategy Implementation
# -----------------------------
class IPCAStrategy:
    def __init__(self, n_factors=3, window=252):
        self.n_factors = n_factors
        self.window = window
        self.pca = PCA(n_components=n_factors)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.beta = None  # Regression coefficients mapping factor scores to returns

    def calculate_characteristics(self, data):
        """
        Calculate instrumented characteristics for each asset.
        Produces 6 features per asset:
         - 1-day, 5-day, and 10-day momentum
         - 5-day and 10-day rolling volatility
         - Mean reversion normalized by rolling std
        """
        n_assets = data.shape[1]
        n_periods = data.shape[0]
        characteristics = np.zeros((n_periods, n_assets, 6))
        
        # Convert to numpy array for faster computation
        data_array = data.values
        
        for i in range(n_assets):
            # Calculate returns
            returns = pd.Series(data_array[:, i]).pct_change().fillna(0).values
            
            # Momentum features
            characteristics[:, i, 0] = returns  # 1-day returns
            characteristics[:, i, 1] = pd.Series(returns).rolling(5).sum().fillna(0).values  # 5-day momentum
            characteristics[:, i, 2] = pd.Series(returns).rolling(10).sum().fillna(0).values  # 10-day momentum
            
            # Volatility features
            characteristics[:, i, 3] = pd.Series(returns).rolling(5).std().fillna(0).values  # 5-day volatility
            characteristics[:, i, 4] = pd.Series(returns).rolling(10).std().fillna(0).values  # 10-day volatility
            
            # Mean reversion feature
            price_series = pd.Series(data_array[:, i])
            rolling_mean = price_series.rolling(5).mean()
            rolling_std = price_series.rolling(5).std()
            z_score = (price_series - rolling_mean) / (rolling_std + 1e-6)
            characteristics[:, i, 5] = z_score.fillna(0).values
        
        return characteristics

    def preprocess_data(self, X):
        """Impute missing values and scale features."""
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        return X_scaled

    def fit(self, data):
        """
        Fit the IPCA model on the training data.
        First, compute the instrumented characteristics, fit PCA, then
        estimate a linear mapping (beta) from factor scores to next-day returns.
        """
        characteristics = self.calculate_characteristics(data)
        n_periods, n_assets, n_chars = characteristics.shape

        # Reshape characteristics: each row is one observation
        X = characteristics.reshape(n_periods * n_assets, n_chars)
        X_processed = self.preprocess_data(X)
        
        # Fit PCA on the processed features
        self.pca.fit(X_processed)

        # Compute factor scores and reshape them back
        factor_scores = self.pca.transform(X_processed)
        factor_scores = factor_scores.reshape(n_periods, n_assets, self.n_factors)

        # Compute target returns: next day returns (shifted by -1)
        returns = data.pct_change().shift(-1).fillna(0).values  # shape: (n_periods, n_assets)
        valid_length = n_periods - 1  # avoid lookahead
        X_reg = factor_scores[:valid_length].reshape(valid_length * n_assets, self.n_factors)
        y_reg = returns[:valid_length].reshape(valid_length * n_assets)
        
        # Solve for beta using ordinary least squares
        self.beta, _, _, _ = np.linalg.lstsq(X_reg, y_reg, rcond=None)

    def predict_returns(self, data):
        """
        Predict returns for each asset using the computed IPCA beta.
        """
        characteristics = self.calculate_characteristics(data)
        n_periods, n_assets, n_chars = characteristics.shape
        X = characteristics.reshape(n_periods * n_assets, n_chars)
        X_processed = self.preprocess_data(X)
        factor_scores = self.pca.transform(X_processed)
        factor_scores = factor_scores.reshape(n_periods, n_assets, self.n_factors)
        
        # Vectorized prediction
        predicted_returns = np.zeros((n_periods, n_assets))
        for t in range(n_periods):
            predicted_returns[t] = np.dot(factor_scores[t], self.beta)
            
        return predicted_returns

    def optimize_weights(self, predicted_returns, risk_aversion=1.0):
        """
        Given predicted returns, solve a mean-variance type optimization:
          maximize (predicted_return - risk_aversion * portfolio_risk)
        with constraints that weights sum to 1 and each weight lies in [-1, 1].
        """
        n_assets = predicted_returns.shape[1]

        def objective(weights):
            portfolio_return = np.sum(weights * predicted_returns)
            portfolio_risk = np.sqrt(np.sum(weights ** 2))
            return -(portfolio_return - risk_aversion * portfolio_risk)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(-1, 1)] * n_assets  # Allow weights in [-1, 1]
        x0 = np.ones(n_assets) / n_assets

        result = minimize(objective, x0, method='SLSQP', constraints=constraints, bounds=bounds)
        if result.success:
            return result.x
        else:
            return np.ones(n_assets) / n_assets  # fallback to equal weights if optimization fails


# ---------------------------------------------
# IPCAAllocator: Integrating IPCA into Framework
# ---------------------------------------------
class IPCAAllocator:
    def __init__(self, train_data):
        """
        Store the training data and fit the IPCA strategy.
        running_price_paths will be updated with each test day.
        """
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        # Initialize and fit IPCA strategy on the training data.
        self.strategy = IPCAStrategy(n_factors=3, window=252)
        self.strategy.fit(train_data)

    def allocate_portfolio(self, asset_prices):
        """
        asset_prices: Series of length 6 (prices for one day)
        Append the new asset prices to running_price_paths,
        predict the next day's returns using the IPCA model,
        and optimize portfolio weights.
        """
        # Append new day's asset prices using concat instead of append
        new_row = pd.DataFrame([asset_prices], columns=self.running_price_paths.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)
        
        # Use the entire running history for prediction
        predicted_returns = self.strategy.predict_returns(self.running_price_paths)
        
        # Get predicted returns for the latest day
        last_pred = predicted_returns[-1, :].reshape(1, -1)
        
        # Optimize weights based on the predicted returns
        weights = self.strategy.optimize_weights(last_pred)
        return weights


# ----------------------------------------------------
# Grading Function (Non-annualized Calculations)
# ----------------------------------------------------
def grading(train_data, test_data):
    # Goal: Design weights to maximize risk-adjusted return (Sharpe ratio)
    print("\nFitting IPCA model on training data...")
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = IPCAAllocator(train_data)

    print("\nGenerating predictions and weights...")
    for i in tqdm(range(len(test_data)), desc="Processing test data"):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :])

    print("\nCalculating returns and Sharpe ratio...")
    capital = [1]  # Start with $1 and aim to grow capital steadily

    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i, :])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i, :]))
        net_change = np.dot(shares, np.array(test_data.iloc[i + 1, :]))
        capital.append(balance + net_change)

    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]

    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0

    return sharpe, capital, weights


# ---------------------------
# Main execution starts here
# ---------------------------
print("\nStarting main execution...")
sharpe, capital, weights = grading(TRAIN, TEST)
print(f"\nFinal Sharpe Ratio: {sharpe:.4f}")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital Growth Over Time")
plt.plot(np.arange(len(TEST)), capital)
plt.xlabel("Time")
plt.ylabel("Capital")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Asset Weights Over Time")
plt.plot(np.arange(len(TEST)), weights)
plt.xlabel("Time")
plt.ylabel("Weight")
plt.legend([f"Asset_{i+1}" for i in range(6)])
plt.grid(True)
plt.show()
