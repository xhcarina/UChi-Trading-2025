#Instrumented Principal Component Analysis

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

print("Starting script...")
start_time = time.time()

# Read the CSV file
print("Reading CSV file...")
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')
print(f"Data loaded. Shape: {data.shape}")

print("Splitting data into train and test sets...")
TRAIN, TEST = train_test_split(data, test_size=0.8, shuffle=False)
print(f"Train set shape: {TRAIN.shape}, Test set shape: {TEST.shape}")

class IPCAStrategy:
    def __init__(self, n_factors=3, window=252):
        self.n_factors = n_factors
        self.window = window
        self.pca = PCA(n_components=n_factors)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def calculate_characteristics(self, data):
        """Calculate asset characteristics"""
        n_assets = data.shape[1]
        n_periods = data.shape[0]
        
        # Initialize characteristics array
        characteristics = np.zeros((n_periods, n_assets, 6))  # 6 characteristics per asset
        
        for i in range(n_assets):
            # Momentum features
            characteristics[:, i, 0] = data.iloc[:, i].pct_change(21)  # 1-month momentum
            characteristics[:, i, 1] = data.iloc[:, i].pct_change(63)  # 3-month momentum
            characteristics[:, i, 2] = data.iloc[:, i].pct_change(126)  # 6-month momentum
            
            # Volatility features
            characteristics[:, i, 3] = data.iloc[:, i].pct_change().rolling(21).std()  # 1-month vol
            characteristics[:, i, 4] = data.iloc[:, i].pct_change().rolling(63).std()  # 3-month vol
            
            # Mean reversion feature
            characteristics[:, i, 5] = (data.iloc[:, i] - data.iloc[:, i].rolling(21).mean()) / data.iloc[:, i].rolling(21).std()
        
        return characteristics
    
    def preprocess_data(self, X):
        """Preprocess data by handling NaN values and scaling"""
        # Handle NaN values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled
    
    def fit(self, data):
        """Fit IPCA model"""
        characteristics = self.calculate_characteristics(data)
        n_periods, n_assets, n_chars = characteristics.shape
        
        # Reshape for PCA
        X = characteristics.reshape(n_periods * n_assets, n_chars)
        
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Fit PCA
        self.pca.fit(X_processed)
        
    def predict_returns(self, data):
        """Predict returns using IPCA factors"""
        characteristics = self.calculate_characteristics(data)
        n_periods, n_assets, n_chars = characteristics.shape
        
        # Transform characteristics
        X = characteristics.reshape(n_periods * n_assets, n_chars)
        
        # Preprocess data
        X_processed = self.preprocess_data(X)
        
        # Get factor loadings
        loadings = self.pca.transform(X_processed)
        loadings = loadings.reshape(n_periods, n_assets, self.n_factors)
        
        # Calculate predicted returns
        predicted_returns = np.zeros((n_periods, n_assets))
        for t in range(n_periods):
            for i in range(n_assets):
                predicted_returns[t, i] = np.sum(loadings[t, i, :] * self.pca.components_[:, 0])
        
        return predicted_returns
    
    def optimize_weights(self, predicted_returns, risk_aversion=1.0):
        """Optimize portfolio weights using mean-variance optimization"""
        n_assets = predicted_returns.shape[1]
        
        def objective(weights):
            portfolio_return = np.sum(weights * predicted_returns)
            portfolio_risk = np.sqrt(np.sum(weights**2))
            return -(portfolio_return - risk_aversion * portfolio_risk)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # weights >= 0
        ]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', constraints=constraints)
        
        if result.success:
            return result.x
        else:
            return np.ones(n_assets) / n_assets  # return equal weights if optimization fails







def grading(train_data, test_data): 
    # Goal: Design weights to maximize risk-adjusted return (Sharpe ratio)
    
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = Allocator(train_data)

    for i in range(0, len(test_data)):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :])
        # Goal: Ensure all weights stay within legal bounds [-1, 1]

    capital = [1]  # Goal: Start with $1 and aim to grow capital steadily

    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i, :])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i, :]))
        net_change = np.dot(shares, np.array(test_data.iloc[i + 1, :]))
        capital.append(balance + net_change)
        # Goal: Maximize capital[i+1] and minimize large swings in capital[i+1] - capital[i]

    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    # Goal: Maximize np.mean(returns), minimize np.std(returns)
    # Goal: Ensure capital[:-1] doesn't drop too low (drawdown protection)
    # Goal: Maximize capital[1:] over time (wealth accumulation)

    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
        # Goal: Sharpe ratio should be as high as possible
    else:
        sharpe = 0  # Goal: Avoid flat return strategies unless safely optimal
    
    return sharpe, capital, weights




sharpe, capital, weights = grading(TRAIN, TEST)
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()