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
    """Grade the strategy"""
    print("\nInitializing IPCA strategy...")
    strategy = IPCAStrategy(n_factors=3, window=252)
    
    print("Fitting on training data...")
    strategy.fit(train_data)
    
    print("Predicting returns for test data...")
    predicted_returns = strategy.predict_returns(test_data)
    
    # Initialize arrays
    n_periods = len(test_data)
    n_assets = test_data.shape[1]
    weights = np.zeros((n_periods, n_assets))
    capital = np.ones(n_periods + 1)  # Initial capital of 1.0
    
    print("\nOptimizing portfolio weights...")
    for i in tqdm(range(n_periods), desc="Allocating portfolio"):
        try:
            w = strategy.optimize_weights(predicted_returns[i:i+1, :])
            # Ensure weights sum to 1 and are non-negative
            w = np.clip(w, 0, None)
            w = w / np.sum(w) if np.sum(w) > 0 else np.ones(n_assets) / n_assets
            weights[i, :] = w
        except Exception as e:
            print(f"\nWarning: Optimization failed at period {i}. Using equal weights. Error: {str(e)}")
            weights[i, :] = np.ones(n_assets) / n_assets
    
    print("\nCalculating returns...")
    # Calculate daily returns properly
    returns = test_data.pct_change().fillna(0).values
    portfolio_returns = np.zeros(n_periods)
    daily_returns = np.zeros(n_periods)
    
    for i in range(n_periods):
        try:
            # Calculate portfolio return for this day
            portfolio_returns[i] = np.sum(weights[i, :] * returns[i, :])
            daily_returns[i] = portfolio_returns[i]  # Store daily return
            capital[i+1] = capital[i] * (1 + portfolio_returns[i])
        except Exception as e:
            print(f"\nWarning: Return calculation failed at period {i}. Error: {str(e)}")
            capital[i+1] = capital[i]
            daily_returns[i] = 0
    
    # Performance metrics with proper annualization
    mean_daily_return = np.mean(daily_returns)
    daily_vol = np.std(daily_returns)
    
    # Annualized metrics
    annual_return = (1 + mean_daily_return) ** 252 - 1
    annual_vol = daily_vol * np.sqrt(252)
    
    # Sharpe ratio (assuming 0 risk-free rate for simplicity)
    if daily_vol > 0:
        sharpe = mean_daily_return / daily_vol * np.sqrt(252)
    else:
        print("\nWarning: Zero volatility detected")
        sharpe = 0.0
    
    # Calculate rolling metrics
    window = 63  # ~3 months
    rolling_returns = pd.Series(daily_returns).rolling(window=window).mean() * 252
    rolling_vol = pd.Series(daily_returns).rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = rolling_returns / rolling_vol
    
    # Calculate drawdown
    peak = np.maximum.accumulate(capital)
    drawdown = (capital - peak) / peak
    max_drawdown = np.min(drawdown) * 100
    
    # Calculate win rate
    winning_days = np.sum(daily_returns > 0)
    win_rate = winning_days / len(daily_returns) * 100
    
    print("\nDetailed Strategy Statistics:")
    print(f"Annualized Return: {annual_return*100:.2f}%")
    print(f"Annualized Volatility: {annual_vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average Daily Return: {mean_daily_return*100:.3f}%")
    print(f"Daily Return Std: {daily_vol*100:.3f}%")
    print(f"Total Return: {(capital[-1]/capital[0]-1)*100:.2f}%")
    
    return sharpe, capital, weights, daily_returns

def main():
    print("\nStarting main execution...")
    
    print("\nStarting portfolio allocation process...")
    sharpe, capital, weights, daily_returns = grading(TRAIN, TEST)
    
    print(f"\nFinal Sharpe Ratio: {sharpe:.4f}")
    print(f"Final Portfolio Value: {capital[-1]:.4f}")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    
    # Calculate rolling metrics
    window = 63  # ~3 months
    rolling_returns = pd.Series(daily_returns).rolling(window=window).mean() * 252
    rolling_vol = pd.Series(daily_returns).rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = rolling_returns / rolling_vol
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Capital Evolution
    plt.subplot(3, 2, 1)
    plt.plot(capital)
    plt.title('Capital Evolution')
    plt.xlabel('Time')
    plt.ylabel('Capital')
    plt.grid(True)
    
    # 2. Daily Returns
    plt.subplot(3, 2, 2)
    plt.plot(daily_returns)
    plt.title('Daily Returns')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.grid(True)
    
    # 3. Rolling Volatility
    plt.subplot(3, 2, 3)
    plt.plot(rolling_vol)
    plt.title('Rolling Volatility (3-month)')
    plt.xlabel('Time')
    plt.ylabel('Annualized Volatility')
    plt.grid(True)
    
    # 4. Rolling Sharpe
    plt.subplot(3, 2, 4)
    plt.plot(rolling_sharpe)
    plt.title('Rolling Sharpe Ratio (3-month)')
    plt.xlabel('Time')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    
    # 5. Portfolio Weights
    plt.subplot(3, 2, 5)
    plt.stackplot(range(len(weights)), weights.T)
    plt.title('Portfolio Weights')
    plt.xlabel('Time')
    plt.ylabel('Weight')
    plt.legend([f'Asset {i+1}' for i in range(weights.shape[1])])
    plt.grid(True)
    
    # 6. Volatility vs Return Scatter
    plt.subplot(3, 2, 6)
    plt.scatter(rolling_vol, rolling_returns, alpha=0.5)
    plt.title('Volatility vs Return')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.grid(True)
    
    # Add a colorbar for time
    scatter = plt.scatter(rolling_vol, rolling_returns, c=range(len(rolling_returns)), 
                         alpha=0.5, cmap='viridis')
    plt.colorbar(scatter, label='Time')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    