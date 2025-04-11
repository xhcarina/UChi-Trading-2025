import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import time

# Add timing
start_time = time.time()

# Read and preprocess data
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')

# Resample to daily data (30 ticks per day)
print("Resampling data to daily frequency...")
data_daily = data.groupby(data.index // 30).last()
print(f"Original shape: {data.shape}, Daily shape: {data_daily.shape}")

# Ensure no NaN values
print("Checking for NaN values in the dataset...")
print(f"Number of NaN values in the dataset: {data_daily.isna().sum().sum()}")
if data_daily.isna().sum().sum() > 0:
    print("Warning: NaN values found in the dataset!")
    data_daily = data_daily.fillna(method='ffill').fillna(method='bfill')

# Split into train and test
TRAIN, TEST = train_test_split(data_daily, test_size=0.2, shuffle=False)

class Allocator:
    def __init__(self, data):
        self.data = data.copy()
        self.n_days = len(data)
        
        # Calculate daily returns
        for col in self.data.columns:
            if col.startswith('Asset'):
                self.data[f'{col}_return'] = self.data[col].pct_change()
                self.data[f'{col}_return'] = self.data[f'{col}_return'].fillna(0)
        
        # Initialize arrays for results
        self.weights = np.zeros((self.n_days, 6))
        self.pair_weights = np.zeros((self.n_days, 6))
        self.momentum_weights = np.zeros((self.n_days, 6))
        self.volatility_weights = np.zeros((self.n_days, 6))
        self.ml_weights = np.zeros((self.n_days, 6))
        
        # Track returns for each strategy
        self.pair_returns = np.zeros(self.n_days)
        self.momentum_returns = np.zeros(self.n_days)
        self.volatility_returns = np.zeros(self.n_days)
        self.ml_returns = np.zeros(self.n_days)
        self.ensemble_returns = np.zeros(self.n_days)
        
        # Capital tracking
        self.pair_capital = [1.0]
        self.momentum_capital = [1.0]
        self.volatility_capital = [1.0]
        self.ml_capital = [1.0]
        self.ensemble_capital = [1.0]
        
        # Initialize ML model
        self.ml_model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        
        # Strategy performance tracking
        self.strategy_sharpes = {
            'pair': np.zeros(self.n_days),
            'momentum': np.zeros(self.n_days),
            'volatility': np.zeros(self.n_days),
            'ml': np.zeros(self.n_days)
        }

    def calculate_features(self, i, window=60):
        """Calculate features for ML model"""
        if i < window:
            return None
            
        features = []
        for j in range(6):
            asset = f'Asset_{j+1}'
            returns = self.data[f'{asset}_return'].iloc[i-window:i]
            
            # Basic features
            momentum_1 = returns.iloc[-1]
            momentum_5 = returns.iloc[-5:].mean()
            momentum_10 = returns.iloc[-10:].mean()
            volatility = returns.std()
            rsi = self.calculate_rsi(returns)
            
            features.extend([momentum_1, momentum_5, momentum_10, volatility, rsi])
        
        return np.array(features)

    def calculate_rsi(self, returns, period=14):
        """Calculate RSI for a series of returns"""
        if len(returns) < period:
            return 50
            
        delta = returns.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    def ml_strategy(self, i, window=60):
        """ML-based strategy using Ridge regression"""
        if i < window:
            return np.zeros(6)
            
        # Prepare features and target
        features = self.calculate_features(i, window)
        if features is None:
            return np.zeros(6)
            
        # Train model on past data
        X = []
        y = []
        for j in range(window, i):
            feat = self.calculate_features(j, window)
            if feat is not None:
                X.append(feat)
                y.append(self.data[f'Asset_1_return'].iloc[j+1])  # Use next day's return as target
                
        if len(X) < window:
            return np.zeros(6)
            
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.ml_model.fit(X_scaled, y)
        
        # Make prediction
        current_features = self.scaler.transform(features.reshape(1, -1))
        predictions = []
        for j in range(6):
            asset = f'Asset_{j+1}_return'
            pred = self.ml_model.predict(current_features)[0]
            predictions.append(pred)
            
        # Convert predictions to weights
        weights = np.zeros(6)
        top_2 = np.argsort(predictions)[-2:]
        bottom_2 = np.argsort(predictions)[:2]
        
        for idx in top_2:
            weights[idx] = 0.5
        for idx in bottom_2:
            weights[idx] = -0.5
            
        return weights

    def rolling_sharpe(self, returns, window=60):
        """Calculate rolling Sharpe ratio"""
        if len(returns) < window:
            return 0
        window_returns = returns[-window:]
        mean_ret = np.mean(window_returns)
        std_ret = np.std(window_returns)
        return mean_ret / std_ret if std_ret > 0 else 0

    def update_strategy_performance(self, i):
        """Update rolling Sharpe ratios for each strategy"""
        if i >= 60:  # Need at least 60 days for Sharpe calculation
            self.strategy_sharpes['pair'][i] = self.rolling_sharpe(self.pair_returns[:i+1])
            self.strategy_sharpes['momentum'][i] = self.rolling_sharpe(self.momentum_returns[:i+1])
            self.strategy_sharpes['volatility'][i] = self.rolling_sharpe(self.volatility_returns[:i+1])
            self.strategy_sharpes['ml'][i] = self.rolling_sharpe(self.ml_returns[:i+1])

    def get_dynamic_weights(self, i):
        """Calculate dynamic weights based on recent performance"""
        if i < 60:
            return np.array([0.4, 0.3, 0.2, 0.1])  # Default weights
            
        # Get recent Sharpe ratios
        sharpes = np.array([
            self.strategy_sharpes['pair'][i],
            self.strategy_sharpes['momentum'][i],
            self.strategy_sharpes['volatility'][i],
            self.strategy_sharpes['ml'][i]
        ])
        
        # Softmax weighting
        exp_sharpes = np.exp(sharpes - np.max(sharpes))
        weights = exp_sharpes / np.sum(exp_sharpes)
        
        return weights

    def pair_trading(self, i, window=60):
        """Implement pair trading strategy primarily for assets 4-5"""
        if i < window:
            return np.zeros(6)
        
        weights = np.zeros(6)
        
        # Focus on volatile assets for pair trading
        pairs = [
            (3, 4),  # Asset_4 and Asset_5 (primary pair)
            (4, 5)   # Asset_5 and Asset_6
        ]
        
        # Track if any pair trading signals were generated
        any_pairs_active = False
        
        for pair in pairs:
            a, b = pair
            asset_a, asset_b = f'Asset_{a+1}', f'Asset_{b+1}'
            
            # Calculate spread, mean, and std for the rolling window
            spread_window = self.data[asset_a].iloc[i-window:i] - self.data[asset_b].iloc[i-window:i]
            spread_mean = spread_window.mean()
            spread_std = spread_window.std()
            
            if spread_std > 0:
                # Current spread
                current_spread = self.data[asset_a].iloc[i] - self.data[asset_b].iloc[i]
                z_score = (current_spread - spread_mean) / spread_std
                
                # Set thresholds - primary pair gets more weight
                if a == 3 and b == 4:  # Asset_4 and Asset_5 (primary pair)
                    entry_threshold = 2.0
                    exit_threshold = 0.5
                    weight_factor = 0.3
                else:
                    entry_threshold = 2.5
                    exit_threshold = 0.7
                    weight_factor = 0.15
                
                # Entering positions
                if z_score > entry_threshold:
                    weights[a] -= weight_factor / spread_std
                    weights[b] += weight_factor / spread_std
                    any_pairs_active = True
                elif z_score < -entry_threshold:
                    weights[a] += weight_factor / spread_std
                    weights[b] -= weight_factor / spread_std
                    any_pairs_active = True
                
                # Exit positions when mean reverts
                if abs(z_score) < exit_threshold:
                    weights[a] = 0
                    weights[b] = 0
                
                # Stop loss if z-score gets extreme
                if abs(z_score) > 4.0:
                    weights[a] = 0
                    weights[b] = 0
        
        # If no pair trading signals, add individual asset trading based on momentum
        if not any_pairs_active:
            # Calculate 5-day momentum for all assets
            momentum = {}
            for j in range(6):
                asset = f'Asset_{j+1}'
                if i >= 5:
                    momentum[j] = self.data[asset].iloc[i] / self.data[asset].iloc[i-5] - 1
                else:
                    momentum[j] = 0
            
            # Long the 2 assets with highest momentum
            sorted_momentum = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
            for j, _ in sorted_momentum[:2]:
                weights[j] = 0.25
            
            # Short the 2 assets with lowest momentum
            for j, _ in sorted_momentum[-2:]:
                weights[j] = -0.25
        
        # Ensure weights are normalized
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))
        
        return weights

    def momentum_strategy(self, i, window=60):
        """Simple momentum strategy"""
        if i < window:
            return np.zeros(6)
        
        weights = np.zeros(6)
        
        # Calculate momentum for all assets
        momentum = {}
        for j in range(6):
            asset = f'Asset_{j+1}'
            if i >= window:
                momentum[j] = self.data[asset].iloc[i] / self.data[asset].iloc[i-window] - 1
            else:
                momentum[j] = 0
        
        # Long the top 2, short the bottom 2
        sorted_momentum = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        for j, _ in sorted_momentum[:2]:
            weights[j] = 0.5
        for j, _ in sorted_momentum[-2:]:
            weights[j] = -0.5
        
        return weights

    def volatility_strategy(self, i, window=60):
        """Simple volatility strategy"""
        if i < window:
            return np.zeros(6)
        
        weights = np.zeros(6)
        
        # Calculate volatility for all assets
        volatility = {}
        for j in range(6):
            asset = f'Asset_{j+1}'
            if i >= window:
                returns = self.data[f'{asset}_return'].iloc[i-window:i]
                volatility[j] = returns.std()
            else:
                volatility[j] = 0.01  # Default small volatility
        
        # Inverse volatility weighting
        inv_vol = {j: 1.0 / (v + 0.001) for j, v in volatility.items()}
        total_inv_vol = sum(inv_vol.values())
        
        if total_inv_vol > 0:
            for j in range(6):
                weights[j] = inv_vol[j] / total_inv_vol
        
        return weights

    def apply_constraints(self, weights):
        """Apply portfolio constraints"""
        # Maximum absolute weight per asset
        max_weight = 0.5
        weights = np.clip(weights, -max_weight, max_weight)
        
        # Normalize if sum of absolute weights > 1
        sum_abs = np.sum(np.abs(weights))
        if sum_abs > 1.0:
            weights = weights / sum_abs
        
        return weights

    def allocate_portfolio(self, current_data):
        """Allocate portfolio weights based on current data"""
        i = len(self.weights) - 1  # Current index
        
        # Get weights from each strategy
        pair_weights = self.pair_trading(i)
        momentum_weights = self.momentum_strategy(i)
        volatility_weights = self.volatility_strategy(i)
        ml_weights = self.ml_strategy(i)
        
        # Get dynamic weights based on recent performance
        strategy_weights = self.get_dynamic_weights(i)
        
        # Blend strategies with dynamic weights
        weights = (
            strategy_weights[0] * pair_weights + 
            strategy_weights[1] * momentum_weights + 
            strategy_weights[2] * volatility_weights +
            strategy_weights[3] * ml_weights
        )
        
        # Apply constraints
        weights = self.apply_constraints(weights)
        
        return weights

def grading(train_data, test_data): 
    # Goal: Design weights to maximize risk-adjusted return (Sharpe ratio)
    
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = Allocator(train_data)

    print("\nGenerating portfolio weights...")
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
        # Calculate annualized Sharpe ratio
        # 1300 days = 5 years, so trading days per year = 260
        # For test period: 130 days = 6 months
        annualization_factor = np.sqrt(260 / 130 * 0.5)  # 0.5 because it's 6 months
        annualized_sharpe = sharpe * annualization_factor
    else:
        sharpe = 0
        annualized_sharpe = 0
    
    return sharpe, annualized_sharpe, capital, weights

# Example usage
if __name__ == "__main__":
    print("Starting strategy execution...")
    sharpe, annualized_sharpe, capital, weights = grading(TRAIN, TEST)
    print(f"\nDaily Sharpe Ratio: {sharpe:.4f}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe:.4f}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    # Plot capital growth
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("Capital Growth Over Time")
    plt.plot(np.arange(len(TEST)), capital)
    plt.xlabel("Time (Days)")
    plt.ylabel("Capital")
    plt.grid(True)

    # Plot asset weights
    plt.subplot(2, 1, 2)
    plt.title("Asset Weights Over Time")
    for i in range(6):
        plt.plot(np.arange(len(TEST)), weights[:, i], label=f"Asset_{i+1}")
    plt.xlabel("Time (Days)")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()