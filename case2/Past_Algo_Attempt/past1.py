import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
from scipy.optimize import minimize
import xgboost as xgb

print("Starting script...")
start_time = time.time()

# Read the CSV file
print("Reading CSV file...")
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')
print(f"Data loaded. Shape: {data.shape}")

# Split the data
print("Splitting data into train and test sets...")
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)
print(f"Train set shape: {TRAIN.shape}, Test set shape: {TEST.shape}")

def calculate_technical_indicators(df, window=30):
    """Calculate technical indicators for each asset."""
    features_dict = {}
    
    # Calculate cross-asset features first
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            # Price spread
            features_dict[f'spread_{i}_{j}'] = df.iloc[:, i] - df.iloc[:, j]
            # Price ratio
            features_dict[f'ratio_{i}_{j}'] = df.iloc[:, i] / df.iloc[:, j]
    
    for col in df.columns:
        prices = df[col]
        features = {}
        
        # Basic price features
        features[f'{col}_returns'] = prices.pct_change()
        features[f'{col}_log_returns'] = np.log1p(prices).diff()
        
        # Lagged returns
        for lag in [1, 3, 5]:
            features[f'{col}_lag_{lag}'] = prices.pct_change(periods=lag)
        
        # Moving averages
        for w in [20, 30, 60]:
            ma = prices.rolling(window=w).mean()
            features[f'{col}_ma_{w}'] = ma
            features[f'{col}_ma_ratio_{w}'] = prices / ma
        
        # Volatility
        returns = features[f'{col}_returns']
        for w in [20, 30, 60]:
            features[f'{col}_vol_{w}'] = returns.rolling(window=w).std()
        
        # Momentum
        for w in [20, 30, 60]:
            features[f'{col}_mom_{w}'] = prices.pct_change(periods=w)
        
        # RSI
        for w in [30]:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / loss
            features[f'{col}_rsi_{w}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features[f'{col}_macd'] = macd
        features[f'{col}_macd_signal'] = signal
        features[f'{col}_macd_hist'] = macd - signal
        
        features_dict.update(features)
    
    return pd.DataFrame(features_dict)

class MLAllocator():
    def __init__(self, train_data, xgb_params=None, pair_params=None):
        self.train_data = train_data.copy()
        self.models = {}  # Changed from list to dict
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.window = 30
        self.running_price_paths = train_data.copy()
        
        # Initialize models for each asset
        for col in train_data.columns:
            self.models[col] = xgb.XGBRegressor(
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                objective='reg:squarederror',
                random_state=42
            )
        
        # Best XGBoost parameters from tuning
        self.xgb_params = xgb_params if xgb_params is not None else {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100
        }
        
        # Best pair trading parameters from tuning
        self.pair_params = pair_params if pair_params is not None else {
            'z_score_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss_threshold': 3.0,
            'strategy_weight': 0.7
        }
        
        # Initialize spread parameters for pair trading
        self.spread_mean = None
        self.spread_std = None
        self.calculate_spread_params()
        
        # Train the models
        print("\nTraining ML models...")
        self.train_models()
    
    def calculate_spread_params(self):
        """Calculate spread parameters for pair trading"""
        # Calculate spread between Asset 4 and Asset 5
        spread = self.train_data['Asset_4'] - self.train_data['Asset_5']
        self.spread_mean = spread.mean()
        self.spread_std = spread.std()
    
    def calculate_z_score(self, spread):
        """Calculate z-score for pair trading."""
        return (spread - self.spread_mean) / self.spread_std
    
    def prepare_ml_data(self, data):
        """Prepare features and targets for ML model."""
        features = calculate_technical_indicators(data, self.window)
        
        # Use 1-day forward returns to reduce target leakage
        forward_returns = {}
        for col in data.columns:
            forward_returns[col] = data[col].pct_change(periods=1).shift(-1)
        
        features = features.fillna(0)
        valid_indices = features.index
        
        y_dict = {}
        for col in forward_returns:
            y = forward_returns[col].loc[valid_indices]
            y = y.fillna(0)
            y_dict[col] = y
            
        return features, y_dict
    
    def train_models(self):
        """Train XGBoost models for each asset"""
        # Calculate features for training data
        features = calculate_technical_indicators(self.train_data, self.window)
        features = features.fillna(0)
        self.feature_columns = features.columns
        
        for col in self.train_data.columns:
            # Prepare target (1-day forward returns)
            target = self.train_data[col].pct_change().shift(-1)
            target = target.fillna(0)
            
            # Remove rows with NaN values
            valid_idx = ~features.isna().any(axis=1) & ~target.isna()
            X = features[valid_idx]
            y = target[valid_idx]
            
            # Train model
            self.models[col].fit(X, y)
    
    def allocate_portfolio(self, current_prices):
        """Allocate portfolio based on model predictions and risk metrics"""
        # Update running price paths
        self.running_price_paths = pd.concat([self.running_price_paths, pd.DataFrame([current_prices])], ignore_index=True)
        
        # Calculate current metrics
        current_vols = self.calculate_volatility(self.running_price_paths).iloc[-1]
        current_returns = self.calculate_returns(self.running_price_paths).iloc[-1]
        
        # Calculate technical indicators
        features = calculate_technical_indicators(self.running_price_paths, self.window)
        features = features.fillna(0)
        
        # Ensure we use the same features as in training
        if self.feature_columns is not None:
            features = features[self.feature_columns]
        
        # Get model predictions
        predictions = {}
        for col in self.train_data.columns:
            predictions[col] = self.models[col].predict(features.iloc[[-1]])[0]
        
        # Convert predictions to array
        ml_weights = np.array([predictions[col] for col in self.train_data.columns])
        
        # Calculate pair trading signals
        spread = current_prices['Asset_4'] - current_prices['Asset_5']
        z_score = (spread - self.spread_mean) / self.spread_std
        
        # Initialize pair trading weights
        pair_weights = np.zeros(len(self.train_data.columns))
        position_size = 1.0
        
        # Pair trading logic
        if z_score > self.pair_params['z_score_threshold']:
            pair_weights[3] = -position_size  # Short Asset_4
            pair_weights[4] = position_size   # Long Asset_5
        elif z_score < -self.pair_params['z_score_threshold']:
            pair_weights[3] = position_size   # Long Asset_4
            pair_weights[4] = -position_size  # Short Asset_5
        
        # Exit positions if spread reverts
        if abs(z_score) < self.pair_params['exit_threshold']:
            pair_weights[3] = 0
            pair_weights[4] = 0
        
        # Stop loss if spread widens too much
        if abs(z_score) > self.pair_params['stop_loss_threshold']:
            pair_weights[3] = 0
            pair_weights[4] = 0
        
        # Combine ML and pair trading strategies
        final_weights = (
            self.pair_params['strategy_weight'] * ml_weights + 
            (1 - self.pair_params['strategy_weight']) * pair_weights
        )
        
        # Normalize weights to ensure they sum to 0 (market neutral)
        final_weights = final_weights - np.mean(final_weights)
        
        # Clip weights to [-1, 1]
        final_weights = np.clip(final_weights, -1, 1)
        
        return final_weights
    
    def calculate_volatility(self, price_data):
        """Calculate rolling volatility for each asset"""
        return price_data.pct_change().rolling(window=20).std()
    
    def calculate_returns(self, price_data):
        """Calculate rolling returns for each asset"""
        return price_data.pct_change().rolling(window=20).mean()

def grading(train_data, test_data): 
    # Goal: Design weights to maximize risk-adjusted return (Sharpe ratio)
    
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = MLAllocator(train_data)

    print("\nGenerating portfolio weights...")
    for i in tqdm(range(0, len(test_data)), desc="Processing"):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :])

    capital = [1]  # Start with $1

    print("\nCalculating returns...")
    for i in tqdm(range(len(test_data) - 1), desc="Calculating"):
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
    
    # Calculate rolling Sharpe ratio
    rolling_sharpe = pd.Series(returns).rolling(63).mean() / pd.Series(returns).rolling(63).std()
    
    return sharpe, capital, weights, rolling_sharpe

def main():
    print("\nStarting main execution...")
    sharpe, capital, weights, rolling_sharpe = grading(TRAIN, TEST)
    print(f"\nSharpe Ratio: {sharpe:.4f}")
    print(f"Average Rolling Sharpe Ratio: {rolling_sharpe.mean():.4f}")
    
    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.title("Cumulative Returns")
    plt.plot(np.arange(len(TEST)), capital - 1, label="Cumulative Returns")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.show()
    
    # Plot portfolio weights
    plt.figure(figsize=(10, 6))
    plt.title("Portfolio Weights")
    plt.plot(np.arange(len(TEST)), weights)
    plt.legend(TEST.columns)
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.show()
    
    # Plot rolling Sharpe ratio
    plt.figure(figsize=(10, 6))
    plt.title("Rolling Sharpe Ratio (63-day window)")
    plt.plot(np.arange(len(rolling_sharpe)), rolling_sharpe)
    plt.xlabel("Time")
    plt.ylabel("Rolling Sharpe Ratio")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()
