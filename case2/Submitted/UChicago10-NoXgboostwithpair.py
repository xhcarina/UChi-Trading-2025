import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

'''
---------------------------------Notes---------------------------------
1. Previous running time record: take 47 seconds to train and 6 mins to run on 9/1 split
2. We also import tqdm 
---------------------------------Notes---------------------------------
'''

print("Starting script...")
start_time = time.time()

# Read CSV file
print("Reading CSV file...")
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')
print(f"Data loaded. Shape: {data.shape}")

# Split the data
print("Splitting data into train and test sets...")
TRAIN, TEST = train_test_split(data, test_size=0.1, shuffle=False)
print(f"Train set shape: {TRAIN.shape}, Test set shape: {TEST.shape}")

def calculate_technical_indicators(df, window=30):
    """Calculate technical indicators for each asset using only pandas and numpy."""
    features_dict = {}
    
    # Calculate cross-asset features
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            features_dict[f'spread_{i}_{j}'] = df.iloc[:, i] - df.iloc[:, j]
            features_dict[f'ratio_{i}_{j}'] = df.iloc[:, i] / df.iloc[:, j]
    
    # Calculate features per asset
    for col in df.columns:
        prices = df[col]
        features = {}
        
        # Basic returns and log returns
        features[f'{col}_returns'] = prices.pct_change()
        features[f'{col}_log_returns'] = np.log1p(prices).diff()
        
        # Lagged returns
        for lag in [1, 3, 5]:
            features[f'{col}_lag_{lag}'] = prices.pct_change(periods=lag)
        
        # Moving averages and ratios
        for w in [20, 30, 60]:
            ma = prices.rolling(window=w).mean()
            features[f'{col}_ma_{w}'] = ma
            features[f'{col}_ma_ratio_{w}'] = prices / ma
        
        # Volatility: standard deviation of returns
        returns = features[f'{col}_returns']
        for w in [20, 30, 60]:
            features[f'{col}_vol_{w}'] = returns.rolling(window=w).std()
        
        # Momentum over different windows
        for w in [20, 30, 60]:
            features[f'{col}_mom_{w}'] = prices.pct_change(periods=w)
        
        # RSI (Relative Strength Index)
        for w in [30]:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / loss
            features[f'{col}_rsi_{w}'] = 100 - (100 / (1 + rs))
        
        # MACD: difference of EMAs
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
    def __init__(self, train_data, gb_params=None, pair_params=None):
        # Use a view instead of copy where possible
        self.train_data = train_data  # Don't copy here, we'll copy only when needed
        self.models = {}  # Dictionary to hold one model per asset
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.window = 30
        self.running_price_paths = train_data.copy()  # Only copy here as we need to modify it
        
        # Initialize models with memory-efficient parameters
        for col in train_data.columns:
            self.models[col] = GradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=50,
                random_state=42,
                subsample=0.8,  # Use subsampling to reduce memory usage
                max_features='sqrt'  # Use sqrt of features to reduce memory
            )
        
        # Use provided parameters or defaults
        self.gb_params = gb_params if gb_params is not None else {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 50,
            'subsample': 0.8,
            'max_features': 'sqrt'
        }
        
        self.pair_params = pair_params if pair_params is not None else {
            'z_score_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss_threshold': 3.0,
            'strategy_weight': 0.7
        }
        
        # Calculate spread parameters for pair trading
        self.spread_mean = None
        self.spread_std = None
        self.calculate_spread_params()
        
        # Train the models with progress updates
        print("\nTraining ML models...")
        self.train_models()
    
    def calculate_spread_params(self):
        """Calculate spread parameters for pair trading using Asset_4 and Asset_5."""
        # Use direct column access instead of creating intermediate series
        self.spread_mean = (self.train_data['Asset_4'] - self.train_data['Asset_5']).mean()
        self.spread_std = (self.train_data['Asset_4'] - self.train_data['Asset_5']).std()
    
    def calculate_z_score(self, spread):
        """Calculate z-score for pair trading."""
        return (spread - self.spread_mean) / self.spread_std
    
    def train_models(self):
        """Train models for each asset using ranking targets with memory optimization."""
        # Calculate features once and reuse
        features = calculate_technical_indicators(self.train_data, self.window)
        features = features.fillna(0)
        self.feature_columns = features.columns
        
        # Calculate forward returns efficiently
        forward_returns = pd.DataFrame()
        for col in self.train_data.columns:
            forward_returns[col] = self.train_data[col].pct_change().shift(-1)
        
        # Create ranking targets efficiently
        ranking_targets = pd.DataFrame(0, index=forward_returns.index, columns=forward_returns.columns)
        
        # Process in chunks to reduce memory usage
        chunk_size = 1000
        for start_idx in range(0, len(forward_returns), chunk_size):
            end_idx = min(start_idx + chunk_size, len(forward_returns))
            chunk = forward_returns.iloc[start_idx:end_idx]
            
            for idx in chunk.index:
                returns = chunk.loc[idx]
                if returns.isna().any():
                    continue
                ranks = returns.rank(ascending=False)
                ranking_targets.loc[idx, ranks <= 2] = 1  # Long top 2
                ranking_targets.loc[idx, ranks >= 5] = -1  # Short bottom 2
        
        # Clean up unused data
        del forward_returns
        
        # Remove rows with NaN values
        valid_idx = ~features.isna().any(axis=1) & ~ranking_targets.isna().any(axis=1)
        features = features[valid_idx]
        ranking_targets = ranking_targets[valid_idx]
        
        # Train one model per asset with memory cleanup
        for col in tqdm(self.train_data.columns, desc="Training models"):
            X = features
            y = ranking_targets[col]
            
            # Train model
            self.models[col].fit(X, y)
            
            # Clear memory after each model
            import gc
            gc.collect()
    
    def allocate_portfolio(self, current_prices):
        """Allocate portfolio based on model predictions and pair trading signals with memory optimization."""
        # Update the running price paths efficiently
        new_row = pd.DataFrame([current_prices], columns=self.train_data.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)
        
        # Calculate technical indicators efficiently
        features = calculate_technical_indicators(self.running_price_paths, self.window)
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        if self.feature_columns is not None:
            features = features[self.feature_columns]
        
        # Get predictions efficiently
        predictions = {}
        for col in self.train_data.columns:
            predictions[col] = self.models[col].predict(features.iloc[[-1]])[0]
        
        # Convert predictions to array and clean up
        ml_weights = np.array([predictions[col] for col in self.train_data.columns])
        del predictions
        
        # Calculate pair trading signal efficiently
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
        
        # Exit positions if the spread reverts
        if abs(z_score) < self.pair_params['exit_threshold']:
            pair_weights[3] = 0
            pair_weights[4] = 0
        
        # Stop loss if the spread widens excessively
        if abs(z_score) > self.pair_params['stop_loss_threshold']:
            pair_weights[3] = 0
            pair_weights[4] = 0
        
        # Combine ML and pair trading strategies
        final_weights = (self.pair_params['strategy_weight'] * ml_weights +
                         (1 - self.pair_params['strategy_weight']) * pair_weights)
        
        # Normalize weights to ensure market neutrality (sum-to-zero)
        final_weights = final_weights - np.mean(final_weights)
        final_weights = np.clip(final_weights, -1, 1)
        
        # Clean up memory
        import gc
        gc.collect()
        
        return final_weights

    def calculate_volatility(self, price_data):
        """Calculate rolling volatility for each asset."""
        return price_data.pct_change().rolling(window=20).std()
    
    def calculate_returns(self, price_data):
        """Calculate rolling returns for each asset."""
        return price_data.pct_change().rolling(window=20).mean()
    
    def optimize_weights(self, predicted_returns, historical_returns):
        """Optimize portfolio weights using predicted returns and risk."""
        num_assets = len(predicted_returns)
        mu = np.array(list(predicted_returns.values()))
        sigma = historical_returns.cov().values
        
        concentration_penalties = [0.1, 0.25, 0.5, 1.0]
        best_sharpe = float('-inf')
        best_weights = None
        
        for penalty in concentration_penalties:
            def neg_sharpe(w):
                port_return = np.dot(w, mu)
                port_vol = np.sqrt(np.dot(w, sigma @ w))
                concentration_penalty = -np.sum(w * w) * penalty
                drawdown_penalty = -np.max(np.abs(w)) * 0.1
                sharpe = port_return / port_vol if port_vol != 0 else -1e6
                return -(sharpe + concentration_penalty + drawdown_penalty)
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w)}]
            bounds = [(-1, 1)] * num_assets
            
            best_result = None
            best_sharpe_local = float('-inf')
            for _ in range(5):
                init_guess = np.random.uniform(-1, 1, num_assets)
                init_guess = init_guess - np.mean(init_guess)
                result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
                if result.success:
                    sharpe_local = -neg_sharpe(result.x)
                    if sharpe_local > best_sharpe_local:
                        best_sharpe_local = sharpe_local
                        best_result = result
            
            if best_result is not None and best_sharpe_local > best_sharpe:
                best_sharpe = best_sharpe_local
                best_weights = best_result.x
        
        if best_weights is not None:
            return best_weights
        
        return np.zeros(num_assets)

def grading(train_data, test_data):
    # Goal: Design weights to maximize risk-adjusted return (Sharpe ratio)
    
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = MLAllocator(train_data)  # Models are trained here during initialization

    print("\nAllocating portfolio weights...")
    for i in tqdm(range(0, len(test_data)), desc="Allocating weights"):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]  # Start with $1

    print("\nCalculating returns...")
    for i in tqdm(range(len(test_data) - 1), desc="Calculating returns"):
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

def main():
    print("\nStarting main execution...")
    sharpe, capital, weights = grading(TRAIN, TEST)
    print(f"\nML Strategy Sharpe Ratio: {sharpe:.4f}")
    
    # Plot capital evolution
    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("Capital Evolution")
    plt.plot(np.arange(len(TEST)), capital)
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.show()
    
    # Plot log capital
    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("Log Capital Evolution")
    plt.plot(np.arange(len(TEST)), np.log(capital))
    plt.xlabel("Time")
    plt.ylabel("Log Capital")
    plt.show()
    
    # Plot portfolio weights
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
