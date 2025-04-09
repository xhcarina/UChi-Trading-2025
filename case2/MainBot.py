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

# Read the CSV file.
print("Reading CSV file...")
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')
print(f"Data loaded. Shape: {data.shape}")

# Split the data (80/20 split without shuffling to maintain time series nature)
print("Splitting data into train and test sets...")
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)
print(f"Train set shape: {TRAIN.shape}, Test set shape: {TEST.shape}")

def calculate_technical_indicators(df, window=30):
    """Calculate technical indicators for each asset."""
    features_dict = {}
    
    for col in df.columns:
        prices = df[col]
        features = {}
        
        # Basic price features
        features[f'{col}_returns'] = prices.pct_change()
        features[f'{col}_log_returns'] = np.log1p(prices).diff()
        
        # Moving averages
        for w in [5, 10, 20, 30]:
            ma = prices.rolling(window=w).mean()
            features[f'{col}_ma_{w}'] = ma
            features[f'{col}_ma_ratio_{w}'] = prices / ma
        
        # Volatility
        returns = features[f'{col}_returns']
        for w in [5, 10, 20, 30]:
            features[f'{col}_vol_{w}'] = returns.rolling(window=w).std()
        
        # Momentum
        for w in [5, 10, 20, 30]:
            features[f'{col}_mom_{w}'] = prices.pct_change(periods=w)
        
        # RSI
        for w in [14, 30]:
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
        
        # Bollinger Bands
        for w in [20]:
            mid = prices.rolling(window=w).mean()
            std = prices.rolling(window=w).std()
            features[f'{col}_bb_upper_{w}'] = mid + (std * 2)
            features[f'{col}_bb_lower_{w}'] = mid - (std * 2)
            features[f'{col}_bb_width_{w}'] = (features[f'{col}_bb_upper_{w}'] - features[f'{col}_bb_lower_{w}']) / mid
            features[f'{col}_bb_position_{w}'] = (prices - mid) / (std * 2)
        
        # Cross-asset correlations
        for other_col in df.columns:
            if other_col != col:
                features[f'{col}_{other_col}_corr_30'] = df[col].rolling(30).corr(df[other_col])
        
        features_dict.update(features)
    
    return pd.DataFrame(features_dict)

class MLAllocator():
    def __init__(self, train_data):
        self.train_data = train_data.copy()
        self.models = {}
        self.scaler = StandardScaler()
        self.window = 30
        self.running_price_paths = train_data.copy()
        self.fit_models()
    
    def prepare_ml_data(self, data):
        """Prepare features and targets for ML model."""
        features = calculate_technical_indicators(data, self.window)
        
        # Forward returns (target) - use 5-day forward returns
        forward_returns = {}
        for col in data.columns:
            forward_returns[col] = data[col].pct_change(periods=5).shift(-5)
        
        # Remove rows with NaN values
        features = features.fillna(0)  # Fill NaN features with 0
        valid_indices = features.index
        
        # Ensure targets align with features and handle NaN values
        y_dict = {}
        for col in forward_returns:
            y = forward_returns[col].loc[valid_indices]
            y = y.fillna(0)  # Fill NaN targets with 0
            y_dict[col] = y
            
        return features, y_dict
    
    def fit_models(self):
        """Train a model for each asset using cross-validation."""
        print("\nTraining ML models...")
        X, y_dict = self.prepare_ml_data(self.train_data)
        X = X.fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 100,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'random_state': 42
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for asset in tqdm(self.train_data.columns, desc="Training models"):
            y = y_dict[asset]
            
            # Train on each fold and keep the best model
            best_score = float('-inf')
            best_model = None
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Evaluate on validation set
                score = model.score(X_val, y_val)
                if score > best_score:
                    best_score = score
                    best_model = model
            
            self.models[asset] = best_model
    
    def predict_returns(self, features):
        """Predict returns for each asset."""
        X_scaled = self.scaler.transform(features)
        predictions = {}
        for asset, model in self.models.items():
            pred = model.predict(X_scaled)[-1]
            # Clip predictions to reasonable range
            pred = np.clip(pred, -0.1, 0.1)
            predictions[asset] = pred
        return predictions
    
    def optimize_weights(self, predicted_returns, historical_returns):
        """Optimize portfolio weights using predicted returns and risk-adjusted optimization."""
        num_assets = len(predicted_returns)
        mu = np.array(list(predicted_returns.values()))
        sigma = historical_returns.cov().values
        
        def neg_sharpe(w):
            port_return = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, sigma @ w))
            
            # Add regularization terms
            concentration_penalty = -np.sum(w * w)  # Penalize concentrated positions
            return -(port_return / port_vol if port_vol != 0 else 1e6) - 0.1 * concentration_penalty
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        bounds = [(0, 1)] * num_assets  # No shorting
        
        # Try multiple initial guesses
        best_result = None
        best_sharpe = float('-inf')
        
        for _ in range(5):
            init_guess = np.random.dirichlet(np.ones(num_assets))
            result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                sharpe = -neg_sharpe(result.x)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = result
        
        if best_result is not None:
            return best_result.x
        
        # Fallback to equal weights if optimization fails
        return np.ones(num_assets) / num_assets
    
    def allocate_portfolio(self, asset_prices, day_idx):
        """Generate portfolio allocation using ML predictions."""
        # Update price history
        new_row = pd.DataFrame([asset_prices], columns=self.train_data.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)
        
        # If not enough history, use equal weights
        if len(self.running_price_paths) < self.window + 2:
            return np.ones(len(self.train_data.columns)) / len(self.train_data.columns)
        
        # Prepare features for prediction
        features = calculate_technical_indicators(self.running_price_paths, self.window)
        if len(features) < 1:
            return np.ones(len(self.train_data.columns)) / len(self.train_data.columns)
        
        # Get predictions and optimize weights
        predicted_returns = self.predict_returns(features)
        historical_returns = self.running_price_paths.pct_change().dropna()
        
        return self.optimize_weights(predicted_returns, historical_returns)

def grading(train_data, test_data): 
    print("\nStarting portfolio allocation process...")
    num_assets = train_data.shape[1]
    weights = np.zeros((len(test_data.index), num_assets))
    alloc = MLAllocator(train_data)
    
    for i in tqdm(range(len(test_data)), desc="Allocating portfolio"):
        w = alloc.allocate_portfolio(test_data.iloc[i, :], i)
        if np.any(w < 0) or np.any(w > 1):
            raise Exception("Weights Outside of Bounds (0-1)")
        weights[i, :] = w
    
    # Simulate capital evolution
    print("\nSimulating capital evolution...")
    capital = [1.0]
    for i in tqdm(range(len(test_data) - 1), desc="Simulating returns"):
        current_prices = np.array(test_data.iloc[i, :])
        next_prices = np.array(test_data.iloc[i+1, :])
        shares = capital[-1] * weights[i, :] / current_prices
        balance = capital[-1] - np.dot(shares, current_prices)
        net_change = np.dot(shares, next_prices)
        capital.append(balance + net_change)
    capital = np.array(capital)
    
    # Compute performance metrics
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    daily_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    
    # Annualize Sharpe ratio for 5 years (1300 trading days)
    annualization_factor = np.sqrt(1300/5)  # sqrt(number of trading days / number of years)
    sharpe = daily_sharpe * annualization_factor
    
    return sharpe, capital, weights

def main():
    print("\nStarting main execution...")
    sharpe, capital, weights = grading(TRAIN, TEST)
    print(f"\nML Strategy Annualized Sharpe Ratio: {sharpe:.4f}")
    
    # Plot capital evolution
    plt.figure(figsize=(10, 6))
    plt.title("ML Strategy - Capital Evolution")
    plt.plot(np.arange(len(TEST)), capital, label="Capital")
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.legend()
    plt.show()
    
    # Plot portfolio weights
    plt.figure(figsize=(10, 6))
    plt.title("ML Strategy - Portfolio Weights")
    plt.plot(np.arange(len(TEST)), weights)
    plt.legend(TEST.columns)
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()
