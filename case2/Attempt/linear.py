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
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import coint
from joblib import Parallel, delayed
import functools
from scipy.cluster.hierarchy import fcluster

print("Starting script...")
start_time = time.time()

# Read the CSV file.
print("Reading CSV file...")
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')
print(f"Data loaded. Shape: {data.shape}")

print("Splitting data into train and test sets...")
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)
print(f"Train set shape: {TRAIN.shape}, Test set shape: {TEST.shape}")

def calculate_technical_indicators(df, window=20):
    """Calculate enhanced technical indicators for each asset."""
    features_dict = {}
    
    for col in df.columns:
        prices = df[col]
        features = {}
        
        # Basic price features
        features[f'{col}_returns'] = prices.pct_change()
        features[f'{col}_log_returns'] = np.log1p(prices).diff()
        
        # Moving averages with different windows - reduced sizes
        for w in [5, 10, 20, 30]:
            ma = prices.rolling(window=w).mean()
            features[f'{col}_ma_{w}'] = ma
            features[f'{col}_ma_ratio_{w}'] = prices / ma
            features[f'{col}_ma_diff_{w}'] = ma.diff()
        
        # Volatility features - reduced sizes
        returns = features[f'{col}_returns']
        for w in [5, 10, 20, 30]:
            features[f'{col}_vol_{w}'] = returns.rolling(window=w).std()
            features[f'{col}_vol_ratio_{w}'] = returns.rolling(window=w).std() / returns.rolling(window=w*2).std()
        
        # Momentum features - reduced sizes
        for w in [5, 10, 20]:
            features[f'{col}_mom_{w}'] = prices.pct_change(periods=w)
            features[f'{col}_mom_ma_{w}'] = features[f'{col}_mom_{w}'].rolling(window=w).mean()
        
        # RSI with different windows - reduced sizes
        for w in [14, 30]:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / loss
            features[f'{col}_rsi_{w}'] = 100 - (100 / (1 + rs))
        
        # MACD with different parameters - keeping just one
        fast, slow, signal = 12, 26, 9
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        features[f'{col}_macd_{fast}_{slow}'] = macd
        features[f'{col}_macd_signal_{fast}_{slow}'] = signal_line
        features[f'{col}_macd_hist_{fast}_{slow}'] = macd - signal_line
        
        # Bollinger Bands with different windows - reduced sizes
        for w in [20, 30]:
            mid = prices.rolling(window=w).mean()
            std = prices.rolling(window=w).std()
            features[f'{col}_bb_upper_{w}'] = mid + (std * 2)
            features[f'{col}_bb_lower_{w}'] = mid - (std * 2)
            features[f'{col}_bb_width_{w}'] = (features[f'{col}_bb_upper_{w}'] - features[f'{col}_bb_lower_{w}']) / mid
            features[f'{col}_bb_position_{w}'] = (prices - mid) / (std * 2)
        
        # Market regime indicators - reduced window
        features[f'{col}_trend_strength'] = prices.rolling(window=30).mean() / prices.rolling(window=10).mean()
        features[f'{col}_volatility_regime'] = returns.rolling(window=30).std() / returns.rolling(window=10).std()
        
        # Cross-asset features - limit to 2 correlations per pair
        for other_col in df.columns:
            if other_col != col:
                # Correlation features - just one time window
                features[f'{col}_{other_col}_corr_30'] = df[col].rolling(30).corr(df[other_col])
                
                # Relative strength
                features[f'{col}_{other_col}_relative_strength'] = df[col] / df[other_col]
                features[f'{col}_{other_col}_relative_strength_ma'] = features[f'{col}_{other_col}_relative_strength'].rolling(window=20).mean()
        
        features_dict.update(features)
    
    return pd.DataFrame(features_dict)

def calculate_risk_parity_weights(returns):
    """Calculate risk parity weights."""
    n = returns.shape[1]
    cov = returns.cov().values
    
    def risk_parity_objective(w):
        w = np.array(w)
        port_vol = np.sqrt(w.T @ cov @ w)
        risk_contrib = (w * (cov @ w)) / port_vol
        return np.sum((risk_contrib - risk_contrib.mean())**2)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    
    # Use equal weights as initial guess for faster convergence
    result = minimize(risk_parity_objective, 
                     x0=np.ones(n)/n,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints,
                     options={'maxiter': 100, 'ftol': 1e-4})
    
    return result.x

def calculate_min_variance_weights(returns):
    """Calculate minimum variance weights."""
    n = returns.shape[1]
    cov = returns.cov().values
    
    def portfolio_vol(w):
        return np.sqrt(w.T @ cov @ w)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    
    result = minimize(portfolio_vol,
                     x0=np.ones(n)/n,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    return result.x

def calculate_hrp_weights(returns):
    """Calculate Hierarchical Risk Parity weights."""
    # Calculate correlation matrix
    corr = returns.corr()
    
    # Calculate distance matrix
    dist = np.sqrt(2*(1 - corr))
    
    # Perform hierarchical clustering
    link = linkage(squareform(dist), 'single')
    
    # Get cluster order
    order = dendrogram(link, no_plot=True)['leaves']
    
    # Reorder correlation matrix
    corr = corr.iloc[order, order]
    
    # Calculate inverse variance weights
    ivols = 1 / returns.std()
    weights = ivols / ivols.sum()
    
    # Adjust weights based on clustering
    ordered_weights = weights.iloc[order]
    
    return ordered_weights.reindex(returns.columns)

class PairTradingStrategy:
    def __init__(self, data):
        self.data = data
        self.pairs = {
            'primary': ('Asset_4', 'Asset_5'),
            'secondary1': ('Asset_1', 'Asset_2'),
            'secondary2': ('Asset_1', 'Asset_3')
        }
        self.spread_stats = {}
        self.positions = {}
        self.initialize_pairs()
    
    def initialize_pairs(self):
        """Initialize pair trading statistics."""
        for pair_name, (asset1, asset2) in self.pairs.items():
            spread = self.data[asset1] - self.data[asset2]
            self.spread_stats[pair_name] = {
                'mean': spread.mean(),
                'std': spread.std(),
                'z_score': (spread - spread.mean()) / spread.std()
            }
            self.positions[pair_name] = None
    
    def calculate_z_score(self, current_prices, pair_name):
        """Calculate current z-score for a pair."""
        asset1, asset2 = self.pairs[pair_name]
        current_spread = current_prices[asset1] - current_prices[asset2]
        stats = self.spread_stats[pair_name]
        return (current_spread - stats['mean']) / stats['std']
    
    def get_pair_weights(self, current_prices):
        """Get weights for pair trading positions."""
        weights = np.zeros(len(current_prices))
        
        # Different thresholds and allocations for different pairs
        pair_configs = {
            'primary': {
                'entry_threshold': 1.5,
                'exit_threshold': 0.5,
                'max_exposure': 0.2  # 20% for primary pair (4-5)
            },
            'secondary1': {
                'entry_threshold': 1.8,
                'exit_threshold': 0.6,
                'max_exposure': 0.05  # 5% for secondary pair (1-2)
            },
            'secondary2': {
                'entry_threshold': 1.8,
                'exit_threshold': 0.6,
                'max_exposure': 0.05  # 5% for secondary pair (1-3)
            }
        }
        
        for pair_name, (asset1, asset2) in self.pairs.items():
            z_score = self.calculate_z_score(current_prices, pair_name)
            config = pair_configs[pair_name]
            
            # Entry/exit logic with pair-specific thresholds
            if abs(z_score) > config['entry_threshold'] and self.positions[pair_name] is None:  # Entry
                if z_score > config['entry_threshold']:  # Long asset1, short asset2
                    self.positions[pair_name] = (1, -1)
                else:  # Long asset2, short asset1
                    self.positions[pair_name] = (-1, 1)
            elif abs(z_score) < config['exit_threshold'] and self.positions[pair_name] is not None:  # Exit
                self.positions[pair_name] = None
            
            # Apply position if active
            if self.positions[pair_name] is not None:
                pos1, pos2 = self.positions[pair_name]
                # Normalize weights to ensure they sum to 0
                total_pos = abs(pos1) + abs(pos2)
                asset1_idx = list(current_prices.index).index(asset1)
                asset2_idx = list(current_prices.index).index(asset2)
                weights[asset1_idx] += (pos1 / total_pos) * config['max_exposure']
                weights[asset2_idx] += (pos2 / total_pos) * config['max_exposure']
        
        return weights

class MLAllocator():
    def __init__(self, train_data):
        self.train_data = train_data.copy()
        self.models = {}
        self.scaler = StandardScaler()
        self.window = 30  # Reduced from 60
        self.running_price_paths = train_data.copy()
        self.vol_target = 0.15
        self.pair_strategy = PairTradingStrategy(train_data)
        self.fit_models()
    
    def prepare_ml_data(self, data):
        """Prepare enhanced features and targets for ML model."""
        features = calculate_technical_indicators(data)
        
        # Forward returns (target) - use multiple horizons
        forward_returns = {}
        for col in data.columns:
            for horizon in [1, 5, 10]:
                forward_returns[f'{col}_ret_{horizon}'] = data[col].pct_change(periods=horizon).shift(-horizon)
        
        # Remove rows with NaN values
        features = features.fillna(0)
        valid_indices = features.index
        
        # Ensure targets align with features
        y_dict = {}
        for col in forward_returns:
            y = forward_returns[col].loc[valid_indices]
            y = y.fillna(0)
            y_dict[col] = y
            
        return features, y_dict
    
    def train_single_asset(self, asset, X_scaled, y_dict, params):
        """Train moels for a single asset."""
        horizon_models = {}
        # Only use 1 and 5 day horizons for faster training
        for horizon in [1, 5]:
            y = y_dict[f'{asset}_ret_{horizon}']
            model = xgb.XGBRegressor(**params)
            model.fit(X_scaled, y, verbose=False)
            horizon_models[horizon] = model
        return asset, horizon_models
    
    def fit_models(self):
        """Train XGBoost models for each asset using parallel processing."""
        print("\nTraining ML models...")
        X, y_dict = self.prepare_ml_data(self.train_data)
        X = X.fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Ultra-optimized XGBoost parameters for speed
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,  # Reduced from 4
            'learning_rate': 0.25,  # Increased from 0.2
            'n_estimators': 30,  # Reduced from 50
            'min_child_weight': 4,
            'subsample': 0.5,  # Reduced from 0.6
            'colsample_bytree': 0.5,  # Reduced from 0.6
            'tree_method': 'hist',
            'gamma': 0.5,  # Increased from 0.3
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Parallel training of models with fewer horizons
        results = Parallel(n_jobs=-1)(
            delayed(self.train_single_asset)(asset, X_scaled, y_dict, params)
            for asset in self.train_data.columns
        )
        
        # Store results
        for asset, models in results:
            self.models[asset] = models
    
    def predict_returns(self, features):
        """Predict returns using XGBoost models."""
        X_scaled = self.scaler.transform(features)
        predictions = {}
        
        for asset in self.models:
            # Get predictions for each horizon
            horizon_preds = []
            for horizon in [1, 5]:  # Reduced from [1, 5, 10]
                pred = self.models[asset][horizon].predict(X_scaled)[-1]
                horizon_preds.append(pred)
            
            # Weight predictions by horizon
            weights = np.array([0.7, 0.3])  # More weight to shorter horizon
            pred = np.average(horizon_preds, weights=weights)
            
            # Clip predictions to reasonable range
            pred = np.clip(pred, -0.1, 0.1)
            predictions[asset] = pred
        
        return predictions
    
    def optimize_weights(self, predicted_returns, historical_returns):
        """Optimize portfolio weights using predicted returns and risk-adjusted optimization."""
        num_assets = len(predicted_returns)
        mu = np.array(list(predicted_returns.values()))
        sigma = historical_returns.cov().values
        
        # Cluster assets based on correlation
        corr = historical_returns.corr()
        dist = np.sqrt(2*(1 - corr))
        link = linkage(squareform(dist), 'single')
        clusters = fcluster(link, 0.5, criterion='distance')
        
        # Calculate cluster weights - fix for inhomogeneous shape
        cluster_weights = {}
        for cluster_id in np.unique(clusters):
            cluster_assets = np.where(clusters == cluster_id)[0]
            if len(cluster_assets) > 0:
                cluster_returns = historical_returns.iloc[:, cluster_assets]
                cluster_cov = cluster_returns.cov()
                cluster_vol = np.sqrt(np.diag(cluster_cov))
                # Calculate scalar weight for cluster
                if len(cluster_vol) > 0:
                    cluster_weights[cluster_id] = 1.0 / cluster_vol.mean()
                else:
                    cluster_weights[cluster_id] = 1.0
        
        # Calculate sum of weights for normalization
        total_cluster_weight = sum(cluster_weights.values())
        
        # Combine cluster weights with individual asset weights
        def neg_sharpe(w):
            # Portfolio return
            port_return = np.dot(w, mu)
            
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(w, sigma @ w))
            
            # Cluster concentration penalty
            cluster_penalty = 0
            for cluster_id in np.unique(clusters):
                cluster_assets = np.where(clusters == cluster_id)[0]
                if len(cluster_assets) > 0:
                    cluster_weight = np.sum(w[cluster_assets])
                    target_weight = cluster_weights[cluster_id] / total_cluster_weight
                    cluster_penalty += (cluster_weight - target_weight)**2
            
            # Individual concentration penalty
            concentration_penalty = np.sum(w * w)
            
            # Combine penalties
            total_penalty = 0.1 * cluster_penalty + 0.1 * concentration_penalty
            
            return -(port_return / port_vol if port_vol != 0 else 1e6) - total_penalty
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(num_assets)]
        
        # Single optimization with early stopping
        result = minimize(neg_sharpe, 
                         np.ones(num_assets) / num_assets, 
                         method='SLSQP', 
                         bounds=bounds, 
                         constraints=constraints,
                         options={'maxiter': 100, 'ftol': 1e-4})
        
        if result.success:
            return result.x
        return np.ones(num_assets) / num_assets
    
    def allocate_portfolio(self, asset_prices, day_idx):
        """Generate portfolio allocation using ensemble of strategies."""
        # Update price history
        new_row = pd.DataFrame([asset_prices], columns=self.train_data.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)
        
        # If not enough history, use equal weights
        if len(self.running_price_paths) < self.window + 2:
            return np.ones(len(self.train_data.columns)) / len(self.train_data.columns)
        
        # Calculate returns for various strategies
        returns = self.running_price_paths.pct_change().dropna()
        recent_returns = returns.tail(self.window)
        
        # Get ML strategy weights
        features = calculate_technical_indicators(self.running_price_paths, self.window)
        predicted_returns = self.predict_returns(features)
        ml_weights = self.optimize_weights(predicted_returns, recent_returns)
        
        # Get Minimum Variance weights
        min_var_weights = calculate_min_variance_weights(recent_returns)
        
        # Get Pair Trading weights
        pair_weights = self.pair_strategy.get_pair_weights(asset_prices)
        
        # Dynamic ensemble weighting based on recent performance
        recent_vol = np.sqrt(252) * recent_returns.std().mean()
        if recent_vol > 0.2:  # High volatility regime
            weights = (0.45 * ml_weights + 
                      0.25 * min_var_weights +
                      0.30 * pair_weights)  # Increased pair trading in high vol
        else:  # Normal volatility regime
            weights = (0.55 * ml_weights + 
                      0.25 * min_var_weights +
                      0.20 * pair_weights)
        
        # Apply volatility targeting with tighter bounds
        vol_scalar = self.vol_target / recent_vol if recent_vol > 0 else 1
        weights = weights * min(vol_scalar, 1.25)  # Tighter constraint for less risk
        
        # Ensure weights sum to 1 and are within bounds
        weights = weights / weights.sum()
        weights = np.clip(weights, 0, 1)
        
        return weights

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
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    return sharpe, capital, weights

def main():
    print("\nStarting main execution...")
    sharpe, capital, weights = grading(TRAIN, TEST)
    print(f"\nML Strategy Sharpe Ratio: {sharpe:.4f}")
    
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
