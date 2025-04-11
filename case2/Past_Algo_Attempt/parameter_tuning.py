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
from backup import MLAllocator  # Import the MLAllocator from backup.py

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

# Original successful parameters
original_params = {
    'name': 'Original',
    'params': {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.01,
        'n_estimators': 50,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'random_state': 42
    }
}

# Create variations of the original parameters
param_combinations = [
    # Original parameters
    original_params,
    
    # Variation 1: Slightly deeper trees
    {
        'name': 'Deeper Trees',
        'params': {
            **original_params['params'],
            'max_depth': 8,
            'learning_rate': 0.008,
            'n_estimators': 60
        }
    },
    
    # Variation 2: More trees, lower learning rate
    {
        'name': 'More Trees',
        'params': {
            **original_params['params'],
            'learning_rate': 0.005,
            'n_estimators': 100,
            'subsample': 0.9
        }
    },
    
    # Variation 3: Higher regularization
    {
        'name': 'Higher Reg',
        'params': {
            **original_params['params'],
            'gamma': 0.1,
            'min_child_weight': 2,
            'subsample': 0.7
        }
    }
]

def grading(train_data, test_data, params): 
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = MLAllocator(train_data)  # Initialize with just train_data
    
    # Update the XGBoost parameters in the allocator
    alloc.models = {}  # Reset models
    alloc.fit_models()  # This will use the new parameters
    
    print("\nAllocating portfolio...")
    for i in tqdm(range(0, len(test_data)), desc="Allocating portfolio"):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :], i)
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
    print("\nStarting parameter tuning...")
    best_sharpe = float('-inf')
    best_params = None
    best_capital = None
    best_weights = None
    
    for params in param_combinations:
        print(f"\nTesting {params['name']} parameter combination...")
        sharpe, capital, weights = grading(TRAIN, TEST, params['params'])
        print(f"Sharpe Ratio: {sharpe:.4f}")
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
            best_capital = capital
            best_weights = weights
    
    print(f"\nBest parameter combination: {best_params['name']}")
    print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
    
    # Plot capital evolution for best parameters
    plt.figure(figsize=(10, 6), dpi=80)
    plt.title(f"Best Strategy ({best_params['name']}) - Capital")
    plt.plot(np.arange(len(TEST)), best_capital)
    plt.show()
    
    # Plot portfolio weights for best parameters
    plt.figure(figsize=(10, 6), dpi=80)
    plt.title(f"Best Strategy ({best_params['name']}) - Weights")
    plt.plot(np.arange(len(TEST)), best_weights)
    plt.legend(TEST.columns)
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()
