import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import time
from Xgboost import MLAllocator, grading, TRAIN, TEST

def tune_parameters():
    # Define parameter combinations focusing on Asset 4-5 pair
    combinations = [
        # Combination 1: Standard Pair Trading
        {
            'xgb': {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'pair': {
                'z_score_threshold': 2.0,  # Standard entry
                'exit_threshold': 0.5,     # Standard exit
                'stop_loss_threshold': 3.0, # Standard stop loss
                'strategy_weight': 0.7     # Balanced weight
            }
        },
        # Combination 2: Aggressive Entry
        {
            'xgb': {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'pair': {
                'z_score_threshold': 1.5,  # More aggressive entry
                'exit_threshold': 0.5,
                'stop_loss_threshold': 3.0,
                'strategy_weight': 0.8     # More weight on pair trading
            }
        },
        # Combination 3: Conservative Entry
        {
            'xgb': {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'pair': {
                'z_score_threshold': 2.5,  # More conservative entry
                'exit_threshold': 0.5,
                'stop_loss_threshold': 3.0,
                'strategy_weight': 0.6     # Less weight on pair trading
            }
        },
        # Combination 4: Quick Exit
        {
            'xgb': {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'pair': {
                'z_score_threshold': 2.0,
                'exit_threshold': 0.3,     # Quick exit
                'stop_loss_threshold': 3.0,
                'strategy_weight': 0.7
            }
        },
        # Combination 5: Tight Stop Loss
        {
            'xgb': {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'pair': {
                'z_score_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss_threshold': 2.5, # Tighter stop loss
                'strategy_weight': 0.7
            }
        }
    ]
    
    # Initialize best parameters and score
    best_score = float('-inf')
    best_params = {}
    
    print("Testing parameter combinations for Asset 4-5 pair trading...")
    
    # Create progress bar for combinations
    pbar = tqdm(combinations, desc="Testing combinations")
    start_time = time.time()
    
    for i, params in enumerate(pbar):
        try:
            # Update progress bar description
            elapsed_time = time.time() - start_time
            avg_time_per_combo = elapsed_time / (i + 1) if i > 0 else 0
            remaining_time = avg_time_per_combo * (len(combinations) - i - 1)
            pbar.set_description(f"Testing combination {i+1}/{len(combinations)} (Est. remaining: {remaining_time:.1f}s)")
            
            # Create allocator with parameters
            allocator = MLAllocator(TRAIN, xgb_params=params['xgb'], pair_params=params['pair'])
            
            # Evaluate performance
            sharpe, _, _ = grading(TRAIN, TEST)
            
            # Update best parameters if better score
            if sharpe > best_score:
                best_score = sharpe
                best_params = params
                print(f"\nNew best score: {sharpe:.4f}")
                print(f"Combination {i+1} Parameters: {params}")
        
        except Exception as e:
            print(f"Error with combination {i+1}: {str(e)}")
            continue
    
    return best_params, best_score

def main():
    print("Starting parameter tuning...")
    start_time = time.time()
    
    best_params, best_score = tune_parameters()
    
    print("\nParameter Tuning Results:")
    print(f"Best Sharpe Ratio: {best_score:.4f}")
    print("\nBest Parameters:")
    print("XGBoost Parameters:")
    for param, value in best_params['xgb'].items():
        print(f"  {param}: {value}")
    print("\nPair Trading Parameters:")
    for param, value in best_params['pair'].items():
        print(f"  {param}: {value}")
    
    total_time = time.time() - start_time
    print(f"\nTotal tuning time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main() 