import numpy as np
import pandas as pd
from pair import Allocator, grading

# Load data
train_data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')
test_data = train_data.copy()  # Use the same data for simplicity

# Define parameter ranges
primary_thresholds = np.arange(1.5, 3.0, 0.5)
secondary_thresholds = np.arange(2.0, 3.5, 0.5)
stop_loss_thresholds = np.arange(2.5, 4.0, 0.5)
exit_thresholds = np.arange(0.3, 0.8, 0.1)

best_sharpe = -np.inf
best_params = {}

# Grid search
for primary in primary_thresholds:
    for secondary in secondary_thresholds:
        for stop_loss in stop_loss_thresholds:
            for exit_thresh in exit_thresholds:
                # Initialize Allocator with current parameters
                allocator = Allocator(train_data, window=30*30)  # 30 rows per day, 30 days window
                allocator.z_score_threshold_primary = primary
                allocator.z_score_threshold_secondary = secondary
                allocator.stop_loss_threshold = stop_loss
                allocator.exit_threshold = exit_thresh

                # Evaluate Sharpe ratio
                sharpe, _, _ = grading(train_data, test_data)

                # Check if this is the best Sharpe ratio
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        'primary': primary,
                        'secondary': secondary,
                        'stop_loss': stop_loss,
                        'exit': exit_thresh
                    }

# Output the best parameters
print("Best Sharpe Ratio:", best_sharpe)
print("Best Parameters:", best_params) 