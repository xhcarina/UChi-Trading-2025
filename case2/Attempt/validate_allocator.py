import numpy as np
import pandas as pd
from pair import Allocator

# Load data
train_data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')

# Initialize Allocator
allocator = Allocator(train_data)

# Validate dynamic parameters
print("Primary Pair (Asset_4, Asset_5):")
print(f"Calculated Mean Spread: {allocator.spread_mean}")
print(f"Calculated Std Dev of Spread: {allocator.spread_std}")

print("\nSecondary Pairs:")
for (asset_a, asset_b), std in allocator.secondary_spread_std.items():
    print(f"Pair ({asset_a}, {asset_b}) Std Dev of Spread: {std}")

# Validate thresholds
print("\nThresholds:")
print(f"Primary Z-Score Threshold: {allocator.z_score_threshold_primary}")
print(f"Secondary Z-Score Threshold: {allocator.z_score_threshold_secondary}")
print(f"Stop-Loss Threshold: {allocator.stop_loss_threshold}")
print(f"Exit Threshold: {allocator.exit_threshold}")

# Test logic with a sample
sample_prices = train_data.iloc[0]
weights = allocator.allocate_portfolio(sample_prices)
print("\nSample Weights:")
print(weights)

# Simulate a few steps
print("\nSimulating a few steps:")
for i in range(5):
    sample_prices = train_data.iloc[i]
    weights = allocator.allocate_portfolio(sample_prices)
    print(f"Step {i+1} Weights: {weights}") 