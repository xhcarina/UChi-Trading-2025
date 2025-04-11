import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')

'''
We recommend that you change your train and test split
'''


TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)


class Allocator():
    def __init__(self, train_data, window=30*30):  # 30 rows per day, 30 days window
        '''
        Initialize the Allocator with training data.
        '''
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.window = window
        
        # Initial dynamic parameters
        self.spread_mean, self.spread_std = self.calculate_spread_params(train_data, 'Asset_4', 'Asset_5')
        self.secondary_spread_std = {
            ('Asset_4', 'Asset_6'): self.calculate_spread_params(train_data, 'Asset_4', 'Asset_6')[1],
            ('Asset_5', 'Asset_6'): self.calculate_spread_params(train_data, 'Asset_5', 'Asset_6')[1]
        }
        # Update with best thresholds
        self.z_score_threshold_primary = 2.5  # Example optimal value
        self.z_score_threshold_secondary = 3.0  # Example optimal value
        self.stop_loss_threshold = 3.5  # Example optimal value
        self.exit_threshold = 0.5  # Example optimal value

    def calculate_spread_params(self, data, asset_a, asset_b):
        '''
        Calculate the mean and standard deviation of the spread between two assets.
        '''
        spread = data[asset_a] - data[asset_b]
        return spread.mean(), spread.std()

    def update_spread_params(self):
        '''
        Update the mean and standard deviation of spreads using a rolling window.
        '''
        if len(self.running_price_paths) >= self.window:
            self.spread_mean, self.spread_std = self.calculate_spread_params(
                self.running_price_paths.iloc[-self.window:], 'Asset_4', 'Asset_5'
            )
            self.secondary_spread_std = {
                ('Asset_4', 'Asset_6'): self.calculate_spread_params(
                    self.running_price_paths.iloc[-self.window:], 'Asset_4', 'Asset_6'
                )[1],
                ('Asset_5', 'Asset_6'): self.calculate_spread_params(
                    self.running_price_paths.iloc[-self.window:], 'Asset_5', 'Asset_6'
                )[1]
            }

    def calculate_z_score(self, spread, mean, std):
        '''
        Calculate the z-score for a given spread.
        '''
        return (spread - mean) / std

    def allocate_portfolio(self, asset_prices):
        '''
        Allocate portfolio based on pair trading strategy.
        '''
        new_row = pd.DataFrame([asset_prices], columns=self.train_data.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)
        
        # Update spread parameters
        self.update_spread_params()
        
        # Calculate spread and z-score for primary pair (Asset_4, Asset_5)
        spread_4_5 = asset_prices.iloc[3] - asset_prices.iloc[4]
        z_score_4_5 = self.calculate_z_score(spread_4_5, self.spread_mean, self.spread_std)

        # Initialize weights
        weights = np.zeros(6)

        # Primary pair trading logic
        if z_score_4_5 > self.z_score_threshold_primary:
            # Short Asset_4, Long Asset_5
            weights[3] = -1 / self.spread_std
            weights[4] = 1 / self.spread_std
        elif z_score_4_5 < -self.z_score_threshold_primary:
            # Long Asset_4, Short Asset_5
            weights[3] = 1 / self.spread_std
            weights[4] = -1 / self.spread_std

        # Exit condition
        if abs(z_score_4_5) < self.exit_threshold:
            weights[3] = 0
            weights[4] = 0

        # Stop-loss condition
        if abs(z_score_4_5) >= self.stop_loss_threshold:
            weights[3] = 0
            weights[4] = 0

        # Secondary pair trading logic
        for (asset_a, asset_b), std in self.secondary_spread_std.items():
            spread = asset_prices.iloc[int(asset_a[-1]) - 1] - asset_prices.iloc[int(asset_b[-1]) - 1]
            z_score = self.calculate_z_score(spread, 0, std)
            
            if z_score > self.z_score_threshold_secondary:
                weights[int(asset_a[-1]) - 1] -= 0.5 / std
                weights[int(asset_b[-1]) - 1] += 0.5 / std
            elif z_score < -self.z_score_threshold_secondary:
                weights[int(asset_a[-1]) - 1] += 0.5 / std
                weights[int(asset_b[-1]) - 1] -= 0.5 / std

            # Exit condition for secondary pairs
            if abs(z_score) < self.exit_threshold:
                weights[int(asset_a[-1]) - 1] = 0
                weights[int(asset_b[-1]) - 1] = 0

            # Stop-loss condition for secondary pairs
            if abs(z_score) >= self.stop_loss_threshold:
                weights[int(asset_a[-1]) - 1] = 0
                weights[int(asset_b[-1]) - 1] = 0

        return weights




def grading(train_data, test_data): 
    # Goal: Design weights to maximize risk-adjusted return (Sharpe ratio)
    
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = Allocator(train_data)

    for i in range(0, len(test_data)):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :])
        # Goal: Ensure all weights stay within legal bounds [-1, 1]

    capital = [1]  # Goal: Start with $1 and aim to grow capital steadily

    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i, :])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i, :]))
        net_change = np.dot(shares, np.array(test_data.iloc[i + 1, :]))
        capital.append(balance + net_change)
        # Goal: Maximize capital[i+1] and minimize large swings in capital[i+1] - capital[i]

    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    # Goal: Maximize np.mean(returns), minimize np.std(returns)
    # Goal: Ensure capital[:-1] doesn't drop too low (drawdown protection)
    # Goal: Maximize capital[1:] over time (wealth accumulation)

    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
        # Goal: Sharpe ratio should be as high as possible
    else:
        sharpe = 0  # Goal: Avoid flat return strategies unless safely optimal
    
    return sharpe, capital, weights




sharpe, capital, weights = grading(TRAIN, TEST)
print(f"Sharpe Ratio: {sharpe:.4f}")


plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()