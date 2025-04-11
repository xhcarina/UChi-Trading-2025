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
    def __init__(self, train_data):
        '''
        Anything data you want to store between days must be stored in a class field
        '''
        
        self.running_price_paths = train_data.copy()
        
        self.train_data = train_data.copy()
        
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        


        
    def allocate_portfolio(self, asset_prices):
        '''
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        '''
        self.running_price_paths.append(asset_prices, ignore_index = True)
    
        ### TODO Implement your code here
        weights = np.array([0,1,-1,0.5,0.1,-0.2])
        
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
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()