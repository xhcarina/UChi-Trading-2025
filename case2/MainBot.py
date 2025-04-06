import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize  # Allowed dependency

# Read the CSV file.
data = pd.read_csv('Case2.csv', index_col=0)

'''
We recommend that you change your train and test split
'''
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
    def __init__(self, train_data):
        '''
        Initialize the allocator with historical data.
        The running_price_paths field will store the entire price history.
        '''
        self.train_data = train_data.copy()
        self.running_price_paths = train_data.copy()
        
    def equal_weight_allocation(self):
        num_assets = self.train_data.shape[1]
        return np.array([1/num_assets] * num_assets)
    
    def markowitz_allocation(self, returns):
        num_assets = returns.shape[1]
        mu = returns.mean().values          # Average daily return per asset
        sigma = returns.cov().values          # Covariance matrix of returns

        # Objective: minimize negative Sharpe ratio (risk-free rate assumed to be zero)
        def neg_sharpe(w):
            port_return = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, sigma.dot(w)))
            # safeguard division by zero
            if port_vol == 0:
                return 1e6
            return -port_return / port_vol
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(-1, 1)] * num_assets
        init_guess = np.array([1/num_assets] * num_assets)
        result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
        else:
            return init_guess
    
    def risk_parity_allocation(self, returns):
        num_assets = returns.shape[1]
        sigma = returns.cov().values
        n = num_assets
        
        # Objective: equalize the risk contributions.
        def risk_parity_obj(w):
            port_var = np.dot(w, sigma.dot(w))
            risk_contrib = w * (sigma.dot(w))
            target = port_var / n
            return np.sum((risk_contrib - target)**2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(-1, 1)] * num_assets
        init_guess = np.array([1/num_assets] * num_assets)
        result = minimize(risk_parity_obj, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
        else:
            return init_guess
    
    def allocate_portfolio(self, asset_prices):
        '''
        asset_prices: np array (or Series) of length 6, prices of the 6 assets on a particular day.
        Returns:
            weights: np array of length 6, portfolio allocation for the next day.
        '''
        # Append the new day's prices to the running price history.
        new_row = pd.DataFrame([asset_prices])
        self.running_price_paths = self.running_price_paths.append(new_row, ignore_index=True)
    
        # If we do not yet have enough history to compute returns, use equal weight.
        if len(self.running_price_paths) < 2:
            return self.equal_weight_allocation()
    
        # Compute daily returns from the updated price history.
        historical_returns = self.running_price_paths.pct_change().dropna()
    
        # Calculate allocations from the three strategies.
        ew = self.equal_weight_allocation()
        mw = self.markowitz_allocation(historical_returns)
        rp = self.risk_parity_allocation(historical_returns)
    
        # Combine the strategies – here we take a simple average.
        combined_weights = (ew + mw + rp) / 3
    
        # Ensure that the weights are within [-1, 1].
        combined_weights = np.clip(combined_weights, -1, 1)
    
        return combined_weights

def grading(train_data, test_data): 
    '''
    Grading Script
    Simulates daily portfolio allocation and computes the Sharpe ratio.
    '''
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(len(test_data)):
        # test_data.iloc[i, :] is assumed to be a Series of asset prices for day i.
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :])
        if np.any(weights[i, :] < -1) or np.any(weights[i, :] > 1):
            raise Exception("Weights Outside of Bounds")
    
    # Simulate capital evolution over the test period.
    capital = [1]
    for i in range(len(test_data) - 1):
        current_prices = np.array(test_data.iloc[i, :])
        next_prices = np.array(test_data.iloc[i+1, :])
        # Calculate number of shares held using current weights.
        shares = capital[-1] * weights[i, :] / current_prices
        balance = capital[-1] - np.dot(shares, current_prices)
        net_change = np.dot(shares, next_prices)
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
        
    return sharpe, capital, weights

def main():
    sharpe, capital, weights = grading(TRAIN, TEST)
    # Sharpe gets printed to command line
    print("Sharpe Ratio:", sharpe)
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("Capital")
    plt.plot(np.arange(len(TEST)), capital)
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.show()
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.title("Weights")
    plt.plot(np.arange(len(TEST)), weights)
    plt.legend(TEST.columns)
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.show()

if __name__ == '__main__':
    main()
