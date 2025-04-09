import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class MarkowitzPortfolio:
    def __init__(self, train_data, risk_aversion=1.0, min_weight=-0.2, max_weight=0.2):
        """
        Initialize the Markowitz portfolio optimizer.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data containing asset prices
        risk_aversion : float
            Risk aversion parameter (higher = more risk averse)
        min_weight : float
            Minimum allowed weight for any asset
        max_weight : float
            Maximum allowed weight for any asset
        """
        self.train_data = train_data
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.n_assets = train_data.shape[1]
        
        # Calculate returns from price data
        self.returns = self.train_data.pct_change().dropna()
        
        # Calculate expected returns and covariance matrix
        self.expected_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        # Add small diagonal term for numerical stability
        self.cov_matrix += np.eye(self.n_assets) * 1e-6
        
    def objective_function(self, weights):
        """
        Objective function for portfolio optimization:
        Maximize: expected_return - risk_aversion * variance
        """
        portfolio_return = np.sum(weights * self.expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        return -(portfolio_return - self.risk_aversion * portfolio_variance)
    
    def optimize_weights(self):
        """
        Optimize portfolio weights using Markowitz mean-variance optimization.
        
        Returns:
        --------
        np.ndarray
            Optimal portfolio weights
        """
        # Initial guess (equal weights)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds for each weight
        bounds = [(self.min_weight, self.max_weight) for _ in range(self.n_assets)]
        
        # Optimize
        result = minimize(
            self.objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            print("Optimization failed, returning equal weights")
            return np.ones(self.n_assets) / self.n_assets

def grading(train_data, test_data):
    """
    Grading function to evaluate the Markowitz portfolio strategy.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
        
    Returns:
    --------
    float
        Sharpe ratio
    np.ndarray
        Capital evolution
    np.ndarray
        Portfolio weights
    """
    # Initialize portfolio
    portfolio = MarkowitzPortfolio(train_data)
    
    # Get optimal weights
    weights = portfolio.optimize_weights()
    
    # Initialize capital and weights array
    capital = [1.0]
    weights_array = np.zeros((len(test_data), len(weights)))
    
    # Simulate trading
    for i in range(len(test_data) - 1):
        # Store weights
        weights_array[i] = weights
        
        # Calculate returns
        current_prices = test_data.iloc[i]
        next_prices = test_data.iloc[i + 1]
        returns = (next_prices - current_prices) / current_prices
        
        # Update capital
        capital.append(capital[-1] * (1 + np.sum(weights * returns)))
    
    # Calculate Sharpe ratio
    returns = np.diff(capital) / capital[:-1]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    
    return sharpe, np.array(capital), weights_array

def main():
    # Read data
    data = pd.read_csv('Case2.csv', index_col=0)
    
    # Split into train and test
    train_size = int(0.7 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Run strategy
    sharpe, capital, weights = grading(train_data, test_data)
    
    print(f"Sharpe Ratio: {sharpe:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot capital evolution
    plt.subplot(2, 1, 1)
    plt.plot(capital)
    plt.title('Capital Evolution')
    plt.xlabel('Time')
    plt.ylabel('Capital')
    
    # Plot weights
    plt.subplot(2, 1, 2)
    plt.plot(weights)
    plt.title('Portfolio Weights')
    plt.xlabel('Time')
    plt.ylabel('Weight')
    plt.legend(data.columns)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 