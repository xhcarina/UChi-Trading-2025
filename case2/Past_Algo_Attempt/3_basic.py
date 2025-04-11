import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.optimize import minimize

print("Starting script...")
start_time = time.time()

# Read the CSV file.
print("Reading CSV file...")
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')
print(f"Data loaded. Shape: {data.shape}")

# Define initial history window (first 10%) and future window (remaining 90%)
initial_window = int(len(data) * 0.1)
history = data.iloc[:initial_window].reset_index(drop=True)
future = data.iloc[initial_window:].reset_index(drop=True)
print(f"History window: {history.shape}, Future window: {future.shape}")

class Allocator:
    def __init__(self, history_data, strategy='equal'):
        self.history_data = history_data.copy()
        self.running_price_paths = history_data.copy()
        self.strategy = strategy

    def equal_weight_allocation(self):
        n = self.history_data.shape[1]
        return np.ones(n) / n

    def markowitz_allocation(self, returns):
        num_assets = returns.shape[1]
        mu = returns.mean().values
        sigma = returns.cov().values

        def neg_sharpe(w):
            port_return = w @ mu
            port_vol = np.sqrt(w @ (sigma @ w))
            return -port_return / port_vol if port_vol > 0 else 1e6

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = [(0, 1)] * num_assets
        init_guess = np.ones(num_assets) / num_assets

        result = minimize(neg_sharpe, init_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else init_guess

    def risk_parity_allocation(self, returns, max_iter=500, tol=1e-8):
        sigma = returns.cov().values
        n = sigma.shape[0]
        w = np.ones(n) / n

        for _ in range(max_iter):
            Sigma_w = sigma @ w
            port_var = w @ Sigma_w
            if port_var < 1e-15:
                return w
            rc = w * Sigma_w
            target = port_var / n

            w_new = np.array([
                w[i] * np.sqrt(target / rc[i]) if rc[i] > 0 else w[i]
                for i in range(n)
            ])
            w_new /= w_new.sum()

            if np.linalg.norm(w_new - w) < tol:
                return w_new
            w = w_new

        return w

    def allocate_portfolio(self, new_prices, day_idx):
        # Append today's prices to the running history
        new_row = pd.DataFrame([new_prices], columns=self.history_data.columns)
        self.running_price_paths = pd.concat(
            [self.running_price_paths, new_row], ignore_index=True
        )

        if len(self.running_price_paths) < 2:
            return self.equal_weight_allocation()

        returns = self.running_price_paths.pct_change().dropna()

        if self.strategy == 'equal':
            return self.equal_weight_allocation()
        elif self.strategy == 'markowitz':
            return self.markowitz_allocation(returns)
        elif self.strategy == 'risk_parity':
            return self.risk_parity_allocation(returns)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

def grading(history_data, future_data, strategy):
    print(f"\n=== Strategy: {strategy.upper()} ===")
    num_assets = history_data.shape[1]
    weights = np.zeros((len(future_data), num_assets))
    alloc = Allocator(history_data, strategy=strategy)

    # Generate weights for each day in the future window
    for i in tqdm(range(len(future_data)), desc=f"Allocating ({strategy})"):
        w = alloc.allocate_portfolio(future_data.iloc[i], i)
        if np.any(w < 0) or np.any(w > 1):
            raise Exception("Weights Outside of Bounds (0-1).")
        weights[i, :] = w

    # Simulate capital evolution over the future window
    capital = [1.0]
    for i in range(len(future_data) - 1):
        cur = future_data.iloc[i].values
        nxt = future_data.iloc[i + 1].values
        shares = capital[-1] * weights[i] / cur
        cash = capital[-1] - (shares @ cur)
        capital.append(cash + (shares @ nxt))
    capital = np.array(capital)

    # Compute Sharpe ratio
    port_returns = np.diff(capital) / capital[:-1]
    sharpe = port_returns.mean() / port_returns.std() if port_returns.std() > 0 else 0
    return sharpe, capital, weights

def main():
    strategies = ['equal', 'markowitz', 'risk_parity']
    results = {}

    for strat in strategies:
        sharpe, capital, wts = grading(history, future, strat)
        results[strat] = (sharpe, capital, wts)
        print(f"{strat.capitalize()} Sharpe Ratio: {sharpe:.4f}")

    # Plot capital evolution
    plt.figure(figsize=(10, 6))
    plt.title("Capital Evolution Comparison")
    for strat in strategies:
        plt.plot(np.arange(len(future)), results[strat][1], label=strat.capitalize())
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.legend()
    plt.show()

    # Plot weights over time
    for strat in strategies:
        plt.figure(figsize=(10, 6))
        plt.title(f"Weights - {strat.capitalize()}")
        plt.plot(np.arange(len(future)), results[strat][2])
        plt.legend(data.columns)
        plt.xlabel("Time")
        plt.ylabel("Weight")
        plt.show()

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
