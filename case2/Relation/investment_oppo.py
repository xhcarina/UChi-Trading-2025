import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ====== 1) Load Data ======
data = pd.read_csv('/Users/apple/Desktop/UChicago-Trading-Competition-2025/case2/Case2.csv')
returns = data.pct_change().dropna()
print("Data loaded. #Days:", len(returns))

# ====== 2) Animation Parameters ======
window = 21       # rolling window length
N = 500           # random portfolios per frame
step = 500         # how many days to advance between frames
fps = 5           # frames per second for the animation

# Prepare figure & scatter
fig, ax = plt.subplots(figsize=(10, 6))
# Initialize an empty scatter
scatter = ax.scatter([], [], c=[], cmap='viridis', alpha=0.6)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Sharpe Ratio (RF=0)")

ax.set_xlim(0, 0.05)   # You may want to auto-scale instead
ax.set_ylim(-0.001, 0.002)  # Adjust to your data
ax.set_xlabel("Portfolio Volatility (Std Dev)")
ax.set_ylabel("Portfolio Expected Return")
ax.set_title("Dynamic Investment Opportunity Set (Overwritten)")

def animate(frame_idx):
    """
    frame_idx: index of the rolling window start
    """
    start_idx = frame_idx * step
    end_idx = start_idx + window
    if end_idx > len(returns):
        # If we go beyond the data, just stop
        return scatter,

    # 1) Slice the rolling window
    window_data = returns.iloc[start_idx:end_idx]

    # 2) Compute mean/cov in that window
    mu = window_data.mean().values
    cov = window_data.cov().values
    num_assets = len(mu)

    # 3) Random portfolios for this window
    results = np.zeros((N, 3))  # vol, ret, sharpe
    for i in range(N):
        w = np.random.random(num_assets)
        w /= w.sum()
        p_ret = np.dot(w, mu)
        p_vol = np.sqrt(w @ cov @ w)
        p_sharpe = p_ret / p_vol if p_vol != 0 else 0
        results[i, 0] = p_vol
        results[i, 1] = p_ret
        results[i, 2] = p_sharpe

    # 4) Update the existing scatter's data
    #    x=vol, y=ret, color=sharpe
    scatter.set_offsets(np.column_stack((results[:,0], results[:,1])))
    scatter.set_array(results[:,2])  # update color

    # Optionally rescale color if needed
    scatter.set_clim(results[:,2].min(), results[:,2].max())

    ax.set_title(f"Dynamic IOS (Days {start_idx} to {end_idx})")
    return scatter,

num_frames = (len(returns) - window) // step
ani = FuncAnimation(fig, animate, frames=num_frames, interval=500, blit=False)

plt.tight_layout()
plt.show()

# If you want to save the animation as an MP4, do:
#writer = FFMpegWriter(fps=fps)
#ani.save("dynamic_markowitz.mp4", writer=writer)
