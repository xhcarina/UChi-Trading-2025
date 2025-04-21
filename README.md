# **UChicago Trading Competition 2025**

## **Project Overview**

### **Case 1: Market-Making and Arbitrage Trading**

#### **Description**
The objective was to trade three stocks (**APT**, **DLR**, **MKJ**) and two ETFs (**AKAV**, **AKIM**). Each asset required different trading strategies based on structured news updates (**APT earnings**, **DLR signatures**) and unstructured news (**MKJ**). Trading sessions were divided into rounds consisting of multiple simulated trading days, each lasting 90 seconds. The goal was to maximize profitability while effectively managing risk and maintaining liquidity.

#### **Strategies Implemented**
- **APT Trading:** Leveraged quarterly earnings announcements and a constant Price-to-Earnings (**P/E**) ratio to predict fair prices.
- **DLR Valuation:** Used a Monte Carlo simulation to estimate the likelihood of achieving petition signature thresholds, dynamically adjusting fair prices.
- **MKJ Market-Making:** Developed a quantitative model using a sliding window algorithm for order book analysis, providing robust fair-price estimations.
- **ETF Arbitrage (AKAV, AKIM):** Capitalized on price discrepancies between ETFs and their underlying stocks through creation/redemption swaps, and executed inverse ETF arbitrage strategies with hedging techniques.
- **Dynamic Market-Making:** Employed adaptive pricing algorithms incorporating **"fade"** adjustments based on inventory positions and market activity, ensuring risk mitigation and market liquidity.

#### **Achievements**
- Successfully integrated quantitative pricing models responsive to structured and unstructured news events.
- Implemented real-time arbitrage identification and execution strategies, significantly reducing market risk exposure.
- Developed adaptive algorithms that dynamically managed spreads and position sizing under varying market volatility.

### **Case 2: Portfolio Optimization Asset Return Prediction**

#### **Description**
The task involved allocating a portfolio across six stocks to maximize returns and minimize variance over a ten-year horizon, leveraging historical price data for training predictive models. The portfolio allocations were updated dynamically at every market tick, ensuring efficient real-time adaptation to market movements.

#### **Strategies Implemented**
- **Machine Learning (Gradient Boosting Regressor):** Constructed predictive models to forecast asset returns based on extensive technical indicators including moving averages, volatility, RSI, MACD, and inter-asset spreads.
- **Pair Trading Strategy:** Integrated pair trading signals specifically between **Asset_4** and **Asset_5** to exploit mean-reversion opportunities based on calculated z-scores.
- **Markowitz Optimization:** Utilized historical return covariance matrices and predicted returns to determine optimal asset weights, optimizing for the Sharpe ratio with penalties for portfolio concentration and risk.
- **Dynamic Risk Management:** Continuously recalibrated portfolio allocations based on rolling predictions, volatility measures, and market neutrality adjustments.

#### **Achievements**
- Achieved an advanced portfolio allocation framework that effectively balanced return predictions and risk management, surpassing baseline equal-weighted strategies.
- Integrated robust feature engineering and ML modeling techniques to enhance predictive accuracy, significantly improving out-of-sample Sharpe ratio performance.
- Successfully navigated portfolio volatility with strategic hedging and dynamic weighting, demonstrating sophisticated risk management capabilities.

### **Final Bot & Performance**
The final integrated bot combined advanced quantitative strategies, dynamic arbitrage execution, predictive modeling, and portfolio optimization to consistently generate positive risk-adjusted returns. The bot exhibited resilience in highly volatile environments and demonstrated strategic adaptability in response to real-time market signals, significantly outperforming simpler benchmark approaches.

---

**Overall**, the developed system represents a comprehensive and nuanced approach to modern quantitative trading, reflecting deep insights into market dynamics, algorithmic adaptability, and effective risk management.
