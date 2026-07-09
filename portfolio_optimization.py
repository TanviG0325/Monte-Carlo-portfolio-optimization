# =============================================================================
# PROJECT 2: Python Portfolio Optimization + Monte Carlo Simulation
# Tools: yfinance, numpy, pandas, matplotlib
# Author: [Your Name]
# =============================================================================
# SETUP: Before running this file, install the required libraries.
# Open your terminal and run:
#   pip install yfinance numpy pandas matplotlib scipy
# Then run this file with: python portfolio_optimization.py
# =============================================================================


# --- STEP 1: IMPORT LIBRARIES ------------------------------------------------
# Think of libraries as toolboxes. Each one gives us specific capabilities.

import numpy as np                  # Numerical math (arrays, matrix operations)
import pandas as pd                 # Data manipulation (like Excel in Python)
import matplotlib.pyplot as plt     # Plotting charts
import matplotlib.gridspec as gridspec
import yfinance as yf               # Downloads free stock price data from Yahoo Finance
import warnings
warnings.filterwarnings('ignore')   # Suppresses non-critical warning messages


# =============================================================================
# STEP 2: CHOOSE YOUR STOCKS
# =============================================================================
# Pick 8-12 stocks across different sectors. This is your investment universe.
# You can change these tickers to anything you want to analyze.

TICKERS = [
    'AAPL',   # Apple          - Technology
    'MSFT',   # Microsoft      - Technology
    'GS',     # Goldman Sachs       - Financials
    'JNJ',    # Johnson & Johnson - Healthcare
    'XOM',    # ExxonMobil     - Energy
    'PG',     # Procter & Gamble - Consumer Staples
    'AMZN',   # Amazon         - Consumer Discretionary
    'NEE',    # NextEra Energy - Utilities
    'BRK-B',  # Berkshire Hathaway - Financials
    'NVDA',   # NVIDIA         - Technology
]

# How many years of historical price data to use
YEARS_OF_DATA = 5

# How many random portfolios to simulate for the Efficient Frontier
NUM_PORTFOLIOS = 10000

# How many future return paths to simulate in Monte Carlo
NUM_SIMULATIONS = 1000

# How many trading days to forecast (252 = 1 year)
FORECAST_DAYS = 252


# =============================================================================
# STEP 3: DOWNLOAD STOCK DATA
# =============================================================================

print("=" * 60)
print("PORTFOLIO OPTIMIZATION + MONTE CARLO SIMULATION")
print("=" * 60)
print(f"\nDownloading {YEARS_OF_DATA} years of price data for {len(TICKERS)} stocks...")

# yf.download pulls daily adjusted closing prices directly from Yahoo Finance
# "Adjusted" close accounts for stock splits and dividends — always use this
raw_data = yf.download(
    tickers=TICKERS,
    period=f"{YEARS_OF_DATA}y",   # e.g. "5y" = last 5 years
    auto_adjust=True,              # Use adjusted prices
    progress=False
)

# We only need the closing price each day, not open/high/low/volume
prices = raw_data['Close']

# Drop any days where data is missing (e.g., different exchange holidays)
prices = prices.dropna()

print(f"Data downloaded: {len(prices)} trading days, {len(prices.columns)} stocks")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")


# =============================================================================
# STEP 4: CALCULATE DAILY RETURNS
# =============================================================================
# A "return" = (today's price - yesterday's price) / yesterday's price
# pandas .pct_change() does this automatically for every stock at once

daily_returns = prices.pct_change().dropna()

# Annualized statistics — scale daily numbers up to a yearly view
# There are ~252 trading days in a year
annual_returns = daily_returns.mean() * 252          # Average yearly return per stock
annual_volatility = daily_returns.std() * np.sqrt(252)  # Yearly risk per stock

print("\n--- Individual Stock Stats (Annualized) ---")
stats_df = pd.DataFrame({
    'Annual Return': (annual_returns * 100).round(2).astype(str) + '%',
    'Annual Volatility': (annual_volatility * 100).round(2).astype(str) + '%'
})
print(stats_df.to_string())


# =============================================================================
# STEP 5: COVARIANCE MATRIX
# =============================================================================
# The covariance matrix shows how stocks move together.
# High covariance = they tend to go up/down at the same time (not ideal for diversification)
# Low/negative covariance = they move independently (good for diversification)

cov_matrix = daily_returns.cov() * 252   # Annualized covariance


# =============================================================================
# STEP 6: SIMULATE RANDOM PORTFOLIOS (EFFICIENT FRONTIER)
# =============================================================================
# We'll try 10,000 random combinations of portfolio weights
# For each combination, we calculate expected return, risk, and Sharpe Ratio

# Risk-free rate: the return you'd get from a "safe" investment (like T-bills)
# Used to calculate the Sharpe Ratio. Update this to the current 3-month T-bill rate.
RISK_FREE_RATE = 0.05   # 5% as of 2024 (approximate)

print(f"\nSimulating {NUM_PORTFOLIOS:,} random portfolios...")

# Arrays to store results for each simulated portfolio
port_returns = np.zeros(NUM_PORTFOLIOS)      # Expected annual return
port_volatility = np.zeros(NUM_PORTFOLIOS)   # Expected annual risk (std deviation)
port_sharpe = np.zeros(NUM_PORTFOLIOS)       # Sharpe Ratio
port_weights = np.zeros((NUM_PORTFOLIOS, len(TICKERS)))  # Weights for each stock

np.random.seed(42)   # Set a seed so results are reproducible

for i in range(NUM_PORTFOLIOS):

    # Generate random weights that sum to 1 (= 100% of portfolio)
    weights = np.random.random(len(TICKERS))
    weights = weights / np.sum(weights)     # Normalize so they sum to 1

    # Portfolio Expected Return = weighted average of individual stock returns
    p_return = np.dot(weights, annual_returns)

    # Portfolio Volatility (Risk) uses the covariance matrix
    # Formula: sqrt(w^T * Cov * w) — accounts for how stocks correlate with each other
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
    # Higher Sharpe = better return for the amount of risk taken
    p_sharpe = (p_return - RISK_FREE_RATE) / p_volatility

    # Store all results
    port_returns[i] = p_return
    port_volatility[i] = p_volatility
    port_sharpe[i] = p_sharpe
    port_weights[i] = weights

print("Simulation complete.")


# =============================================================================
# STEP 7: IDENTIFY KEY PORTFOLIOS
# =============================================================================

# Maximum Sharpe Ratio Portfolio — best risk-adjusted return
max_sharpe_idx = np.argmax(port_sharpe)
max_sharpe_return = port_returns[max_sharpe_idx]
max_sharpe_vol = port_volatility[max_sharpe_idx]
max_sharpe_ratio = port_sharpe[max_sharpe_idx]
max_sharpe_weights = port_weights[max_sharpe_idx]

# Minimum Variance Portfolio — lowest possible risk
min_vol_idx = np.argmin(port_volatility)
min_vol_return = port_returns[min_vol_idx]
min_vol_vol = port_volatility[min_vol_idx]
min_vol_ratio = port_sharpe[min_vol_idx]
min_vol_weights = port_weights[min_vol_idx]

print("\n--- Max Sharpe Ratio Portfolio ---")
print(f"  Expected Annual Return : {max_sharpe_return*100:.2f}%")
print(f"  Annual Volatility      : {max_sharpe_vol*100:.2f}%")
print(f"  Sharpe Ratio           : {max_sharpe_ratio:.4f}")
print("\n  Portfolio Weights:")
for ticker, weight in zip(TICKERS, max_sharpe_weights):
    if weight > 0.01:   # Only show positions > 1%
        print(f"    {ticker:<8}: {weight*100:.2f}%")

print("\n--- Minimum Variance Portfolio ---")
print(f"  Expected Annual Return : {min_vol_return*100:.2f}%")
print(f"  Annual Volatility      : {min_vol_vol*100:.2f}%")
print(f"  Sharpe Ratio           : {min_vol_ratio:.4f}")
print("\n  Portfolio Weights:")
for ticker, weight in zip(TICKERS, min_vol_weights):
    if weight > 0.01:
        print(f"    {ticker:<8}: {weight*100:.2f}%")


# =============================================================================
# STEP 8: MONTE CARLO SIMULATION
# =============================================================================
# Now we take the Max Sharpe portfolio and simulate 1,000 possible futures.
# This helps us understand the range of outcomes (best case, worst case, likely case).

print(f"\nRunning Monte Carlo Simulation ({NUM_SIMULATIONS:,} paths, {FORECAST_DAYS} days)...")

# Starting investment amount
INITIAL_INVESTMENT = 10000   # $10,000

# Calculate the daily expected return and volatility for our chosen portfolio
mc_weights = max_sharpe_weights
mc_daily_return = np.dot(mc_weights, daily_returns.mean())      # Expected daily return
mc_daily_vol = np.sqrt(np.dot(mc_weights.T, np.dot(daily_returns.cov(), mc_weights)))  # Daily vol

# Each simulation generates a random daily return path using a normal distribution
# Geometric Brownian Motion formula: price(t) = price(t-1) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
# where Z is a random standard normal number (mean 0, std 1)

simulation_results = np.zeros((FORECAST_DAYS, NUM_SIMULATIONS))

for sim in range(NUM_SIMULATIONS):
    portfolio_value = INITIAL_INVESTMENT
    daily_path = [portfolio_value]

    for day in range(FORECAST_DAYS - 1):
        # Random daily shock drawn from a normal distribution
        random_shock = np.random.normal(0, 1)

        # Daily return: drift + random component
        daily_ret = mc_daily_return + mc_daily_vol * random_shock

        # Update portfolio value
        portfolio_value = portfolio_value * (1 + daily_ret)
        daily_path.append(portfolio_value)

    simulation_results[:, sim] = daily_path

# Final portfolio values at end of simulation period
final_values = simulation_results[-1, :]

# Value at Risk (VaR) — worst outcome at a given confidence level
# 5th percentile = there's a 5% chance the portfolio falls below this value
var_95 = np.percentile(final_values, 5)
var_99 = np.percentile(final_values, 1)

print("\n--- Monte Carlo Results ($10,000 investment, 1-year horizon) ---")
print(f"  Mean outcome     : ${np.mean(final_values):>10,.2f}   (${np.mean(final_values)-INITIAL_INVESTMENT:+,.2f})")
print(f"  Median outcome   : ${np.median(final_values):>10,.2f}   (${np.median(final_values)-INITIAL_INVESTMENT:+,.2f})")
print(f"  Best case  (95th): ${np.percentile(final_values, 95):>10,.2f}   (${np.percentile(final_values, 95)-INITIAL_INVESTMENT:+,.2f})")
print(f"  Worst case  (5th): ${var_95:>10,.2f}   (${var_95-INITIAL_INVESTMENT:+,.2f})")
print(f"\n  VaR (95% confidence): There is a 5% chance of losing more than ${INITIAL_INVESTMENT - var_95:,.2f}")
print(f"  VaR (99% confidence): There is a 1% chance of losing more than ${INITIAL_INVESTMENT - var_99:,.2f}")


# =============================================================================
# STEP 9: CHARTS
# =============================================================================
print("\nGenerating charts...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Portfolio Optimization & Monte Carlo Simulation', fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)


# --- Chart 1: Efficient Frontier ---
ax1 = fig.add_subplot(gs[0, 0])
scatter = ax1.scatter(
    port_volatility * 100,
    port_returns * 100,
    c=port_sharpe,               # Color each dot by its Sharpe Ratio
    cmap='viridis',              # Color map: purple (low Sharpe) → yellow (high Sharpe)
    alpha=0.5,
    s=10
)
# Plot the two key portfolios as big stars
ax1.scatter(max_sharpe_vol * 100, max_sharpe_return * 100, marker='*', color='red', s=300, zorder=5, label='Max Sharpe')
ax1.scatter(min_vol_vol * 100, min_vol_return * 100, marker='*', color='blue', s=300, zorder=5, label='Min Variance')
plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
ax1.set_xlabel('Annual Volatility (%)')
ax1.set_ylabel('Annual Return (%)')
ax1.set_title('Efficient Frontier')
ax1.legend()


# --- Chart 2: Max Sharpe Portfolio Weights ---
ax2 = fig.add_subplot(gs[0, 1])
# Only show tickers with weight > 1%
labels = [t for t, w in zip(TICKERS, max_sharpe_weights) if w > 0.01]
values = [w for w in max_sharpe_weights if w > 0.01]
colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
wedges, texts, autotexts = ax2.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Max Sharpe Portfolio Weights')


# --- Chart 3: Monte Carlo Simulation Paths ---
ax3 = fig.add_subplot(gs[1, 0])
# Plot a sample of simulation paths (plotting all 1000 is slow)
sample_paths = np.random.choice(NUM_SIMULATIONS, 200, replace=False)
for sim in sample_paths:
    ax3.plot(simulation_results[:, sim], alpha=0.05, color='blue', linewidth=0.5)

# Overlay the key percentile paths
ax3.plot(np.percentile(simulation_results, 95, axis=1), color='green', linewidth=2, label='95th Pct')
ax3.plot(np.percentile(simulation_results, 50, axis=1), color='orange', linewidth=2, label='Median')
ax3.plot(np.percentile(simulation_results, 5, axis=1), color='red', linewidth=2, label='5th Pct (VaR)')
ax3.axhline(y=INITIAL_INVESTMENT, color='black', linestyle='--', linewidth=1, label='Initial Investment')
ax3.set_xlabel('Trading Days')
ax3.set_ylabel('Portfolio Value ($)')
ax3.set_title(f'Monte Carlo Simulation ({NUM_SIMULATIONS:,} paths)')
ax3.legend()
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))


# --- Chart 4: Distribution of Final Portfolio Values ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(final_values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax4.axvline(x=INITIAL_INVESTMENT, color='black', linestyle='--', linewidth=1.5, label='Initial ($10k)')
ax4.axvline(x=var_95, color='red', linestyle='--', linewidth=1.5, label=f'VaR 95% (${var_95:,.0f})')
ax4.axvline(x=np.mean(final_values), color='green', linestyle='--', linewidth=1.5, label=f'Mean (${np.mean(final_values):,.0f})')
ax4.set_xlabel('Final Portfolio Value ($)')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Final Portfolio Values (1 Year)')
ax4.legend()
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.savefig('portfolio_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as 'portfolio_results.png'")


# =============================================================================
# STEP 10: EXPORT RESULTS TO CSV (for your resume / portfolio writeup)
# =============================================================================

# Save the simulated portfolio data
results_df = pd.DataFrame({
    'Annual Return (%)': (port_returns * 100).round(4),
    'Annual Volatility (%)': (port_volatility * 100).round(4),
    'Sharpe Ratio': port_sharpe.round(4)
})
results_df.to_csv('efficient_frontier_data.csv', index=False)

# Save the weight allocations for key portfolios
weights_df = pd.DataFrame({
    'Ticker': TICKERS,
    'Max Sharpe Weight (%)': (max_sharpe_weights * 100).round(2),
    'Min Variance Weight (%)': (min_vol_weights * 100).round(2)
})
weights_df.to_csv('portfolio_weights.csv', index=False)

print("\nData exported:")
print("  - efficient_frontier_data.csv  (all simulated portfolios)")
print("  - portfolio_weights.csv        (key portfolio allocations)")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
