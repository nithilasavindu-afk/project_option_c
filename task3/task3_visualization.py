# Task 3: Data Visualization Functions
# Simple candlestick and boxplot functions for financial data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from datetime import datetime
import os

def display_candlestick_chart(data, title="Stock Analysis", n_days=None):
    """
    Display stock market financial data using candlestick chart.

    Args:
        data: DataFrame with OHLC columns (Open, High, Low, Close)
        title: Chart title
        n_days: Number of recent days to show (n >= 1), None for all data
    """
    chart_data = data.copy()

    # Handle MultiIndex columns from yfinance
    if isinstance(chart_data.columns, pd.MultiIndex):
        chart_data.columns = chart_data.columns.get_level_values(0)

    # Remove any rows with NaN values
    chart_data = chart_data.dropna()

    if n_days is not None and n_days >= 1:
        chart_data = chart_data.tail(int(n_days))
        title = f"{title} (Last {int(n_days)} Days)"

    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Create candlestick chart using mplfinance
    fig, axes = mpf.plot(chart_data,
                         type='candle',
                         title=title,
                         ylabel='Price ($)',
                         volume=True,
                         style='charles',
                         figsize=(12, 8),
                         returnfig=True)

    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figures/candlestick_chart_{timestamp}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Candlestick chart saved to {filename}")

    plt.show()

def display_boxplot_chart(data, window_size=30, title="Price Movement Analysis"):
    """
    Display stock market financial data using boxplot chart.
    Shows price distribution for moving windows of consecutive trading days.

    Args:
        data: DataFrame with Close column
        window_size: Size of rolling window for analysis
        title: Chart title
    """
    chart_data = data.copy()

    # Handle MultiIndex columns from yfinance
    if isinstance(chart_data.columns, pd.MultiIndex):
        chart_data.columns = chart_data.columns.get_level_values(0)

    close_prices = chart_data['Close']
    
    # Create rolling windows
    rolling_data = []
    dates = []
    
    for i in range(window_size, len(close_prices), 5):  # Step by 5 days
        window = close_prices.iloc[i-window_size:i]
        rolling_data.append(window.values)
        dates.append(close_prices.index[i])
    
    # Create boxplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Price timeline
    ax1.plot(close_prices.index, close_prices.values, 'b-', alpha=0.7)
    ax1.set_title(f"{title} - Price Timeline")
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Boxplot of recent windows
    recent_count = min(15, len(rolling_data))
    boxplot_data = rolling_data[-recent_count:]
    labels = [d.strftime('%m/%d') for d in dates[-recent_count:]]
    
    bp = ax2.boxplot(boxplot_data, tick_labels=labels, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax2.set_title(f"Rolling {window_size}-Day Price Distribution")
    ax2.set_ylabel('Price ($)')
    ax2.set_xlabel('Date')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Save the figure
    os.makedirs('figures', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figures/boxplot_chart_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Boxplot chart saved to {filename}")

    plt.show()

def demo_task3():
    """Demo function to test the visualization functions"""
    print("Task 3: Data Visualization Demo")

    # Download sample data
    data = yf.download("META", start="2023-01-01", end="2024-01-01")
    print(f"Downloaded {len(data)} records")
    print("Data columns:", data.columns.tolist())
    print("Data shape:", data.shape)

    # Test candlestick chart
    print("Creating candlestick chart...")
    display_candlestick_chart(data, "META Stock Analysis", n_days=60)

    # Test boxplot chart
    print("Creating boxplot chart...")
    display_boxplot_chart(data, window_size=30, title="META Price Movement")

    print("Demo complete!")

if __name__ == "__main__":
    demo_task3()
