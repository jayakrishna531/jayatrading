"""
S&P 500 Stock Screener - Find stocks with RSI below 20 on daily timeframe
This script identifies oversold stocks in the S&P 500 index
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: pandas Series of closing prices
        period: RSI period (default 14)
    
    Returns:
        RSI values as pandas Series
    """
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    
    return pd.Series(rsi, index=prices.index)


def get_sp500_symbols():
    """
    Get list of S&P 500 stock symbols
    
    Returns:
        List of ticker symbols
    """
    # Fetch S&P 500 constituents from Wikipedia
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = tables[0]
    symbols = df['Symbol'].tolist()
    return symbols


def check_rsi_below_20(symbol, rsi_threshold=20):
    """
    Check if a stock's RSI is below threshold on daily timeframe
    
    Args:
        symbol: Stock ticker symbol
        rsi_threshold: RSI threshold (default 20)
    
    Returns:
        Dictionary with stock info and RSI value, or None
    """
    try:
        # Download daily data for the last 100 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty or len(data) < 14:
            return None
        
        # Calculate RSI
        rsi = calculate_rsi(data['Close'], period=14)
        current_rsi = rsi.iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Check if RSI is below threshold
        if current_rsi < rsi_threshold:
            return {
                'Symbol': symbol,
                'Price': round(current_price, 2),
                'RSI': round(current_rsi, 2),
                'Date': data.index[-1].strftime('%Y-%m-%d')
            }
        
        return None
    
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None


def scan_sp500_for_low_rsi(rsi_threshold=20):
    """
    Scan all S&P 500 stocks for RSI below threshold
    
    Args:
        rsi_threshold: RSI threshold to filter by
    
    Returns:
        DataFrame with stocks matching criteria
    """
    print("Fetching S&P 500 constituents...")
    symbols = get_sp500_symbols()
    print(f"Found {len(symbols)} stocks in S&P 500")
    
    results = []
    
    print(f"\nScanning for stocks with RSI below {rsi_threshold}...\n")
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Checking {symbol}...", end='\r')
        
        stock_data = check_rsi_below_20(symbol, rsi_threshold)
        if stock_data:
            results.append(stock_data)
        
        # Rate limiting to avoid API issues
        time.sleep(0.1)
    
    print(" " * 50, end='\r')  # Clear the line
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('RSI')
        return df
    else:
        return pd.DataFrame()


def main():
    """Main execution"""
    print("=" * 60)
    print("S&P 500 RSI Screener - Find Oversold Stocks")
    print("=" * 60)
    
    # Scan for stocks with RSI below 20
    results = scan_sp500_for_low_rsi(rsi_threshold=20)
    
    if not results.empty:
        print(f"\nFound {len(results)} stocks with RSI below 20:\n")
        print(results.to_string(index=False))
        
        # Save to CSV
        filename = f"sp500_rsi_below_20_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
    else:
        print("\nNo stocks found with RSI below 20")


if __name__ == "__main__":
    main()
