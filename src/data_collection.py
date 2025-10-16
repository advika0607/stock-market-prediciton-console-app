"""
Data Collection Module - FIXED & ROBUST VERSION

Fetches stock data from Yahoo Finance API with error handling
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

class StockDataCollector:
    """
    Class to handle stock data collection from Yahoo Finance
    """

    def __init__(self, ticker_symbol, start_date=None, end_date=None):
        """
        Initialize the data collector

        Parameters:
        -----------
        ticker_symbol : str
            Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        """
        self.ticker = ticker_symbol.upper()

        # Set default dates if not provided
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date

        if start_date is None:
            # Default to 5 years of data
            self.start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
        else:
            self.start_date = start_date

        self.data = None

    def fetch_data(self, retry_count=3):
        """
        Fetch stock data from Yahoo Finance with retry logic

        Parameters:
        -----------
        retry_count : int
            Number of retry attempts

        Returns:
        --------
        pd.DataFrame : Stock data with OHLCV columns
        """
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")

        for attempt in range(retry_count):
            try:
                # Method 1: Using period parameter (more reliable)
                stock = yf.Ticker(self.ticker)

                # Try downloading with period first
                print(f" Attempt {attempt + 1}/{retry_count}...")
                self.data = stock.history(period="5y", interval="1d")

                # If period method fails, try with date range
                if self.data.empty:
                    print(f" Trying alternative method...")
                    self.data = yf.download(
                        self.ticker,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        show_errors=False
                    )

                # Fallback to direct CSV download if above fails (added fix)
                if self.data is None or self.data.empty:
                    print(" Trying fallback direct CSV download from Yahoo Finance...")
                    url = f"https://query1.finance.yahoo.com/v7/finance/download/{self.ticker}?period1=0&period2=9999999999&interval=1d&events=history"
                    try:
                        self.data = pd.read_csv(url)
                        print(f"✓ Fallback CSV download succeeded with {len(self.data)} rows.")
                    except Exception as e:
                        print(f"✗ Fallback CSV download failed: {e}")

                # Check if data is empty after all attempts
                if self.data is None or self.data.empty:
                    if attempt < retry_count - 1:
                        print(f" No data received, retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    else:
                        raise ValueError(f"No data found for ticker {self.ticker}")

                # Reset index to make Date a column
                self.data.reset_index(inplace=True)

                # Ensure we have the required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in self.data.columns:
                        raise ValueError(f"Missing required column: {col}")

                # Remove any rows with NaN in critical columns
                self.data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

                if len(self.data) < 100:
                    raise ValueError(f"Insufficient data: only {len(self.data)} rows found. Need at least 100.")

                print(f"✓ Successfully fetched {len(self.data)} rows of data")
                print(f"✓ Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
                return self.data

            except Exception as e:
                print(f" Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    print(f" Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    print(f"\n✗ Error fetching data after {retry_count} attempts")
                    print(f"✗ Error details: {str(e)}")
                    print(f"\nPossible reasons:")
                    print(f" 1. Invalid ticker symbol: {self.ticker}")
                    print(f" 2. Stock may be delisted")
                    print(f" 3. Yahoo Finance API is temporarily down")
                    print(f" 4. Network connection issues")
                    print(f"\nTry:")
                    print(f" - Verify ticker symbol at https://finance.yahoo.com")
                    print(f" - Check your internet connection")
                    print(f" - Try again in a few minutes")
                    return None

    def get_stock_info(self):
        """
        Get additional stock information

        Returns:
        --------
        dict : Stock information
        """
        try:
            stock = yf.Ticker(self.ticker)
            # Try to get info with error handling
            try:
                info = stock.info
            except:
                print("⚠ Could not fetch detailed stock info, using basic data...")
                return {
                    'company_name': self.ticker,
                    'sector': 'N/A',
                    'industry': 'N/A',
                    'market_cap': 'N/A',
                    'currency': 'USD',
                    'exchange': 'N/A',
                    'country': 'N/A'
                }
            # Extract key information with fallbacks
            stock_info = {
                'company_name': info.get('longName', info.get('shortName', self.ticker)),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'country': info.get('country', 'N/A')
            }
            return stock_info

        except Exception as e:
            print(f"⚠ Warning: Could not fetch stock info: {str(e)}")
            return {
                'company_name': self.ticker,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 'N/A',
                'currency': 'USD',
                'exchange': 'N/A',
                'country': 'N/A'
            }

    def validate_ticker(self):
        """
        Validate if ticker exists
        
        Returns:
        --------
        bool : True if valid, False otherwise
        """
        try:
            stock = yf.Ticker(self.ticker)
            # Try to get basic info
            hist = stock.history(period="5d")
            return not hist.empty
        except:
            return False

    def save_data(self, filepath='data/raw/stock_data.csv'):
        """
        Save the fetched data to CSV

        Parameters:
        -----------
        filepath : str
            Path to save the CSV file
        """
        if self.data is None:
            print("✗ No data to save. Fetch data first.")
            return
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save to CSV
        self.data.to_csv(filepath, index=False)
        print(f"✓ Data saved to {filepath}")

    def load_data(self, filepath='data/raw/stock_data.csv'):
        """
        Load data from CSV

        Parameters:
        -----------
        filepath : str
            Path to the CSV file

        Returns:
        --------
        pd.DataFrame : Loaded stock data
        """
        try:
            self.data = pd.read_csv(filepath)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            print(f"✓ Data loaded from {filepath}")
            print(f"✓ Loaded {len(self.data)} rows")
            return self.data
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None

    def get_multiple_stocks(self, tickers_list, start_date=None, end_date=None):
        """
        Fetch data for multiple stocks

        Parameters:
        -----------
        tickers_list : list
            List of ticker symbols
        start_date : str
            Start date
        end_date : str
            End date

        Returns:
        --------
        dict : Dictionary of DataFrames for each ticker
        """
        all_data = {}
        for ticker in tickers_list:
            print(f"\n{'='*60}")
            print(f"Fetching data for {ticker}...")
            print('='*60)
            try:
                collector = StockDataCollector(ticker, start_date, end_date)
                data = collector.fetch_data()
                if data is not None:
                    all_data[ticker] = data
                    print(f"✓ Successfully fetched {ticker}")
                else:
                    print(f"✗ Failed to fetch {ticker}")
            except Exception as e:
                print(f"✗ Failed to fetch {ticker}: {str(e)}")
        return all_data

def test_ticker(ticker):
    """
    Quick test function to check if a ticker is valid

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol

    Returns:
    --------
    bool : True if valid, False otherwise
    """
    print(f"\nTesting ticker: {ticker}")
    print("-" * 60)
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            print(f"✗ No data found for {ticker}")
            print(f" Ticker may be invalid or delisted")
            return False
        else:
            print(f"✓ Ticker {ticker} is valid!")
            print(f" Latest price: ${hist['Close'].iloc[-1]:.2f}")
            print(f" Data points: {len(hist)}")
            return True
    except Exception as e:
        print(f"✗ Error testing {ticker}: {str(e)}")
        return False

def main():
    """
    Example usage of StockDataCollector
    """
    print("="*60)
    print("STOCK DATA COLLECTION - ROBUST VERSION")
    print("="*60 + "\n")

    # Test a ticker first
    ticker = input("Enter stock ticker to test (e.g., AAPL, GOOGL, MSFT): ").upper().strip()
    if not ticker:
        print("✗ No ticker provided. Using default: AAPL")
        ticker = 'AAPL'

    # Quick validation
    print(f"\nValidating ticker {ticker}...")
    if not test_ticker(ticker):
        print(f"\n⚠ Warning: {ticker} may not be a valid ticker.")
        proceed = input("Do you want to proceed anyway? (y/n): ").lower()
        if proceed != 'y':
            print("Exiting...")
            return

    # Initialize collector
    collector = StockDataCollector(ticker)
    # Fetch data with retries
    data = collector.fetch_data(retry_count=3)

    if data is not None:
        print("\n" + "="*60)
        print("DATA PREVIEW")
        print("="*60)
        print("\nFirst 5 rows:")
        print(data.head())
        print("\nLast 5 rows:")
        print(data.tail())
        print(f"\nData shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print("\nData types:")
        print(data.dtypes)
        print("\nBasic statistics:")
        print(data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

        # Get stock info
        print("\n" + "="*60)
        print("STOCK INFORMATION")
        print("="*60)
        info = collector.get_stock_info()
        if info:
            for key, value in info.items():
                print(f"{key.replace('_', ' ').title()}: {value}")

        # Save data
        print("\n" + "="*60)
        print("SAVING DATA")
        print("="*60)
        collector.save_data()

        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETED SUCCESSFULLY!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("DATA COLLECTION FAILED")
        print("="*60)
        print("\nPlease check:")
        print(" 1. Ticker symbol is correct")
        print(" 2. Internet connection is working")
        print(" 3. Yahoo Finance is accessible")
        print("\nTry these working tickers: AAPL, MSFT, GOOGL, AMZN, META")

if __name__ == "__main__":
    main()
