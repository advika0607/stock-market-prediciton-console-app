"""
Data Collection Module - COMPLETE WORKING CODE
Fetches stock data from Yahoo Finance API
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta


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
        self.ticker = ticker_symbol
        
        # Set default dates if not provided
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            # Default to 5 years of data
            self.start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        self.data = None
        
    def fetch_data(self):
        """
        Fetch stock data from Yahoo Finance
        
        Returns:
        --------
        pd.DataFrame : Stock data with OHLCV columns
        """
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        
        try:
            # Download stock data
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            
            # Check if data is empty
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Reset index to make Date a column
            self.data.reset_index(inplace=True)
            
            print(f"✓ Successfully fetched {len(self.data)} rows of data")
            print(f"✓ Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
            
            return self.data
            
        except Exception as e:
            print(f"✗ Error fetching data: {str(e)}")
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
            info = stock.info
            
            # Extract key information
            stock_info = {
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
                'country': info.get('country', 'N/A')
            }
            
            return stock_info
            
        except Exception as e:
            print(f"✗ Error fetching stock info: {str(e)}")
            return None
    
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
            print(f"\nFetching data for {ticker}...")
            try:
                collector = StockDataCollector(ticker, start_date, end_date)
                data = collector.fetch_data()
                if data is not None:
                    all_data[ticker] = data
            except Exception as e:
                print(f"✗ Failed to fetch {ticker}: {str(e)}")
        
        return all_data


def main():
    """
    Example usage of StockDataCollector
    """
    print("="*60)
    print("STOCK DATA COLLECTION - COMPLETE EXAMPLE")
    print("="*60 + "\n")
    
    # Initialize collector for Apple stock
    collector = StockDataCollector('AAPL')
    
    # Fetch data
    data = collector.fetch_data()
    
    # Display first few rows
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


if __name__ == "__main__":
    main()