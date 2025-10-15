"""
Data Preprocessing Module - COMPLETE WORKING CODE
Handles data cleaning, transformation, and preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os


class DataPreprocessor:
    """
    Class to handle data preprocessing tasks - ALL FUNCTIONS COMPLETE
    """
    
    def __init__(self, data):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw stock data
        """
        self.data = data.copy()
        self.scalers = {}
        self.original_data = data.copy()
        
    def check_missing_values(self):
        """
        Check for missing values in the dataset
        
        Returns:
        --------
        pd.Series : Count of missing values per column
        """
        print("\n" + "="*60)
        print("CHECKING MISSING VALUES")
        print("="*60)
        
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        
        print("\nMissing values per column:")
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        total_missing = missing.sum()
        if total_missing == 0:
            print("\n✓ No missing values found!")
        else:
            print(f"\n⚠ Total missing values: {total_missing}")
        
        return missing
    
    def handle_missing_values(self, method='ffill'):
        """
        Handle missing values - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        method : str
            Method to handle missing values ('ffill', 'bfill', 'interpolate', 'drop')
        """
        print(f"\nHandling missing values using method: '{method}'")
        
        initial_missing = self.data.isnull().sum().sum()
        
        if method == 'ffill':
            # Forward fill - propagate last valid observation forward
            self.data.fillna(method='ffill', inplace=True)
            # Backward fill remaining (if any at the start)
            self.data.fillna(method='bfill', inplace=True)
            
        elif method == 'bfill':
            # Backward fill - use next valid observation
            self.data.fillna(method='bfill', inplace=True)
            # Forward fill remaining (if any at the end)
            self.data.fillna(method='ffill', inplace=True)
            
        elif method == 'interpolate':
            # Linear interpolation for numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].interpolate(method='linear', limit_direction='both')
            
        elif method == 'drop':
            # Drop rows with missing values
            initial_rows = len(self.data)
            self.data.dropna(inplace=True)
            dropped_rows = initial_rows - len(self.data)
            print(f"  Dropped {dropped_rows} rows")
        
        final_missing = self.data.isnull().sum().sum()
        print(f"✓ Missing values reduced from {initial_missing} to {final_missing}")
    
    def check_outliers(self, column='Close'):
        """
        Check for outliers using IQR method - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        column : str
            Column name to check for outliers
            
        Returns:
        --------
        pd.DataFrame : Rows containing outliers
        """
        print(f"\n" + "="*60)
        print(f"CHECKING OUTLIERS IN '{column}'")
        print("="*60)
        
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers = self.data[(self.data[column] < lower_bound) | 
                             (self.data[column] > upper_bound)]
        
        print(f"\nQ1 (25th percentile): {Q1:.2f}")
        print(f"Q3 (75th percentile): {Q3:.2f}")
        print(f"IQR: {IQR:.2f}")
        print(f"Lower bound: {lower_bound:.2f}")
        print(f"Upper bound: {upper_bound:.2f}")
        print(f"\nNumber of outliers: {len(outliers)} ({len(outliers)/len(self.data)*100:.2f}%)")
        
        if len(outliers) > 0:
            print("\nSample outliers:")
            print(outliers[[column]].head())
        
        return outliers
    
    def remove_outliers(self, column='Close', method='iqr'):
        """
        Remove outliers from data - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        column : str
            Column to check
        method : str
            'iqr' or 'zscore'
        """
        initial_rows = len(self.data)
        
        if method == 'iqr':
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.data = self.data[
                (self.data[column] >= lower_bound) & 
                (self.data[column] <= upper_bound)
            ]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.data[column]))
            self.data = self.data[z_scores < 3]
        
        removed_rows = initial_rows - len(self.data)
        print(f"✓ Removed {removed_rows} outlier rows")
    
    def create_returns(self):
        """
        Calculate daily returns - COMPLETE IMPLEMENTATION
        """
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        print("\n✓ Daily returns calculated")
        print(f"  Mean return: {self.data['Returns'].mean()*100:.4f}%")
        print(f"  Std return: {self.data['Returns'].std()*100:.4f}%")
    
    def create_moving_averages(self, windows=[5, 10, 20, 50, 200]):
        """
        Create moving averages - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        windows : list
            List of window sizes for moving averages
        """
        for window in windows:
            # Simple Moving Average
            self.data[f'MA_{window}'] = self.data['Close'].rolling(window=window).mean()
            
            # Exponential Moving Average
            self.data[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
        
        print(f"\n✓ Moving averages created for windows: {windows}")
        print(f"  Created {len(windows)*2} MA/EMA features")
    
    def normalize_data(self, columns, method='minmax'):
        """
        Normalize specified columns - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        columns : list
            List of column names to normalize
        method : str
            Normalization method ('minmax' or 'standard')
            
        Returns:
        --------
        dict : Dictionary of scalers for each column
        """
        for col in columns:
            if col not in self.data.columns:
                print(f"⚠ Warning: Column '{col}' not found, skipping...")
                continue
                
            if method == 'minmax':
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = StandardScaler()
            
            # Reshape for scaler
            values = self.data[col].values.reshape(-1, 1)
            self.data[f'{col}_scaled'] = scaler.fit_transform(values)
            self.scalers[col] = scaler
        
        print(f"\n✓ Normalized {len(columns)} columns using '{method}' method")
        return self.scalers
    
    def create_train_test_split(self, train_size=0.8):
        """
        Split data into training and testing sets - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        train_size : float
            Proportion of data for training (0 to 1)
            
        Returns:
        --------
        tuple : (train_data, test_data)
        """
        split_index = int(len(self.data) * train_size)
        
        train_data = self.data[:split_index].copy()
        test_data = self.data[split_index:].copy()
        
        print("\n" + "="*60)
        print("TRAIN-TEST SPLIT")
        print("="*60)
        print(f"Training set: {len(train_data)} rows ({train_size*100:.0f}%)")
        print(f"Testing set: {len(test_data)} rows ({(1-train_size)*100:.0f}%)")
        
        if 'Date' in train_data.columns:
            print(f"\nTraining period: {train_data['Date'].min()} to {train_data['Date'].max()}")
            print(f"Testing period: {test_data['Date'].min()} to {test_data['Date'].max()}")
        
        return train_data, test_data
    
    def get_processed_data(self):
        """
        Get the processed data
        
        Returns:
        --------
        pd.DataFrame : Processed data
        """
        return self.data
    
    def save_processed_data(self, filepath='data/processed/processed_data.csv'):
        """
        Save processed data to CSV - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        filepath : str
            Path to save the CSV file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.data.to_csv(filepath, index=False)
        print(f"\n✓ Processed data saved to {filepath}")
        print(f"  Shape: {self.data.shape}")
    
    def get_data_summary(self):
        """
        Get summary statistics of the data - COMPLETE IMPLEMENTATION
        
        Returns:
        --------
        pd.DataFrame : Summary statistics
        """
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"\nShape: {self.data.shape}")
        print(f"Rows: {self.data.shape[0]}")
        print(f"Columns: {self.data.shape[1]}")
        
        if 'Date' in self.data.columns:
            print(f"\nDate range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        
        print(f"\nColumn names ({len(self.data.columns)}):")
        for i, col in enumerate(self.data.columns, 1):
            print(f"  {i}. {col}")
        
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        return self.data.describe()
    
    def process_pipeline(self, train_size=0.8):
        """
        Run complete preprocessing pipeline - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        train_size : float
            Train-test split ratio
            
        Returns:
        --------
        tuple : (train_data, test_data, scalers)
        """
        print("\n" + "="*60)
        print("RUNNING COMPLETE PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Check missing values
        self.check_missing_values()
        
        # Step 2: Handle missing values
        self.handle_missing_values(method='ffill')
        
        # Step 3: Check outliers
        self.check_outliers(column='Close')
        
        # Step 4: Create returns
        self.create_returns()
        
        # Step 5: Create moving averages
        self.create_moving_averages(windows=[5, 10, 20, 50])
        
        # Step 6: Get summary
        summary = self.get_data_summary()
        print("\n", summary)
        
        # Step 7: Split data
        train_data, test_data = self.create_train_test_split(train_size=train_size)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return train_data, test_data, self.scalers


def main():
    """
    Example usage of DataPreprocessor - COMPLETE WORKING EXAMPLE
    """
    try:
        # Load data
        print("Loading raw data...")
        data = pd.read_csv('data/raw/stock_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        print(f"✓ Loaded {len(data)} rows")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(data)
        
        # Run complete pipeline
        train_data, test_data, scalers = preprocessor.process_pipeline(train_size=0.8)
        
        # Normalize close prices
        print("\nNormalizing Close prices...")
        scalers = preprocessor.normalize_data(['Close'], method='minmax')
        
        # Save processed data
        preprocessor.save_processed_data()
        
        # Save train and test sets separately
        train_data.to_csv('data/processed/train_data.csv', index=False)
        test_data.to_csv('data/processed/test_data.csv', index=False)
        print("✓ Train and test data saved separately")
        
        print("\n" + "="*60)
        print("ALL PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except FileNotFoundError:
        print("\n✗ Error: data/raw/stock_data.csv not found!")
        print("Please run src/data_collection.py first to fetch the data.")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")


if __name__ == "__main__":
    main()