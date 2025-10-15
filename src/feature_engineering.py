"""
Feature Engineering Module - COMPLETE WORKING CODE
Creates technical indicators and features for prediction
ALL FUNCTIONS FULLY IMPLEMENTED
"""

import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Class to create technical indicators and features
    ALL METHODS COMPLETE AND WORKING
    """
    
    def __init__(self, data):
        """
        Initialize feature engineer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed stock data
        """
        self.data = data.copy()
        self.initial_rows = len(self.data)
    
    def add_technical_indicators(self):
        """
        Add various technical indicators - COMPLETE IMPLEMENTATION
        """
        print("\n" + "="*60)
        print("ADDING TECHNICAL INDICATORS")
        print("="*60)
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
        print("✓ Simple Moving Averages (SMA) - 6 features")
        
        # Exponential Moving Averages
        for window in [12, 26, 50]:
            self.data[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
        print("✓ Exponential Moving Averages (EMA) - 3 features")
        
        # Weighted Moving Average
        self.data['WMA_10'] = self.data['Close'].rolling(window=10).apply(
            lambda x: (x * np.arange(1, 11)).sum() / np.arange(1, 11).sum(), raw=True
        )
        print("✓ Weighted Moving Average (WMA) - 1 feature")
    
    def add_momentum_indicators(self):
        """
        Add momentum-based indicators - COMPLETE IMPLEMENTATION
        """
        print("\n" + "="*60)
        print("ADDING MOMENTUM INDICATORS")
        print("="*60)
        
        # Relative Strength Index (RSI)
        self.data['RSI_14'] = ta.momentum.RSIIndicator(
            close=self.data['Close'], window=14
        ).rsi()
        self.data['RSI_7'] = ta.momentum.RSIIndicator(
            close=self.data['Close'], window=7
        ).rsi()
        print("✓ Relative Strength Index (RSI) - 2 features")
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=14,
            smooth_window=3
        )
        self.data['STOCH_K'] = stoch.stoch()
        self.data['STOCH_D'] = stoch.stoch_signal()
        print("✓ Stochastic Oscillator - 2 features")
        
        # Williams %R
        self.data['WILLIAMS_R'] = ta.momentum.WilliamsRIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            lbp=14
        ).williams_r()
        print("✓ Williams %R - 1 feature")
        
        # Rate of Change (ROC)
        self.data['ROC_12'] = ta.momentum.ROCIndicator(
            close=self.data['Close'], window=12
        ).roc()
        self.data['ROC_25'] = ta.momentum.ROCIndicator(
            close=self.data['Close'], window=25
        ).roc()
        print("✓ Rate of Change (ROC) - 2 features")
        
        # Ultimate Oscillator
        self.data['UO'] = ta.momentum.UltimateOscillator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        ).ultimate_oscillator()
        print("✓ Ultimate Oscillator - 1 feature")
    
    def add_volatility_indicators(self):
        """
        Add volatility-based indicators - COMPLETE IMPLEMENTATION
        """
        print("\n" + "="*60)
        print("ADDING VOLATILITY INDICATORS")
        print("="*60)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            close=self.data['Close'],
            window=20,
            window_dev=2
        )
        self.data['BB_HIGH'] = bollinger.bollinger_hband()
        self.data['BB_LOW'] = bollinger.bollinger_lband()
        self.data['BB_MID'] = bollinger.bollinger_mavg()
        self.data['BB_WIDTH'] = bollinger.bollinger_wband()
        self.data['BB_PCT'] = bollinger.bollinger_pband()
        print("✓ Bollinger Bands - 5 features")
        
        # Average True Range (ATR)
        self.data['ATR_14'] = ta.volatility.AverageTrueRange(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=14
        ).average_true_range()
        print("✓ Average True Range (ATR) - 1 feature")
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=20
        )
        self.data['KC_HIGH'] = keltner.keltner_channel_hband()
        self.data['KC_LOW'] = keltner.keltner_channel_lband()
        self.data['KC_MID'] = keltner.keltner_channel_mband()
        print("✓ Keltner Channel - 3 features")
        
        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=20
        )
        self.data['DC_HIGH'] = donchian.donchian_channel_hband()
        self.data['DC_LOW'] = donchian.donchian_channel_lband()
        self.data['DC_MID'] = donchian.donchian_channel_mband()
        print("✓ Donchian Channel - 3 features")
    
    def add_trend_indicators(self):
        """
        Add trend-based indicators - COMPLETE IMPLEMENTATION
        """
        print("\n" + "="*60)
        print("ADDING TREND INDICATORS")
        print("="*60)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(
            close=self.data['Close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        self.data['MACD'] = macd.macd()
        self.data['MACD_SIGNAL'] = macd.macd_signal()
        self.data['MACD_DIFF'] = macd.macd_diff()
        print("✓ MACD - 3 features")
        
        # Average Directional Index (ADX)
        self.data['ADX'] = ta.trend.ADXIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=14
        ).adx()
        print("✓ Average Directional Index (ADX) - 1 feature")
        
        # Commodity Channel Index (CCI)
        self.data['CCI'] = ta.trend.CCIIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=20
        ).cci()
        print("✓ Commodity Channel Index (CCI) - 1 feature")
        
        # Aroon Indicator - FIXED: removed 'close' parameter
        aroon = ta.trend.AroonIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            window=25
        )
        self.data['AROON_UP'] = aroon.aroon_up()
        self.data['AROON_DOWN'] = aroon.aroon_down()
        self.data['AROON_IND'] = aroon.aroon_indicator()
        print("✓ Aroon Indicator - 3 features")
        
        # Parabolic SAR
        self.data['PSAR'] = ta.trend.PSARIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        ).psar()
        print("✓ Parabolic SAR - 1 feature")
        
        # Ichimoku Indicator
        ichimoku = ta.trend.IchimokuIndicator(
            high=self.data['High'],
            low=self.data['Low']
        )
        self.data['ICHIMOKU_A'] = ichimoku.ichimoku_a()
        self.data['ICHIMOKU_B'] = ichimoku.ichimoku_b()
        print("✓ Ichimoku - 2 features")
    
    def add_volume_indicators(self):
        """
        Add volume-based indicators - COMPLETE IMPLEMENTATION
        """
        print("\n" + "="*60)
        print("ADDING VOLUME INDICATORS")
        print("="*60)
        
        # On-Balance Volume (OBV)
        self.data['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=self.data['Close'],
            volume=self.data['Volume']
        ).on_balance_volume()
        print("✓ On-Balance Volume (OBV) - 1 feature")
        
        # Chaikin Money Flow (CMF)
        self.data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            volume=self.data['Volume'],
            window=20
        ).chaikin_money_flow()
        print("✓ Chaikin Money Flow (CMF) - 1 feature")
        
        # Volume Price Trend (VPT)
        self.data['VPT'] = ta.volume.VolumePriceTrendIndicator(
            close=self.data['Close'],
            volume=self.data['Volume']
        ).volume_price_trend()
        print("✓ Volume Price Trend (VPT) - 1 feature")
        
        # Money Flow Index (MFI)
        self.data['MFI'] = ta.volume.MFIIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            volume=self.data['Volume'],
            window=14
        ).money_flow_index()
        print("✓ Money Flow Index (MFI) - 1 feature")
        
        # Force Index
        self.data['FI'] = ta.volume.ForceIndexIndicator(
            close=self.data['Close'],
            volume=self.data['Volume'],
            window=13
        ).force_index()
        print("✓ Force Index - 1 feature")
        
        # Volume Weighted Average Price
        self.data['VWAP'] = (self.data['Volume'] * (self.data['High'] + self.data['Low'] + self.data['Close']) / 3).cumsum() / self.data['Volume'].cumsum()
        print("✓ VWAP - 1 feature")
    
    def add_custom_features(self):
        """
        Add custom engineered features - COMPLETE IMPLEMENTATION
        """
        print("\n" + "="*60)
        print("ADDING CUSTOM FEATURES")
        print("="*60)
        
        # Price changes
        self.data['Price_Change'] = self.data['Close'].diff()
        self.data['Price_Change_Pct'] = self.data['Close'].pct_change() * 100
        print("✓ Price changes - 2 features")
        
        # High-Low range
        self.data['High_Low_Range'] = self.data['High'] - self.data['Low']
        self.data['High_Low_Pct'] = (self.data['High_Low_Range'] / self.data['Low']) * 100
        print("✓ High-Low range - 2 features")
        
        # Open-Close range
        self.data['Open_Close_Range'] = self.data['Close'] - self.data['Open']
        self.data['Open_Close_Pct'] = (self.data['Open_Close_Range'] / self.data['Open']) * 100
        print("✓ Open-Close range - 2 features")
        
        # Gap (difference between today's open and yesterday's close)
        self.data['Gap'] = self.data['Open'] - self.data['Close'].shift(1)
        self.data['Gap_Pct'] = (self.data['Gap'] / self.data['Close'].shift(1)) * 100
        print("✓ Gap features - 2 features")
        
        # Intraday features
        self.data['High_Close_Diff'] = self.data['High'] - self.data['Close']
        self.data['Close_Low_Diff'] = self.data['Close'] - self.data['Low']
        print("✓ Intraday features - 2 features")
        
        # Volume features
        self.data['Volume_Change'] = self.data['Volume'].pct_change() * 100
        self.data['Volume_MA_Ratio'] = self.data['Volume'] / self.data['Volume'].rolling(window=20).mean()
        print("✓ Volume features - 2 features")
        
        # Temporal features
        if 'Date' in self.data.columns:
            self.data['Day_of_Week'] = pd.to_datetime(self.data['Date']).dt.dayofweek
            self.data['Month'] = pd.to_datetime(self.data['Date']).dt.month
            self.data['Quarter'] = pd.to_datetime(self.data['Date']).dt.quarter
            self.data['Day_of_Month'] = pd.to_datetime(self.data['Date']).dt.day
            self.data['Week_of_Year'] = pd.to_datetime(self.data['Date']).dt.isocalendar().week
            print("✓ Temporal features - 5 features")
        
        # Lag features (previous days' prices)
        for lag in [1, 2, 3, 5, 7, 14]:
            self.data[f'Close_Lag_{lag}'] = self.data['Close'].shift(lag)
            self.data[f'Volume_Lag_{lag}'] = self.data['Volume'].shift(lag)
        print(f"✓ Lag features - {6*2} features")
        
        # Rolling statistics (7-day window)
        self.data['Close_Rolling_Mean_7'] = self.data['Close'].rolling(window=7).mean()
        self.data['Close_Rolling_Std_7'] = self.data['Close'].rolling(window=7).std()
        self.data['Close_Rolling_Min_7'] = self.data['Close'].rolling(window=7).min()
        self.data['Close_Rolling_Max_7'] = self.data['Close'].rolling(window=7).max()
        print("✓ Rolling statistics (7-day) - 4 features")
        
        # Rolling statistics (30-day window)
        self.data['Close_Rolling_Mean_30'] = self.data['Close'].rolling(window=30).mean()
        self.data['Close_Rolling_Std_30'] = self.data['Close'].rolling(window=30).std()
        print("✓ Rolling statistics (30-day) - 2 features")
        
        # Price momentum
        self.data['Momentum_5'] = self.data['Close'] - self.data['Close'].shift(5)
        self.data['Momentum_10'] = self.data['Close'] - self.data['Close'].shift(10)
        self.data['Momentum_20'] = self.data['Close'] - self.data['Close'].shift(20)
        print("✓ Price momentum - 3 features")
    
    def create_target_variable(self, horizon=1):
        """
        Create target variable for prediction - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        horizon : int
            Number of days ahead to predict
        """
        print(f"\n" + "="*60)
        print(f"CREATING TARGET VARIABLE ({horizon}-day ahead)")
        print("="*60)
        
        # Future close price
        self.data[f'Target_Close_{horizon}d'] = self.data['Close'].shift(-horizon)
        
        # Price change percentage
        self.data[f'Target_Change_{horizon}d'] = ((self.data[f'Target_Close_{horizon}d'] - self.data['Close']) / self.data['Close']) * 100
        
        # Binary target: 1 if price goes up, 0 if down
        self.data[f'Target_Direction_{horizon}d'] = (
            self.data[f'Target_Close_{horizon}d'] > self.data['Close']
        ).astype(int)
        
        print(f"✓ Target variables created:")
        print(f"  - Target_Close_{horizon}d: Future closing price")
        print(f"  - Target_Change_{horizon}d: Price change percentage")
        print(f"  - Target_Direction_{horizon}d: Direction (1=up, 0=down)")
    
    def remove_nan_rows(self):
        """
        Remove rows with NaN values - COMPLETE IMPLEMENTATION
        """
        print("\n" + "="*60)
        print("REMOVING NaN VALUES")
        print("="*60)
        
        initial_rows = len(self.data)
        self.data.dropna(inplace=True)
        final_rows = len(self.data)
        removed_rows = initial_rows - final_rows
        
        print(f"Initial rows: {initial_rows}")
        print(f"Removed rows: {removed_rows} ({removed_rows/initial_rows*100:.2f}%)")
        print(f"Final rows: {final_rows}")
    
    def get_feature_names(self, exclude_cols=None):
        """
        Get list of feature column names - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        exclude_cols : list
            Columns to exclude from features
            
        Returns:
        --------
        list : Feature column names
        """
        if exclude_cols is None:
            exclude_cols = ['Date', 'Target_Close_1d', 'Target_Change_1d', 'Target_Direction_1d']
        
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        return feature_cols
    
    def get_engineered_data(self):
        """
        Get the data with engineered features
        
        Returns:
        --------
        pd.DataFrame : Data with features
        """
        return self.data
    
    def save_features(self, filepath='data/processed/features.csv'):
        """
        Save engineered features to CSV - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        filepath : str
            Path to save the CSV file
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.data.to_csv(filepath, index=False)
        print(f"\n✓ Features saved to {filepath}")
        print(f"  Shape: {self.data.shape}")
    
    def build_all_features(self, target_horizon=1):
        """
        Build all features at once - COMPLETE IMPLEMENTATION
        
        Parameters:
        -----------
        target_horizon : int
            Number of days ahead to predict
            
        Returns:
        --------
        pd.DataFrame : Data with all features
        """
        print("\n" + "="*70)
        print("  BUILDING ALL FEATURES - COMPLETE PIPELINE")
        print("="*70)
        
        # Add all technical indicators
        self.add_technical_indicators()
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_trend_indicators()
        self.add_volume_indicators()
        self.add_custom_features()
        
        # Create target variable
        self.create_target_variable(horizon=target_horizon)
        
        # Remove NaN rows
        self.remove_nan_rows()
        
        print("\n" + "="*70)
        print("  FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nInitial rows: {self.initial_rows}")
        print(f"Final rows: {len(self.data)}")
        print(f"Total features created: {len(self.data.columns)}")
        print(f"Final dataset shape: {self.data.shape}")
        
        # IMPORTANT: Return the data
        return self.data


def main():
    """
    Example usage of FeatureEngineer - COMPLETE WORKING EXAMPLE
    """
    try:
        print("\n" + "="*70)
        print("  FEATURE ENGINEERING - STARTING")
        print("="*70)
        
        # Load preprocessed data
        data = pd.read_csv('data/processed/processed_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        print(f"\n✓ Loaded data with {len(data)} rows")
        
        # Initialize feature engineer
        engineer = FeatureEngineer(data)
        
        # Build all features
        featured_data = engineer.build_all_features(target_horizon=1)
        
        # Check if data was returned
        if featured_data is None:
            print("✗ Error: build_all_features returned None")
            return
        
        print("\n" + "="*70)
        print("  FEATURE SUMMARY")
        print("="*70)
        
        # Get feature names
        features = engineer.get_feature_names()
        print(f"\nTotal feature columns: {len(features)}")
        print(f"\nSample features (first 20):")
        for i, feat in enumerate(features[:20], 1):
            print(f"  {i}. {feat}")
        
        if len(features) > 20:
            print(f"\n... and {len(features) - 20} more features")
        
        # Display sample data
        print("\n" + "="*70)
        print("  SAMPLE DATA")
        print("="*70)
        print("\nFirst 3 rows:")
        print(featured_data.head(3))
        
        print("\n" + "="*70)
        print("  TARGET VARIABLE DISTRIBUTION")
        print("="*70)
        if 'Target_Direction_1d' in featured_data.columns:
            target_dist = featured_data['Target_Direction_1d'].value_counts()
            print(f"\nTarget Distribution:")
            print(f"  Up (1): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(featured_data)*100:.2f}%)")
            print(f"  Down (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(featured_data)*100:.2f}%)")
        
        # Save features
        print("\n" + "="*70)
        print("  SAVING FEATURES")
        print("="*70)
        engineer.save_features()
        
        print("\n" + "="*70)
        print("  FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except FileNotFoundError:
        print("\n✗ Error: data/processed/processed_data.csv not found!")
        print("Please run src/data_preprocessing.py first!")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()