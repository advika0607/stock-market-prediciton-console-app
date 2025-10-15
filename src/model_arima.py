"""
ARIMA Model Module - COMPLETE WORKING CODE
Implements ARIMA (AutoRegressive Integrated Moving Average) model
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import joblib
import os


class ARIMAModel:
    """Class to implement ARIMA model for stock price prediction"""
    
    def __init__(self, data, target_col='Close'):
        self.data = data.copy()
        self.target_col = target_col
        self.model = None
        self.fitted_model = None
        self.predictions = None
        self.train_data = None
        self.test_data = None
        
    def check_stationarity(self, plot=True):
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        print("="*60)
        print("STATIONARITY TEST")
        print("="*60)
        result = adfuller(self.data[self.target_col].dropna())
        test_results = {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Is Stationary': result[1] < 0.05
        }
        print(f"\nADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.3f}")
        if result[1] < 0.05:
            print("\n✓ Series is stationary (p-value < 0.05)")
        else:
            print("\n✗ Series is non-stationary (p-value >= 0.05)")
            print("  Consider differencing the series")
        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            axes[0].plot(self.data[self.target_col])
            axes[0].set_title(f'{self.target_col} - Original Series', fontweight='bold')
            axes[0].set_ylabel('Price')
            axes[0].grid(True, alpha=0.3)
            rolling_mean = self.data[self.target_col].rolling(window=20).mean()
            rolling_std = self.data[self.target_col].rolling(window=20).std()
            axes[1].plot(self.data[self.target_col], label='Original')
            axes[1].plot(rolling_mean, label='Rolling Mean', color='red')
            axes[1].plot(rolling_std, label='Rolling Std', color='green')
            axes[1].set_title('Rolling Mean & Standard Deviation', fontweight='bold')
            axes[1].set_ylabel('Price')
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        return test_results
    
    def plot_acf_pacf(self, lags=40):
        """Plot ACF and PACF to determine ARIMA parameters"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(self.data[self.target_col].dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold')
        plot_pacf(self.data[self.target_col].dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold')
        plt.tight_layout()
        plt.show()
        print("\nHow to interpret ACF/PACF:")
        print("- ACF cuts off after lag q → MA(q) model")
        print("- PACF cuts off after lag p → AR(p) model")
        print("- Both tail off → ARMA model")
    
    def split_data(self, train_size=0.8):
        """Split data into train and test sets"""
        split_idx = int(len(self.data) * train_size)
        self.train_data = self.data[:split_idx]
        self.test_data = self.data[split_idx:]
        print(f"\n" + "="*60)
        print("DATA SPLIT")
        print("="*60)
        print(f"Training set: {len(self.train_data)} samples ({train_size*100:.0f}%)")
        print(f"Testing set: {len(self.test_data)} samples ({(1-train_size)*100:.0f}%)")
        return self.train_data, self.test_data
    
    def fit_model(self, order=(1, 1, 1)):
        """Fit ARIMA model"""
        print(f"\n" + "="*60)
        print(f"FITTING ARIMA MODEL")
        print("="*60)
        print(f"Order (p, d, q): {order}")
        try:
            self.model = ARIMA(self.train_data[self.target_col], order=order)
            self.fitted_model = self.model.fit()
            print("\n✓ Model fitted successfully!")
            print("\n" + "="*60)
            print("MODEL SUMMARY")
            print("="*60)
            print(self.fitted_model.summary())
        except Exception as e:
            print(f"✗ Error fitting model: {str(e)}")
    
    def predict(self, steps=None):
        """Make predictions"""
        if self.fitted_model is None:
            print("✗ Error: Model not fitted. Call fit_model() first.")
            return None
        print(f"\n" + "="*60)
        print("MAKING PREDICTIONS")
        print("="*60)
        if steps is None:
            steps = len(self.test_data)
            start = len(self.train_data)
            end = len(self.train_data) + steps - 1
            self.predictions = self.fitted_model.predict(start=start, end=end)
        else:
            self.predictions = self.fitted_model.forecast(steps=steps)
        print(f"✓ Generated {len(self.predictions)} predictions")
        return self.predictions
    
    def evaluate(self):
        """Evaluate model performance"""
        if self.predictions is None:
            print("✗ Error: No predictions available. Call predict() first.")
            return None
        actual = self.test_data[self.target_col].values
        predicted = self.predictions.values
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)
        metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2 Score': r2}
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*60)
        return metrics
    
    def plot_predictions(self, save_path=None):
        """Plot actual vs predicted values"""
        if self.predictions is None:
            print("✗ Error: No predictions available.")
            return
        plt.figure(figsize=(14, 7))
        plt.plot(self.train_data.index, self.train_data[self.target_col], label='Training Data', color='blue', alpha=0.7)
        plt.plot(self.test_data.index, self.test_data[self.target_col], label='Actual Test Data', color='green', linewidth=2)
        plt.plot(self.test_data.index, self.predictions, label='ARIMA Predictions', color='red', linestyle='--', linewidth=2)
        plt.title('ARIMA Model - Predictions vs Actual', fontsize=16, fontweight='bold')
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()
    
    def plot_residuals(self, save_path=None):
        """Plot model residuals"""
        if self.fitted_model is None:
            print("✗ Error: Model not fitted.")
            return
        fig = self.fitted_model.plot_diagnostics(figsize=(14, 10))
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()
    
    def save_model(self, filepath='models/arima_model.pkl'):
        """Save the fitted model"""
        if self.fitted_model is None:
            print("✗ Error: No model to save.")
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.fitted_model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/arima_model.pkl'):
        """Load a saved model"""
        try:
            self.fitted_model = joblib.load(filepath)
            print(f"✓ Model loaded from {filepath}")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")


def auto_arima_grid_search(data, target_col='Close', max_p=3, max_d=2, max_q=3):
    """Perform grid search to find best ARIMA parameters"""
    print("\n" + "="*60)
    print("AUTO ARIMA - GRID SEARCH")
    print("="*60)
    print("Searching for best parameters...")
    print(f"Search space: p=[0-{max_p}], d=[0-{max_d}], q=[0-{max_q}]")
    print("This may take a few minutes...\n")
    
    best_aic = np.inf
    best_order = None
    split_idx = int(len(data) * 0.8)
    train = data[:split_idx][target_col]
    
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1)
    current = 0
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                current += 1
                try:
                    model = ARIMA(train, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        print(f"[{current}/{total_combinations}] New best: ARIMA{best_order} - AIC: {best_aic:.2f}")
                except:
                    continue
    
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETED")
    print("="*60)
    print(f"✓ Best ARIMA order: {best_order}")
    print(f"✓ Best AIC: {best_aic:.2f}")
    print("="*60)
    
    return best_order, best_aic


def main():
    """Example usage of ARIMAModel - COMPLETE WORKING EXAMPLE"""
    try:
        print("\n" + "="*70)
        print("  ARIMA MODEL - STOCK PRICE PREDICTION")
        print("="*70)
        
        # Load data
        data = pd.read_csv('data/processed/processed_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        print(f"\n✓ Loaded {len(data)} rows of data")
        
        # Initialize ARIMA model
        arima_model = ARIMAModel(data, target_col='Close')
        
        # Check stationarity
        arima_model.check_stationarity(plot=True)
        
        # Plot ACF and PACF
        print("\n" + "="*60)
        print("ACF & PACF ANALYSIS")
        print("="*60)
        arima_model.plot_acf_pacf(lags=40)
        
        # Split data
        train_data, test_data = arima_model.split_data(train_size=0.8)
        
        # Option 1: Use manual order
        print("\n" + "="*60)
        print("OPTION 1: MANUAL ORDER")
        print("="*60)
        print("Using ARIMA(1,1,1) as default...")
        
        # Fit model
        arima_model.fit_model(order=(1, 1, 1))
        
        # Make predictions
        predictions = arima_model.predict()
        
        # Evaluate model
        metrics = arima_model.evaluate()
        
        # Plot predictions
        print("\n" + "="*60)
        print("PLOTTING RESULTS")
        print("="*60)
        arima_model.plot_predictions(save_path='visualizations/arima_predictions.png')
        
        # Plot residuals
        arima_model.plot_residuals(save_path='visualizations/arima_residuals.png')
        
        # Save model
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        arima_model.save_model()
        
        # Option 2: Auto ARIMA (Grid Search)
        print("\n" + "="*60)
        print("OPTION 2: AUTO ARIMA (OPTIONAL)")
        print("="*60)
        user_input = input("Run Auto ARIMA grid search? (y/n): ").lower()
        
        if user_input == 'y':
            best_order, best_aic = auto_arima_grid_search(data, target_col='Close', max_p=3, max_d=2, max_q=3)
            
            # Fit model with best parameters
            print(f"\nFitting model with best order: {best_order}")
            arima_model_best = ARIMAModel(data, target_col='Close')
            arima_model_best.split_data(train_size=0.8)
            arima_model_best.fit_model(order=best_order)
            arima_model_best.predict()
            metrics_best = arima_model_best.evaluate()
            arima_model_best.plot_predictions(save_path='visualizations/arima_predictions_best.png')
            arima_model_best.save_model(filepath='models/arima_model_best.pkl')
        
        print("\n" + "="*70)
        print("  ARIMA MODEL COMPLETED SUCCESSFULLY!")
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