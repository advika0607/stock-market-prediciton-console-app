"""
Main Pipeline - COMPLETE WORKING CODE
Runs the entire stock prediction pipeline from data collection to model training
"""

import sys
import os
sys.path.append('src')

from data_collection import StockDataCollector
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from visualization import StockVisualizer
from model_arima import ARIMAModel
from model_lstm import LSTMModel

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def step_1_data_collection(ticker='AAPL'):
    """Step 1: Collect stock data"""
    print_header("STEP 1: DATA COLLECTION")
    
    # Validate ticker first
    print(f"Validating ticker: {ticker}...")
    try:
        import yfinance as yf
        test_stock = yf.Ticker(ticker)
        test_data = test_stock.history(period="5d")
        
        if test_data.empty:
            print(f"✗ Ticker {ticker} appears to be invalid or has no data")
            print(f"\nTrying alternative tickers...")
            
            alternatives = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
            for alt in alternatives:
                print(f"  Testing {alt}...")
                alt_stock = yf.Ticker(alt)
                alt_data = alt_stock.history(period="5d")
                if not alt_data.empty:
                    print(f"✓ {alt} is valid! Using {alt} instead.")
                    ticker = alt
                    break
            else:
                print("✗ Could not find any valid ticker")
                return None
        else:
            print(f"✓ Ticker {ticker} is valid")
    except Exception as e:
        print(f"⚠ Warning during validation: {str(e)}")
    
    # Fetch data
    collector = StockDataCollector(ticker)
    data = collector.fetch_data(retry_count=3)
    
    if data is not None:
        print(f"\n✓ Data shape: {data.shape}")
        print(f"✓ Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Get stock info
        info = collector.get_stock_info()
        if info:
            print(f"\n✓ Company: {info['company_name']}")
            print(f"✓ Sector: {info['sector']}")
        
        # Save data
        collector.save_data()
        return data
    else:
        print("✗ Failed to collect data")
        print("\n" + "="*80)
        print("TROUBLESHOOTING:")
        print("="*80)
        print("1. Check if ticker symbol is correct (e.g., AAPL, GOOGL, MSFT)")
        print("2. Verify internet connection")
        print("3. Try these known working tickers: AAPL, MSFT, GOOGL, AMZN, META")
        print("4. Wait a few minutes and try again (Yahoo Finance may be busy)")
        print("\nExample command: python main.py --ticker AAPL")
        return None


def step_2_data_preprocessing(data):
    """Step 2: Preprocess data"""
    print_header("STEP 2: DATA PREPROCESSING")
    
    preprocessor = DataPreprocessor(data)
    
    # Check and handle missing values
    preprocessor.check_missing_values()
    preprocessor.handle_missing_values(method='ffill')
    
    # Check outliers
    preprocessor.check_outliers(column='Close')
    
    # Create returns and moving averages
    preprocessor.create_returns()
    preprocessor.create_moving_averages(windows=[5, 10, 20, 50])
    
    # Split data
    train_data, test_data = preprocessor.create_train_test_split(train_size=0.8)
    
    # Save processed data
    preprocessor.save_processed_data()
    
    print(f"\n✓ Preprocessing completed")
    print(f"✓ Training set: {len(train_data)} samples")
    print(f"✓ Testing set: {len(test_data)} samples")
    
    return preprocessor.get_processed_data()


def step_3_feature_engineering(data):
    """Step 3: Engineer features"""
    print_header("STEP 3: FEATURE ENGINEERING")
    
    engineer = FeatureEngineer(data)
    
    # Build all features
    featured_data = engineer.build_all_features(target_horizon=1)
    
    # Check if data was returned
    if featured_data is None:
        print("✗ Error: Feature engineering returned None")
        # Try to get data from engineer object
        featured_data = engineer.get_engineered_data()
        if featured_data is None:
            raise ValueError("Failed to generate features")
    
    # Save features
    engineer.save_features()
    
    print(f"\n✓ Feature engineering completed")
    print(f"✓ Total features: {len(featured_data.columns)}")
    print(f"✓ Dataset shape: {featured_data.shape}")
    
    return featured_data


def step_4_visualization(data, ticker='AAPL'):
    """Step 4: Create visualizations"""
    print_header("STEP 4: DATA VISUALIZATION")
    
    viz = StockVisualizer(data, ticker=ticker)
    
    print("Creating visualizations...")
    
    # Create plots
    viz.plot_price_history(save_path='visualizations/price_history.png')
    viz.plot_moving_averages(save_path='visualizations/moving_averages.png')
    viz.plot_technical_indicators(save_path='visualizations/technical_indicators.png')
    viz.plot_correlation_heatmap(save_path='visualizations/correlation_heatmap.png')
    viz.plot_returns_distribution(save_path='visualizations/returns_distribution.png')
    
    # Create interactive plots
    viz.plot_candlestick(days=60, save_path='visualizations/candlestick.html')
    viz.create_dashboard(save_path='visualizations/dashboard.html')
    
    print("\n✓ All visualizations created successfully!")


def step_5_arima_model(data):
    """Step 5: Train ARIMA model"""
    print_header("STEP 5: ARIMA MODEL")
    
    arima_model = ARIMAModel(data, target_col='Close')
    
    # Check stationarity
    arima_model.check_stationarity(plot=False)
    
    # Split data
    arima_model.split_data(train_size=0.8)
    
    # Fit model
    arima_model.fit_model(order=(1, 1, 1))
    
    # Make predictions
    predictions = arima_model.predict()
    
    # Evaluate
    metrics = arima_model.evaluate()
    
    # Plot results
    arima_model.plot_predictions(save_path='visualizations/arima_predictions.png')
    arima_model.plot_residuals(save_path='visualizations/arima_residuals.png')
    
    # Save model
    arima_model.save_model()
    
    print("\n✓ ARIMA model completed")
    print(f"✓ RMSE: {metrics['RMSE']:.4f}")
    print(f"✓ MAPE: {metrics['MAPE']:.4f}%")
    
    return arima_model, metrics


def step_6_lstm_model(data):
    """Step 6: Train LSTM model"""
    print_header("STEP 6: LSTM MODEL")
    
    lstm_model = LSTMModel(data, target_col='Close', sequence_length=60)
    
    # Prepare data
    X_train, y_train, X_test, y_test = lstm_model.prepare_data(train_size=0.8)
    
    # Build model
    lstm_model.build_model(units=[50, 50, 25], dropout=0.2)
    
    # Train model
    history = lstm_model.train_model(epochs=50, batch_size=32, validation_split=0.1)
    
    # Plot training history
    lstm_model.plot_training_history(save_path='visualizations/lstm_training_history.png')
    
    # Make predictions
    predictions = lstm_model.predict()
    
    # Evaluate
    metrics = lstm_model.evaluate()
    
    # Plot results
    lstm_model.plot_predictions(save_path='visualizations/lstm_predictions.png')
    lstm_model.plot_prediction_error(save_path='visualizations/lstm_errors.png')
    
    # Save model
    lstm_model.save_model()
    
    # Predict future
    future_predictions = lstm_model.predict_future(days=30)
    
    print("\n✓ LSTM model completed")
    print(f"✓ RMSE: {metrics['RMSE']:.4f}")
    print(f"✓ MAPE: {metrics['MAPE']:.4f}%")
    print(f"\n✓ Future predictions (next 30 days) generated")
    
    return lstm_model, metrics, future_predictions


def compare_models(arima_metrics, lstm_metrics):
    """Compare model performance"""
    print_header("MODEL COMPARISON")
    
    print(f"{'Metric':<15} {'ARIMA':<15} {'LSTM':<15} {'Winner':<15}")
    print("-" * 60)
    
    for metric in ['RMSE', 'MAE', 'MAPE', 'R2 Score']:
        arima_val = arima_metrics[metric]
        lstm_val = lstm_metrics[metric]
        
        if metric == 'R2 Score':
            winner = 'ARIMA' if arima_val > lstm_val else 'LSTM'
        else:
            winner = 'ARIMA' if arima_val < lstm_val else 'LSTM'
        
        print(f"{metric:<15} {arima_val:<15.4f} {lstm_val:<15.4f} {winner:<15}")
    
    print("-" * 60)


def main():
    """Run complete pipeline"""
    print("\n" + "="*80)
    print("  STOCK PRICE PREDICTION MODEL - COMPLETE PIPELINE")
    print("="*80)
    
    # Configuration - ASK USER FOR TICKER
    print("\n📊 Welcome to Stock Price Predictor!")
    print("-" * 80)
    
    # Get ticker from user
    TICKER = input("\n🔍 Enter stock ticker (e.g., AAPL, GOOGL, TSLA, MSFT): ").upper().strip()
    
    if not TICKER:
        print("❌ No ticker provided. Using default: AAPL")
        TICKER = 'AAPL'
    
    # Ask if user wants to run all steps
    run_choice = input("\n🚀 Run complete pipeline? (y/n) [default: y]: ").lower().strip()
    RUN_ALL_STEPS = run_choice != 'n'
    
    print(f"\n✅ Configuration:")
    print(f"  📈 Stock Ticker: {TICKER}")
    print(f"  🔄 Run All Steps: {RUN_ALL_STEPS}")
    print("="*80)
    
    try:
        # Step 1: Data Collection
        if RUN_ALL_STEPS:
            data = step_1_data_collection(ticker=TICKER)
            if data is None:
                print("✗ Pipeline failed at data collection")
                return
        else:
            # Load existing data
            print_header("LOADING EXISTING DATA")
            data = pd.read_csv('data/raw/stock_data.csv')
            data['Date'] = pd.to_datetime(data['Date'])
            print(f"✓ Loaded {len(data)} rows from saved data")
        
        # Step 2: Data Preprocessing
        processed_data = step_2_data_preprocessing(data)
        
        # Step 3: Feature Engineering
        featured_data = step_3_feature_engineering(processed_data)
        
        # Step 4: Visualization
        step_4_visualization(featured_data, ticker=TICKER)
        
        # Step 5: ARIMA Model
        arima_model, arima_metrics = step_5_arima_model(processed_data)
        
        # Step 6: LSTM Model
        lstm_model, lstm_metrics, future_predictions = step_6_lstm_model(processed_data)
        
        # Compare Models
        compare_models(arima_metrics, lstm_metrics)
        
        # Final Summary
        print_header("PIPELINE COMPLETED SUCCESSFULLY!")
        
        print("Summary:")
        print(f"  ✓ Data collected and preprocessed")
        print(f"  ✓ {len(featured_data.columns)} features engineered")
        print(f"  ✓ Visualizations created in 'visualizations/' folder")
        print(f"  ✓ ARIMA model trained and saved")
        print(f"  ✓ LSTM model trained and saved")
        print(f"  ✓ Models compared and evaluated")
        
        print("\n" + "="*80)
        print("  All files saved in respective folders:")
        print("  - data/raw/ : Raw stock data")
        print("  - data/processed/ : Processed data and features")
        print("  - models/ : Trained models")
        print("  - visualizations/ : All plots and charts")
        print("="*80)
        
        # Display future predictions
        print("\n" + "="*80)
        print(f"  FUTURE PRICE PREDICTIONS (NEXT 30 DAYS) - LSTM MODEL")
        print("="*80)
        
        print(f"\n{'Day':<10} {'Predicted Price':<20}")
        print("-" * 30)
        for i, pred in enumerate(future_predictions[:10], 1):
            print(f"Day {i:<6} ${pred:.2f}")
        print(f"... and {len(future_predictions) - 10} more days")
        
        print("\n" + "="*80)
        print("  🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY! 🎉")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("  ✗ PIPELINE FAILED")
        print("="*80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


def run_quick_prediction(ticker='AAPL'):
    """Quick prediction without full pipeline (uses saved models)"""
    print_header("QUICK PREDICTION MODE")
    
    try:
        # Load data
        print("Loading data...")
        data = pd.read_csv('data/processed/processed_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        print(f"✓ Loaded {len(data)} rows")
        
        # Load LSTM model
        print("\nLoading LSTM model...")
        lstm_model = LSTMModel(data, target_col='Close', sequence_length=60)
        lstm_model.load_model('models/lstm_model.h5')
        lstm_model.prepare_data(train_size=0.8)
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = lstm_model.predict()
        metrics = lstm_model.evaluate()
        
        # Predict future
        future_predictions = lstm_model.predict_future(days=30)
        
        print("\n" + "="*60)
        print("QUICK PREDICTION RESULTS")
        print("="*60)
        print(f"\nModel Performance:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.4f}%")
        print(f"  R2 Score: {metrics['R2 Score']:.4f}")
        
        print(f"\nFuture Predictions (Next 30 days):")
        for i, pred in enumerate(future_predictions[:10], 1):
            print(f"  Day {i}: ${pred:.2f}")
        print(f"  ... and {len(future_predictions) - 10} more days")
        
        print("\n✓ Quick prediction completed!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {str(e)}")
        print("Please run the full pipeline first to train and save models.")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Prediction Pipeline')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'quick'],
                       help='Run mode: full pipeline or quick prediction')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker symbol')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        main()
    elif args.mode == 'quick':
        run_quick_prediction(ticker=args.ticker)