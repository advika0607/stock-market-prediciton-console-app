"""
Batch Analysis - Analyze Multiple Stocks at Once
Run: python batch_analysis.py
"""

import sys
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


def analyze_stock(ticker):
    """Analyze a single stock"""
    print("\n" + "="*80)
    print(f"  ANALYZING: {ticker}")
    print("="*80)
    
    try:
        # Step 1: Data Collection
        print(f"\n📥 Fetching data for {ticker}...")
        collector = StockDataCollector(ticker)
        data = collector.fetch_data()
        
        if data is None:
            print(f"❌ Failed to fetch data for {ticker}")
            return None
        
        collector.save_data(filepath=f'data/raw/{ticker}_stock_data.csv')
        print(f"✅ Data fetched: {len(data)} rows")
        
        # Step 2: Preprocessing
        print(f"\n🔧 Preprocessing data...")
        preprocessor = DataPreprocessor(data)
        preprocessor.check_missing_values()
        preprocessor.handle_missing_values(method='ffill')
        preprocessor.create_returns()
        preprocessor.create_moving_averages(windows=[5, 10, 20, 50])
        processed_data = preprocessor.get_processed_data()
        preprocessor.save_processed_data(filepath=f'data/processed/{ticker}_processed.csv')
        print(f"✅ Preprocessing completed")
        
        # Step 3: Feature Engineering
        print(f"\n⚙️ Engineering features...")
        engineer = FeatureEngineer(processed_data)
        featured_data = engineer.build_all_features(target_horizon=1)
        engineer.save_features(filepath=f'data/processed/{ticker}_features.csv')
        print(f"✅ Features created: {len(featured_data.columns)} features")
        
        # Step 4: Visualization
        print(f"\n📊 Creating visualizations...")
        viz = StockVisualizer(featured_data, ticker=ticker)
        viz.plot_price_history(save_path=f'visualizations/{ticker}_price_history.png')
        viz.plot_technical_indicators(save_path=f'visualizations/{ticker}_technical.png')
        viz.create_dashboard(save_path=f'visualizations/{ticker}_dashboard.html')
        print(f"✅ Visualizations saved")
        
        # Step 5: LSTM Model
        print(f"\n🤖 Training LSTM model...")
        lstm_model = LSTMModel(processed_data, target_col='Close', sequence_length=60)
        lstm_model.prepare_data(train_size=0.8)
        lstm_model.build_model(units=[50, 50, 25], dropout=0.2)
        lstm_model.train_model(epochs=30, batch_size=32, validation_split=0.1)
        lstm_model.predict()
        metrics = lstm_model.evaluate()
        lstm_model.plot_predictions(save_path=f'visualizations/{ticker}_lstm_predictions.png')
        lstm_model.save_model(filepath=f'models/{ticker}_lstm_model.h5')
        
        # Step 6: Future Predictions
        future_predictions = lstm_model.predict_future(days=30)
        
        print(f"\n✅ Analysis completed for {ticker}")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   Future prediction (Day 1): ${future_predictions[0]:.2f}")
        
        return {
            'ticker': ticker,
            'metrics': metrics,
            'future_predictions': future_predictions
        }
        
    except Exception as e:
        print(f"\n❌ Error analyzing {ticker}: {str(e)}")
        return None


def main():
    """Main batch analysis function"""
    print("\n" + "="*80)
    print("  📊 BATCH STOCK ANALYSIS")
    print("="*80)
    
    # Get stock list from user
    print("\n🔍 Enter stock tickers to analyze:")
    print("   (Separate multiple tickers with commas)")
    print("   Example: AAPL, GOOGL, TSLA, MSFT")
    print()
    
    user_input = input("Tickers: ").upper().strip()
    
    if not user_input:
        print("\n⚠️  No tickers provided. Using defaults: AAPL, GOOGL, TSLA")
        tickers = ['AAPL', 'GOOGL', 'TSLA']
    else:
        tickers = [t.strip() for t in user_input.split(',')]
    
    print(f"\n✅ Will analyze {len(tickers)} stocks: {', '.join(tickers)}")
    
    # Confirm
    confirm = input("\n🚀 Proceed with analysis? (y/n) [y]: ").lower().strip()
    if confirm == 'n':
        print("❌ Analysis cancelled")
        return
    
    # Analyze each stock
    results = []
    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'='*80}")
        print(f"  Progress: {i}/{len(tickers)}")
        print(f"{'='*80}")
        
        result = analyze_stock(ticker)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("  📊 ANALYSIS SUMMARY")
    print("="*80)
    
    if results:
        print(f"\n✅ Successfully analyzed {len(results)}/{len(tickers)} stocks\n")
        
        print(f"{'Ticker':<10} {'RMSE':<12} {'MAPE':<12} {'Day 1 Pred':<15}")
        print("-" * 60)
        
        for result in results:
            ticker = result['ticker']
            rmse = result['metrics']['RMSE']
            mape = result['metrics']['MAPE']
            pred = result['future_predictions'][0]
            
            print(f"{ticker:<10} {rmse:<12.2f} {mape:<12.2f} ${pred:<14.2f}")
        
        print("\n" + "="*80)
        print("  🎉 BATCH ANALYSIS COMPLETED!")
        print("="*80)
        print("\n📁 Files saved in:")
        print("   • data/raw/ - Raw stock data")
        print("   • data/processed/ - Processed data and features")
        print("   • models/ - Trained models")
        print("   • visualizations/ - Charts and dashboards")
        print()
    else:
        print("\n❌ No stocks were successfully analyzed")


if __name__ == "__main__":
    main()