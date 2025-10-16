# Stock Market Analysis Console Application

A comprehensive Python application for end-to-end stock market analysis and prediction using time series forecasting and deep learning models.

## Features

- Automated data collection from stock tickers.
- Data preprocessing including cleaning, missing value handling, and returns calculation.
- Advanced feature engineering with technical, momentum, volatility, trend, and volume indicators.
- Batch analysis to process multiple stock tickers efficiently.
- Time series forecasting using ARIMA and LSTM deep learning models.
- Interactive visualization of stock prices, returns, volume, and technical indicators.
- Scalable and reproducible pipeline with modular design.

## Technology Stack

- Python 3.x
- Pandas, NumPy for data handling
- Scikit-learn for preprocessing
- TensorFlow/Keras for LSTM modeling
- Statsmodels for ARIMA forecasting
- Matplotlib, Seaborn, Plotly for visualizations

## File Structure

- `data_collection.py` - Fetches stock data for analysis.
- `data_preprocessing.py` - Cleans and prepares data for modeling.
- `feature_engineering.py` - Creates and adds numerous financial features.
- `batch_analysis.py` - Manages batch processing of multiple stocks.
- `model_arima.py` - Implements ARIMA forecasting model.
- `model_lstm.py` - Implements LSTM deep learning model.
- `visualization.py` - Contains plotting utilities for analysis.
- `main.py` - The main pipeline orchestrating the analysis.

## Installation

Clone the repository:


