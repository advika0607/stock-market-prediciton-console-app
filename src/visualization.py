"""
Visualization Module - COMPLETE WORKING CODE
Creates various plots and charts for data analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)


class StockVisualizer:
    """
    Class to create visualizations for stock data
    """
    
    def __init__(self, data, ticker='STOCK'):
        self.data = data.copy()
        self.ticker = ticker
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
    
    def plot_price_history(self, save_path=None):
        """Plot historical price with volume"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(self.data['Date'], self.data['Close'], label='Close Price', linewidth=2, color='#2E86C1')
        ax1.set_title(f'{self.ticker} - Historical Stock Price', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(self.data['Date'], self.data['Volume'], color='#85C1E9', alpha=0.7, label='Volume')
        ax2.set_title('Trading Volume', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()
    
    def plot_candlestick(self, days=60, save_path=None):
        """Create interactive candlestick chart"""
        recent_data = self.data.tail(days)
        fig = go.Figure(data=[go.Candlestick(
            x=recent_data['Date'],
            open=recent_data['Open'],
            high=recent_data['High'],
            low=recent_data['Low'],
            close=recent_data['Close'],
            name='OHLC'
        )])
        fig.update_layout(
            title=f'{self.ticker} - Candlestick Chart (Last {days} days)',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template='plotly_white',
            height=600
        )
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"✓ Interactive plot saved to {save_path}")
        fig.show()
    
    def plot_moving_averages(self, save_path=None):
        """Plot price with moving averages"""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Date'], self.data['Close'], label='Close Price', linewidth=2, color='black', alpha=0.7)
        ma_cols = [col for col in self.data.columns if 'SMA' in col or 'EMA' in col]
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, col in enumerate(ma_cols[:6]):
            if col in self.data.columns:
                plt.plot(self.data['Date'], self.data[col], label=col, linewidth=1.5, color=colors[i % len(colors)], alpha=0.7)
        plt.title(f'{self.ticker} - Price with Moving Averages', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()
    
    def plot_technical_indicators(self, save_path=None):
        """Plot technical indicators (RSI, MACD, Bollinger Bands)"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        if all(col in self.data.columns for col in ['BB_HIGH', 'BB_LOW', 'BB_MID']):
            axes[0].plot(self.data['Date'], self.data['Close'], label='Close', linewidth=2)
            axes[0].plot(self.data['Date'], self.data['BB_HIGH'], label='Upper BB', linestyle='--', alpha=0.7)
            axes[0].plot(self.data['Date'], self.data['BB_MID'], label='Middle BB', linestyle='--', alpha=0.7)
            axes[0].plot(self.data['Date'], self.data['BB_LOW'], label='Lower BB', linestyle='--', alpha=0.7)
            axes[0].fill_between(self.data['Date'], self.data['BB_HIGH'], self.data['BB_LOW'], alpha=0.1)
            axes[0].set_title('Price with Bollinger Bands', fontweight='bold')
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)
        
        if 'RSI_14' in self.data.columns:
            axes[1].plot(self.data['Date'], self.data['RSI_14'], label='RSI', linewidth=2, color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
            axes[1].set_title('Relative Strength Index (RSI)', fontweight='bold')
            axes[1].set_ylim([0, 100])
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
        
        if all(col in self.data.columns for col in ['MACD', 'MACD_SIGNAL', 'MACD_DIFF']):
            axes[2].plot(self.data['Date'], self.data['MACD'], label='MACD', linewidth=2, color='blue')
            axes[2].plot(self.data['Date'], self.data['MACD_SIGNAL'], label='Signal', linewidth=2, color='red')
            axes[2].bar(self.data['Date'], self.data['MACD_DIFF'], label='Histogram', alpha=0.3, color='gray')
            axes[2].set_title('MACD', fontweight='bold')
            axes[2].legend(loc='best')
            axes[2].grid(True, alpha=0.3)
        
        axes[3].bar(self.data['Date'], self.data['Volume'], alpha=0.7, color='steelblue')
        axes[3].set_title('Trading Volume', fontweight='bold')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()
    
    def plot_correlation_heatmap(self, features=None, save_path=None):
        """Plot correlation heatmap of features"""
        if features is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col not in ['Day_of_Week', 'Month', 'Quarter']][:20]
        corr_matrix = self.data[features].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()
    
    def plot_returns_distribution(self, save_path=None):
        """Plot distribution of returns"""
        if 'Returns' not in self.data.columns:
            self.data['Returns'] = self.data['Close'].pct_change()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(self.data['Returns'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_title('Distribution of Daily Returns', fontweight='bold')
        axes[0].set_xlabel('Returns')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        stats.probplot(self.data['Returns'].dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()
    
    def plot_prediction_vs_actual(self, actual, predicted, dates=None, save_path=None):
        """Plot predicted vs actual prices"""
        plt.figure(figsize=(14, 7))
        if dates is None:
            dates = range(len(actual))
        plt.plot(dates, actual, label='Actual Price', linewidth=2, color='blue', alpha=0.7)
        plt.plot(dates, predicted, label='Predicted Price', linewidth=2, color='red', alpha=0.7, linestyle='--')
        plt.title(f'{self.ticker} - Predicted vs Actual Prices', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.show()
    
    def create_dashboard(self, save_path='visualizations/dashboard.html'):
        """Create interactive dashboard with multiple charts"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price History', 'Volume', 'RSI', 'MACD', 'Candlestick (Last 60 days)', 'Returns Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "candlestick"}, {"secondary_y": False}]]
        )
        fig.add_trace(go.Scatter(x=self.data['Date'], y=self.data['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Bar(x=self.data['Date'], y=self.data['Volume'], name='Volume', marker=dict(color='lightblue')), row=1, col=2)
        if 'RSI_14' in self.data.columns:
            fig.add_trace(go.Scatter(x=self.data['Date'], y=self.data['RSI_14'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
        if 'MACD' in self.data.columns:
            fig.add_trace(go.Scatter(x=self.data['Date'], y=self.data['MACD'], name='MACD', line=dict(color='blue')), row=2, col=2)
            fig.add_trace(go.Scatter(x=self.data['Date'], y=self.data['MACD_SIGNAL'], name='Signal', line=dict(color='red')), row=2, col=2)
        recent = self.data.tail(60)
        fig.add_trace(go.Candlestick(x=recent['Date'], open=recent['Open'], high=recent['High'], low=recent['Low'], close=recent['Close'], name='OHLC'), row=3, col=1)
        if 'Returns' in self.data.columns:
            fig.add_trace(go.Histogram(x=self.data['Returns'].dropna(), name='Returns', marker=dict(color='steelblue')), row=3, col=2)
        fig.update_layout(title_text=f"{self.ticker} - Stock Analysis Dashboard", showlegend=True, height=1200, template='plotly_white')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"✓ Dashboard saved to {save_path}")
        fig.show()


def main():
    """Example usage of StockVisualizer"""
    try:
        data = pd.read_csv('data/processed/features.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        viz = StockVisualizer(data, ticker='AAPL')
        print("Creating visualizations...")
        viz.plot_price_history(save_path='visualizations/price_history.png')
        viz.plot_moving_averages(save_path='visualizations/moving_averages.png')
        viz.plot_technical_indicators(save_path='visualizations/technical_indicators.png')
        viz.plot_correlation_heatmap(save_path='visualizations/correlation_heatmap.png')
        viz.plot_returns_distribution(save_path='visualizations/returns_distribution.png')
        viz.plot_candlestick(days=60, save_path='visualizations/candlestick.html')
        viz.create_dashboard(save_path='visualizations/dashboard.html')
        print("\n✓ All visualizations created successfully!")
    except FileNotFoundError:
        print("✗ Error: Please run feature engineering first!")


if __name__ == "__main__":
    main()