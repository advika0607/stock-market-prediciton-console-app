"""
LSTM Model Module - COMPLETE WORKING CODE
Implements LSTM (Long Short-Term Memory) neural network for stock prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Use the alias for Keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import warnings
warnings.filterwarnings('ignore')


class LSTMModel:
    """Class to implement LSTM model for stock price prediction"""
    
    def __init__(self, data, target_col='Close', sequence_length=60):
        self.data = data.copy()
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.y_test_actual = None
        
    def prepare_data(self, train_size=0.8, features=None):
        """Prepare and scale data for LSTM"""
        print("\n" + "="*60)
        print("PREPARING DATA FOR LSTM")
        print("="*60)
        
        if features is None:
            features = [self.target_col]
        
        print(f"Using features: {features}")
        print(f"Sequence length: {self.sequence_length}")
        
        data_values = self.data[features].values
        scaled_data = self.scaler.fit_transform(data_values)
        
        train_size_idx = int(len(scaled_data) * train_size)
        train_data = scaled_data[:train_size_idx]
        test_data = scaled_data[train_size_idx - self.sequence_length:]
        
        print(f"\nTrain data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")
        
        self.X_train, self.y_train = self._create_sequences(train_data)
        self.X_test, self.y_test = self._create_sequences(test_data)
        
        print(f"\nTraining sequences: {self.X_train.shape}")
        print(f"Training targets: {self.y_train.shape}")
        print(f"Testing sequences: {self.X_test.shape}")
        print(f"Testing targets: {self.y_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def _create_sequences(self, data):
        """Create sequences for LSTM input"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, units=None, dropout=0.2):
        """Build LSTM model architecture"""
        if units is None:
            units = [50, 50]
            
        print("\n" + "="*60)
        print("BUILDING LSTM MODEL")
        print("="*60)
        
        self.model = Sequential()
        
        self.model.add(LSTM(units=units[0], return_sequences=True, 
                           input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(dropout))
        
        for i in range(1, len(units)):
            return_seq = (i < len(units) - 1)
            self.model.add(LSTM(units=units[i], return_sequences=return_seq))
            self.model.add(Dropout(dropout))
        
        self.model.add(Dense(units=1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        print("\nModel Architecture:")
        self.model.summary()
        
        print(f"\n✓ Model built successfully")
        print(f"  LSTM layers: {len(units)}")
        print(f"  Units per layer: {units}")
        print(f"  Dropout rate: {dropout}")
        
        return self.model
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.1):
        """Train the LSTM model"""
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL")
        print("="*60)
        
        if self.model is None:
            print("✗ Error: Model not built. Call build_model() first.")
            return None
        
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Validation split: {validation_split}")
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        os.makedirs('models', exist_ok=True)
        checkpoint = ModelCheckpoint('models/lstm_best_model.h5', monitor='val_loss', 
                                    save_best_only=True, verbose=0)
        
        print("\nTraining in progress...")
        print("-" * 60)
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETED")
        print("="*60)
        
        return self.history
    
    def predict(self):
        """Make predictions on test set"""
        print("\n" + "="*60)
        print("MAKING PREDICTIONS")
        print("="*60)
        
        if self.model is None:
            print("✗ Error: Model not trained.")
            return None
        
        predictions_scaled = self.model.predict(self.X_test, verbose=0)
        
        predictions_scaled_full = np.zeros((len(predictions_scaled), self.X_test.shape[2]))
        predictions_scaled_full[:, 0] = predictions_scaled[:, 0]
        self.predictions = self.scaler.inverse_transform(predictions_scaled_full)[:, 0]
        
        y_test_scaled_full = np.zeros((len(self.y_test), self.X_test.shape[2]))
        y_test_scaled_full[:, 0] = self.y_test
        self.y_test_actual = self.scaler.inverse_transform(y_test_scaled_full)[:, 0]
        
        print(f"✓ Generated {len(self.predictions)} predictions")
        
        return self.predictions
    
    def evaluate(self):
        """Evaluate model performance"""
        if self.predictions is None:
            print("✗ Error: No predictions available. Call predict() first.")
            return None
        
        mse = mean_squared_error(self.y_test_actual, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test_actual, self.predictions)
        mape = np.mean(np.abs((self.y_test_actual - self.predictions) / self.y_test_actual)) * 100
        r2 = r2_score(self.y_test_actual, self.predictions)
        
        metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2 Score': r2}
        
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*60)
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """Plot training and validation loss"""
        if self.history is None:
            print("✗ Error: No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss During Training', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Model MAE During Training', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, save_path=None):
        """Plot predicted vs actual prices"""
        if self.predictions is None:
            print("✗ Error: No predictions available.")
            return
        
        plt.figure(figsize=(14, 7))
        
        indices = range(len(self.y_test_actual))
        
        plt.plot(indices, self.y_test_actual, label='Actual Price', 
                linewidth=2, color='blue', alpha=0.7)
        plt.plot(indices, self.predictions, label='LSTM Predictions', 
                linewidth=2, color='red', alpha=0.7, linestyle='--')
        
        plt.title('LSTM Model - Predicted vs Actual Prices', fontsize=16, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_error(self, save_path=None):
        """Plot prediction errors"""
        if self.predictions is None:
            print("✗ Error: No predictions available.")
            return
        
        errors = self.y_test_actual - self.predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_title('Prediction Error Distribution', fontweight='bold')
        axes[0].set_xlabel('Error')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(errors, color='purple', alpha=0.7)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title('Prediction Errors Over Time', fontweight='bold')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath='models/lstm_model.h5'):
        """Save the trained model"""
        if self.model is None:
            print("✗ Error: No model to save.")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/lstm_model.h5'):
        """Load a saved model"""
        try:
            self.model = load_model(filepath)
            print(f"✓ Model loaded from {filepath}")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
    
    def predict_future(self, days=30):
        """Predict future stock prices"""
        if self.model is None:
            print("✗ Error: Model not trained.")
            return None
        
        print(f"\n" + "="*60)
        print(f"PREDICTING NEXT {days} DAYS")
        print("="*60)
        
        last_sequence = self.scaler.transform(self.data[[self.target_col]].tail(self.sequence_length).values)
        last_sequence = last_sequence.reshape(1, self.sequence_length, 1)
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            next_pred = self.model.predict(current_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_pred_full = np.zeros((len(future_predictions), 1))
        future_pred_full[:, 0] = future_predictions[:, 0]
        future_predictions_actual = self.scaler.inverse_transform(future_pred_full)
        
        print(f"✓ Generated {days} future predictions")
        
        return future_predictions_actual.flatten()


def main():
    """Example usage of LSTMModel - COMPLETE WORKING EXAMPLE"""
    try:
        print("\n" + "="*70)
        print("  LSTM MODEL - STOCK PRICE PREDICTION")
        print("="*70)
        
        data = pd.read_csv('data/processed/processed_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        print(f"\n✓ Loaded {len(data)} rows of data")
        
        lstm_model = LSTMModel(data, target_col='Close', sequence_length=60)
        
        lstm_model.prepare_data(train_size=0.8)
        
        lstm_model.build_model(units=[50, 50, 25], dropout=0.2)
        
        lstm_model.train_model(epochs=50, batch_size=32, validation_split=0.1)
        
        print("\n" + "="*60)
        print("PLOTTING TRAINING HISTORY")
        print("="*60)
        lstm_model.plot_training_history(save_path='visualizations/lstm_training_history.png')
        
        lstm_model.predict()
        
        metrics = lstm_model.evaluate()
        
        print("\n" + "="*60)
        print("PLOTTING PREDICTIONS")
        print("="*60)
        lstm_model.plot_predictions(save_path='visualizations/lstm_predictions.png')
        lstm_model.plot_prediction_error(save_path='visualizations/lstm_errors.png')
        
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        lstm_model.save_model()
        
        future_days = 30
        future_predictions = lstm_model.predict_future(days=future_days)
        
        print("\n" + "="*60)
        print(f"FUTURE PREDICTIONS (NEXT {future_days} DAYS)")
        print("="*60)
        for i, pred in enumerate(future_predictions[:10], 1):
            print(f"Day {i}: ${pred:.2f}")
        print(f"... and {future_days - 10} more days")
        
        print("\n" + "="*70)
        print("  LSTM MODEL COMPLETED SUCCESSFULLY!")
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