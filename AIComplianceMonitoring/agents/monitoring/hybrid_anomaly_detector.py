"""
Hybrid Anomaly Detection System combining LSTM and Autoencoder for improved compliance monitoring.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional


class HybridAnomalyDetector:
    """Hybrid Anomaly Detection System using LSTM and Autoencoder."""
    
    def __init__(self, lstm_units: int = 64, ae_units: int = 32):
        """Initialize the hybrid detector."""
        self.lstm_units = lstm_units
        self.ae_units = ae_units
        self.lstm_model = None
        self.ae_model = None
        self.combined_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _build_lstm_model(self, input_shape: Tuple[int, int]):
        """Build LSTM-based anomaly detection model."""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            LSTM(self.lstm_units // 2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy')
        return model
    
    def _build_autoencoder(self, input_dim: int):
        """Build Autoencoder model."""
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoder = Dense(self.ae_units * 2, activation='relu')(input_layer)
        encoder = Dense(self.ae_units, activation='relu')(encoder)
        
        # Decoder
        decoder = Dense(self.ae_units * 2, activation='relu')(encoder)
        decoder = Dense(input_dim, activation='sigmoid')(decoder)
        
        autoencoder = Model(input_layer, decoder)
        autoencoder.compile(optimizer=Adam(), loss='mse')
        
        return autoencoder
    
    def _build_combined_model(self):
        """Build combined model that integrates LSTM and Autoencoder predictions."""
        lstm_input = Input(shape=(None, self.lstm_model.input_shape[2]))
        ae_input = Input(shape=(self.ae_model.input_shape[1],))
        
        lstm_pred = self.lstm_model(lstm_input)
        ae_pred = self.ae_model(ae_input)
        
        combined = concatenate([lstm_pred, ae_pred])
        final_output = Dense(1, activation='sigmoid')(combined)
        
        model = Model([lstm_input, ae_input], final_output)
        model.compile(optimizer=Adam(), loss='binary_crossentropy')
        
        return model
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for both LSTM and Autoencoder."""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Prepare LSTM sequence data
        lstm_sequences = []
        for i in range(len(data) - 10):
            lstm_sequences.append(scaled_data[i:i+10])
        lstm_sequences = np.array(lstm_sequences)
        
        # Prepare Autoencoder data
        ae_data = scaled_data[10:]  # Remove the first 10 samples used for LSTM
        
        return lstm_sequences, ae_data
    
    def train(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        """Train the hybrid anomaly detection system."""
        # Preprocess data
        lstm_sequences, ae_data = self.preprocess_data(data)
        
        # Build models if not already built
        if self.lstm_model is None:
            self.lstm_model = self._build_lstm_model(lstm_sequences.shape[1:])
        
        if self.ae_model is None:
            self.ae_model = self._build_autoencoder(ae_data.shape[1])
        
        # Train LSTM
        lstm_labels = np.zeros((lstm_sequences.shape[0], 1))
        self.lstm_model.fit(lstm_sequences, lstm_labels, epochs=epochs, batch_size=batch_size)
        
        # Train Autoencoder
        self.ae_model.fit(ae_data, ae_data, epochs=epochs, batch_size=batch_size)
        
        # Build and train combined model
        if self.combined_model is None:
            self.combined_model = self._build_combined_model()
        
        # Generate combined training data
        lstm_preds = self.lstm_model.predict(lstm_sequences)
        ae_preds = self.ae_model.predict(ae_data)
        
        # Train combined model
        combined_labels = np.zeros((ae_data.shape[0], 1))
        self.combined_model.fit(
            [lstm_sequences, ae_data],
            combined_labels,
            epochs=epochs,
            batch_size=batch_size
        )
        
        self.is_trained = True
    
    def detect_anomalies(self, data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """Detect anomalies using the hybrid model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
        
        lstm_sequences, ae_data = self.preprocess_data(data)
        
        # Get LSTM predictions
        lstm_preds = self.lstm_model.predict(lstm_sequences)
        
        # Get Autoencoder predictions
        ae_preds = self.ae_model.predict(ae_data)
        ae_reconstruction_error = np.mean(np.abs(ae_data - ae_preds), axis=1)
        
        # Get combined predictions
        combined_preds = self.combined_model.predict([lstm_sequences, ae_data])
        
        # Determine anomalies
        anomalies = {
            'lstm': (lstm_preds > threshold).flatten(),
            'ae': (ae_reconstruction_error > threshold).flatten(),
            'combined': (combined_preds > threshold).flatten()
        }
        
        return {
            'anomalies': anomalies,
            'scores': {
                'lstm': lstm_preds.flatten(),
                'ae': ae_reconstruction_error,
                'combined': combined_preds.flatten()
            },
            'confidence': self._calculate_confidence(anomalies)
        }
    
    def _calculate_confidence(self, anomalies: Dict[str, np.ndarray]) -> float:
        """Calculate confidence score based on multiple model agreements."""
        lstm_anom = anomalies['lstm']
        ae_anom = anomalies['ae']
        combined_anom = anomalies['combined']
        
        # Agreement score
        agreement = np.mean(
            (lstm_anom == combined_anom) & 
            (ae_anom == combined_anom)
        )
        
        return float(agreement)
