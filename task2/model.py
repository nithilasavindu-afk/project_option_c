# File: model.py
# Task 2: LSTM Model Architecture
# Building upon v0.1.py model structure with class-based approach like P1

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import os
import matplotlib.pyplot as plt

class StockPredictionModel:
    """
    LSTM Model class for stock price prediction
    Based on v0.1 architecture with P1 model enhancements
    """

    def __init__(self, sequence_length, n_features, units=50, n_layers=3, dropout=0.2):
        """
        Initialize the model parameters

        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of features per time step
            units: Number of LSTM units per layer
            n_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units
        self.n_layers = n_layers
        self.dropout = dropout
        self.model = None
        self.history = None

    def build_model(self):
        """
        Build LSTM model with same architecture as v0.1
        """
        self.model = Sequential()

        # First LSTM layer - same as v0.1
        self.model.add(LSTM(units=self.units, return_sequences=True,
                           input_shape=(self.sequence_length, self.n_features)))
        self.model.add(Dropout(self.dropout))

        # Second LSTM layer - same as v0.1
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(Dropout(self.dropout))

        # Third LSTM layer - same as v0.1
        self.model.add(LSTM(units=self.units))
        self.model.add(Dropout(self.dropout))

        # Output layer - same as v0.1
        self.model.add(Dense(units=1))

        # Compile model - same as v0.1
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        return self.model

    def train_model(self, X_train, y_train, X_val=None, y_val=None,
                   epochs=25, batch_size=32, verbose=1):
        """
        Train the model
        """
        if self.model is None:
            self.build_model()

        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose
        )

        return self.history

    def save_model_weights(self, filepath):
        """
        Save model weights (like P1 model)
        """
        if self.model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save weights
            weights_path = filepath.replace('.h5', '.weights.h5')
            self.model.save_weights(weights_path)
            print(f"Model weights saved to {weights_path}")

            # Also save full model
            self.model.save(filepath)
            print(f"Full model saved to {filepath}")
        else:
            print("No model to save. Please build and train the model first.")

    def load_model_weights(self, filepath):
        """
        Load model weights
        """
        if self.model is None:
            self.build_model()

        weights_path = filepath.replace('.h5', '.weights.h5')
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"Model weights loaded from {weights_path}")
        elif os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f"Full model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")

    def save_training_plots(self, save_dir="results"):
        """
        Save training history plots (like P1 model)
        """
        if self.history is None:
            print("No training history to plot")
            return

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Plot training history
        plt.figure(figsize=(15, 5))

        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Training progress
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['loss'])
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        # Final loss comparison
        plt.subplot(1, 3, 3)
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1] if 'val_loss' in self.history.history else 0

        plt.bar(['Training', 'Validation'], [final_train_loss, final_val_loss])
        plt.title('Final Loss Comparison')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {plot_path}")
        plt.show()

    def predict(self, X):
        """
        Make predictions
        """
        if self.model is None:
            print("No model available. Please build and train the model first.")
            return None

        return self.model.predict(X)

    def get_model_summary(self):
        """
        Get model summary
        """
        if self.model is not None:
            return self.model.summary()
        else:
            print("No model built yet")
            return None

# Convenience functions for backward compatibility
def create_lstm_model(input_shape, units=50):
    """
    Create LSTM model - backward compatibility function
    """
    model_instance = StockPredictionModel(
        sequence_length=input_shape[0],
        n_features=input_shape[1],
        units=units
    )
    return model_instance.build_model()

def save_model(model, filepath):
    """
    Save trained model - backward compatibility function
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load saved model - backward compatibility function
    """
    model = tf.keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model
