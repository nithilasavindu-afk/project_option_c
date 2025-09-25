"""
Task C.5 - Machine Learning 2
Model Factory for sequence models (LSTM / GRU / SimpleRNN) with flexible depth.
Enhanced for multivariate and multistep predictions.
"""

import io
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
from tensorflow.keras import regularizers

# Map short names to Keras layers
LAYER_MAP = {
    "lstm": LSTM,
    "gru": GRU,
    "rnn": SimpleRNN,
}

def _get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)

def build_model(
    model_type: str,
    input_shape: Tuple[int, int],
    units: int = 64,
    n_layers: int = 3,
    dropout: float = 0.2,
    bidirectional: bool = False,
    recurrent_dropout: float = 0.0,
    l2: float | None = None,
    dense_units: int | None = None,
    loss: str = "mse",
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    prediction_horizon: int = 1,  # NEW: For multistep prediction
) -> tf.keras.Model:
    """
    Build and compile a sequence model for regression.
    Enhanced for multivariate and multistep predictions (Task 5).
    """
    model_type = model_type.lower().strip()
    if model_type not in LAYER_MAP:
        raise ValueError(f"Unknown model_type={model_type}. Choose from {list(LAYER_MAP)}")

    RLayer = LAYER_MAP[model_type]
    reg = regularizers.l2(l2) if l2 is not None and l2 > 0 else None

    model = Sequential()
    for i in range(n_layers):
        return_sequences = i < (n_layers - 1)
        layer = RLayer(
            units,
            return_sequences=return_sequences,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=reg,
            name=f"{model_type.upper()}_{i+1}"
        )
        if bidirectional:
            layer = Bidirectional(layer, name=f"Bi{layer.name}")
        if i == 0:
            model.add(layer)
            model.build((None, *input_shape))
        else:
            model.add(layer)
        if dropout and dropout > 0:
            model.add(Dropout(dropout, name=f"dropout_{i+1}"))

    if dense_units is not None and dense_units > 0:
        model.add(Dense(dense_units, activation="relu", name="dense_hidden"))

    # Output layer: single value or multistep sequence
    if prediction_horizon == 1:
        model.add(Dense(1, name="output"))
    else:
        model.add(Dense(prediction_horizon, name=f"multistep_output_{prediction_horizon}"))

    loss_fn = tf.keras.losses.Huber() if loss.lower() == "huber" else "mse"
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss=loss_fn)
    return model

def model_summary_str(model: tf.keras.Model) -> str:
    """Return model summary as string."""
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()