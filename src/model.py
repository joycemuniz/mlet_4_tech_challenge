import tensorflow as tf
from tensorflow.keras import layers

def build_lstm_multistep(window_size: int, horizon: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(horizon)  
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
