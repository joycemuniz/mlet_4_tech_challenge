import os
import yfinance as yf
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from .features import create_windows_multistep
from .model import build_lstm_multistep

def sanitize_ticker(ticker: str) -> str:
    return ticker.replace("-", "_").replace(".", "_").replace("^", "")

def train_and_save_for_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    window_size: int = 90,
    horizon: int = 14,
    models_dir: str = "models",
    epochs: int = 40,
    batch_size: int = 32,
):
    os.makedirs(models_dir, exist_ok=True)
    safe = sanitize_ticker(ticker)

    model_path = os.path.join(models_dir, f"{safe}_w{window_size}_h{horizon}.keras")
    scaler_path = os.path.join(models_dir, f"{safe}_scaler.pkl")

    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[["Close"]].dropna()

    if len(df) < (window_size + horizon + 10):
        raise ValueError(f"Dados insuficientes para treinar {ticker}. Linhas: {len(df)}")

    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)
    train_df = df.iloc[:split_index]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[["Close"]])

    X_train, y_train = create_windows_multistep(train_scaled, window_size, horizon)

    tf.random.set_seed(42)
    model = build_lstm_multistep(window_size, horizon)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path
