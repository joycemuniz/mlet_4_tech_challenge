import os
from datetime import date
import numpy as np
import yfinance as yf
import joblib
from tensorflow import keras

def sanitize_ticker(ticker: str) -> str:
    return ticker.replace("-", "_").replace(".", "_").replace("^", "")

def resolve_paths(ticker: str, window_size: int, horizon: int, models_dir: str = "models"):
    safe = sanitize_ticker(ticker)
    model_path = os.path.join(models_dir, f"{safe}_w{window_size}_h{horizon}.keras")
    scaler_path = os.path.join(models_dir, f"{safe}_scaler.pkl")
    return model_path, scaler_path

def fetch_close_df(ticker: str, start_date: str, end_date: str):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df is None or df.empty:
        raise ValueError("Sem dados no yfinance para esse ticker/período.")
    df = df[["Close"]].dropna()
    if df.empty:
        raise ValueError("Série Close vazia após limpeza.")
    return df

def predict_for_ticker(
    ticker: str,
    start_date: str,
    end_date: str | None,
    window_size: int,
    horizon: int,
    models_dir: str = "models",
):
    """
    - Baixa dados do yfinance (Close)
    - Usa scaler e modelo salvos para o ticker
    - Retorna previsão em escala ORIGINAL (desnormalizada)
    """
    end_date = end_date or date.today().strftime("%Y-%m-%d")

    df = fetch_close_df(ticker, start_date, end_date)
    if len(df) < window_size:
        raise ValueError(f"Dados insuficientes: tem {len(df)} pontos e precisa de {window_size}.")

    model_path, scaler_path = resolve_paths(ticker, window_size, horizon, models_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")

    scaler = joblib.load(scaler_path)
    model = keras.models.load_model(model_path)

    window_vals = df["Close"].values[-window_size:].reshape(-1, 1)

    window_scaled = scaler.transform(window_vals)

    X = window_scaled.reshape(1, window_size, 1)

    yhat = model.predict(X, verbose=0)

    yhat = np.array(yhat).reshape(-1, 1)

    yhat_inv = scaler.inverse_transform(yhat).reshape(-1)

    preds = yhat_inv[:horizon].tolist()

    return {
        "ticker": ticker,
        "window_size": window_size,
        "horizon": horizon,
        "predictions": [float(p) for p in preds],
        "model_path": model_path,
        "scaler_path": scaler_path,
    }
