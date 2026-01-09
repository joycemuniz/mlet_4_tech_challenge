import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def backtest_metrics_multistep(
    model,
    scaler,
    close_df: pd.DataFrame,
    window_size: int = 90,
    horizon: int = 14,
    backtest_days: int = 180
):
    """
    Faz backtest nos últimos `backtest_days` dias (aprox).
    Para cada ponto t, prevê t+1..t+14 e compara com valores reais.
    Retorna métricas globais e por dia.
    """
    df = close_df.copy()
    df = df[["Close"]].dropna()

    if len(df) < window_size + horizon + 10:
        raise ValueError("Dados insuficientes para backtest.")

    start_idx = max(window_size, len(df) - backtest_days)
    end_idx = len(df) - horizon

    if end_idx <= start_idx:
        raise ValueError("backtest_days muito grande para o tamanho do dataset.")

    scaled_all = scaler.transform(df[["Close"]].values)

    y_true_all = []
    y_pred_all = []

    for i in range(start_idx, end_idx):
        X = scaled_all[i-window_size:i, 0].reshape(1, window_size, 1)
        pred_scaled = model.predict(X, verbose=0)

        true_scaled = scaled_all[i:i+horizon, 0]  

        pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true_real = scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()

        y_pred_all.append(pred_real)
        y_true_all.append(true_real)

    y_pred_all = np.array(y_pred_all)  
    y_true_all = np.array(y_true_all)

    mae_global = mean_absolute_error(y_true_all.flatten(), y_pred_all.flatten())
    rmse_global = np.sqrt(mean_squared_error(y_true_all.flatten(), y_pred_all.flatten()))
    mape_global = mape(y_true_all.flatten(), y_pred_all.flatten())

    mae_by_day = [mean_absolute_error(y_true_all[:, d], y_pred_all[:, d]) for d in range(horizon)]
    rmse_by_day = [np.sqrt(mean_squared_error(y_true_all[:, d], y_pred_all[:, d])) for d in range(horizon)]
    mape_by_day = [mape(y_true_all[:, d], y_pred_all[:, d]) for d in range(horizon)]

    return {
        "mae_global": float(mae_global),
        "rmse_global": float(rmse_global),
        "mape_global": float(mape_global),
        "mae_by_day": [float(x) for x in mae_by_day],
        "rmse_by_day": [float(x) for x in rmse_by_day],
        "mape_by_day": [float(x) for x in mape_by_day],
        "samples": int(y_true_all.shape[0])
    }

def append_metrics_log(
    log_path: str,
    row: dict
):
    """
    Salva histórico de métricas em CSV.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    df_new = pd.DataFrame([row])
    if os.path.exists(log_path):
        df_old = pd.read_csv(log_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(log_path, index=False)
    return df_all
