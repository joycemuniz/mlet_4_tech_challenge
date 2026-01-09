import numpy as np

def create_windows_multistep(series_scaled: np.ndarray, window_size: int, horizon: int):
    X, y = [], []
    n = len(series_scaled)

    for i in range(window_size, n - horizon + 1):
        X.append(series_scaled[i-window_size:i, 0])
        y.append(series_scaled[i:i+horizon, 0])

    X = np.array(X).reshape(-1, window_size, 1)  # 3D para LSTM
    y = np.array(y)                              # (amostras, horizon)
    return X, y
