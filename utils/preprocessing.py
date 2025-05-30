import pandas as pd
import numpy as np
from brian2 import ms

def prepare_features(data, window_size=5):
    closes = data["Close"].values
    X = []
    y = []
    for i in range(len(closes) - window_size - 1):
        window = closes[i:i+window_size]
        target = int(closes[i+window_size] > closes[i+window_size-1])
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

def rate_code(data, max_rate=100, duration=100*ms):
    from brian2 import Hz
    from numpy import interp
    data = data.astype(float)  # <--- защита от строк
    if np.isnan(data).any():
        raise ValueError("NaN в данных! Проверь входные данные.")

    rates = interp(data, (data.min(), data.max()), (0, max_rate)) * Hz
    return rates