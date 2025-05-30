import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def prepare_features(data, window_size=5):
    # Убедимся, что столбец 'Close' является числовым
    # errors='coerce' заменит все нечисловые значения на NaN
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    # Удалим строки с NaN значениями после преобразования
    data.dropna(subset=['Close'], inplace=True)

    closes = data["Close"].values
    X = []
    y = []
    # len(closes) - window_size - 1, потому что нам нужно window_size дней для X и 1 день для y (следующий день)
    for i in range(len(closes) - window_size - 1):
        window = closes[i:i+window_size]
        # target = 1, если цена следующего дня выше текущего дня в окне, иначе 0
        target = int(closes[i+window_size] > closes[i+window_size-1])
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

def rate_code_snn(data, num_steps=100):
    """
    Преобразует входные данные в тензор спайков для snnTorch.
    Использует Min-Max масштабирование и затем конвертирует в частоту.
    """
    # Убедимся, что data является numpy массивом числового типа
    # Если data уже тензор PyTorch, это условие будет False
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32) # Преобразуем в numpy массив
    
    # Теперь, когда мы уверены, что data - это numpy массив, преобразуем его в float32
    data = data.astype(np.float32)

    # Преобразуем numpy массив в тензор PyTorch
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Применяем Min-Max масштабирование к данным, чтобы они были в диапазоне [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    # reshape(-1, 1) нужен для scaler, потому что он ожидает 2D-массив
    scaled_data = torch.tensor(scaler.fit_transform(data_tensor.reshape(-1, 1)).flatten(), dtype=torch.float32)

    # Создаем тензор спайков: (num_steps, input_features)
    spike_data = torch.rand((num_steps, len(scaled_data))) < scaled_data

    return spike_data.float() # Возвращаем float, так как snnTorch ожидает float32