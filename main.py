from utils.data_loader import download_sberbank_data
import pandas as pd
from utils.preprocessing import prepare_features, rate_code_snn # Изменили функцию кодирования
from snn_model.snn_classifier import run_snn_snn # Изменили функцию запуска SNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import torch # Добавляем PyTorch
from sklearn.model_selection import train_test_split # Добавляем для разделения данных

# Определяем количество временных шагов для симуляции
NUM_STEPS = 100 # Эквивалент duration=100*ms

# 1. Скачиваем данные
download_sberbank_data()
df = pd.read_csv("data/sber.csv")

# 2. Готовим данные
X, y = prepare_features(df, window_size=5)

# Разделяем данные на обучающую и тестовую выборки
# Пока что просто разделим, чтобы показать, как это будет выглядеть
# Для реальной оценки нужно сделать более грамотное разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Пока что будем использовать весь X и y для простоты, как в твоем исходном коде,
# но в будущем будем работать с train/test.

# 3. Возьмём один пример для визуализации и теста (из X_test, если бы мы его использовали)
sample_input = X[100] # Или X_test[0] если будем использовать тестовые данные
# Кодируем входной пример в тензор спайков
sample_spike_input = rate_code_snn(sample_input, num_steps=NUM_STEPS)

# 4. Пропускаем через спайковую сеть
# run_snn_snn принимает тензор спайков
spikes_count = run_snn_snn(sample_spike_input) # Возвращает numpy массив

# 5. Визуализация входных данных и частот
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title("Цены за 5 дней (sample_input)")
plt.plot(sample_input, marker='o')
plt.xlabel("День")
plt.ylabel("Цена")

plt.subplot(1, 2, 2)
plt.title(f"Вероятности спайков (масштабированные данные, {NUM_STEPS} шагов)")
# Для визуализации вероятностей, нам нужно заново масштабировать sample_input
from sklearn.preprocessing import MinMaxScaler
scaler_viz = MinMaxScaler(feature_range=(0, 1))
scaled_sample_input = scaler_viz.fit_transform(sample_input.reshape(-1, 1)).flatten()
plt.bar(range(len(scaled_sample_input)), scaled_sample_input)
plt.xlabel("Нейрон")
plt.ylabel("Вероятность спайка (от 0 до 1)") # Изменили под rate coding
plt.tight_layout()
plt.show()

# 6. Визуализация количества спайков
plt.figure(figsize=(6, 4))
plt.title("Количество спайков на нейрон")
plt.bar(range(len(spikes_count)), spikes_count)
plt.xlabel("Нейрон")
plt.ylabel("Число спайков")
plt.show()

print(f"Total spikes: {np.sum(spikes_count)}")

# 7. Подсчёт точности на всем датасете (начинаем с 100-го примера, как и раньше)
threshold = 240
y_true = y[100:]
y_pred = []

for i in range(100, len(X)):
    # Кодируем каждый пример в спайки
    current_spike_input = rate_code_snn(X[i], num_steps=NUM_STEPS)
    # Пропускаем через SNN
    spikes_count_i = run_snn_snn(current_spike_input)
    total_spikes_i = np.sum(spikes_count_i)
    prediction = int(total_spikes_i > threshold)
    y_pred.append(prediction)

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.2f}")

# 8. Матрица ошибок (confusion matrix)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()