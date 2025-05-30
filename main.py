from utils.data_loader import download_sberbank_data
import pandas as pd
from utils.preprocessing import prepare_features, rate_code
from snn_model.snn_classifier import run_snn
from brian2 import ms, Hz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 1. Скачиваем данные
download_sberbank_data()
df = pd.read_csv("data/sber.csv")

# 2. Готовим данные
X, y = prepare_features(df, window_size=5)

# 3. Возьмём один пример для визуализации и теста
sample_input = X[100]
rates = rate_code(sample_input, max_rate=100)

# 4. Пропускаем через спайковую сеть
spikes = run_snn(rates, duration=100*ms)

# 5. Визуализация входных данных и частот
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title("Цены за 5 дней (sample_input)")
plt.plot(sample_input, marker='o')
plt.xlabel("День")
plt.ylabel("Цена")

plt.subplot(1, 2, 2)
plt.title("Частоты спайков (rates)")
plt.bar(range(len(rates)), [r / Hz for r in rates])  # делим на Hz для чисел
plt.xlabel("Нейрон")
plt.ylabel("Частота (Гц)")

plt.tight_layout()
plt.show()

# 6. Визуализация количества спайков
plt.figure(figsize=(6, 4))
plt.title("Количество спайков на нейрон")
plt.bar(range(len(spikes)), spikes)
plt.xlabel("Нейрон")
plt.ylabel("Число спайков")
plt.show()

print(f"Total spikes: {np.sum(spikes)}")

# 7. Подсчёт точности на всем датасете (начинаем с 100-го примера, чтобы не использовать слишком маленькие окна)
threshold = 20  # Можно подобрать, чтобы улучшить точность
y_true = y[100:]
y_pred = []

for i in range(100, len(X)):
    rates = rate_code(X[i], max_rate=100)
    spikes = run_snn(rates, duration=100*ms)
    total_spikes = np.sum(spikes)
    prediction = int(total_spikes > threshold)
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
