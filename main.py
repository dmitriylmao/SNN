import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from utils.data_processing import load_stock_data, create_features, create_target, scale_features, create_sequences
from snn_model.snn_classifier import SNNClassifier

import random 

def set_seed(seed=66):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(66)


SEQUENCE_LENGTH = 20
INPUT_FEATURES = 4
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2
NUM_EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
SNN_BETA = 0.95
TRAIN_TEST_SPLIT_RATIO = 0.8

SNN_NUM_STEPS = SEQUENCE_LENGTH


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sequences.size(0)
            
            _, predicted_classes = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        print(f"Эпоха [{epoch+1}/{num_epochs}], Потери: {epoch_loss:.4f}, Точность: {epoch_accuracy:.4f}")
    
    return epoch_losses, epoch_accuracies

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    all_labels = []
    all_predictions = []
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * sequences.size(0)

            _, predicted_classes = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_classes.cpu().numpy())
            
    avg_test_loss = test_loss / total_samples
    accuracy = correct_predictions / total_samples
    print(f"Средние потери на тесте: {avg_test_loss:.4f}")
    print(f"Точность на тесте: {accuracy:.4f}")
    
    return all_labels, all_predictions, accuracy, avg_test_loss


def plot_metrics(train_losses, train_accuracies, val_accuracy, val_loss, num_epochs):
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Потери на обучении')
    plt.axhline(y=val_loss, color='r', linestyle='-', label=f'Потери на тесте: {val_loss:.4f}')
    plt.title('Потери (Loss)')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Точность на обучении')
    plt.axhline(y=val_accuracy, color='r', linestyle='-', label=f'Точность на тесте: {val_accuracy:.4f}')
    plt.title('Точность (Accuracy)')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predicted_labels, class_names=['Падение/Без изм.', 'Рост']):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    stock_df_raw = load_stock_data()
    if stock_df_raw is None:
        print("Не удалось загрузить данные. Завершение работы.")
        exit()

    features_df = create_features(stock_df_raw.copy())
    
    common_index = features_df.index.intersection(stock_df_raw.index)
    features_df = features_df.loc[common_index]
    aligned_close_prices = stock_df_raw.loc[common_index, 'Close']
    
    target_series = create_target(aligned_close_prices)
    target_series = target_series.loc[features_df.index]

    target_series.dropna(inplace=True)
    features_df = features_df.loc[target_series.index]

    if features_df.empty or target_series.empty or len(features_df) < SEQUENCE_LENGTH + 50:
        print("Недостаточно данных после обработки для создания последовательностей или обучения.")
        print(f"Размер признаков: {features_df.shape}, Размер таргета: {target_series.shape}")
        exit()
        
    if features_df.shape[1] != INPUT_FEATURES:
        print(f"ОШИБКА: Количество созданных признаков ({features_df.shape[1]}) не совпадает с INPUT_FEATURES ({INPUT_FEATURES}).")
        print("Созданные признаки:", features_df.columns.tolist())
        print("Пожалуйста, обновите INPUT_FEATURES в main.py или проверьте create_features в data_processing.py")
        exit()


    scaled_features_np, scaler = scale_features(features_df)
    
    X_sequences_np, y_sequences_np = create_sequences(scaled_features_np, target_series, SEQUENCE_LENGTH)

    if X_sequences_np.size == 0:
        print("Не удалось создать последовательности. Возможно, данных слишком мало.")
        exit()

    X_tensor = torch.tensor(X_sequences_np, dtype=torch.float)
    y_tensor = torch.tensor(y_sequences_np, dtype=torch.long)

    print(f"Форма тензора X (последовательности): {X_tensor.shape}")
    print(f"Форма тензора y (метки): {y_tensor.shape}")

    dataset = TensorDataset(X_tensor, y_tensor)
    total_samples = len(dataset)
    train_size = int(TRAIN_TEST_SPLIT_RATIO * total_samples)
    test_size = total_samples - train_size
    
    if train_size == 0 or test_size == 0:
        print(f"Недостаточно данных для разделения на train/test. Всего сэмплов: {total_samples}")
        exit()
        
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    if len(train_loader) == 0 or len(test_loader) == 0:
        print("Один из DataLoader пуст. Проверьте размеры выборок и BATCH_SIZE.")
        exit()

    model = SNNClassifier(input_size=INPUT_FEATURES, 
                            hidden_size=HIDDEN_SIZE, 
                            num_snn_steps=SNN_NUM_STEPS,
                            output_size=OUTPUT_SIZE,
                            beta=SNN_BETA).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Начало обучения ---")
    train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, device)
    print("--- Обучение завершено ---")

    print("\n--- Оценка модели на тестовых данных ---")
    true_labels, predicted_labels, test_accuracy, test_loss_val = evaluate_model(model, test_loader, criterion, device)
    
    plot_metrics(train_losses, train_accuracies, test_accuracy, test_loss_val, NUM_EPOCHS)
    plot_confusion_matrix(true_labels, predicted_labels)

    print("\n--- Анализ точности ---")
    print(f"Финальная точность на тестовой выборке: {test_accuracy*100:.2f}%")