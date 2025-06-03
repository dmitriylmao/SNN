import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from snntorch import spikegen

def download_sber_data(ticker="SBER.ME", start_date="2010-01-01", end_date=None, filepath="data/SBER_stock_data.csv"):
    pass

def load_stock_data(filepath="data/SBER_stock_data.csv"):
    try:
        column_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df = pd.read_csv(filepath, skiprows=4, header=None, names=column_names, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV файл должен содержать колонки: {', '.join(required_cols)}")
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df
    except FileNotFoundError:
        print(f"Файл не найден: {filepath}. Пожалуйста, скачай его и помести в папку 'data'.")
        print("Ты можешь попробовать использовать функцию download_sber_data() или скачать вручную.")
        return None

def create_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Price_Diff'] = df['Close'].diff()
    df['MA5'] = df['Close'].rolling(window=5).mean().pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean().pct_change()
    df['Volatility'] = df['Return'].rolling(window=5).std()

    df.dropna(inplace=True)
    
    features = df[['Return', 'MA5', 'MA10', 'Volatility']].copy()
    
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    
    return features

def create_target(df_close_price, forecast_horizon=1):
    target = (df_close_price.shift(-forecast_horizon) > df_close_price).astype(int)
    return target

def scale_features(features_df):
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)
    return scaled_features, scaler

def encode_to_spikes(scaled_data, num_steps):
    spike_data = spikegen.rate(torch.tensor(scaled_data, dtype=torch.float), num_steps=num_steps)
    return spike_data.permute(1, 0, 2)


def create_sequences(features, target, sequence_length):
    X, y = [], []
    target_np = target.to_numpy() if isinstance(target, pd.Series) else target

    for i in range(len(features) - sequence_length):
        X.append(features[i:(i + sequence_length)])
        y.append(target_np[i + sequence_length -1])

    return np.array(X), np.array(y)

if __name__ == '__main__':
    download_sber_data()

    stock_df = load_stock_data()
    if stock_df is not None:
        features_df = create_features(stock_df.copy())

        common_index = features_df.index.intersection(stock_df.index)
        features_df = features_df.loc[common_index]
        aligned_close_prices = stock_df.loc[common_index, 'Close']
        
        target_series = create_target(aligned_close_prices)
        target_series = target_series.loc[features_df.index]

        target_series.dropna(inplace=True)
        features_df = features_df.loc[target_series.index]

        if features_df.empty or target_series.empty:
            print("Недостаточно данных после обработки. Проверьте ваш CSV файл и параметры.")
        else:
            print(f"Размер датасета признаков после обработки: {features_df.shape}")
            print(f"Размер датасета таргета после обработки: {target_series.shape}")
            print("\nПервые 5 строк признаков:")
            print(features_df.head())
            print("\nПервые 5 строк таргета:")
            print(target_series.head())

            scaled_features, scaler = scale_features(features_df)
            print(f"\nРазмер масштабированных признаков: {scaled_features.shape}")

            sequence_length = 10
            X_seq, y_seq = create_sequences(scaled_features, target_series, sequence_length)
            
            if X_seq.size == 0:
                print("Не удалось создать последовательности. Возможно, данных слишком мало для указанной sequence_length.")
            else:
                print(f"\nФорма X_seq (до кодирования в спайки): {X_seq.shape}")
                print(f"Форма y_seq: {y_seq.shape}")

                num_sequences, seq_len, num_original_features = X_seq.shape
                
                X_reshaped_for_spike_encoding = X_seq.reshape(-1, num_original_features)
                
                num_spike_encoding_steps_per_feature_vector = 1
                
                print(f"\nФинальная форма X_seq для подачи в SNN (как аналоговые значения): {X_seq.shape}")
                print(f"Финальная форма y_seq: {torch.tensor(y_seq, dtype=torch.long).shape}")
                X_final_data = torch.tensor(X_seq, dtype=torch.float)
                y_final_data = torch.tensor(y_seq, dtype=torch.long)