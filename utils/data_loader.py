import yfinance as yf
import pandas as pd
import os

def load_and_save_data(ticker_sber="SBER.ME", ticker_usd_rub="USDRUB=X", start_date="2018-01-01", end_date=None, data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(f"Загрузка данных по Сбербанку ({ticker_sber}) с {start_date} до {end_date if end_date else 'сегодня'}...")
    sber_data = yf.download(ticker_sber, start=start_date, end=end_date)
    sber_data.index.name = 'Date' 
    sber_file_path = os.path.join(data_dir, "sberbank_stock_data.csv")
    sber_data.to_csv(sber_file_path)
    print(f"Данные по Сбербанку сохранены в {sber_file_path}")

    print(f"Загрузка данных по USD/RUB ({ticker_usd_rub}) с {start_date} до {end_date if end_date else 'сегодня'}...")
    usd_rub_data = yf.download(ticker_usd_rub, start=start_date, end=end_date)
    usd_rub_data.index.name = 'Date'
    usd_rub_file_path = os.path.join(data_dir, "usd_rub_exchange_rate.csv")
    usd_rub_data.to_csv(usd_rub_file_path)
    print(f"Данные по USD/RUB сохранены в {usd_rub_file_path}")

def load_merged_data(sber_file_path="data/sberbank_stock_data.csv", usd_file_path="data/usd_rub_exchange_rate.csv"):
    try:
        sber_df = pd.read_csv(sber_file_path, skiprows=2, header=None, index_col=0, parse_dates=True)
        usd_df = pd.read_csv(usd_file_path, skiprows=2, header=None, index_col=0, parse_dates=True)

        sber_close = sber_df[[3]].rename(columns={3: 'SBER_Close'})
        usd_close = usd_df[[3]].rename(columns={3: 'USD_RUB_Close'})

    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл данных. Запустите `load_and_save_data()` сначала. {e}")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке или обработке данных: {e}")
        print("Пожалуйста, проверьте структуру ваших CSV-файлов.")
        return None

    merged_df = pd.merge(sber_close, usd_close, left_index=True, right_index=True, how='inner')

    merged_df.dropna(inplace=True)

    print(f"Объединенные данные загружены. Размер: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    load_and_save_data()
    df = load_merged_data()
    if df is not None:
        print(df.head())