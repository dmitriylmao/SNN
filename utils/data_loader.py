import yfinance as yf
import pandas as pd

def download_sberbank_data(start="2020-01-01", end="2024-12-31", save_path="data/sber.csv"):
    ticker = "SBER.ME"
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(save_path)
    print(f"Saved to {save_path}")