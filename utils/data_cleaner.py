# utils.data_cleaner.py
import pandas as pd

def clean_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Chuẩn hóa dataframe từ yfinance cho multi-output OHLCV.
    
    - Nếu MultiIndex, lấy các cột Open, High, Low, Close, Volume của ticker
    - Nếu single symbol, kiểm tra đủ các cột OHLCV
    - Chuẩn hóa Date, drop NaN, sort theo Date
    - Kết quả chuẩn hóa phù hợp cho pipeline multi-output LSTM
    """
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # Reset index nếu index là Date
    if df.index.name == "Date":
        df = df.reset_index()

    # MultiIndex columns (ví dụ: các cột từ yfinance với nhiều ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Lấy các cột Date + OHLCV của ticker
        cols_to_take = [("Date", "")] + [(col, ticker) for col in ["Open", "High", "Low", "Close", "Volume"]]
        try:
            df = df.loc[:, cols_to_take]
        except KeyError as e:
            raise KeyError(f"Không tìm thấy các cột OHLCV của ticker '{ticker}': {e}")
        # Đổi tên cột thành chuẩn ["Date", "Open", "High", "Low", "Close", "Volume"]
        df.columns = required_cols
    else:
        # Single symbol
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Dữ liệu phải chứa các cột: {required_cols}. Thiếu: {missing_cols}"
            )
        df = df[required_cols]

    # Chuẩn hóa dữ liệu
    df = df.dropna(subset=required_cols).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


