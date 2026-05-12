import pandas as pd
import streamlit as st
import numpy as np

def backtest_strategy(df, forecast_days=180, strategy="buy_and_hold", short_window=5, long_window=20, threshold=0.05):
    """
    Backtest chiến lược đầu tư trên dữ liệu dự báo giá cổ phiếu.

    Args:
        df (pd.DataFrame): DataFrame phải có các cột: 'Date', 'Close', 'Predicted_Close'
        forecast_days (int): Số ngày thực hiện backtest
        strategy (str): Tên chiến lược, ví dụ: 'buy_and_hold', 'predict_signal', 'moving_average', 'momentum', 'bollinger', 'rsi'
        short_window (int): Sử dụng cho chiến lược MA ngắn hạn (dùng cho MA crossover)
        long_window (int): Sử dụng cho chiến lược MA dài hạn
        threshold (float): Ngưỡng dùng cho chiến lược momentum

    Returns:
        pd.DataFrame: DataFrame kết quả với lợi nhuận cộng dồn theo chiến lược và mua giữ
    """
    if df is None or not isinstance(df, pd.DataFrame):
        st.error("❌ Dữ liệu đầu vào không hợp lệ!")
        return pd.DataFrame()

    # Kiểm tra xem có đầy đủ cột không
    required_cols = {"Date", "Close", "Predicted_Close"}
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        st.error(f"❌ Dữ liệu thiếu các cột cần thiết: {', '.join(missing_cols)}")
        return pd.DataFrame()

    df = df.copy()
    df = df.sort_values("Date")  # Sắp xếp theo ngày
    df = df.tail(forecast_days).reset_index(drop=True)

    if len(df) < forecast_days:
        st.warning(f"⚠️ Dữ liệu không đủ {forecast_days} ngày để backtest.")
        return pd.DataFrame()

    # Tính lợi nhuận hàng ngày
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)
    df["Buy_and_Hold_Return"] = df["Daily_Return"]
    df["Cumulative_BuyHold"] = (1 + df["Buy_and_Hold_Return"]).cumprod()

    # Xử lý các chiến lược
    if strategy == "predict_signal":
        df["Signal"] = (df["Predicted_Close"] > df["Close"].shift(1)).astype(int)
        df["Strategy_Return"] = df["Signal"] * df["Daily_Return"]

    elif strategy == "moving_average" or strategy == "ma_crossover":
        df["SMA_Short"] = df["Close"].rolling(window=short_window).mean()
        df["SMA_Long"] = df["Close"].rolling(window=long_window).mean()
        df["Signal"] = (df["SMA_Short"] > df["SMA_Long"]).astype(int)
        df["Strategy_Return"] = df["Signal"] * df["Daily_Return"]

    elif strategy == "momentum":
        base_price = df["Close"].iloc[0]
        df["Signal"] = (df["Close"] >= base_price * (1 + threshold)).astype(int)
        df["Strategy_Return"] = df["Signal"] * df["Daily_Return"]

    elif strategy == "bollinger":
        window = long_window
        df["MA"] = df["Close"].rolling(window=window).mean()
        df["STD"] = df["Close"].rolling(window=window).std()
        df["Upper"] = df["MA"] + 2 * df["STD"]
        df["Lower"] = df["MA"] - 2 * df["STD"]
        df["Signal"] = np.where(df["Close"] < df["Lower"], 1, np.where(df["Close"] > df["Upper"], -1, 0))
        df["Strategy_Return"] = df["Signal"].shift(1) * df["Daily_Return"]

    elif strategy == "rsi":
        window = short_window
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["Signal"] = np.where(df["RSI"] < 30, 1, np.where(df["RSI"] > 70, -1, 0))
        df["Strategy_Return"] = df["Signal"].shift(1) * df["Daily_Return"]

    elif strategy == "buy_and_hold":
        df["Signal"] = 1
        df["Strategy_Return"] = df["Buy_and_Hold_Return"]

    else:
        df["Signal"] = 0
        df["Strategy_Return"] = 0.0

    # Tính toán lợi nhuận cộng dồn và Portfolio Value
    df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod().fillna(1)
    df["Cumulative_BuyHold"] = (1 + df["Buy_and_Hold_Return"]).cumprod().fillna(1)
    df["Portfolio_Value"] = df["Cumulative_Strategy"]

    # In ra kết quả chi tiết
    # st.write(f"🔍 Kết quả backtest cho chiến lược '{strategy}':")
    # st.write(df.head())  # Debug: Hiển thị 5 dòng đầu của kết quả backtest

    return df[[
        "Date", "Close", "Predicted_Close",
        "Signal", "Strategy_Return", "Cumulative_Strategy",
        "Buy_and_Hold_Return", "Cumulative_BuyHold",
        "Portfolio_Value"
    ]]
