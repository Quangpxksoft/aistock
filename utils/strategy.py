import pandas as pd

def backtest_strategy(df, forecast_days=180, strategy="buy_and_hold", short_window=5, long_window=20):
    """
    Backtest chiến lược đầu tư trên dữ liệu dự báo giá cổ phiếu.

    Args:
        df (pd.DataFrame): DataFrame phải có các cột: 'Date', 'Close', 'Predicted_Close'
        forecast_days (int): Số ngày thực hiện backtest
        strategy (str): Tên chiến lược, ví dụ: 'buy_and_hold', 'predict_signal', 'moving_average'
        short_window (int): Sử dụng cho chiến lược MA ngắn hạn (dùng cho MA crossover)
        long_window (int): Sử dụng cho chiến lược MA dài hạn

    Returns:
        pd.DataFrame: DataFrame kết quả với lợi nhuận cộng dồn theo chiến lược và mua giữ
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input phải là DataFrame hợp lệ.")

    required_cols = {"Date", "Close", "Predicted_Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dữ liệu thiếu các cột cần thiết: {required_cols}")

    df = df.copy()
    df = df.sort_values("Date")
    df = df.tail(forecast_days).reset_index(drop=True)

    # Tính lợi nhuận hàng ngày
    df["Daily_Return"] = df["Close"].pct_change()

    # --- Chiến lược 1: Mua và giữ (Buy & Hold) ---
    df["Buy_and_Hold_Return"] = df["Daily_Return"]
    df["Cumulative_BuyHold"] = (1 + df["Buy_and_Hold_Return"]).cumprod()

    # --- Chiến lược 2: Theo tín hiệu dự báo ---
    if strategy == "predict_signal":
        df["Signal"] = (df["Predicted_Close"] > df["Close"].shift(1)).astype(int)
        df["Strategy_Return"] = df["Signal"] * df["Daily_Return"]

    # --- Chiến lược 3: MA crossover ---
    elif strategy == "moving_average":
        df["SMA_Short"] = df["Close"].rolling(window=short_window).mean()
        df["SMA_Long"] = df["Close"].rolling(window=long_window).mean()
        df["Signal"] = (df["SMA_Short"] > df["SMA_Long"]).astype(int)
        df["Strategy_Return"] = df["Signal"] * df["Daily_Return"]

    # --- Chiến lược mặc định: Mua và giữ (không giao dịch) ---
    else:  # 'buy_and_hold'
        df["Strategy_Return"] = 0.0

    # --- Tính lợi nhuận cộng dồn ---
    df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod()

    # --- Trả về kết quả ---
    return df[[
        "Date", "Close", "Predicted_Close",
        "Strategy_Return", "Cumulative_Strategy",
        "Buy_and_Hold_Return", "Cumulative_BuyHold"
    ]]
