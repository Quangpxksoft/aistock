import pandas as pd
import numpy as np

def optimize_portfolio(df: pd.DataFrame):
    """
    Tối ưu danh mục cơ bản theo phân bổ đều và tính các chỉ số rủi ro.

    Tham số:
        df (pd.DataFrame): DataFrame gồm các cột 'Date', 'Ticker', 'Close'.

    Trả về:
        pd.DataFrame: Bảng kết quả gồm các cột:
            - 'Ticker': Mã cổ phiếu
            - 'Weight': Tỷ trọng phân bổ (đều)
            - 'Expected_Return': Lợi suất kỳ vọng (năm)
            - 'Volatility': Độ biến động (năm)
    """
    if 'Ticker' not in df.columns:
        raise ValueError("DataFrame phải chứa cột 'Ticker'.")

    returns = {}

    for ticker in df['Ticker'].unique():
        sub_df = df[df["Ticker"] == ticker].copy()
        sub_df = sub_df.sort_values("Date")
        sub_df["Return"] = np.log(sub_df["Close"] / sub_df["Close"].shift(1))
        ret_series = sub_df["Return"].dropna()
        if not ret_series.empty and ret_series.std() > 0:
            returns[ticker] = ret_series

    returns_df = pd.DataFrame(returns).dropna()
    tickers = returns_df.columns.tolist()
    n_assets = len(tickers)

    if n_assets < 1:
        raise ValueError("❌ Không có tài sản hợp lệ sau khi xử lý returns.")

    equal_weights = np.repeat(1 / n_assets, n_assets)
    exp_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    if cov_matrix.isnull().values.any():
        raise ValueError("❌ Covariance matrix có giá trị null.")

    volatility = np.sqrt(np.diag(cov_matrix))
    if len(volatility) != n_assets:
        raise ValueError("❌ Thiếu dữ liệu Volatility.")

    results = {
        "Ticker": tickers,
        "Weight": equal_weights,
        "Expected_Return": exp_returns.values,
        "Volatility": volatility
    }

    return pd.DataFrame(results)
