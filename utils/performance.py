import pandas as pd
import numpy as np

def calculate_performance_metrics(df, risk_free_rate=0.0, mar=0.1):
    """
    Tính toán các chỉ số đánh giá hiệu suất đầu tư.
    
    Args:
        df (pd.DataFrame): Phải có 'Strategy_Return' và 'Date'
        risk_free_rate (float): Lãi suất phi rủi ro (% năm), mặc định 0
        mar (float): Mức lợi nhuận kỳ vọng tối thiểu cho Calmar & Sterling (default 10%)

    Returns:
        pd.DataFrame: Một bảng các chỉ số hiệu suất
    """
    metrics = {}

    # Đảm bảo kiểu ngày
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    returns = df["Strategy_Return"]
    cumulative = (1 + returns).cumprod()

    # Thời gian
    days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
    years = days / 365.0

    # CAGR — guard years=0 (data chỉ có 1 ngày hoặc cùng ngày) tránh ZeroDivisionError
    if years > 0:
        cagr = cumulative.iloc[-1] ** (1 / years) - 1
    else:
        cagr = np.nan
    metrics["CAGR"] = cagr

    # Volatility
    volatility = returns.std() * np.sqrt(252)
    metrics["Volatility"] = volatility

    # Sharpe Ratio — thêm 1e-9 để tránh chia 0 khi returns.std()==0, nhất quán với Sortino
    sharpe = (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252) + 1e-9)
    metrics["Sharpe Ratio"] = sharpe

    # Sortino Ratio
    downside_std = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252 - risk_free_rate) / (downside_std + 1e-9)
    metrics["Sortino Ratio"] = sortino

    # Max Drawdown
    cummax = cumulative.cummax()
    drawdown = (cumulative - cummax) / cummax
    max_dd = drawdown.min()
    metrics["Max Drawdown"] = max_dd

    # Win Rate
    win_rate = (returns > 0).mean()
    metrics["Win Rate"] = win_rate

    # Avg Gain / Loss
    metrics["Average Gain"] = returns[returns > 0].mean()
    metrics["Average Loss"] = returns[returns < 0].mean()

    # === 📌 Các chỉ số mở rộng ===

    # Calmar Ratio = CAGR / abs(Max Drawdown)
    if max_dd != 0:
        metrics["Calmar Ratio"] = cagr / abs(max_dd)
    else:
        metrics["Calmar Ratio"] = np.nan

    # Omega Ratio = (lợi nhuận > MAR) / (lợi nhuận < MAR)
    threshold = mar / 252
    excess = returns - threshold
    omega = excess[excess > 0].sum() / abs(excess[excess < 0].sum() + 1e-9)
    metrics["Omega Ratio"] = omega

    # Sterling Ratio = CAGR / (Avg Max Drawdown > 10% + 0.1)
    dd_10 = drawdown[drawdown < -0.1].abs()
    if not dd_10.empty:
        sterling = cagr / (dd_10.mean() + 0.1)
    else:
        sterling = np.nan
    metrics["Sterling Ratio"] = sterling

    return pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
