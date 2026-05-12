

import os
import joblib
import pandas as pd
from prophet import Prophet

# -------------------------------
# 📂 Đường dẫn lưu mô hình Prophet
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_PROPHET = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "prophet"))
os.makedirs(MODEL_DIR_PROPHET, exist_ok=True)


# -------------------------------
# 🎯 Huấn luyện Prophet
# -------------------------------
def train_prophet(df, ticker):
    """
    Huấn luyện mô hình Prophet trên dữ liệu giá đóng cửa.

    Parameters:
        df (DataFrame): Dữ liệu có cột 'Date' và 'Close'.
        ticker (str): Mã cổ phiếu để đặt tên mô hình lưu.

    Returns:
        Prophet: Mô hình Prophet đã huấn luyện.
    """
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Thiếu cột 'Date' hoặc 'Close'.")

    # Chuẩn hóa dữ liệu cho Prophet
    prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    # Lưu mô hình
    model_path = os.path.join(MODEL_DIR_PROPHET, f"prophet_{ticker}.pkl")
    joblib.dump(model, model_path)

    print(f"✅ Đã lưu Prophet model tại: {model_path}")
    return model


# -------------------------------
# 📥 Load Prophet
# -------------------------------
def load_prophet_model(ticker):
    """
    Tải mô hình Prophet đã lưu từ đĩa.

    Parameters:
        ticker (str): Mã cổ phiếu để xác định mô hình.

    Returns:
        Prophet | None: Mô hình Prophet hoặc None nếu không tồn tại.
    """
    model_path = os.path.join(MODEL_DIR_PROPHET, f"prophet_{ticker}.pkl")
    if not os.path.exists(model_path):
        print(f"⚠️ Không tìm thấy Prophet model: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        print(f"✅ Đã tải Prophet model từ: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải Prophet model: {e}")
        return None


# -------------------------------
# 🔮 Dự báo với Prophet
# -------------------------------
def forecast_prophet(model, forecast_days=7):
    """
    Dự báo n ngày tiếp theo bằng mô hình Prophet.

    Parameters:
        model (Prophet): Mô hình Prophet đã huấn luyện.
        forecast_days (int): Số ngày cần dự báo.

    Returns:
        DataFrame: Kết quả dự báo gồm các cột ds, yhat, yhat_lower, yhat_upper.
    """
    if model is None:
        raise ValueError("Model Prophet chưa được khởi tạo hoặc load.")

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)
