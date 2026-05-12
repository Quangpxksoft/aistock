

import os
import joblib
import pandas as pd
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

# -------------------------------
# 📂 Thư mục lưu ARIMA models
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_ARIMA = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "arima"))
os.makedirs(MODEL_DIR_ARIMA, exist_ok=True)


# -------------------------------
# 📉 Train ARIMA
# -------------------------------
def train_arima(df, ticker, order=(5, 1, 0)):
    """
    Huấn luyện mô hình ARIMA và lưu lại model.
    """
    if not {"Date", "Close"}.issubset(df.columns):
        raise ValueError("⚠️ Dữ liệu không hợp lệ: thiếu 'Date' hoặc 'Close'.")

    # Chuẩn hoá series
    series = df.set_index("Date")["Close"].asfreq("D").interpolate()

    # Train ARIMA
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
    except Exception as e:
        raise RuntimeError(f"❌ Lỗi khi huấn luyện ARIMA: {e}")

    # Lưu model
    model_path = os.path.join(MODEL_DIR_ARIMA, f"arima_{ticker}.pkl")
    joblib.dump(model_fit, model_path)
    print(f"✅ Đã lưu ARIMA model tại: {model_path}")

    return model_fit


# -------------------------------
# 📂 Load ARIMA
# -------------------------------
def load_arima_model(ticker):
    """
    Tải mô hình ARIMA đã lưu. Trả về model hoặc None nếu chưa có.
    """
    model_path = os.path.join(MODEL_DIR_ARIMA, f"arima_{ticker}.pkl")
    if not os.path.exists(model_path):
        print(f"⚠️ Không tìm thấy ARIMA model: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        print(f"✅ Đã tải ARIMA model từ: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải ARIMA model: {e}")
        return None


# -------------------------------
# 🔮 Forecast ARIMA
# -------------------------------
# def forecast_arima(df, ticker, forecast_days=7, use_saved_model=True, order=(5, 1, 0)):
#     """
#     Dự báo giá bằng mô hình ARIMA đã huấn luyện.
#     """
#     if not {"Date", "Close"}.issubset(df.columns):
#         raise ValueError("⚠️ Dữ liệu không hợp lệ: thiếu 'Date' hoặc 'Close'.")

#     series = df.set_index("Date")["Close"].asfreq("D").interpolate()

#     if use_saved_model:
#         model_fit = load_arima_model(ticker)
#         if model_fit is None:
#             raise ValueError(f"⚠️ Model ARIMA chưa được huấn luyện cho {ticker}")
#     else:
#         model_fit = ARIMA(series, order=order).fit()

#     forecast = model_fit.forecast(steps=forecast_days)
#     last_date = pd.to_datetime(series.index[-1])
#     future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

#     return pd.DataFrame({
#         "Date": future_dates,
#         "Predicted_Close": forecast.values
#     })
# -------------------------------
# 🔮 Forecast ARIMA
# -------------------------------
def forecast_arima(df, forecast_days=7, use_saved_model=True, ticker=None, order=(5, 1, 0)):
    """
    Dự báo giá bằng mô hình ARIMA đã huấn luyện.
    """
    print(">>> forecast_arima loaded from utils/arima_model.py")

    if not {"Date", "Close"}.issubset(df.columns):
        raise ValueError("⚠️ Dữ liệu không hợp lệ: thiếu 'Date' hoặc 'Close'.")

    series = df.set_index("Date")["Close"].asfreq("D").interpolate()

    if use_saved_model:
        model_fit = load_arima_model(ticker)
        if model_fit is None:
            raise ValueError(f"⚠️ Model ARIMA chưa được huấn luyện cho {ticker}")
    else:
        model_fit = ARIMA(series, order=order).fit()

    forecast = model_fit.forecast(steps=forecast_days)
    last_date = pd.to_datetime(series.index[-1])
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": forecast.values
    })
