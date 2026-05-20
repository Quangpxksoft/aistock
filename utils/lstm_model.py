# -------------------------------
# 📂 Train LSTM multi-output → dự báo đồng thời [Open, High, Low, Close, Volume] cho ngày tiếp theo.
# Lưu 1 file .h5 (model) + 1 file .pkl (scaler).
# -------------------------------

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense




# Models
Sequential = tf.keras.models.Sequential
load_model = tf.keras.models.load_model

# Layers
Dense = tf.keras.layers.Dense
LSTM = tf.keras.layers.LSTM
Input = tf.keras.layers.Input

# Callbacks
EarlyStopping = tf.keras.callbacks.EarlyStopping


# -------------------------------
# 📂 Thư mục lưu LSTM models
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_LSTM = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "lstm"))
os.makedirs(MODEL_DIR_LSTM, exist_ok=True)


# # -------------------------------
# # 🏋️ Train LSTM multi-output
# # -------------------------------
def train_lstm_model(df, ticker, epochs=10, batch_size=16, look_back=60):
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in required_cols):
        raise ValueError(f"⚠️ Dữ liệu không hợp lệ: thiếu {required_cols}")

    # Chuẩn hoá dữ liệu (5 cột) và lưu tên cột cho scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[required_cols])
    # scaler.feature_names_in_ = required_cols  # Thêm dòng này
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[required_cols])

    # Tạo dữ liệu huấn luyện
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, :])
        y.append(scaled_data[i, :])
    X, y = np.array(X), np.array(y)

    # Xây dựng mô hình multi-output
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50),
        Dense(5)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Huấn luyện
    early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

    # Lưu mô hình
    model_path = os.path.join(MODEL_DIR_LSTM, f"lstm_model_{ticker}.h5")
    scaler_path = os.path.join(MODEL_DIR_LSTM, f"lstm_scaler_{ticker}.pkl")

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Đã lưu LSTM multi-output model tại: {model_path}")
    print(f"✅ Đã lưu scaler tại: {scaler_path}")

    return model, scaler


# -------------------------------
# 📂 Load LSTM
# -------------------------------

def load_lstm_model(ticker):
    """
    Tải mô hình LSTM + scaler cho ticker.
    Đảm bảo scaler có feature_names_in_ để dự báo multi-output đúng cột.
    """
    model_path = os.path.join(MODEL_DIR_LSTM, f"lstm_model_{ticker}.h5")
    scaler_path = os.path.join(MODEL_DIR_LSTM, f"lstm_scaler_{ticker}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"⚠️ Không tìm thấy model hoặc scaler cho {ticker}")
        return None, None

    try:
        model = load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # Kiểm tra scaler có feature_names_in_ không
        if not hasattr(scaler, "feature_names_in_"):
            raise AttributeError("⚠️ Scaler không có feature_names_in_. Vui lòng train lại LSTM.")

        print(f"✅ Đã tải LSTM model + scaler cho {ticker}")
        return model, scaler
    except Exception as e:
        print(f"❌ Lỗi khi tải model hoặc scaler: {e}")
        return None, None


# # -------------------------------
# # 🔮 Forecast LSTM multi-output
# # -------------------------------

def predict_lstm(df, model, scaler, ticker, n_days=7, look_back=60):
    """
    Dự báo n_days tiếp theo cho tất cả các cột đã train LSTM (OHLCV).
    Tuân thủ nguyên tắc: đầu vào bao nhiêu cột, đầu ra bấy nhiêu cột.
    """
    # Lấy danh sách cột scaler đã fit
    if hasattr(scaler, "feature_names_in_"):
        required_cols = list(scaler.feature_names_in_)
    else:
        raise ValueError("⚠️ Không xác định được các cột scaler đã fit. Vui lòng train lại LSTM.")

    # Kiểm tra df có đủ các cột cần thiết không
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"⚠️ Dữ liệu dự báo thiếu các cột quan trọng: {missing_cols}. Không thể dự báo.")

    # Chuẩn bị dữ liệu sạch
    df_clean = df[required_cols].dropna()
    if df_clean.shape[0] < look_back:
        raise ValueError(f"⚠️ Không đủ dữ liệu để dự báo. Cần ít nhất {look_back} dòng dữ liệu.")

    if hasattr(scaler, "feature_names_in_"):
        if list(df_clean.columns) != list(scaler.feature_names_in_):
            raise ValueError("Feature mismatch giữa train và predict")

    # data = df_clean.values
    # data_scaled = scaler.transform(data)
    data = df_clean[required_cols]
    data_scaled = scaler.transform(data)

    predictions = []
    last_sequence = data_scaled[-look_back:].copy()

    for _ in range(n_days):
        X_input = last_sequence.reshape(1, look_back, len(required_cols))
        predicted_scaled = model.predict(X_input, verbose=0)[0]  # đầu ra cho tất cả cột
        predictions.append(predicted_scaled)
        last_sequence = np.vstack([last_sequence[1:], predicted_scaled])

    predicted_values = scaler.inverse_transform(np.array(predictions))
    forecast_dates = pd.date_range(start=pd.to_datetime(df["Date"].iloc[-1]) + pd.Timedelta(days=1),
                                   periods=n_days)

    # Sinh ra Predicted_* cho từng cột
    forecast_df = pd.DataFrame(predicted_values, columns=[f"Predicted_{col}" for col in required_cols])
    forecast_df.insert(0, "Date", forecast_dates)

    return forecast_df

