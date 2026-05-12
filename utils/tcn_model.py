import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tcn import TCN

# -------------------------------
# 📂 Thư mục lưu TCN models
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_TCN = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "tcn"))
os.makedirs(MODEL_DIR_TCN, exist_ok=True)


# -------------------------------
# 🏋️ Train TCN multi-output (single step)
# -------------------------------
def train_tcn_model(df, ticker, look_back=60, epochs=10, batch_size=16):
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in required_cols):
        raise ValueError(f"⚠️ Dữ liệu không hợp lệ: thiếu {required_cols}")

    # Chuẩn hoá dữ liệu
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[required_cols])
    scaler.feature_names_in_ = required_cols  # lưu lại để dùng khi dự báo

    # Chuẩn bị X, y
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, :])
        y.append(scaled_data[i, :])
    X, y = np.array(X), np.array(y)

    # Kiến trúc TCN
    input_layer = Input(shape=(look_back, len(required_cols)))
    x = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4], return_sequences=False)(input_layer)
    output_layer = Dense(len(required_cols))(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Lưu model và scaler
    model_path = os.path.join(MODEL_DIR_TCN, f"tcn_model_{ticker}.h5")
    scaler_path = os.path.join(MODEL_DIR_TCN, f"tcn_scaler_{ticker}.pkl")

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Đã lưu TCN model tại: {model_path}")
    print(f"✅ Đã lưu scaler tại: {scaler_path}")

    return model, scaler


# -------------------------------
# 📂 Load TCN model
# -------------------------------
def load_tcn_model(ticker):
    model_path = os.path.join(MODEL_DIR_TCN, f"tcn_model_{ticker}.h5")
    scaler_path = os.path.join(MODEL_DIR_TCN, f"tcn_scaler_{ticker}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"⚠️ Không tìm thấy model hoặc scaler cho {ticker}")
        return None, None

    try:
        model = load_model(model_path, custom_objects={"TCN": TCN})
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        if not hasattr(scaler, "feature_names_in_"):
            raise AttributeError("⚠️ Scaler không có feature_names_in_. Vui lòng train lại TCN.")

        print(f"✅ Đã tải TCN model + scaler cho {ticker}")
        return model, scaler
    except Exception as e:
        print(f"❌ Lỗi khi tải model hoặc scaler: {e}")
        return None, None


# -------------------------------
# 🔮 Forecast TCN multi-output (autoregressive n_days)
# -------------------------------
def predict_tcn(df, model, scaler, ticker, n_days=7, look_back=60):
    """
    Dự báo n_days tiếp theo cho OHLCV bằng TCN.
    Multi-output (OHLCV), autoregressive loop giống LSTM.
    """
    if hasattr(scaler, "feature_names_in_"):
        required_cols = list(scaler.feature_names_in_)
    else:
        raise ValueError("⚠️ Không xác định được các cột scaler đã fit. Vui lòng train lại TCN.")

    df_clean = df[required_cols].dropna()
    if df_clean.shape[0] < look_back:
        raise ValueError(f"⚠️ Không đủ dữ liệu để dự báo. Cần ít nhất {look_back} dòng dữ liệu.")

    data = df_clean.values
    data_scaled = scaler.transform(data)

    predictions = []
    last_sequence = data_scaled[-look_back:].copy()

    # autoregressive loop
    for _ in range(n_days):
        X_input = last_sequence.reshape(1, look_back, len(required_cols))
        predicted_scaled = model.predict(X_input, verbose=0)[0]  # multi-output (5 cột)
        predictions.append(predicted_scaled)
        last_sequence = np.vstack([last_sequence[1:], predicted_scaled])

    # inverse transform
    predicted_values = scaler.inverse_transform(np.array(predictions))
    forecast_dates = pd.date_range(
        start=pd.to_datetime(df["Date"].iloc[-1]) + pd.Timedelta(days=1),
        periods=n_days
    )

    forecast_df = pd.DataFrame(predicted_values, columns=[f"Predicted_{col}" for col in required_cols])
    forecast_df.insert(0, "Date", forecast_dates)

    return forecast_df
