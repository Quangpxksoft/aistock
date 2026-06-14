
import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# Giới hạn RAM TensorFlow — tránh OOM trên server RAM thấp
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tcn import TCN

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# -------------------------------
# 📂 Thư mục lưu TCN models
# -------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_TCN = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "tcn"))
os.makedirs(MODEL_DIR_TCN, exist_ok=True)

PRICE_COLS    = ["Open", "High", "Low", "Close"]
VOLUME_COL    = "Volume"
REQUIRED_COLS = PRICE_COLS + [VOLUME_COL]   # thứ tự cố định: O H L C V

# Ngưỡng winsorize log return — clip outlier vượt quá ±WINSOR_CLIP
# Thị trường VN giới hạn biên độ ±7% (HOSE), ±10% (HNX/UPCOM).
# Đặt WINSOR_CLIP=0.15 để giữ các gap hợp lệ nhưng loại bỏ outlier
# do lỗi dữ liệu hoặc sự kiện corporate action chưa được điều chỉnh.
WINSOR_CLIP = 0.15

# Số ngày tối thiểu bắt buộc để train — bằng SCALER_WINDOW cũ (252)
# Dưới ngưỡng này distribution log return chưa ổn định.
MIN_ROWS = 252

@tf.keras.utils.register_keras_serializable(package="custom")
def _combined_loss(y_true, y_pred):
    huber = tf.keras.losses.Huber(delta=0.5)(y_true, y_pred)
    pred_std = tf.math.reduce_std(y_pred, axis=-1)
    true_std = tf.math.reduce_std(y_true, axis=-1)
    variance_penalty = tf.reduce_mean(tf.nn.relu(true_std - pred_std))
    return huber + 0.15 * variance_penalty

# ================================================================
# 🔧 Helpers nội bộ — nhất quán với lstm_model.py
# ================================================================

def _validate_df(df: pd.DataFrame) -> None:
    """Kiểm tra DataFrame đầu vào có đủ cột và dữ liệu hợp lệ."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df phải là pandas DataFrame")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"⚠️ Thiếu cột: {missing}")
    if not np.isfinite(df[REQUIRED_COLS].to_numpy()).all():
        raise ValueError("⚠️ Dữ liệu chứa NaN hoặc Inf")
    if (df[VOLUME_COL] < 0).any():
        raise ValueError("⚠️ Volume chứa giá trị âm")
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"⚠️ Không đủ dữ liệu: cần tối thiểu {MIN_ROWS} rows, "
            f"hiện có {len(df)}. Distribution log return chưa ổn định."
        )


def _compute_log_returns(df: pd.DataFrame) -> np.ndarray:
    """
    Tính log return cho 5 chiều theo encoding tài chính đúng:

    Chiều 0 — close_return  = log(C[t] / C[t-1])
        Anchor chính. Stationary, không phụ thuộc vùng giá tuyệt đối.
        Inverse: C[t] = C[t-1] * exp(close_return[t])

    Chiều 1 — high_ratio    = log(H[t] / C[t])  ≥ 0
        H luôn ≥ C (định nghĩa tài chính) → ratio ≥ 1 → log ≥ 0.
        Model học target ≥ 0 tự nhiên — không cần fix hậu kỳ.
        Inverse: H[t] = C[t] * exp(high_ratio[t])

    Chiều 2 — low_ratio     = log(C[t] / L[t])  ≥ 0
        C luôn ≥ L (định nghĩa tài chính) → ratio ≥ 1 → log ≥ 0.
        Model học target ≥ 0 tự nhiên — không cần fix hậu kỳ.
        Inverse: L[t] = C[t] / exp(low_ratio[t])  = C[t] * exp(-low_ratio[t])

    Chiều 3 — open_return   = log(O[t] / C[t-1])
        Anchor vào Close ngày trước (cùng anchor với close_return).
        Có thể âm (gap down) hoặc dương (gap up).
        Inverse: O[t] = C[t-1] * exp(open_return[t])

    Chiều 4 — vol_delta     = log1p(V[t]) - log1p(V[t-1])
        Delta log volume — stationary, scale tương đồng giữa các mã.
        Inverse: V[t] = expm1(log1p(V[t-1]) + vol_delta[t])

    Lưu ý:
    - Row đầu tiên (t=0) không có t-1 → bỏ (drop first row).
    - Sau khi tính, winsorize tất cả 5 chiều tại ±WINSOR_CLIP
      để loại outlier do lỗi dữ liệu.
    - Trả về array shape (n-1, 5) — n là số rows df.
    """
    closes  = df["Close"].to_numpy(dtype=np.float64)
    highs   = df["High"].to_numpy(dtype=np.float64)
    lows    = df["Low"].to_numpy(dtype=np.float64)
    opens   = df["Open"].to_numpy(dtype=np.float64)
    volumes = df["Volume"].to_numpy(dtype=np.float64)

    # Tránh log(0) hoặc log âm — giá phải > 0
    for name, arr in [("Close", closes), ("High", highs),
                      ("Low", lows), ("Open", opens)]:
        if (arr <= 0).any():
            raise ValueError(f"⚠️ Cột {name} chứa giá trị ≤ 0, không thể tính log return.")

    # Tính 5 chiều — mỗi chiều shape (n-1,)
    close_return = np.log(closes[1:] / closes[:-1])          # log(C[t]/C[t-1])
    high_ratio   = np.log(highs[1:]  / closes[1:])           # log(H[t]/C[t]) ≥ 0
    low_ratio    = np.log(closes[1:] / lows[1:])             # log(C[t]/L[t]) ≥ 0
    open_return  = np.log(opens[1:]  / closes[:-1])          # log(O[t]/C[t-1])
    vol_log      = np.log1p(volumes)
    vol_delta    = vol_log[1:] - vol_log[:-1]                 # delta log volume

    returns = np.column_stack([
        close_return,   # col 0
        high_ratio,     # col 1
        low_ratio,      # col 2
        open_return,    # col 3
        vol_delta,      # col 4
    ])                  # shape: (n-1, 5)

    # Winsorize — clip outlier tại ±WINSOR_CLIP
    returns = np.clip(returns, -WINSOR_CLIP, WINSOR_CLIP)

    return returns.astype(np.float64)


def _make_scaler_bundle(df: pd.DataFrame) -> dict:
    """
    Fit RobustScaler trên log return của toàn bộ df.

    Lý do dùng toàn bộ df (không dùng SCALER_WINDOW nữa):
    - Log return stationary — distribution không drift theo thời gian.
    - Median log return luôn gần 0 cho mọi ticker, mọi thời kỳ.
    - Dùng toàn bộ lịch sử cho scaler ổn định hơn.

    Lý do vẫn dùng RobustScaler:
    - Log return vẫn có outlier dù đã winsorize (tail fat distribution).
    - RobustScaler dùng median+IQR → ít nhạy cảm với tail hơn StandardScaler.

    Bundle lưu thêm last_close và last_log_volume để anchor inverse trong predict:
    - last_close      : Close thực tế của row cuối cùng trong df train
    - last_log_volume : log1p(Volume) của row cuối cùng trong df train
    """
    returns = _compute_log_returns(df)   # (n-1, 5)

    scaler = RobustScaler()
    scaler.fit(returns)

    last_close      = float(df["Close"].iloc[-1])
    last_log_volume = float(np.log1p(df["Volume"].iloc[-1]))

    return {
        "scaler":               scaler,
        "last_close":           last_close,
        "last_log_volume":      last_log_volume,
        "price_cols":           PRICE_COLS,
        "volume_col":           VOLUME_COL,
        "required_cols":        REQUIRED_COLS,
        "encoding":             "log_return_v2",   # versioning để detect bundle cũ
    }


def _scale(df: pd.DataFrame, bundle: dict) -> tuple:
    """
    Scale df OHLCV → log return scaled.

    Trả về:
        scaled  : np.ndarray shape (n-1, 5) — log return đã scale
        anchors : dict chứa arrays Close và log1p(Volume) của df gốc
                  dùng để anchor inverse từng bước trong predict

    anchors["closes"]      : shape (n,)   — Close gốc từng row
    anchors["log_volumes"] : shape (n,)   — log1p(Volume) gốc từng row

    Lưu ý: scaled[i] tương ứng với return từ df.iloc[i] → df.iloc[i+1]
    Tức là row 0 của scaled = return của df.iloc[1] so với df.iloc[0].
    """
    returns = _compute_log_returns(df)   # (n-1, 5)
    scaled  = bundle["scaler"].transform(returns).astype(np.float32)

    closes      = df["Close"].to_numpy(dtype=np.float64)
    log_volumes = np.log1p(df["Volume"].to_numpy(dtype=np.float64))

    anchors = {
        "closes":      closes,       # shape (n,)
        "log_volumes": log_volumes,  # shape (n,)
    }
    return scaled, anchors


def _inverse_scale_sequence(
    preds_scaled: np.ndarray,
    bundle: dict,
    anchor_close: float,
    anchor_log_volume: float,
    clip_val: float = WINSOR_CLIP,
) -> np.ndarray:
    """
    Inverse scale chuỗi dự báo n_days bước — anchor-based, không tích lũy sai số.

    Parameters
    ----------
    preds_scaled       : shape (n_days, 5) — output model đã scale
    bundle             : scaler bundle
    anchor_close       : Close thực tế của ngày cuối cùng đã biết (ngày t=0)
    anchor_log_volume  : log1p(Volume) thực tế của ngày cuối cùng đã biết

    Cơ chế anchor-based — tại sao không tích lũy sai số:
    - Cách cũ (recursive): C[t+2] = C[t+1] * exp(r[t+2])
        → C[t+1] là giá trị dự báo, đã chứa sai số ε[t+1]
        → C[t+2] tích lũy thêm ε[t+2] lên ε[t+1]
        → Sai số tăng theo n

    - Cách này (anchor): C[t+k] = C[t] * exp(sum(r[t+1..t+k]))
        → C[t] là giá thực tế đã biết, không có sai số
        → Mỗi bước k chỉ chứa sai số dự báo của chính bước k
        → Không tích lũy

    OHLC constraint đảm bảo bằng toán học:
    - H[t] = C[t] * exp(|high_ratio[t]|)  ≥ C[t]  vì exp ≥ 1
    - L[t] = C[t] * exp(-|low_ratio[t]|) ≤ C[t]  vì exp ≤ 1
    - abs() để đảm bảo ratio ≥ 0 dù model predict âm
      (model học target ≥ 0 nhưng không có hard constraint trong activation)

    Returns
    -------
    np.ndarray shape (n_days, 5) — OHLCV thực tế đã inverse
    Thứ tự cột: Open, High, Low, Close, Volume
    """
    n_days     = preds_scaled.shape[0]
    n_features = preds_scaled.shape[1]   # 5

    # Inverse scale về log return space
    returns = bundle["scaler"].inverse_transform(preds_scaled)   # (n_days, 5)

    # Winsorize sau inverse — clip_val nới hơn WINSOR_CLIP khi predict
    returns = np.clip(returns, -clip_val, clip_val)

    result = np.zeros((n_days, n_features), dtype=np.float64)

    # Tích lũy log return từ anchor — anchor-based, không recursive
    # close_return tích lũy: C[t+k] = anchor_close * exp(sum(close_return[0..k]))
    cumsum_close = np.cumsum(returns[:, 0])   # shape (n_days,)

    for k in range(n_days):
        # Close — anchor-based cumsum
        C = anchor_close * np.exp(cumsum_close[k])

        # High — C[t+k] * exp(|high_ratio|) — abs() đảm bảo H ≥ C
        H = C * np.exp(np.abs(returns[k, 1]))

        # Low — C[t+k] * exp(-|low_ratio|) — abs() đảm bảo L ≤ C
        L = C * np.exp(-np.abs(returns[k, 2]))

        # Open — anchor vào C[t+k-1]
        # k=0: anchor vào anchor_close (ngày thực tế cuối)
        # k>0: anchor vào C[t+k-1] = anchor_close * exp(cumsum_close[k-1])
        if k == 0:
            prev_close = anchor_close
        else:
            prev_close = anchor_close * np.exp(cumsum_close[k - 1])
        O = prev_close * np.exp(returns[k, 3])

        # Volume — anchor-based cumsum từ anchor_log_volume
        log_vol = anchor_log_volume + np.sum(returns[:k + 1, 4])
        log_vol = np.clip(log_vol, 0.0, None)   # log1p(V) ≥ 0
        V = np.expm1(log_vol)
        V = max(V, 0.0)                          # Volume ≥ 0

        result[k] = [O, H, L, C, V]

    return result


def _build_sequences_direct(scaled: np.ndarray, look_back: int, n_days: int):
    """
    Tạo sequences cho Direct Multi-Output từ log return scaled.

    scaled : (n-1, 5) — log return đã scale, row i = return của df.iloc[i+1]

    X shape: (samples, look_back, 5)
    y shape: (samples, n_days * 5) — flatten C order

    Encoding y (5 chiều):
        col 0: close_return  — log(C[t]/C[t-1]), scaled
        col 1: high_ratio    — log(H[t]/C[t]) ≥ 0, scaled
        col 2: low_ratio     — log(C[t]/L[t]) ≥ 0, scaled
        col 3: open_return   — log(O[t]/C[t-1]), scaled
        col 4: vol_delta     — delta log volume, scaled

    Direct forecasting: model predict thẳng n_days bước trong 1 forward pass.
    Không recursive → không tích lũy sai số trong quá trình training.
    """
    X, y = [], []
    total = len(scaled)   # n-1 rows
    for i in range(look_back, total - n_days + 1):
        X.append(scaled[i - look_back:i, :])
        y.append(scaled[i:i + n_days, :].flatten())   # (n_days*5,) — C order
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ================================================================
# 🏗️ Xây dựng kiến trúc model
# ================================================================

def _build_model(look_back: int, n_features: int, n_days: int) -> Model:
    """
    Kiến trúc TCN: TCN block + Dropout + Dense head → Direct multi-output.

    TCN-specific params (giữ nguyên, không đồng bộ sang LSTM):
      nb_filters=64, kernel_size=3, dilations=[1,2,4,8,16], dropout_rate=0.2
      return_sequences=False: TCN tự pool về vector → không cần GlobalAvgPool
      (khác LSTM dùng return_sequences=True + GlobalAveragePooling1D)

    Dense head đồng bộ với lstm_model.py:
      Dense(64, relu) → Dropout(0.4) → Dense(32, relu) → output
    """
    input_layer = Input(shape=(look_back, n_features), name="input_seq")

    x = TCN(
        nb_filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16],
        dropout_rate=0.2,
        return_sequences=False,
        kernel_initializer="he_normal",
        name="tcn_block"
    )(input_layer)

    x = Dropout(0.2, name="drop_1")(x)

    # --- Dense head đồng bộ với lstm_model.py ---
    x = Dense(64, activation="relu", name="dense_1")(x)
    x = Dropout(0.4, name="drop_2")(x)
    x = Dense(32, activation="relu", name="dense_2")(x)

    # --- Output: dự báo thẳng n_days × n_features ---
    output_layer = Dense(n_days * n_features, activation="linear", name="output")(x)

    model = Model(inputs=input_layer, outputs=output_layer, name="tcn_direct")
    return model


# ================================================================
# 🏋️ Train
# ================================================================

def train_tcn_model(df: pd.DataFrame, ticker: str,
                    epochs: int = 50,
                    batch_size: int = 16,
                    look_back: int = 30,
                    n_days: int = 7) -> tuple:
    """
    Train TCN multi-output dự báo OHLCV.

    Đồng bộ với lstm_model.py:
    - look_back default: 60 → 30
        Lý do: log return look_back=30 đủ capture short/medium term pattern.
        Giảm look_back tăng số sequences: (n-1-30-7+1) > (n-1-60-7+1).
    - min_required = look_back + n_days + 2
        +2: 1 cho drop first row (log return), 1 cho split
    - _validate_df kiểm tra MIN_ROWS=252 trước khi làm bất cứ điều gì.
    - Bundle lưu last_close và last_log_volume của toàn bộ df
      (không phải chỉ train set) — để predict dùng anchor đúng.

    Parameters
    ----------
    df         : DataFrame có cột OHLCV + Date (hoặc index)
    ticker     : mã cổ phiếu
    epochs     : số epoch tối đa
    batch_size : batch size
    look_back  : số ngày nhìn lại (default giảm từ 60 → 30)
    n_days     : số ngày dự báo — phải khớp với predict_tcn

    Returns
    -------
    (model, scaler_bundle)
    """
    # tf.config.experimental.enable_op_determinism()  # ← chuyển vào đây
    tf.keras.backend.clear_session()

    _validate_df(df)
    df = df[REQUIRED_COLS].copy().reset_index(drop=True)

    # Log return có n-1 rows — cần đủ sequences sau khi tính return
    min_required = look_back + n_days + 2   # +2: 1 cho drop first row, 1 cho split
    if len(df) < min_required:
        raise ValueError(
            f"Không đủ dữ liệu train. "
            f"Cần tối thiểu {min_required} rows (look_back={look_back} + "
            f"n_days={n_days} + 2), hiện có {len(df)}"
        )

    # --- Fit scaler trên log return toàn bộ df ---
    bundle = _make_scaler_bundle(df)
    scaled, _ = _scale(df, bundle)   # (n-1, 5) — anchors không dùng trong train

    # --- Sequences ---
    X, y = _build_sequences_direct(scaled, look_back, n_days)
    # X: (samples, look_back, 5)
    # y: (samples, n_days * 5)

    # --- Walk-forward split: 80% train / 20% val (giữ temporal order) ---
    split = int(len(X) * 0.8)
    if split < 1 or (len(X) - split) < 1:
        raise ValueError(
            f"Không đủ samples sau khi split. "
            f"Tổng samples={len(X)}, split={split}. "
            f"Tăng dữ liệu hoặc giảm look_back/n_days."
        )
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # --- Build model ---
    n_features = len(REQUIRED_COLS)   # 5
    model = _build_model(look_back, n_features, n_days)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=_combined_loss)

    # --- Metadata gán trước fit ---
    model.lookback   = look_back
    model.n_features = n_features
    model.n_days     = n_days

    # --- Callbacks đồng bộ với lstm_model.py ---
    early_stop = EarlyStopping(
        monitor="val_loss", patience=15,
        restore_best_weights=True, verbose=0
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=8, min_lr=1e-6, verbose=0
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
        callbacks=[early_stop, reduce_lr]
    )

    # --- Lưu model + scaler bundle ---
    model_path  = os.path.join(MODEL_DIR_TCN, f"tcn_model_{ticker}.keras")
    scaler_path = os.path.join(MODEL_DIR_TCN, f"tcn_scaler_{ticker}.pkl")

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        bundle["n_days"] = n_days  # lưu để kiểm tra model_exists
        bundle["trained_date"] = __import__("datetime").date.today().isoformat()  # ngày train
        pickle.dump(bundle, f)

    print(f"✅ Đã lưu TCN model tại:        {model_path}")
    print(f"✅ Đã lưu scaler bundle tại:     {scaler_path}")
    print(f"   look_back={look_back}, n_days={n_days}, "
          f"train_samples={len(X_train)}, val_samples={len(X_val)}")

    return model, bundle


# ================================================================
# 📂 Load
# ================================================================

def load_tcn_model(ticker: str) -> tuple:
    """
    Tải TCN model + scaler bundle cho ticker.

    Đồng bộ với lstm_model.py:
    - Validate bundle key "encoding" == "log_return_v2"
      → Detect và reject bundle cũ (dual RobustScaler giá tuyệt đối)
      → Buộc train lại thay vì dùng bundle sai encoding
    - Validate "last_close" và "last_log_volume" có trong bundle
    - Bỏ validate "price_scaler"/"volume_scaler" (không còn dual scaler)
    - Bỏ validate "center_" trên hai scaler riêng lẻ
    - TCN vẫn cần custom_objects={"TCN": TCN} khi load
    """
    model_path  = os.path.join(MODEL_DIR_TCN, f"tcn_model_{ticker}.keras")
    scaler_path = os.path.join(MODEL_DIR_TCN, f"tcn_scaler_{ticker}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"ℹ️ Chưa có model/scaler cho {ticker}, cần train mới")
        return None, None

    try:
        # TCN cần custom_objects khi load
        model = load_model(
            model_path,
            custom_objects={"TCN": TCN},
            compile=False
        )

        # Kiểm tra input shape: (None, look_back, n_features)
        if model.input_shape is None or len(model.input_shape) != 3:
            raise ValueError(
                f"Model input_shape không hợp lệ: {model.input_shape}. "
                f"Cần shape (None, look_back, n_features)."
            )

        model.lookback   = model.input_shape[1]
        model.n_features = model.input_shape[2]

        # Suy ra n_days từ output shape: (None, n_days * n_features)
        output_units = model.output_shape[1]
        if output_units % model.n_features != 0:
            raise ValueError(
                f"output_units={output_units} không chia hết cho "
                f"n_features={model.n_features}. Model không hợp lệ."
            )
        model.n_days = output_units // model.n_features

        with open(scaler_path, "rb") as f:
            bundle = pickle.load(f)

        # Validate encoding version — reject bundle cũ
        if bundle.get("encoding") != "log_return_v2":
            raise ValueError(
                f"⚠️ Scaler bundle dùng encoding cũ: '{bundle.get('encoding')}'. "
                f"Vui lòng train lại model để dùng log_return_v2."
            )

        # Validate keys bắt buộc
        required_keys = {
            "scaler", "last_close", "last_log_volume",
            "price_cols", "volume_col", "required_cols", "encoding"
        }
        missing_keys = required_keys - set(bundle.keys())
        if missing_keys:
            raise KeyError(
                f"⚠️ Scaler bundle thiếu keys: {missing_keys}. "
                f"Vui lòng train lại model."
            )

        # Validate scaler là RobustScaler đã fit
        if not hasattr(bundle["scaler"], "center_"):
            raise AttributeError(
                f"⚠️ bundle['scaler'] không phải RobustScaler đã fit. "
                f"Vui lòng train lại model."
            )

        # Validate n_features khớp với scaler (fit trên 5 chiều log return)
        if bundle["scaler"].n_features_in_ != model.n_features:
            raise ValueError(
                f"Feature mismatch: scaler fit trên {bundle['scaler'].n_features_in_} chiều, "
                f"model nhận {model.n_features} features."
            )

        # Validate anchor values hợp lệ
        if not np.isfinite(bundle["last_close"]) or bundle["last_close"] <= 0:
            raise ValueError(
                f"⚠️ bundle['last_close']={bundle['last_close']} không hợp lệ. "
                f"Vui lòng train lại model."
            )
        if not np.isfinite(bundle["last_log_volume"]) or bundle["last_log_volume"] < 0:
            raise ValueError(
                f"⚠️ bundle['last_log_volume']={bundle['last_log_volume']} không hợp lệ. "
                f"Vui lòng train lại model."
            )

        print(f"✅ Đã tải TCN model + scaler bundle cho {ticker}")
        print(f"   look_back={model.lookback}, n_features={model.n_features}, "
              f"n_days={model.n_days}")
        return model, bundle

    except Exception as e:
        print(f"❌ Lỗi khi tải model hoặc scaler: {e}")
        return None, None


# ================================================================
# 🔮 Predict
# ================================================================

def predict_tcn(df: pd.DataFrame, model: Model,
                bundle: dict, ticker: str,
                n_days: int = 7) -> pd.DataFrame:
    """
    Dự báo OHLCV n_days tiếp theo — Direct Multi-Output + anchor-based inverse.

    Đồng bộ với lstm_model.py:
    - Input model là log return scaled thay vì OHLCV scaled tuyệt đối.
    - Inverse dùng _inverse_scale_sequence với anchor_close và anchor_log_volume
      từ df thực tế (ngày cuối đã biết) — không dùng bundle["last_close"]
      vì df trong predict có thể chứa dữ liệu mới hơn lúc train.
    - Không còn _inverse_scale cũ — thay hoàn toàn bằng _inverse_scale_sequence.
    - OHLC constraint đảm bảo bằng toán học trong _inverse_scale_sequence,
      không cần post-processing.

    Parameters
    ----------
    df     : DataFrame gốc, cần có cột REQUIRED_COLS và 'Date'
    model  : model đã load bằng load_tcn_model
    bundle : scaler bundle đã load bằng load_tcn_model
    ticker : mã cổ phiếu
    n_days : số ngày dự báo

    Returns
    -------
    DataFrame với cột: Date, Predicted_Open, Predicted_High,
                       Predicted_Low, Predicted_Close,
                       Predicted_Volume, Ticker
    """
    if n_days != model.n_days:
        raise ValueError(
            f"n_days={n_days} không khớp với model.n_days={model.n_days}. "
            f"Model này chỉ dự báo được đúng {model.n_days} ngày. "
            f"Vui lòng train lại với n_days={n_days}."
        )

    _validate_df(df)

    if "Date" not in df.columns:
        raise ValueError("DataFrame cần có cột 'Date' để tạo output dates.")

    df = df.sort_values("Date").reset_index(drop=True)

    look_back  = model.lookback
    n_features = model.n_features

    # Scale → log return (n-1 rows)
    scaled, anchors = _scale(df, bundle)   # scaled: (n-1, 5)

    # Cần ít nhất look_back rows trong scaled
    if scaled.shape[0] < look_back:
        raise ValueError(
            f"Không đủ dữ liệu dự báo. "
            f"Sau khi tính log return còn {scaled.shape[0]} rows, "
            f"cần ít nhất {look_back}."
        )

    # Lấy look_back rows cuối của log return scaled → input model
    last_seq = scaled[-look_back:].copy()                          # (look_back, 5)
    X        = last_seq.reshape(1, look_back, n_features).astype(np.float32)

    # --- MC Dropout: N lần forward pass với training=True → giữ Dropout active ---
    N_MC = 20
    mc_preds = np.array([
        model(X, training=True).numpy()[0]
        for _ in range(N_MC)
    ])
    raw_pred = mc_preds.mean(axis=0)                               # (n_days*5,)

    # Kiểm tra NaN/Inf — raise ngay, không che lỗi
    if not np.isfinite(raw_pred).all():
        bad_indices = np.where(~np.isfinite(raw_pred))[0].tolist()
        raise ValueError(
            f"Model output chứa NaN hoặc Inf tại indices {bad_indices}. "
            f"Kiểm tra lại model hoặc input data cho {ticker}."
        )

    # Reshape về (n_days, 5) — C order, khớp với _build_sequences_direct
    preds_scaled = raw_pred.reshape(n_days, n_features)            # (n_days, 5)

    # Anchor: Close và log1p(Volume) của ngày thực tế cuối cùng trong df
    # Dùng df trực tiếp thay vì bundle["last_close"] — df trong predict
    # có thể chứa dữ liệu mới hơn thời điểm train
    anchor_close      = float(df["Close"].iloc[-1])
    anchor_log_volume = float(np.log1p(df["Volume"].iloc[-1]))

    # --- Anchor-based inverse — không tích lũy sai số ---
    preds = _inverse_scale_sequence(
        preds_scaled, bundle,
        anchor_close, anchor_log_volume,
        clip_val=0.10
    )   # (n_days, 5): O, H, L, C, V

    # --- Business days output dates ---
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    dates     = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    # --- Xây output DataFrame ---
    out = pd.DataFrame(preds, columns=REQUIRED_COLS)
    out.insert(0, "Date", dates)
    out = out.rename(columns={c: f"Predicted_{c}" for c in REQUIRED_COLS})
    out.insert(len(out.columns), "Ticker", ticker)

    return out
