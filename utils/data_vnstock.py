# utils/data_vnstock.py
import pandas as pd
from datetime import datetime

def load_data_vnstock(ticker: str, start_date: str = "2010-01-01", end_date: str = None):
    """
    Tải dữ liệu OHLCV từ vnstock (nguồn VCI) thay thế VNDIRECT.

    Thay đổi so với load_data_vnd:
    - Dùng vnstock.api.quote.Quote thay vì requests đến finfo-api.vndirect.com.vn
      → VNDIRECT API không ổn định, hay bị chặn từ môi trường local/server
    - Nhân 1000 cho Open, High, Low, Close
      → vnstock trả về đơn vị nghìn đồng (VD: 24.65 = 24,650 VND)
      → Đồng bộ đơn vị với yfinance (trả về VND thực tế)
    - Giữ nguyên cấu trúc output: columns, Predicted_*, Ticker
      → Không phá các hàm downstream (insert_new_rows, load_data, v.v.)
    - Giữ nguyên tên hàm pattern: load_data_vnstock — nhất quán với load_data_vnd, load_data_yf

    Parameters
    ----------
    ticker     : mã chứng khoán không có suffix .VN (VD: "ACB", "FPT", "VIC")
    start_date : ngày bắt đầu format YYYY-MM-DD
    end_date   : ngày kết thúc format YYYY-MM-DD (mặc định = hôm nay)

    Returns
    -------
    DataFrame chuẩn: Date, Open, High, Low, Close, Volume,
                     Predicted_Open, Predicted_High, Predicted_Low,
                     Predicted_Close, Predicted_Volume, Ticker
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"📥 Tải dữ liệu vnstock (VCI): {ticker} từ {start_date} đến {end_date}...")

    _EMPTY = pd.DataFrame(columns=[
        "Date", "Open", "High", "Low", "Close", "Volume",
        "Predicted_Open", "Predicted_High", "Predicted_Low",
        "Predicted_Close", "Predicted_Volume", "Ticker"
    ])

    try:
        from vnstock.api.quote import Quote
    except ImportError as e:
        print(f"❌ Không import được vnstock: {e}. Chạy: pip install vnstock -U")
        return _EMPTY.copy()

    try:
        q  = Quote(symbol=ticker, source="VCI")
        df = q.history(start=start_date, end=end_date, interval="1D")
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu vnstock ({ticker}): {e}")
        return _EMPTY.copy()

    if df is None or df.empty:
        print(f"⚠️ Không có dữ liệu cho {ticker} từ vnstock")
        return _EMPTY.copy()

    # 🔹 Chuẩn hóa cột — vnstock trả về: time, open, high, low, close, volume
    df = df.rename(columns={
        "time":   "Date",
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    })

    # 🔹 Chỉ giữ cột chuẩn OHLCV
    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # 🔹 Đảm bảo cột Volume luôn tồn tại
    if "Volume" not in df.columns:
        print(f"⚠️ {ticker}: Không tìm thấy cột Volume từ vnstock, gán về 0")
        df["Volume"] = 0

    # 🔹 Chuẩn hóa kiểu dữ liệu trước khi nhân
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 🔹 Nhân 1000 cho OHLC — vnstock trả về đơn vị nghìn đồng
    # Volume giữ nguyên — đã đúng đơn vị cổ phiếu
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col] * 1000

    # 🔹 Bổ sung cột Predicted_*
    df["Predicted_Open"]   = pd.NA
    df["Predicted_High"]   = pd.NA
    df["Predicted_Low"]    = pd.NA
    df["Predicted_Close"]  = pd.NA
    df["Predicted_Volume"] = pd.NA
    df["Ticker"] = ticker.upper()

    # 🔹 Chuẩn hóa Predicted_* về numeric
    for col in ["Predicted_Open", "Predicted_High", "Predicted_Low",
                "Predicted_Close", "Predicted_Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 🔹 Loại bỏ dòng lỗi
    df = df.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)

    if df.empty:
        print(f"⚠️ Không có dữ liệu hợp lệ cho {ticker}")
        return _EMPTY.copy()

    # 🔹 Sort tăng dần theo Date — nhất quán với load_data_yf
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"✅ Tải thành công {len(df)} rows cho {ticker}")

    return df[[
        "Date", "Open", "High", "Low", "Close", "Volume",
        "Predicted_Open", "Predicted_High", "Predicted_Low",
        "Predicted_Close", "Predicted_Volume", "Ticker"
    ]]