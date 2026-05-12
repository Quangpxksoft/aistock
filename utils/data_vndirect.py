import pandas as pd
import requests
from datetime import datetime
from utils.db_manager import insert_new_rows

def load_data_vnd(ticker: str, start_date: str = "2010-01-01", end_date: str = None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"📥 Tải dữ liệu VNDIRECT: {ticker} từ {start_date} đến {end_date}...")

    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    url = (
        f"https://finfo-api.vndirect.com.vn/v4/stock_prices"
        f"?sort=date&size=10000&page=1"
        f"&q=code:{ticker}+date:gte:{start}+date:lte:{end}"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"⏱️ Timeout khi kết nối tới VNDIRECT cho mã {ticker}")
        return pd.DataFrame(columns=["Date", "Open","High","Low","Close","Volume",
                                     "Predicted_Open","Predicted_High","Predicted_Low",
                                     "Predicted_Close","Predicted_Volume","Ticker"])
    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi khi gọi API VNDIRECT ({ticker}): {e}")
        return pd.DataFrame(columns=["Date", "Open","High","Low","Close","Volume",
                                     "Predicted_Open","Predicted_High","Predicted_Low",
                                     "Predicted_Close","Predicted_Volume","Ticker"])

    data = resp.json().get("data", [])
    if not data:
        print(f"⚠️ Không có dữ liệu cho {ticker} từ VNDIRECT")
        return pd.DataFrame(columns=["Date", "Open","High","Low","Close","Volume",
                                     "Predicted_Open","Predicted_High","Predicted_Low",
                                     "Predicted_Close","Predicted_Volume","Ticker"])

    df = pd.DataFrame(data)

    # 🔹 Chuẩn hóa cột
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "average": "AvgPrice",
        "nmVolume": "Volume"
    })

    # 🔹 Chỉ giữ cột chuẩn OHLCV
    keep_cols = ["Date","Open","High","Low","Close","Volume"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # 🔹 Bổ sung cột Predicted_*
    df["Predicted_Open"] = pd.NA
    df["Predicted_High"] = pd.NA
    df["Predicted_Low"] = pd.NA
    df["Predicted_Close"] = pd.NA
    df["Predicted_Volume"] = pd.NA
    df["Ticker"] = ticker.upper()

    # 🔹 Chuẩn hóa kiểu dữ liệu
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["Open","High","Low","Close","Volume",
                "Predicted_Open","Predicted_High","Predicted_Low",
                "Predicted_Close","Predicted_Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 🔹 Loại bỏ dòng lỗi
    df = df.dropna(subset=["Date","Close"]).reset_index(drop=True)

    if df.empty:
        print(f"⚠️ Không có dữ liệu hợp lệ cho {ticker}")
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume",
                                     "Predicted_Open","Predicted_High","Predicted_Low",
                                     "Predicted_Close","Predicted_Volume","Ticker"])

    # ✅ Lưu vào SQLite
    insert_new_rows(ticker, df)

    return df[["Date","Open","High","Low","Close","Volume",
               "Predicted_Open","Predicted_High","Predicted_Low",
               "Predicted_Close","Predicted_Volume","Ticker"]]

