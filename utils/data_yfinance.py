
# import pandas as pd
# import yfinance as yf

# def load_data_yf(ticker: str, start_date: str = "2010-01-01", end_date: str = None):
#     if end_date is None:
#         end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

#     try:
#         # df = yf.download(ticker, start=start_date, end=end_date, progress=False)
#         df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)


#         # 🔹 Hỗ trợ MultiIndex columns từ yf.download
#         if isinstance(df.columns, pd.MultiIndex):
#             df.columns = [' '.join(col).strip() for col in df.columns.values]

#         # 🔹 Kiểm tra cột Close
#         close_col = next((c for c in df.columns if "Close" in c), None)
#         if df.empty or close_col is None:
#             print(f"❌ Không có dữ liệu Close cho {ticker}")
#             return pd.DataFrame(columns=[
#                 "Date","Open","High","Low","Close","Volume",
#                 "Predicted_Open","Predicted_High","Predicted_Low","Predicted_Close","Predicted_Volume","Ticker"
#             ])

#         # 🔹 Chuẩn hóa DataFrame
#         df.reset_index(inplace=True)

#         # Mapping cột OHLCV
#         col_map = {}
#         for col in ["Open","High","Low","Close","Volume"]:
#             real_col = next((c for c in df.columns if col in c), None)
#             if real_col:
#                 col_map[real_col] = col

#         df = df[list(col_map.keys()) + ["Date"]].copy()
#         df.rename(columns=col_map, inplace=True)

#         # 🔹 Thêm cột Predicted tương ứng
#         df["Predicted_Open"] = pd.NA
#         df["Predicted_High"] = pd.NA
#         df["Predicted_Low"] = pd.NA
#         df["Predicted_Close"] = pd.NA
#         df["Predicted_Volume"] = pd.NA
#         df["Ticker"] = ticker.upper()

#         # 🔹 Chuẩn hoá kiểu dữ liệu
#         for col in ["Open","High","Low","Close","Volume"]:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#         for col in ["Predicted_Open","Predicted_High","Predicted_Low","Predicted_Close","Predicted_Volume"]:
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#         df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

#         return df[[
#             "Date","Open","High","Low","Close","Volume",
#             "Predicted_Open","Predicted_High","Predicted_Low","Predicted_Close","Predicted_Volume","Ticker"
#         ]]

#     except Exception as e:
#         print(f"❌ Lỗi tải dữ liệu {ticker}: {e}")
#         return pd.DataFrame(columns=[
#             "Date","Open","High","Low","Close","Volume",
#             "Predicted_Open","Predicted_High","Predicted_Low","Predicted_Close","Predicted_Volume","Ticker"
#         ])


import pandas as pd
import yfinance as yf

def load_data_yf(ticker: str, start_date: str = "2010-01-01", end_date: str = None):
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    # ✅ Đồng bộ với load_data_vnd: dùng _EMPTY thay vì khai báo lặp lại
    _EMPTY = pd.DataFrame(columns=[
        "Date","Open","High","Low","Close","Volume",
        "Predicted_Open","Predicted_High","Predicted_Low",
        "Predicted_Close","Predicted_Volume","Ticker"
    ])

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        # 🔹 Hỗ trợ MultiIndex columns từ yf.download
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]

        # 🔹 Kiểm tra cột Close
        close_col = next((c for c in df.columns if "Close" in c), None)
        if df.empty or close_col is None:
            print(f"❌ Không có dữ liệu Close cho {ticker}")
            return _EMPTY.copy()

        # 🔹 Chuẩn hóa DataFrame
        df.reset_index(inplace=True)

        # Mapping cột OHLCV
        col_map = {}
        for col in ["Open","High","Low","Close","Volume"]:
            real_col = next((c for c in df.columns if col in c), None)
            if real_col:
                col_map[real_col] = col

        df = df[list(col_map.keys()) + ["Date"]].copy()
        df.rename(columns=col_map, inplace=True)

        # 🔹 Thêm cột Predicted_*
        df["Predicted_Open"]   = pd.NA
        df["Predicted_High"]   = pd.NA
        df["Predicted_Low"]    = pd.NA
        df["Predicted_Close"]  = pd.NA
        df["Predicted_Volume"] = pd.NA
        df["Ticker"] = ticker.upper()

        # 🔹 Chuẩn hoá kiểu dữ liệu
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["Predicted_Open","Predicted_High","Predicted_Low",
                    "Predicted_Close","Predicted_Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

        return df[[
            "Date","Open","High","Low","Close","Volume",
            "Predicted_Open","Predicted_High","Predicted_Low",
            "Predicted_Close","Predicted_Volume","Ticker"
        ]]

    except Exception as e:
        print(f"❌ Lỗi tải dữ liệu {ticker}: {e}")
        return _EMPTY.copy()