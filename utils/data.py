
# utils/data.py
import pandas as pd
from datetime import date
from utils.data_loader import load_data_yf
from utils.data_vnstock import load_data_vnstock
from utils.db_manager import load_data as load_from_db, insert_new_rows, list_tables

def load_or_download_data(ticker, start_date="2010-01-01", end_date=None, source="yf"):
    """
    Tải dữ liệu từ SQLite nếu có, ngược lại tải từ Yahoo Finance hoặc VSTOCK rồi lưu vào DB.
    Giữ nguyên API cũ, thêm param `source` để chọn nguồn dữ liệu.
    - source="yf" -> Yahoo Finance
    - source="vnd" -> VSTOCK
    """
    ticker = ticker.upper()

    # 1. Thử lấy dữ liệu từ SQLite
    try:
        df = load_from_db(ticker)
        if not df.empty:
            return df
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc dữ liệu {ticker} từ SQLite: {e}")

    # 2. Nếu chưa có thì tải mới
    if source in ("vnd", "vnstock"):
        df = load_data_vnstock(ticker, start_date=start_date, end_date=end_date)
    else:
        df = load_data_yf(ticker, start_date=start_date, end_date=end_date)

    # 3. Nếu tải thành công -> lưu vào DB
    if not df.empty:
        insert_new_rows(ticker, df)

    return df


def merge_all_csv(data_dir=None):
    """
    Trước đây merge CSV, giờ merge tất cả bảng trong SQLite.
    Giữ tên hàm để không phá code cũ.
    """
    try:
        tables = list_tables()
        if not tables:
            raise ValueError("⚠️ Không có bảng nào trong SQLite để gộp.")

        dataframes = []
        for ticker in tables:
            df = load_from_db(ticker)
            if not df.empty:
                dataframes.append(df)

        if not dataframes:
            raise ValueError("⚠️ Không có dữ liệu hợp lệ trong SQLite.")

        df_all = pd.concat(dataframes, ignore_index=True)
        df_all = df_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        return df_all

    except Exception as e:
        print(f"❌ Lỗi khi merge dữ liệu từ SQLite: {e}")
        return pd.DataFrame()
