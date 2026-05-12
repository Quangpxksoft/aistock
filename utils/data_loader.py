# utils/data_loader.py
from datetime import date
from utils.data_yfinance import load_data_yf
from utils.data_vndirect import load_data_vnd
from utils.date_utils import to_yyyymmdd_str


def load_data(ticker, start_date="2010-01-01", end_date=None, source="yf"):

    """
    Giao diện duy nhất để tải dữ liệu theo nguồn (Yahoo Finance hoặc VNDIRECT).

    Args:
        ticker (str): Mã chứng khoán
        start_date (str|date|datetime): Ngày bắt đầu
        end_date (str|date|datetime): Ngày kết thúc (mặc định = hôm nay)
        source (str): "yf" = Yahoo Finance, "vnd" = VNDIRECT

    Returns:
        pd.DataFrame: DataFrame chuẩn gồm Date, Close, Predicted_Close, Ticker
    """

    # ✅ Chuẩn hoá ngày tháng
    start_date = to_yyyymmdd_str(start_date)
    end_date = to_yyyymmdd_str(end_date) if end_date else to_yyyymmdd_str(date.today())

    # ✅ Điều phối theo nguồn
    if source == "vnd":
        return load_data_vnd(ticker, start_date, end_date)
    elif source == "yf":
        return load_data_yf(ticker, start_date, end_date)
    else:
        raise ValueError(f"❌ Nguồn dữ liệu không hợp lệ: {source}")

