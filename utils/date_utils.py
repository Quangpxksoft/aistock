from datetime import datetime, date
import streamlit as st


def get_date_range_from_session(default_start: date = date(2015, 1, 1)) -> tuple[datetime, datetime]:
    """
    Lấy from_date và to_date từ session, chuyển từ date → datetime.

    Args:
        default_start (date): Ngày mặc định nếu không có trong session.

    Returns:
        tuple(datetime, datetime): (from_date, to_date) dạng datetime.
    """
    from_date_raw = st.session_state.get("from_date", default_start)
    to_date_raw   = st.session_state.get("to_date", date.today())

    from_date = datetime.combine(from_date_raw, datetime.min.time())
    to_date   = datetime.combine(to_date_raw, datetime.min.time())

    return from_date, to_date


def get_date_range_str(default_start: date = date(2015, 1, 1)) -> tuple[str, str]:
    """
    Lấy ngày từ session và trả về định dạng chuỗi YYYY-MM-DD.

    Returns:
        tuple(str, str): (from_date_str, to_date_str)
    """
    from_dt, to_dt = get_date_range_from_session(default_start)
    return from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d")


def get_today_str() -> str:
    """Trả về ngày hôm nay dạng chuỗi 'YYYY-MM-DD'."""
    return date.today().strftime("%Y-%m-%d")


def to_datetime_safe(d: date | str | datetime) -> datetime:
    """
    Chuyển một biến `date`, `datetime` hoặc `str` (dạng 'YYYY-MM-DD') sang datetime.

    Args:
        d (date | str | datetime): Dữ liệu đầu vào.

    Returns:
        datetime: Kết quả đã convert.
    """
    if isinstance(d, datetime):
        return d
    elif isinstance(d, date):
        return datetime.combine(d, datetime.min.time())
    elif isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d")
    else:
        raise ValueError(f"Không hỗ trợ kiểu dữ liệu: {type(d)}")


def to_yyyymmdd_str(d: date | datetime | str) -> str:
    """
    Chuyển `date`, `datetime` hoặc `str` sang chuỗi 'YYYY-MM-DD'.

    Args:
        d (date | datetime | str): Dữ liệu đầu vào.

    Returns:
        str: Chuỗi ngày dạng 'YYYY-MM-DD'
    """
    return to_datetime_safe(d).strftime("%Y-%m-%d")


__all__ = [
    "get_date_range_from_session",
    "get_date_range_str",
    "get_today_str",
    "to_datetime_safe",
    "to_yyyymmdd_str",
]
