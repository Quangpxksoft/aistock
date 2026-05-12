#================================================
from reportlab.platypus import (
    BaseDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, Frame, PageTemplate, NextPageTemplate, SimpleDocTemplate, KeepTogether
)

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4, landscape
import matplotlib.dates as mdates


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager
from io import BytesIO
import os
import re

from datetime import datetime, date
import plotly.express as px
import io
import numpy as np

from reportlab.platypus import Image as RLImage





# Đăng ký font hỗ trợ tiếng Việt
pdfmetrics.registerFont(TTFont("DejaVu", "fonts/DejaVuSans.ttf"))
addMapping("DejaVu", 0, 0, "DejaVu")

# Styles sử dụng Unicode
styles = getSampleStyleSheet()


# Thêm các style Unicode nếu chưa tồn tại
if 'Normal_Unicode' not in styles:
    styles.add(ParagraphStyle(name='Normal_Unicode', fontName='DejaVu', fontSize=10, leading=14))

if 'Title_Unicode' not in styles:
    styles.add(ParagraphStyle(name='Title_Unicode', parent=styles['Title'], fontName='DejaVu'))

if 'Heading1_Unicode' not in styles:
    styles.add(ParagraphStyle(name='Heading1_Unicode', parent=styles['Heading1'], fontName='DejaVu'))

if 'Heading2_Unicode' not in styles:
    styles.add(ParagraphStyle(name='Heading2_Unicode', parent=styles['Heading2'], fontName='DejaVu'))

# 🔹 Canh giữa cho Heading1
if 'Heading1_Center' not in styles:
    styles.add(ParagraphStyle(
        name="Heading1_Center",
        parent=styles["Heading1_Unicode"],
        alignment=TA_CENTER
    ))

# 🔹 Canh giữa cho Heading2
if 'Heading2_Center' not in styles:
    styles.add(ParagraphStyle(
        name="Heading2_Center",
        parent=styles["Heading2_Unicode"],
        alignment=TA_CENTER
    ))

@contextmanager
def safe_open_pdf(path):
    try:
        f = open(path, "rb")
        yield f
        f.close()
    except FileNotFoundError:
        raise FileNotFoundError("Không tìm thấy file PDF để tải về.")

def make_table_from_df(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return [Paragraph("Không có dữ liệu", styles["Normal_Unicode"])]

    # Xác định chính xác tên cột
    col_names = list(df.columns)

    # Dữ liệu để build bảng
    rows = []

    # Nếu chỉ có 2 cột (Metric, Value) hoặc thêm cột Ticker thì xử lý riêng
    if set(col_names) <= {"Ticker", "Metric", "Value"}:
        header = [c for c in col_names if c != "Ticker"]
        rows.append(header)

        for _, r in df.iterrows():
            row = [r[c] for c in header]
            rows.append(row)
    else:
        # DataFrame bình thường
        rows = [col_names] + df.fillna("-").astype(str).values.tolist()

    # Convert toàn bộ dữ liệu thành string để tránh lỗi kiểu dữ liệu
    rows = [[str(cell) for cell in row] for row in rows]

    table = Table(rows, hAlign='LEFT', colWidths='*')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVu'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ]))

    return [table, Spacer(1, 0.3 * cm)]



from typing import List


def generate_pdf_filename(tickers: list, report_date, report_types: list) -> str:
    """
    Sinh tên file PDF theo chuẩn:
    - Một ticker & một loại báo cáo: AAPL_05082025_forecast.pdf
    - Một ticker & nhiều loại báo cáo: AAPL_05082025_forecast_backtest.pdf
    - Một ticker & đủ 6 loại báo cáo: AAPL_05082025_all.pdf
    - Nhiều ticker & một loại báo cáo: AAPL_MSFT_05082025_forecast.pdf

    Args:
        tickers (list): Danh sách ticker (chuỗi).
        report_date (datetime.date | str): Ngày báo cáo (kiểu `date` hoặc chuỗi ISO).
        report_types (list): Danh sách loại báo cáo ('forecast', 'risk',...).

    Returns:
        str: Tên file PDF đúng chuẩn.
    """
    # Ép kiểu report_date nếu cần
    if isinstance(report_date, str):
        try:
            report_date = datetime.strptime(report_date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Ngày không hợp lệ: {report_date}")

    if not isinstance(report_date, date):
        raise TypeError("report_date phải là datetime.date hoặc str định dạng YYYY-MM-DD")

    if not tickers or not isinstance(tickers, list):
        raise ValueError("tickers phải là list chứa ít nhất 1 ticker")

    if not report_types or not isinstance(report_types, list):
        raise ValueError("report_types phải là list chứa ít nhất 1 loại báo cáo")

    # Đảm bảo thứ tự cố định và không trùng
    tickers_sorted = sorted(set(tickers))
    report_types_sorted = sorted(set(report_types))

    date_str = report_date.strftime("%d%m%Y")

    # Prefix: tên ticker (ghép nếu nhiều)
    prefix = "_".join(tickers_sorted)

    # Suffix
    if len(report_types_sorted) == 6:
        suffix = "all"
    else:
        suffix = "_".join(report_types_sorted)

    return f"{prefix}_{date_str}_{suffix}.pdf"


# ----- Hàm helper format số -----
def format_number(x, decimals=0):
    try:
        return "{:,.{dec}f}".format(float(str(x).replace(',', '').replace('%','')), dec=decimals)
    except:
        return "-"

def format_percent(x, decimals=3):
    try:
        return "{:.{dec}f} %".format(float(str(x).replace(',', '').replace('%','')), dec=decimals)
    except:
        return "-"


def extract_number(val):
    """Lấy số float đầu tiên trong chuỗi"""
    if pd.isnull(val):
        return None
    match = re.search(r"\d+(\.\d+)?", str(val))
    return float(match.group()) if match else None

def format_currency(x):
    """
    Định dạng số tiền nguyên:
    - Dấu '.' phân nghìn
    - Không có thập phân
    Ví dụ:
        1234567 -> '1.234.567'
        123456  -> '123.456'
        12345   -> '12.345'
        1234    -> '1.234'
    """
    try:
        if pd.isna(x):
            return ""
        x = int(round(float(x), 0))
        return f"{x:,}".replace(",", ".")
    except:
        return ""

# -------------------- HÀM ĐỊNH DẠNG KHỐI LƯỢNG / SỐ NGUYÊN --------------------
def format_volume(x):
    """
    Định dạng số lượng / khối lượng: 123456 -> '123.456'
    """
    try:
        if pd.isna(x):
            return ""
        x = int(round(float(x), 0))
        return f"{x:,}".replace(",", ".")
    except:
        return ""

# -------------------- HÀM ĐỊNH DẠNG NGÀY --------------------
def format_date(x):
    """
    Định dạng ngày: datetime hoặc string -> 'dd/mm/yyyy'
    """
    try:
        x = pd.to_datetime(x, errors="coerce")
        if pd.isna(x):
            return ""
        return x.strftime("%d/%m/%Y")
    except:
        return ""


def get_all_tickers_in_reports_custom(folder_path: str) -> list:
    """
    Lấy danh sách tất cả ticker từ các file PDF trong thư mục báo cáo,
    dựa trên định dạng file:
    {ticker1}_{ticker2}_..._{ngay(ddmmyyyy)}_{report_type1}_{report_type2}_...pdf
    Ví dụ: AAPL_05082025_backtest.pdf
    """
    tickers = set()
    pattern = re.compile(r"^(.*?)_(\d{2}\d{2}\d{4})_.*\.pdf$")  # định dạng DDMMYYYY

    for f in os.listdir(folder_path):
        match = pattern.match(f)
        if match:
            ticker_part = match.group(1)
            tickers_in_file = ticker_part.split("_")
            tickers.update(filter(None, tickers_in_file))  # loại bỏ chuỗi rỗng

    return sorted(tickers)



def export_report_pdf(
    ticker,
    bt_df=None,
    perf_df=None,
    risk_df=None,
    weights_df=None,
    
    forecast_df=None,
    alloc_display=None,   # thêm
    adj_display=None,     # thêm
    output_path=None,
    forecast_result=None,
    include_backtest=True,
    include_perf=True,
    include_risk=True,
    include_rebalance=False,
    include_forecast=False,
    include_optimization=False
    
):
    try:
        if output_path is None:
            today_str = datetime.today().strftime("%d%m%Y")
            filename = f"{ticker}_{today_str}_report.pdf"
            output_path = os.path.join("reports", filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        flow = []

        tickers = ticker if isinstance(ticker, list) else [ticker]

        for ticker in tickers:
            t_norm = ticker.upper()

            #flow.append(Paragraph(f"Báo cáo cho mã: {t_norm}", styles["Heading1_Unicode"]))
            flow.append(Paragraph(f"Báo cáo cho mã: {t_norm}", styles["Heading1_Center"]))

            flow.append(Spacer(1, 0.3 * cm))



            
            # ===== FORECAST =====
            if include_forecast and forecast_df is not None:
                # --- Lấy dữ liệu dự báo cho ticker một cách resilient ---
                df_tk = None
                if isinstance(forecast_df, dict):
                    # thử các key khác nhau
                    for key_try in (ticker, ticker.upper(), ticker.lower()):
                        if key_try in forecast_df:
                            df_tk = forecast_df.get(key_try)
                            break
                    if df_tk is None:
                        # thử khớp không phân biệt hoa thường trên tên key
                        for k in forecast_df.keys():
                            try:
                                if isinstance(k, str) and k.lower() == ticker.lower():
                                    df_tk = forecast_df.get(k)
                                    break
                            except Exception:
                                continue
                elif isinstance(forecast_df, pd.DataFrame):
                    # nếu forecast_df là DataFrame có cột Ticker/ticker thì lọc, nếu không thì coi là đã được lọc sẵn
                    if "Ticker" in forecast_df.columns:
                        df_tk = forecast_df[forecast_df["Ticker"].astype(str).str.lower() == ticker.lower()].copy()
                    elif "ticker" in forecast_df.columns:
                        df_tk = forecast_df[forecast_df["ticker"].astype(str).str.lower() == ticker.lower()].copy()
                    else:
                        df_tk = forecast_df.copy()
                else:
                    df_tk = None

                # Nếu là Series (cá biệt), chuyển về DataFrame 1 hàng
                if isinstance(df_tk, pd.Series):
                    df_tk = df_tk.to_frame().T

                # Nếu không có dữ liệu -> ghi note ngắn và bỏ qua
                if not (isinstance(df_tk, pd.DataFrame) and not df_tk.empty):
                    flow.append(Paragraph(f"⚠️ Không tìm thấy dữ liệu dự báo cho {ticker.upper()} (kiểu nguồn: {type(forecast_df).__name__}).", styles["Normal"]))
                    flow.append(Spacer(1, 0.2 * cm))
                    flow.append(PageBreak())
                else:
                    # --- Chuẩn hoá tên cột sang chuẩn nội bộ ---
                    orig_cols = list(df_tk.columns)
                    col_map = {}
                    for col in orig_cols:
                        lc = col.strip().lower()
                        if lc in ("ds", "date"):
                            col_map[col] = "Date"
                        elif ("predicted" in lc and "close" in lc) or (lc in ("yhat", "yhat1", "yhat_lower", "yhat_upper")):
                            col_map[col] = "Predicted_Close"
                        elif ("predicted" in lc and "open" in lc) or ("predicted_open" in lc):
                            col_map[col] = "Predicted_Open"
                        elif ("predicted" in lc and "high" in lc) or ("predicted_high" in lc):
                            col_map[col] = "Predicted_High"
                        elif ("predicted" in lc and "low" in lc) or ("predicted_low" in lc):
                            col_map[col] = "Predicted_Low"
                        elif ("predicted" in lc and ("vol" in lc or "volume" in lc)) or ("predicted_volume" in lc):
                            col_map[col] = "Predicted_Volume"
                        elif lc == "close":
                            col_map[col] = "Close"
                        elif "volume" in lc or lc == "vol":
                            col_map[col] = "Volume"
                    if col_map:
                        df_tk.rename(columns=col_map, inplace=True)

                    # --- Giữ bản raw numeric để vẽ (df_tk_raw) ---
                    df_tk_raw = df_tk.copy()

                    # Nếu Date chưa có, thử lấy từ index hoặc các cột có dạng date
                    if "Date" not in df_tk_raw.columns:
                        try:
                            if isinstance(df_tk_raw.index, pd.DatetimeIndex):
                                df_tk_raw = df_tk_raw.reset_index()
                                if "Date" not in df_tk_raw.columns:
                                    idx_name = df_tk_raw.columns[0]
                                    df_tk_raw.rename(columns={idx_name: "Date"}, inplace=True)
                            else:
                                for c in df_tk_raw.columns:
                                    if "date" in c.lower() or c.lower() == "ds":
                                        df_tk_raw.rename(columns={c: "Date"}, inplace=True)
                                        break
                        except Exception:
                            pass

                    # Ép kiểu Date và các cột predicted về numeric
                    if "Date" in df_tk_raw.columns:
                        df_tk_raw["Date"] = pd.to_datetime(df_tk_raw["Date"], errors="coerce")

                    for col in ["Predicted_Open", "Predicted_High", "Predicted_Low", "Predicted_Close", "Predicted_Volume", "Close", "Volume"]:
                        if col in df_tk_raw.columns:
                            df_tk_raw[col] = pd.to_numeric(df_tk_raw[col], errors="coerce")

                    # --- Chuẩn bị df_tk_display (bảng in) từ df_tk_raw ---
                    cols_keep = ["Date", "Predicted_Open", "Predicted_High", "Predicted_Low", "Predicted_Close", "Predicted_Volume"]
                    df_tk_for_display = df_tk_raw[[c for c in cols_keep if c in df_tk_raw.columns]].copy()

                    vietnamese_map = {
                        "Date": "Ngày",
                        "Predicted_Open": "Giá mở cửa",
                        "Predicted_High": "Cao nhất",
                        "Predicted_Low": "Thấp nhất",
                        "Predicted_Close": "Giá đóng cửa",
                        "Predicted_Volume": "Khối lượng"
                    }
                    df_tk_display = df_tk_for_display.rename(columns=vietnamese_map)

                    # 🔹 Dùng hàm format chung
                    if "Ngày" in df_tk_display.columns:
                        df_tk_display["Ngày"] = df_tk_display["Ngày"].apply(format_date)

                    for col in ["Giá mở cửa", "Cao nhất", "Thấp nhất", "Giá đóng cửa"]:
                        if col in df_tk_display.columns:
                            df_tk_display[col] = df_tk_display[col].apply(format_currency)

                    if "Khối lượng" in df_tk_display.columns:
                        df_tk_display["Khối lượng"] = df_tk_display["Khối lượng"].apply(format_volume)

                    # ===== Ghi ra report (bảng) =====
                    flow.append(Paragraph(f"Dự báo chi tiết – {ticker.upper()}", styles["Heading2_Unicode"]))
                    flow.append(Spacer(1, 0.2 * cm))
                    if not df_tk_display.empty:
                        flow.extend(make_table_from_df(df_tk_display))
                        flow.append(Spacer(1, 0.3 * cm))
                    else:
                        flow.append(Paragraph("⚠️ Bảng dự báo rỗng (không có cột phù hợp để hiển thị).", styles["Normal"]))
                        flow.append(Spacer(1, 0.3 * cm))

                    # ===== Biểu đồ 1: Matplotlib (Giá đóng cửa dự báo) =====
                    fig, ax = plt.subplots(figsize=(6, 3))
                    if "Predicted_Close" in df_tk_raw.columns:
                        df_plot = df_tk_raw[["Date", "Predicted_Close"]].dropna()
                        if not df_plot.empty:
                            ax.plot(df_plot["Date"], df_plot["Predicted_Close"],
                                    label="Giá đóng cửa dự báo", marker="o", linestyle="-")

                    ax.set_title(f"Biểu đồ giá đóng cửa dự báo – {ticker.upper()}")
                    ax.set_xlabel("Ngày")
                    ax.set_ylabel("Giá đóng cửa")
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                    ax.legend()
                    plt.tight_layout()

                    img_buffer = BytesIO()
                    plt.savefig(img_buffer, format="png")
                    plt.close()
                    img_buffer.seek(0)
                    img = Image(img_buffer, width=16 * cm, height=7 * cm)
                    flow.append(img)
                    flow.append(Spacer(1, 0.3 * cm))

                    # ===== Biểu đồ 2: Xu hướng giá đóng cửa & SMA =====
                    try:
                        if "Predicted_Close" in df_tk_raw.columns:
                            df_chart = df_tk_raw[["Date", "Predicted_Close"]].dropna()
                            forecast_days = st.session_state.get("forecast_days_slider", 7)
                            if not df_chart.empty:
                                df_chart["SMA"] = df_chart["Predicted_Close"].rolling(window=forecast_days).mean()

                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.plot(df_chart["Date"], df_chart["Predicted_Close"],
                                        label="Giá đóng cửa", color="blue", marker="o", linestyle="-")
                                ax.plot(df_chart["Date"], df_chart["SMA"],
                                        label=f"SMA {forecast_days}", color="orange", linestyle="--")

                                ax.set_title(f"Xu hướng giá đóng cửa & SMA ({forecast_days} kỳ) - {ticker.upper()}")
                                ax.set_xlabel("Ngày")
                                ax.set_ylabel("Giá đóng cửa (VNĐ)")
                                ax.legend()
                                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                                plt.tight_layout()

                                img_buffer = BytesIO()
                                plt.savefig(img_buffer, format="png")
                                plt.close()
                                img_buffer.seek(0)
                                img = Image(img_buffer, width=16 * cm, height=7 * cm)
                                flow.append(img)
                                flow.append(Spacer(1, 0.5 * cm))
                            else:
                                flow.append(Paragraph("⚠️ Không đủ dữ liệu để vẽ biểu đồ xu hướng.", styles["Normal"]))
                                flow.append(Spacer(1, 0.2 * cm))
                        else:
                            flow.append(Paragraph("⚠️ Không tìm thấy cột 'Predicted_Close' để vẽ biểu đồ.", styles["Normal"]))
                            flow.append(Spacer(1, 0.2 * cm))

                    except Exception as e:
                        flow.append(Paragraph(f"⚠️ Lỗi khi vẽ biểu đồ SMA cho {ticker}: {e}", styles["Normal"]))

                    flow.append(PageBreak())

            # ===== BACKTEST =====
            if include_backtest and isinstance(bt_df, dict) and ticker in bt_df:
                _bt_df = bt_df[ticker]
                if isinstance(_bt_df, pd.DataFrame) and not _bt_df.empty and "Date" in _bt_df.columns:
                    equity_col = next((col for col in _bt_df.columns if "Equity" in col or "Portfolio" in col), None)
                    if equity_col:
                        # Tạo figure và axis
                        fig, ax = plt.subplots(figsize=(10, 4))
                        
                        # Vẽ đường equity
                        ax.plot(
                            _bt_df["Date"], 
                            _bt_df[equity_col], 
                            label=equity_col, 
                            color='#1f77b4', 
                            linewidth=2
                        )
                        
                        # Highlight điểm cuối
                        ax.scatter(
                            _bt_df["Date"].iloc[-1], 
                            _bt_df[equity_col].iloc[-1], 
                            color='red', 
                            s=40, 
                            zorder=5
                        )
                        
                        # Tiêu đề và nhãn
                        ax.set_title(f"{t_norm} - {equity_col}", fontsize=12, fontname="DejaVu")
                        ax.set_xlabel("Ngày", fontsize=10, fontname="DejaVu")
                        ax.set_ylabel("Giá trị", fontsize=10, fontname="DejaVu")
                        
                        # Grid mềm, nền nhẹ
                        ax.grid(True, alpha=0.3)
                        ax.set_facecolor('#f9f9f9')
                        
                        # Xóa viền để nhìn hiện đại
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        
                        # Legend
                        ax.legend(fontsize=9)
                        
                        # Lưu hình vào buffer
                        buf = BytesIO()
                        fig.tight_layout()
                        fig.savefig(buf, format='png', dpi=200)
                        plt.close(fig)
                        buf.seek(0)
                        
                        # Thêm vào PDF flow
                        flow.append(Paragraph("Backtest", styles["Normal_Unicode"]))
                        flow.append(Image(buf, width=16 * cm, height=6 * cm))
                        flow.append(Spacer(1, 0.4 * cm))

                        # --- Thêm lệnh PageBreak để mỗi ticker 1 trang ---
                        # flow.append(PageBreak())
            

            # # ===== HIỆU SUẤT =====

            if include_perf and isinstance(perf_df, dict) and ticker in perf_df:
                _perf_df = perf_df[ticker]
                if isinstance(_perf_df, pd.DataFrame) and not _perf_df.empty:
                    df_perf_show = _perf_df.copy()

                    # --- Việt hóa metric dài, giữ metric gốc cho các metric viết tắt ---
                    metric_map = {
                        "Average Gain": "Lợi nhuận trung bình khi thắng",
                        "Average Loss": "Lỗ trung bình khi thua",
                        "Win Rate": "Tỷ lệ thắng",
                        "Calmar Ratio": "Tỷ lệ Calmar",
                        "Omega Ratio": "Tỷ lệ Omega",
                        "Sterling Ratio": "Tỷ lệ Sterling"
                        # Metric viết tắt như CAGR, Sharpe Ratio, Volatility... giữ nguyên
                    }
                    df_perf_show["Metric"] = df_perf_show["Metric"].map(metric_map).fillna(df_perf_show["Metric"])

                    # --- Định dạng số ---
                    for col in df_perf_show.select_dtypes(include=['float', 'int']).columns:
                        df_perf_show[col] = df_perf_show[col].apply(lambda x: f"{x:.4f}")

                    # --- Chuyển thành rows chuẩn hệ thống ---
                    rows = [[str(cell) for cell in row] for row in df_perf_show.values.tolist()]
                    rows = [df_perf_show.columns.tolist()] + rows

                    # --- Chiều rộng cột cố định ---
                    table_width = 16 * cm
                    val_col_width = 5 * 0.35 * cm  # fix 5 ký tự ~1.75 cm
                    index_col_width = table_width - val_col_width  # cột Chỉ số chiếm phần còn lại
                    col_widths = [index_col_width, val_col_width]

                    # --- Tạo Table chuẩn hệ thống ---
                    table = Table(rows, hAlign='LEFT', colWidths=col_widths)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), 'DejaVu'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                    ]))

                    # --- Thêm vào flow PDF ---
                    flow.append(Paragraph("Hiệu suất", styles["Normal_Unicode"]))
                    flow.append(table)
                    flow.append(Spacer(1, 0.3 * cm))
                    flow.append(PageBreak())
          
            # ===== RISK =====
            if include_risk and isinstance(risk_df, dict) and ticker in risk_df:
                _risk_df = risk_df[ticker]
                if isinstance(_risk_df, pd.DataFrame) and not _risk_df.empty:
                    # flow.append(Paragraph(f"Phân tích rủi ro – {t_norm}", styles["Heading1_Unicode"]))
                    flow.append(Spacer(1, 12))

                    # --- Biểu đồ cột ---
                    fig, ax = plt.subplots(figsize=(6.5, 4))
                    bars = ax.bar(_risk_df["Metric"], _risk_df["Value"], color="#4F81BD", width=0.5)
                    for bar in bars:
                        h = bar.get_height()
                        ax.annotate(f"{h:.4f}",
                                    xy=(bar.get_x() + bar.get_width() / 2, h),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=8, fontname="DejaVu")
                    ax.set_xlabel("Chỉ số", fontname="DejaVu")
                    ax.set_ylabel("Giá trị", fontname="DejaVu")
                    ax.set_title(f"Biểu đồ phân tích rủi ro – {t_norm}", fontname="DejaVu")
                    ax.grid(axis='y', linestyle='--', alpha=0.6)
                    fig.tight_layout()

                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=200)
                    plt.close(fig)
                    buf.seek(0)
                    flow.append(Image(buf, width=14*cm, height=6*cm))
                    flow.append(Spacer(1, 20))

                    # --- Bảng dữ liệu ---
                    table_data = [["Mã", "Metric", "Value"]]
                    for _, row in _risk_df.iterrows():
                        table_data.append([
                            row.get("Ticker", t_norm),
                            row["Metric"],
                            f"{row['Value']:.4f}"
                        ])

                    table = Table(table_data, colWidths=[60, 150, 80])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                        ('FONTNAME', (0, 0), (-1, -1), 'DejaVu'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ]))
                    flow.append(table)

                    flow.append(PageBreak())


            # --- Bảng phân bổ vốn ---
            if include_optimization:
                if alloc_display is not None:
                    alloc_pdf = alloc_display[alloc_display["Mã cổ phiếu"].str.strip().str.upper() == t_norm].copy()
                    if not alloc_pdf.empty:
                        # Bỏ cột "Mã cổ phiếu" nếu tồn tại
                        if "Mã cổ phiếu" in alloc_pdf.columns:
                            alloc_pdf = alloc_pdf.drop(columns=["Mã cổ phiếu"])

                        # Định dạng cột
                        if "Tỷ trọng" in alloc_pdf.columns:
                            alloc_pdf["Tỷ trọng"] = alloc_pdf["Tỷ trọng"].apply(lambda x: format_percent(x, 3))
                        if "Expected_Return" in alloc_pdf.columns:
                            alloc_pdf["Expected_Return"] = alloc_pdf["Expected_Return"].apply(lambda x: format_percent(x, 3))
                        if "Volatility" in alloc_pdf.columns:
                            alloc_pdf["Volatility"] = alloc_pdf["Volatility"].apply(lambda x: format_percent(x, 3))
                        if "Vốn phân bổ" in alloc_pdf.columns:
                            alloc_pdf["Vốn phân bổ"] = alloc_pdf["Vốn phân bổ"].apply(lambda x: format_number(x, 0))
                        


                        # Việt hoá cột
                        alloc_pdf = alloc_pdf.rename(columns={
                            "Tỷ trọng": "Tỷ trọng",
                            "Expected_Return": "Lợi nhuận kỳ vọng",
                            "Volatility": "Độ biến động gốc",
                            "Vốn phân bổ": "Vốn phân bổ"
                        })

                        flow.append(Paragraph("Phân bổ vốn", styles["Normal_Unicode"]))
                        flow.extend(make_table_from_df(alloc_pdf))
                        flow.append(Paragraph("Đơn vị vốn phân bổ = đơn vị của Tổng vốn nhập vào", styles["Normal_Unicode"]))
                        flow.append(Spacer(1, 0.3 * cm))

                # ----- Bảng phân bổ nâng cao -----
                if adj_display is not None:
                    adj_pdf = adj_display[adj_display["Mã cổ phiếu"].str.strip().str.upper() == t_norm].copy()
                    if not adj_pdf.empty:
                        if "Mã cổ phiếu" in adj_pdf.columns:
                            adj_pdf = adj_pdf.drop(columns=["Mã cổ phiếu"])

                        # Định dạng cột
                        for col in ["Tỷ trọng gốc", "Tỷ trọng điều chỉnh", "Expected_Return", "Volatility", "Độ biến động"]:
                            if col in adj_pdf.columns:
                                adj_pdf[col] = adj_pdf[col].apply(lambda x: format_percent(x, 3))
                        if "Vốn phân bổ điều chỉnh" in adj_pdf.columns:
                            adj_pdf["Vốn phân bổ điều chỉnh"] = adj_pdf["Vốn phân bổ điều chỉnh"].apply(lambda x: format_number(x, 0))

                        # Việt hoá
                        adj_pdf = adj_pdf.rename(columns={
                            "Tỷ trọng gốc": "Tỷ trọng gốc",
                            "Tỷ trọng điều chỉnh": "Tỷ trọng ĐC",
                            "Expected_Return": "Lợi nhuận kỳ vọng",
                            "Volatility": "Độ biến động",
                            "Độ biến động": "Độ biến động ĐC",
                            "Vốn phân bổ điều chỉnh": "Vốn phân bổ ĐC"
                        })

                        flow.append(Paragraph("Phân bổ nâng cao theo rủi ro", styles["Normal_Unicode"]))
                        flow.extend(make_table_from_df(adj_pdf))
                        flow.append(Paragraph("Đơn vị vốn phân bổ điều chỉnh (ĐC) = đơn vị của Tổng vốn nhập vào", styles["Normal_Unicode"]))
                        flow.append(Paragraph("Lợi nhuận kỳ vọng tính theo 1 năm, dựa trên log return hàng ngày và annualized.", styles["Normal_Unicode"]))
                        flow.append(Spacer(1, 0.3 * cm))

                flow.append(PageBreak())

            
            
            # ===== REBALANCE =====
            if include_rebalance and isinstance(weights_df, pd.DataFrame) and 'Ticker' in weights_df.columns:
                # Bảo đảm dùng đúng DF của TAB 2 (rebalance)
                required_cols = {'Ticker', 'Tỷ trọng hiện tại', 'Tỷ trọng mục tiêu (chiến lược)'}
                missing = [c for c in required_cols if c not in weights_df.columns]
                if missing:
                    flow.append(Paragraph(
                        f"Thiếu cột cho Tái cân bằng: {', '.join(missing)}", styles["Normal_Unicode"]
                    ))
                else:
                    # Chuẩn hóa Ticker
                    weights_df['Ticker'] = weights_df['Ticker'].astype(str).str.upper()

                    # --- BẢNG CHO MỖI TICKER ---
                    _reb_df = weights_df[weights_df['Ticker'] == t_norm].copy()
                    if not _reb_df.empty:
                        # Nếu có DataFrame kế hoạch thì merge thêm SL hiện tại & Chênh lệch SL
                        if "rebalance_plan_df" in st.session_state:
                            plan_df = st.session_state["rebalance_plan_df"].copy()
                            plan_df['Ticker'] = plan_df['Ticker'].astype(str).str.upper()
                            _reb_df = _reb_df.merge(
                                plan_df[['Ticker', 'SL hiện tại', 'Chênh lệch SL']],
                                on='Ticker', how='left'
                            )

                        # Lưu series tỷ trọng để vẽ pie chart
                        cur_series = (
                            weights_df[['Ticker', 'Tỷ trọng hiện tại']]
                            .dropna()
                            .groupby('Ticker', as_index=True)['Tỷ trọng hiện tại']
                            .sum()
                        )
                        tgt_series = (
                            weights_df[['Ticker', 'Tỷ trọng mục tiêu (chiến lược)']]
                            .dropna()
                            .groupby('Ticker', as_index=True)['Tỷ trọng mục tiêu (chiến lược)']
                            .sum()
                        )

                        # Convert về số nếu cột là chuỗi có %
                        for s in (cur_series, tgt_series):
                            if s.dtype == object:
                                s = s.str.replace('%', '', regex=False).str.replace(',', '', regex=False)
                                s = pd.to_numeric(s, errors='coerce') / 100.0

                        # Bỏ cột Ticker và cột Tỷ trọng khỏi bảng PDF
                        drop_cols = [c for c in _reb_df.columns if (c == 'Ticker') or ('Tỷ trọng' in c)]
                        table_df = _reb_df.drop(columns=drop_cols, errors='ignore')

                        # Định dạng số nguyên có dấu phẩy cho các cột số
                        for col in table_df.columns:
                            if pd.api.types.is_numeric_dtype(table_df[col]):
                                table_df[col] = table_df[col].apply(
                                    lambda x: "{:,.0f}".format(x) if pd.notnull(x) else ""
                                )

                        # Thêm bảng vào PDF
                        flow.append(Paragraph(f"Tái cân bằng - {t_norm}", styles["Normal_Unicode"]))
                        flow.extend(make_table_from_df(table_df))
                        flow.append(Spacer(1, 0.3 * cm))

                        # --- Biểu đồ tròn trước/sau ---
                        figs = []
                        fig1, ax1 = plt.subplots(figsize=(3, 3))
                        ax1.pie(cur_series.values, labels=cur_series.index, autopct='%1.1f%%')
                        ax1.set_title("Trước tái cân bằng")
                        buf1 = io.BytesIO()
                        plt.savefig(buf1, format="png", bbox_inches="tight")
                        plt.close(fig1)
                        buf1.seek(0)
                        figs.append(RLImage(buf1, width=7*cm))

                        fig2, ax2 = plt.subplots(figsize=(3, 3))
                        ax2.pie(tgt_series.values, labels=tgt_series.index, autopct='%1.1f%%')
                        ax2.set_title("Sau tái cân bằng")
                        buf2 = io.BytesIO()
                        plt.savefig(buf2, format="png", bbox_inches="tight")
                        plt.close(fig2)
                        buf2.seek(0)
                        figs.append(RLImage(buf2, width=7*cm))

                        # Đặt 2 pie chart ngang nhau
                        flow.append(Table([[figs[0], figs[1]]]))
                        flow.append(Spacer(1, 0.5 * cm))

                        # Kết thúc trang
                        flow.append(PageBreak())
            
            
            


        doc = SimpleDocTemplate(output_path, pagesize=A4)
        if not flow:
           
            flow.append(Paragraph("Không có dữ liệu nào để xuất báo cáo.", styles["Normal"]))

        doc.build(flow)
        return output_path   # luôn trả về path nếu build thành công

    except Exception as e:
        import traceback
        st.error(f"[ERROR] Lỗi khi xuất PDF: {e}")
        st.write(traceback.format_exc())  # in đầy stack trace
        return None

