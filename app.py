# =================================================================
# app.py – AIS-Analysis Investment Stock - Ksoft Software Soluttion
# =================================================================

from __future__ import annotations

import os
import gc
import smtplib
import ssl
import sys
import shutil
import time
import traceback
import numpy as np
import threading
from datetime import datetime, date, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List
import plotly.graph_objects as go
import matplotlib.pyplot as plt


import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from utils.chatbot import chatbot_answer

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

try:
    import tensorflow as tf
except ImportError:
    tf = None
# ------------------- Local imports -------------------

# Utilities and modules
from utils.permissions import ROLE_PERMISSIONS, get_permissions, can_access, get_permissions_by_username, get_permissions_by_role

from utils.reporting import export_report_pdf, safe_open_pdf
import utils.db_manager as db

from utils import db_manager
from utils.data_cleaner import clean_dataframe

from utils.db_manager import load_data, save_forecast, get_connection, load_forecast, list_tables, save_forecast_last



from utils.user_history import record_user_ticker_view

from utils.user_manager import init_db, login_user, register_user
from register_page import register_page

#from utils.upgrade_page import upgrade_page
from init_db import get_role_by_username

# from utils.email_scheduler import run_email_scheduler

from utils.reporting import (
    get_all_tickers_in_reports_custom,
    generate_pdf_filename,
    export_report_pdf,
    safe_open_pdf,
)
from utils.user_manager import upgrade_user
from utils.permissions import get_permissions
from utils.admin_activate_page import admin_activate_page


from utils.email_sender import send_report_via_email
from utils.batch_utils import process_batch

# Forecasting & Modeling

from utils.arima_model import load_arima_model, train_arima, forecast_arima
from utils.prophet_model import load_prophet_model, train_prophet, forecast_prophet
from utils.lstm_model import load_lstm_model, train_lstm_model, predict_lstm
from utils.tcn_model import load_tcn_model, train_tcn_model, predict_tcn


# Data loading and processing (Note: duplicate imports merged)

from utils.data_loader import load_data as load_data_dl

from datetime import date, timedelta

today = date.today()
cutoff = today - timedelta(days=180)


# Pages (Streamlit UI)
sys.path.append(os.path.dirname(__file__))
from login_page import login_page
from register_page import register_page

# ---- local utils -----------------------------------------------------------
from utils import (
    data_cleaner, portfolio_optimizer, risk_metrics,
    backtest, performance, strategy, reporting
)
# from init_db import create_db
# create_db()

# Khởi tạo ngày duy nhất cho toàn bộ session
if "today_str" not in st.session_state:
    st.session_state["today_str"] = datetime.today().strftime("%d%m%Y")



MEM_SAFE_MODE = True   # Hoặc False, tùy nhu cầu
BATCH_SIZE = 50        # Số lượng mỗi batch, tùy nhu cầu xử lý

# ========= Streamlit setup ==================================================
st.set_page_config(page_title="AIS phân tích đầu tư chứng khoán", layout="wide")

# st.title("📈 Hệ thống đầu tư AIS‑Ksoft")
# ========= Thư mục ==========================================================

REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)


# # ================== KHỞI TẠO TRẠNG THÁI ==================
init_db()

if "page" not in st.session_state:
    st.session_state["page"] = "login"

if "user" not in st.session_state:
    st.session_state["user"] = None

if "enter_home" not in st.session_state:
    st.session_state["enter_home"] = False

# ---------------------------------------------------------------------------
# Hàm phụ trợ để lấy danh sách mã hợp lệ
def get_valid_tickers(portfolio_df):
        """Trích xuất danh sách mã cổ phiếu hợp lệ từ DataFrame."""
        if portfolio_df is not None and not portfolio_df.empty and "Ticker" in portfolio_df.columns:
            return portfolio_df["Ticker"].dropna().unique().tolist()
        return []   
# ---- Hàm tiện ích show_notification ---------------------------------------
def show_notification(ph, notif_type: str, msg: str):
    """Ghi đè nội dung placeholder với thông báo mới."""
    ph.empty()          # xoá thông báo cũ
    if notif_type == "success":
        ph.success(msg)
    elif notif_type == "warning":
        ph.warning(msg)
    elif notif_type == "info":
        ph.info(msg)

# # 🚀 Khởi động lịch gửi email tự động (chỉ gọi 1 lần)

# if "email_scheduler_started" not in st.session_state:
#     st.session_state["email_scheduler_started"] = True
#     threading.Thread(target=run_email_scheduler, daemon=True).start()


# ========= Email helper =====================================================
def send_email(subject: str, body: str, attachment_paths: List[str],
            smtp_server: str, smtp_port: int,
            username: str, password: str, receiver: str):
    """Gửi email với nhiều file đính kèm (PDF/CSV…)."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = username
    msg["To"]      = receiver
    msg.set_content(body)

    for p in attachment_paths:
        with open(p, "rb") as f:
            data = f.read()
        msg.add_attachment(data, maintype="application",
                        subtype="octet-stream", filename=os.path.basename(p))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
        server.login(username, password)
        server.send_message(msg)

# ======================================================
# ===== Khởi tạo session_state =====
def init_session_keys():
    defaults = {
        "user": None,
        "download_frames": [],
        "portfolio_data": None,
        "last_update": {},
        "invalid_tickers": [],
        "page": "login",
        "enter_home": False,
        "batch_size": 20,
        "mem_safe": True,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "from_date" not in st.session_state:
        st.session_state["from_date"] = date(2023, 1, 1)

    if "to_date" not in st.session_state:
        st.session_state["to_date"] = date.today()
# -------------------- HÀM ĐỊNH DẠNG TIỀN NGUYÊN --------------------
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


# ========= Helper: tải dữ liệu một ticker ===================================

# Biến toàn cục giữ tất cả DataFrame đã tải
_download_frames: list[pd.DataFrame] = []
def _download_one(ticker: str, from_date, to_date, source="yf") -> bool:
    try:
        # Load dữ liệu từ data_loader
        df = load_data_dl(ticker, start_date=from_date, end_date=to_date, source=source)
        if df.empty:
            return False
        # Append vào frame toàn cục
        _download_frames.append(df)
        # Lưu vào DB
        db.insert_new_rows(ticker, df)
        # Cập nhật session_state last_update nếu có Streamlit
        try:
            if "last_update" not in st.session_state:
                st.session_state["last_update"] = {}
            st.session_state["last_update"][ticker] = datetime.now()
        except Exception:
            pass  # Nếu không có Streamlit, bỏ qua
        return True
    except Exception:
        return False



# ===== Download batch =====

def run_download(target_list, from_date, to_date):
    """Tải dữ liệu cho danh sách tickers và cập nhật session_state."""
    loaded, failed = [], []

    if not target_list:
        st.warning("⚠️ Không có tickers để tải!")
        return

    # Reset _download_frames trước mỗi lần download thủ công
    global _download_frames
    _download_frames = []

    for i in range(0, len(target_list),
                   BATCH_SIZE if MEM_SAFE_MODE else len(target_list)):
        batch = target_list[i : i + (BATCH_SIZE if MEM_SAFE_MODE else len(target_list))]
        for tk in batch:
            ok = _download_one(tk, from_date, to_date)  # Truyền từ ngày đến ngày vào _download_one
            (loaded if ok else failed).append(tk)

        gc.collect()
        if "tf" in globals() and tf:
            tf.keras.backend.clear_session()

    # —— Thông báo gọn —— 
    if loaded:
        show_notification(
            st.session_state.get("load_msg", "Đã tải dữ liệu"),
            "success",
            "✅ Đã tải dữ liệu: " + ", ".join(loaded)
        )
    if failed:
        show_notification(
            st.session_state.get("update_msg_ph", "Cập nhật không thành công"),
            "warning",
            "⚠️ Không có dữ liệu cho " + ", ".join(failed)
        )
        st.session_state["invalid_tickers"] = list(
            set(st.session_state.get("invalid_tickers", [])) | set(failed)
        )

    # —— Ghép dữ liệu —— 
    if _download_frames:
        new_data = pd.concat(_download_frames)
        if st.session_state.get("portfolio_data") is None:
            st.session_state["portfolio_data"] = new_data
        else:
            st.session_state["portfolio_data"] = (
                pd.concat([st.session_state["portfolio_data"], new_data])
                  .drop_duplicates(subset=["Date", "Ticker"])
            )

# ===================== TRANG CHÍNH ====================

def page_home():
    
    init_session_keys()

    user = st.session_state["user"]
    if not user:
        st.warning("Chưa đăng nhập.")
        st.session_state["page"] = "login"
        st.rerun()
        return

    st.write(f"Xin chào **{user['full_name']}**!")
    st.write(f"Bạn đang đăng nhập với quyền: `{user['role']}`")
    
    # ===== Chỉ admin mới thấy nút =====
    # if user.get("role") == "admin":
    #     #col1, col2 = st.columns([9, 1])  # dồn nút sang phải
    #     #with col2:
    #     if st.button("Pending"):
    #         st.session_state["page"] = "admin_activate"
    #         st.rerun()
    # ==================================
    # if st.button("Đăng xuất", key="btn_logout"):
    #     st.session_state["user"] = None
    #     st.session_state["enter_home"] = False
    #     st.session_state["page"] = "login"
    #     st.rerun()

    
    # # ===========st.subheader("📌 Các chức năng của bạn:")=============

    # # Nút tải dữ liệu danh mục
    # # --- khai báo  ngay từ đầu ---
    global MEM_SAFE_MODE, BATCH_SIZE  # <-- thêm dòng này

    # Tạo 10 cột, col1 rộng hơn, col9 và col10 chuẩn, các cột còn lại nhỏ
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([12,2,1,1,1,2,2,2,8,8])

    # Nút Phân tích & dự báo ở col1
    with col1:
        if st.button("📊 Phân tích & dự báo", key="btn_download"):
            tickers = st.session_state.get("tickers", [])
            if user and tickers:
                record_user_ticker_view(user["id"], tickers)
            run_download(
                tickers,
                st.session_state.get("from_date", "2020-01-01"),
                st.session_state.get("to_date", datetime.now().date())
            )

    # col2 → col8 trống
    # Nút Pending ở col9 (chỉ admin)
    with col9:
        if user.get("role") == "admin":
            if st.button("📌 Pending", key="btn_pending"):
                st.session_state["page"] = "admin_activate"
                st.rerun()

    # Nút Đăng xuất ở col10
    with col10:
        if st.button("🚀 Đăng xuất", key="btn_logout"):
            st.session_state["user"] = None
            st.session_state["enter_home"] = False
            st.session_state["page"] = "login"
            st.rerun()

    
    # ========= Sidebar ==========================================================
    
    # ✅ Chọn nguồn dữ liệu
    data_source = st.sidebar.selectbox(
        "Nguồn dữ liệu:",
        ["yfinance", "vndirect"],
        index=0,
        help="Chọn nguồn để tải dữ liệu: Yahoo Finance hoặc VNDIRECT"
    )

    # ✅ Nhập danh sách ticker
    tickers_input = st.sidebar.text_input(
        "Nhập ticker (cách bởi dấu phẩy)",
        "VIC.VN",
        help="Ví dụ: SHB.VN,ACB.VN,VIC.VN",
        key="tickers_input"
    )
    # Ép input sang chữ hoa ngay khi nhập
    tickers_input = tickers_input.upper()

    # ✅ Tách danh sách ticker
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    st.session_state["tickers"] = tickers
    # ============
    
    # ---- Date range ------------------------------------------------------------
    date_cols  = st.sidebar.columns(2)
    #====
    from_date = date_cols[0].date_input(
        "📅 Từ ngày",
        value=st.session_state.get("from_date", date(2023, 1, 1)),
        format="DD/MM/YYYY",
        key="from_date"
    )

    to_date = date_cols[1].date_input(
        "📅 Đến ngày",
        value=st.session_state.get("to_date", date.today()),
        format="DD/MM/YYYY",
        key="to_date"
    )
    from_date = pd.to_datetime(from_date).date()
    to_date = pd.to_datetime(to_date).date()


    # ---- Auto‑update + Drawdown slider ----------------------------------------
    auto_update = st.sidebar.checkbox("Tự động cập nhật mỗi 15 phút", True, key="auto_update")
    max_drawdown_percent = st.sidebar.slider("🎯 Giới hạn Max Drawdown (%)", 0.0, 100.0, 25.0,
                                            help="Chỉ giữ lại mã có mức giảm tối đa ≤ giá trị này",
                                            key="max_dd_pct")
    max_drawdown = max_drawdown_percent / 100.0

    # ---- RAM optimisation controls --------------------------------------------
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Quản lý bộ nhớ")
    MEM_SAFE_MODE = st.sidebar.checkbox("🚦 Bật chế độ tiết kiệm RAM", value=True, key="mem_safe")

    BATCH_SIZE = st.sidebar.slider(
        "Số mã xử lý mỗi batch",           # label
        5, 200, 20,                        # min, max, value
        disabled=not MEM_SAFE_MODE,
        key="batch_size",
        help="Giới hạn số mã được xử lý trong một lần batch để tiết kiệm RAM. "
            "Batch nhỏ hơn = ít RAM hơn nhưng thời gian tổng thể dài hơn."
    )

    # ========= Session init =====================================================
    for k, v in {
        "last_update":      {},
        "portfolio_data":   None,
        "invalid_tickers":  [],
        "saved_msg_ph":     None,
        "load_msg":         None,
        "update_msg_ph":    None,
    }.items():
        st.session_state.setdefault(k, v)

    

    # ========= Notification placeholders (khởi tạo 1 lần) ======================
    if st.session_state.get("load_msg") is None:          # 👈 dùng .get()
        st.session_state["load_msg"] = st.empty()         # placeholder tải dữ liệu

    if st.session_state.get("update_msg_ph") is None:     # 👈 dùng .get()
        st.session_state["update_msg_ph"] = st.empty()    # placeholder cảnh báo/auto‑update


    # ========= Auto‑update list ================================================
    tickers_to_update: List[str] = []
    if auto_update:
        now = datetime.now()
        for t in tickers:
            last = st.session_state["last_update"].get(t)
            if last is None or (now - last).total_seconds() > 900:
                tickers_to_update.append(t)

    
    # ── Placeholder THÔNG BÁO nằm ngay dưới nút ────────────────────────
    # Gán lại vào session_state để mọi nơi dùng chung
    st.session_state["update_msg_ph"] = st.empty()
    ph_upd = st.session_state["update_msg_ph"]


    # ========= Auto‑update 15 phút (tối ưu thông báo) ==========================
    if tickers_to_update:
        ph_upd = st.session_state["update_msg_ph"]

        # # dòng đầu tiên: đang chạy
        ph_upd.info("🕒 Đang tự động cập nhật dữ liệu…")

        success_messages, failed_messages = [], []
        source = "yf"
        for t in tickers_to_update:
            try:
                # ⚡ Load dữ liệu chuẩn từ load_data
                df_raw = load_data_dl(t, start_date=str(from_date), end_date=str(to_date), source=source)
                if df_raw.empty:
                    failed_messages.append(t)
                    continue

                # --- Append vào frame toàn cục để dùng tiếp ---
                _download_frames.append(df_raw)

                # --- Lưu vào DB ---
                db.insert_new_rows(t, df_raw)

                # --- Merge / cập nhật session_state nếu dùng Streamlit ---
                try:
                    if "tickers_data" not in st.session_state:
                        st.session_state["tickers_data"] = {}
                    st.session_state["tickers_data"][t] = df_raw
                except Exception:
                    pass  # Nếu không dùng Streamlit, bỏ qua

                # --- Ghi success ---
                success_messages.append(t)

            except Exception:
                failed_messages.append(t)


        # # ---- Ghi đè placeholder bằng danh sách mã đã cập nhật ----
        if success_messages:
            ph_upd.info(
                "🕒 Đang tự động cập nhật dữ liệu… "
                + ", ".join(success_messages)
                + "..."
            )

        # ---- Nếu có mã thất bại, ghi đè tiếp bằng cảnh báo ----
        if failed_messages:
            ph_upd.warning("⚠️ Không có dữ liệu cho " + ", ".join(failed_messages))
            st.session_state["invalid_tickers"] = list(
                set(st.session_state.get("invalid_tickers", [])) | set(failed_messages)
            )
        

    # —— giữ nguyên đoạn kiểm tra dữ liệu ——
    portfolio_data: pd.DataFrame | None = st.session_state["portfolio_data"]
    if portfolio_data is None or portfolio_data.empty:
        st.stop()
    
    # ====================CHATBOT=================================
    # --- Khởi tạo lịch sử chat chung ---
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with st.expander("💬 Trợ lý đầu tư AIS", expanded=False):
        # Hiển thị toàn bộ lịch sử chat
        for msg in st.session_state.chat_messages:
            st.markdown(f"**{msg['role']}:** {msg['content']}")

        # Nhận câu hỏi người dùng
        user_q = st.chat_input("Hỏi tôi bất cứ điều gì thuộc phạm vi ứng dụng...")

        if user_q:
            # Lưu câu hỏi
            st.session_state.chat_messages.append({"role": "Bạn", "content": user_q})

            # Sinh câu trả lời
            ans = chatbot_answer(user_q)
            st.session_state.chat_messages.append({"role": "Trợ lý", "content": ans})

            # 🔄 Bắt buộc rerun ngay để hiển thị đúng (tránh trễ 1 câu)
            st.rerun()


    # ==============Tabs phân quyền=====================
    # Lấy username từ session (hoặc từ login)
    role = user.get("role")
    permissions = get_permissions_by_role(role)

    FEATURE_TABS = {
        "forecast": "📊 Dự báo",
        "train": "⚙️ Huấn luyện mô hình",
        "risk": "📉 Phân tích rủi ro",
        "backtest_perf": "🔄 Backtest & Hiệu suất",
        "optimize": "📈 Tối ưu danh mục",
        "rebalance": "♻️ Tái cân bằng",
        "report": "🎯 Báo cáo",
    }

    # Lọc tab labels
    tab_labels = [FEATURE_TABS[f] for f in permissions if f in FEATURE_TABS]

    if tab_labels:
        tabs = st.tabs(tab_labels)

        # Map feature -> index
        tab_map = {f: i for i, f in enumerate(permissions) if f in FEATURE_TABS}


        # ===================================

        # --- 📊 Tab dự báo ---
        if "forecast" in tab_map:
            with tabs[tab_map["forecast"]]:
                
                # ====================DỰ BÁO CHỨC NĂNG=================================        
                sub_tabs = st.tabs([
                    "📈 Dự báo chi tiết",
                    "📊 Thống kê mô tả dự báo",
                    "🧾 Dự báo hành vi",
                    "🌍 Thị trường & Giao dịch"
                ])

                # ====================DỰ BÁO=================================
                with sub_tabs[0]:
                    # model_choice = st.selectbox(
                    #     "Mô hình dự báo",
                    #     ["LSTM", "Prophet", "ARIMA"],
                    #     key="forecast_model_select",
                    #     help=(
                    #         "• LSTM – Mạng nơ-ron hồi quy cho chuỗi thời gian, có thể dự báo nhiều thông số (Giá mở cửa, Cao nhất, Thấp nhất, Đóng cửa, Khối lượng).\n"
                    #         "• Prophet – Mô hình của Meta, mạnh về seasonality, nhưng CHỈ dự báo Giá đóng cửa (không có Giá mở cửa, Cao nhất, Thấp nhất, Khối lượng).\n"
                    #         "• ARIMA – Mô hình thống kê cổ điển, cũng CHỈ dự báo Giá đóng cửa."
                    #     )
                    # )
                    
                    model_choice = st.selectbox(
                        "Mô hình dự báo",
                        ["LSTM", "TCN"],
                        key="forecast_model_select",
                        help=(
                            "• LSTM – (Long Short-Term Memory) là một kiến trúc mạng RNN (Recurrent Neural Network) đặc biệt, được thiết kế để học chuỗi thời gian hoặc dữ liệu tuần tự, multi-output, multi steps cần ít nhất 60 dòng dữ liệu.\n"
                            "• TCN – (Temporal Convolutional Network) là một mô hình học sâu cho chuỗi thời gian, phù hợp với dự báo OHLCV, dựa trên Convolutional Neural Network (CNN) nhưng được thiết kế đặc biệt để xử lý dữ liệu tuần tự, mạnh với chuỗi dài và quan hệ phức tạp."
                        )
                    )


                    forecast_days = st.slider("Số ngày dự báo", 1, 30, 7, key="forecast_days_slider")
                    use_saved = st.checkbox("Dùng mô hình đã huấn luyện", value=False, key="forecast_use_saved")

                    @st.cache_data(show_spinner=False)
                    def cached_forecast(choice, df_json, n_days, use_saved_model, mdl_name):
                         
                        import traceback
                        # df = pd.read_json(df_json)
                        df = pd.read_json(df_json, orient="records")

                        try:
                            # tf.keras.backend.clear_session()
                            if choice == "LSTM":
                                # from utils.lstm_model import load_lstm_model, train_lstm_model, predict_lstm
                                ticker = mdl_name.replace("lstm_", "")
                                if use_saved_model:
                                    model, scaler = load_lstm_model(ticker)
                                    if model is None or scaler is None:
                                        return None, "ModelNotTrained"
                                else:
                                    # model, scaler = train_lstm_model(df, ticker, look_back=cfg["lookback"])
                                    model, scaler = train_lstm_model(df, ticker)

                                if "Close" not in df.columns or df["Close"].isnull().all():
                                    raise ValueError("Dữ liệu không hợp lệ: thiếu hoặc toàn bộ giá đóng cửa.")
                                if len(df) < 60:
                                    raise ValueError("Không đủ dữ liệu cho LSTM (cần ít nhất 60 dòng).")

                                fc = predict_lstm(df, model, scaler, ticker, n_days)
                                fc["Date"] = pd.to_datetime(fc["Date"])

                                # Chuẩn hóa tên cột dự báo (phiên bản cũ có cột Forecast)
                                if "Forecast" in fc.columns:
                                    fc = fc.rename(columns={"Forecast": "Predicted_Close"})

                                predicted_cols = [col for col in fc.columns if col.startswith("Predicted_")]

                                return fc[["Date"] + predicted_cols], predicted_cols

                            # ==========TCN==========
                            elif choice == "TCN":
                                ticker = mdl_name.replace("tcn_", "")
                                if use_saved_model:
                                    model, scaler = load_tcn_model(ticker)
                                    if model is None or scaler is None:
                                        return None, "ModelNotTrained"
                                else:
                                    model, scaler = train_tcn_model(df, ticker)

                                if "Close" not in df.columns or df["Close"].isnull().all():
                                    raise ValueError("Dữ liệu không hợp lệ: thiếu hoặc toàn bộ giá đóng cửa.")
                                if len(df) < 60:
                                    raise ValueError("Không đủ dữ liệu cho TCN (cần ít nhất 60 dòng).")

                                fc = predict_tcn(df, model, scaler, ticker, n_days)
                                fc["Date"] = pd.to_datetime(fc["Date"])
                                predicted_cols = [col for col in fc.columns if col.startswith("Predicted_")]

                                return fc[["Date"] + predicted_cols], predicted_cols
                            
                            # elif choice == "Prophet":
                            #     # from utils.prophet_model import load_prophet_model, train_prophet, forecast_prophet
                            #     ticker = mdl_name.replace("prophet_", "")

                            #     if use_saved_model:
                            #         model = load_prophet_model(ticker)
                            #         if model is None:
                            #             return None, "ModelNotTrained"
                            #     else:
                            #         model = train_prophet(df, ticker)

                            #     fc = forecast_prophet(model, n_days)
                            #     fc = fc.rename(columns={"ds": "Date", "yhat": "Predicted_Close"})

                            #     return fc[["Date", "Predicted_Close"]], "Predicted_Close"


                            # elif choice == "ARIMA":
                            #     # from utils.arima_model import load_arima_model, train_arima, forecast_arima
                            #     ticker = mdl_name.replace("arima_", "")

                            #     if use_saved_model:
                            #         model = load_arima_model(ticker)
                            #         if model is None:
                            #             return None, "ModelNotTrained"
                            #     else:
                            #         model = train_arima(df, ticker)

                            #     fc = forecast_arima(df, n_days, use_saved_model=True, ticker=ticker)
                            #     fc = fc.rename(columns={"Predicted_Close": "Predicted_Close"})

                            #     return fc[["Date", "Predicted_Close"]], "Predicted_Close"

                            else:
                                raise ValueError(f"Mô hình không hỗ trợ: {choice}")

                        except Exception as e:
                            st.error(f"❌ Lỗi khi dự báo với model `{mdl_name}`: {str(e)}")
                            st.code(traceback.format_exc())
                            return None, "DataError"

                    
                    valid_tickers = get_valid_tickers(portfolio_data)
                    processed, failed_tickers = [], []
                    status_ph = st.empty()

                    for tk in valid_tickers:
                        df_tk = portfolio_data[portfolio_data["Ticker"] == tk]
                        if df_tk.empty or "Close" not in df_tk.columns:
                            st.warning(f"{tk}: Dữ liệu không hợp lệ hoặc thiếu")
                            failed_tickers.append(tk)
                            continue
                        #===
                        if model_choice == "LSTM":
                            model_file = f"models/lstm/lstm_model_{tk}.h5"
                            scaler_file = f"models/lstm/lstm_scaler_{tk}.pkl"  # scaler cũng cần check
                            if use_saved and (not os.path.exists(model_file) or not os.path.exists(scaler_file)):
                                st.warning(f"{tk}: Model hoặc scaler chưa huấn luyện")
                                failed_tickers.append(tk)
                                continue
                        elif model_choice == "TCN":
                            model_file = f"models/tcn/tcn_model_{tk}.h5"
                            scaler_file = f"models/tcn/tcn_scaler_{tk}.pkl"
                            if use_saved and (not os.path.exists(model_file) or not os.path.exists(scaler_file)):
                                st.warning(f"{tk}: Model hoặc scaler chưa huấn luyện")
                                failed_tickers.append(tk)
                                continue
                        # elif model_choice == "Prophet":
                        #     model_file = f"models/prophet/prophet_{tk}.pkl"
                        #     if use_saved and not os.path.exists(model_file):
                        #         st.warning(f"{tk}: Model chưa huấn luyện")
                        #         failed_tickers.append(tk)
                        #         continue
                        # elif model_choice == "ARIMA":
                        #     model_file = f"models/arima/arima_{tk}.pkl"
                        #     if use_saved and not os.path.exists(model_file):
                        #         st.warning(f"{tk}: Model chưa huấn luyện")
                        #         failed_tickers.append(tk)
                        #         continue
                        else:
                            st.warning(f"Mô hình {model_choice} không hỗ trợ.")
                            failed_tickers.append(tk)
                            continue

                        
                        status_ph.info(f"⏳ Đang dự báo {tk}…, thời gian phụ thuộc vào khối lượng dữ liệu của bạn.")

                       
                        df_tk = df_tk.reset_index(drop=True)

                        fc_df, error_type = cached_forecast(
                            model_choice, df_tk.to_json(orient="columns"),  # mặc định, không cần ghi cũng được
                            forecast_days,
                            use_saved,
                            f"{model_choice.lower()}_{tk}"
                        )

                        if fc_df is None or fc_df.empty:
                            if error_type == "ModelNotTrained":
                                st.warning(f"{tk}: Model chưa được huấn luyện hoặc không tồn tại.")
                            elif error_type == "DataError":
                                st.warning(f"{tk}: Không thể dự báo – dữ liệu đầu vào lỗi hoặc model không dùng được.")
                            else:
                                st.warning(f"{tk}: Lỗi không xác định.")
                            failed_tickers.append(tk)
                            continue

                        
                        # CHUẨN HOÁ FORECAST RESULT CHO CÁC MÔ HÌNH
                        
                        
                        fc_df = fc_df.copy()
                        rename_map = {
                            "date": "Date",
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                            "forecast": "Predicted_Close",
                            "predicted_close": "Predicted_Close"
                        }
                        fc_df.rename(columns=rename_map, inplace=True)
                        if "Date" in fc_df.columns:
                            fc_df["Date"] = pd.to_datetime(fc_df["Date"])

                        # # Với LSTM (có thể multi-output)
                        # if model_choice == "LSTM":
                        #     base_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
                        #     predicted_cols = [col for col in fc_df.columns if col.startswith("Predicted_")]
                        #     for col in base_cols:
                        #         if col not in fc_df.columns:
                        #             fc_df[col] = None
                        #     fc_df = fc_df[base_cols + predicted_cols]
                        # Với LSTM hoặc TCN (có thể multi-output)
                        if model_choice in ["LSTM", "TCN"]:
                            base_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
                            predicted_cols = [col for col in fc_df.columns if col.startswith("Predicted_")]
                            for col in base_cols:
                                if col not in fc_df.columns:
                                    fc_df[col] = None
                            fc_df = fc_df[base_cols + predicted_cols]

                        # # Với Prophet/ARIMA
                        # else:
                            
                        #     required_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Predicted_Close"]
                        #     for col in required_cols:
                        #         if col not in fc_df.columns:
                        #             fc_df[col] = None
                        #     fc_df = fc_df[required_cols]


                        # Lưu vào session_state
                        if "forecast_result" not in st.session_state:
                            st.session_state["forecast_result"] = {}
                        st.session_state["forecast_result"][tk] = fc_df

                         # --- Lưu vào database ---
                        save_forecast(tk, fc_df)
                        
                        processed.append(tk)
                   
                        with st.expander(f"\U0001F4C8 Dự báo chi tiết danh mục - {tk}"):

                            # ================== BIỂU ĐỒ PLOTLY DỰ BÁO GIÁ ==================
                            try:
                                if "Date" not in fc_df.columns or "Predicted_Close" not in fc_df.columns:
                                    raise ValueError(f"{tk}: fc_df thiếu cột 'Date' hoặc 'Predicted_Close'")

                                fc_df["Date"] = pd.to_datetime(fc_df["Date"], errors="coerce")
                                if fc_df["Date"].isna().all():
                                    raise ValueError(f"{tk}: Không thể chuyển 'Date' sang datetime")

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=fc_df["Date"],
                                    y=fc_df["Predicted_Close"],
                                    mode="lines+markers",
                                    name="Dự báo"
                                ))

                                if "Date" in df_tk.columns and "Close" in df_tk.columns:
                                    last_date = pd.to_datetime(df_tk["Date"].iloc[-1], errors="coerce")
                                    last_close = pd.to_numeric(df_tk["Close"].iloc[-1], errors="coerce")
                                    if pd.notna(last_date) and pd.notna(last_close):
                                        fig.add_trace(go.Scatter(
                                            x=[last_date],
                                            y=[last_close],
                                            mode="markers",
                                            name="Giá hiện tại",
                                            marker=dict(color='red', size=10)
                                        ))

                                fig.update_layout(
                                    title=f"{tk} – Dự báo giá",
                                    xaxis_title="Ngày",
                                    yaxis_title="Giá đóng cửa"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.error(f"⚠️ Lỗi khi vẽ biểu đồ Plotly dự báo cho {tk}: {e}")

                            # ================== BẢNG DỰ BÁO (VIỆT HÓA & ĐỊNH DẠNG) ==================
                            raw_fc = st.session_state["forecast_result"].get(tk)  # giữ nguyên bản raw (numeric)
                            if raw_fc is not None and not raw_fc.empty:
                                # 1) Chuẩn bị dataframe để hiển thị (copy từ raw, chỉ chọn cột cần thiết)
                                cols_needed = ["Date", "Predicted_Open", "Predicted_High", "Predicted_Low", "Predicted_Close", "Predicted_Volume"]
                                display_df = raw_fc[[c for c in cols_needed if c in raw_fc.columns]].copy()

                                # Việt hóa cột (chỉ đổi tên hiển thị)
                                display_df = display_df.rename(columns={
                                    "Date": "Ngày",
                                    "Predicted_Open": "Giá mở cửa",
                                    # giữ tên thống nhất với danh sách format phía dưới
                                    "Predicted_High": "Giá cao nhất",
                                    "Predicted_Low": "Giá thấp nhất",
                                    "Predicted_Close": "Giá đóng cửa",
                                    "Predicted_Volume": "Khối lượng"
                                })

                                # Định dạng cho hiển thị (không ảnh hưởng raw_fc)
                                # - Ngày
                                if "Ngày" in display_df.columns:
                                    display_df["Ngày"] = display_df["Ngày"].apply(format_date)

                                # - Giá (chỉ format những cột tồn tại)
                                cols_to_format = ["Giá mở cửa", "Giá cao nhất", "Giá thấp nhất", "Giá đóng cửa"]
                                for col in cols_to_format:
                                    if col in display_df.columns:
                                        display_df[col] = display_df[col].apply(format_currency)

                                # - Khối lượng
                                if "Khối lượng" in display_df.columns:
                                    display_df["Khối lượng"] = display_df["Khối lượng"].apply(format_volume)
                        

                                # Hiển thị bảng (đã format)
                                st.dataframe(display_df, use_container_width=True)


                            else:
                                st.warning(f"⚠️ {tk}: Dữ liệu bảng dự báo chưa sẵn sàng.")

                            # ================== BIỂU ĐỒ XU HƯỚNG GIÁ ĐÓNG CỬA & SMA ==================
                            try:
                                # Dùng raw_fc (numeric) để vẽ biểu đồ & tính SMA — KHÔNG dùng bản đã format để tránh lỗi chuyển đổi
                                if raw_fc is None or raw_fc.empty:
                                    raise ValueError("Dữ liệu dự báo trống (raw)")

                                # Chuẩn hóa Date và giá numeric
                                df_chart = raw_fc.copy()
                                df_chart["Date"] = pd.to_datetime(df_chart["Date"], errors="coerce")
                                # Đảm bảo Predicted_Close tồn tại và là numeric
                                if "Predicted_Close" not in df_chart.columns:
                                    raise ValueError("Thiếu cột 'Predicted_Close' trong dữ liệu raw để vẽ chart")
                                df_chart["Predicted_Close"] = pd.to_numeric(df_chart["Predicted_Close"], errors="coerce")

                                # Lấy số ngày forecast (dùng để tính SMA)
                                forecast_days = st.session_state.get("forecast_days_slider", 7)
                                # Nếu số dòng < forecast_days, rolling sẽ trả NaN — điều này chấp nhận được
                                df_chart["SMA"] = df_chart["Predicted_Close"].rolling(window=forecast_days).mean()

                                # Vẽ biểu đồ (dùng các giá trị numeric, không qua format string)
                                fig2 = go.Figure()
                                fig2.add_trace(go.Scatter(
                                    x=df_chart["Date"],
                                    y=df_chart["Predicted_Close"],
                                    mode="lines+markers",
                                    name="Giá đóng cửa"
                                ))
                                fig2.add_trace(go.Scatter(
                                    x=df_chart["Date"],
                                    y=df_chart["SMA"],
                                    mode="lines",
                                    name=f"SMA {forecast_days}",
                                    line=dict(dash="dash")
                                ))

                                fig2.update_layout(
                                    title=f"📈 Xu hướng giá đóng cửa & SMA ({forecast_days} kỳ) - {tk}",
                                    xaxis_title="Ngày dự báo",
                                    yaxis_title="Giá đóng cửa (VNĐ)",
                                    width=900,
                                    height=400,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom", y=-0.25,
                                        xanchor="right", x=1
                                    )
                                )

                                st.plotly_chart(fig2, use_container_width=True)

                            except Exception as e:
                                st.error(f"⚠️ Lỗi khi vẽ biểu đồ Plotly cho {tk}: {e}")

                    if processed:
                        status_ph.success("✅ Hoàn thành dự báo cho: " + ", ".join(processed))
                    elif not failed_tickers:
                        status_ph.warning("⚠️ Không có mã nào được dự báo.")

                    if failed_tickers:
                        invalid_tickers = set(st.session_state.get("invalid_tickers", []))
                        new_invalid_tickers = set(failed_tickers)
                        if new_invalid_tickers != invalid_tickers:
                            st.session_state["invalid_tickers"] = list(invalid_tickers | new_invalid_tickers)
                            if not st.session_state.get('rerun_called', False):
                                st.session_state['rerun_called'] = True
                                time.sleep(1)
                                st.rerun()

                    #==================Báo cáo dự báo==================

                    if st.button("📥 Xuất báo cáo PDF Dự báo", help="Tạo báo cáo PDF từ dữ liệu dự báo chi tiết"):
                        # import os  # đảm bảo os luôn có sẵn trong hàm
                        forecast_result = st.session_state.get("forecast_result", {})

                        if not isinstance(forecast_result, dict) or not forecast_result:
                            st.warning("⚠️ Không có dữ liệu dự báo để xuất báo cáo.")
                        else:
                            
                            today_str = st.session_state.get("today_str")
                            if not today_str:  # fallback nếu session chưa có giá trị
                                today_str = datetime.today().strftime("%d%m%Y")
                                st.session_state["today_str"] = today_str

                            valid_data = []
                            for ticker, df in forecast_result.items():
                                if df is None or df.empty:
                                    continue
                                df = df.copy()
                                df["Ticker"] = ticker
                                valid_data.append({
                                    "ticker": ticker,
                                    "forecast_df": df
                                })

                            if not valid_data:
                                st.error("❌ Không có mã nào đủ điều kiện để xuất báo cáo.")
                            else:
                                # ✅ Chuẩn hóa dữ liệu cho export_report_pdf
                                tickers = [d["ticker"] for d in valid_data]  # list ticker
                                forecast_dict = {d["ticker"]: d["forecast_df"] for d in valid_data}  # dict {ticker: DataFrame}

                                # ✅ Định dạng tên file theo yêu cầu
                                ticker_part = "_".join(tickers)
                                
                                filename = f"{ticker_part}_{today_str}_forecast.pdf"
                                output_path = os.path.join("reports", filename)
                                os.makedirs("reports", exist_ok=True)

                                try:
                                    pdf_file_path = export_report_pdf(
                                        ticker=tickers,                 # list ticker
                                        forecast_df=forecast_dict,      # dict dữ liệu
                                        output_path=output_path,
                                        include_backtest=False,
                                        include_perf=False,
                                        include_risk=False,
                                        include_rebalance=False,
                                        include_forecast=True,
                                        include_optimization=False
                                    )

                                    # ✅ Nút tải về
                                    with open(pdf_file_path, "rb") as f:
                                        st.download_button(
                                            label="📥 Tải về báo cáo Dự báo chi tiết",
                                            data=f,
                                            file_name=os.path.basename(output_path),
                                            mime="application/pdf"
                                        )

                                    st.success(f"✅ Đã xuất báo cáo dự báo: {output_path}")

                                except Exception as e:
                                    st.error(f"❌ Lỗi khi xuất báo cáo: {e}")

                    

                # --- Sub-tab 2: 📊 Thống kê mô tả theo mã ---
                with sub_tabs[1]:
                    forecast_days = st.session_state.get("forecast_days_slider", 7)

                    for tk in valid_tickers:
                        if tk not in st.session_state.get("invalid_tickers", []):
                            if "forecast_result" in st.session_state:
                                df = st.session_state["forecast_result"].get(tk)
                                if df is not None:
                                    if isinstance(df.index, pd.DatetimeIndex):
                                        df = df.reset_index().rename(columns={"index": "date"})

                                    df.columns = [col.strip().lower() for col in df.columns]

                                    if "date" not in df.columns:
                                        st.warning(f"⚠️ {tk}: Dữ liệu dự báo thiếu cột 'Ngày'. Không thể hiển thị dữ liệu cho ticker này.")
                                        continue

                                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                                    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
                                    df["date"] = df["date"].dt.strftime("%d/%m/%Y")

                                    # Chỉ giữ predicted_close và predicted_volume
                                    # df = df[["date", "predicted_close", "predicted_volume"]]
                                    if "predicted_volume" not in df.columns:
                                        df["predicted_volume"] = None
                                    df = df[["date", "predicted_close", "predicted_volume"]]


                                    df["predicted_close"] = pd.to_numeric(df["predicted_close"], errors="coerce")
                                    df["predicted_volume"] = pd.to_numeric(df["predicted_volume"], errors="coerce")

                                    st.session_state[f"forecast_{tk}"] = df

                    for tk in valid_tickers:
                        if tk in st.session_state.get("invalid_tickers", []):
                            st.info(f"ℹ️ {tk} nằm trong danh sách ticker không hợp lệ, bỏ qua.")
                            continue

                        df_forecast = st.session_state.get(f"forecast_{tk}")

                        if df_forecast is None or df_forecast.empty:
                            st.warning(f"⚠️ {tk}: Dữ liệu dự báo chưa sẵn sàng.")
                            continue

                        with st.expander(f"📊 Thống kê mô tả dự báo theo mô hình của: {tk}"):
                            df_display = df_forecast.copy()

                            # Numeric để tính thống kê
                            df_display["predicted_close_num"] = df_display["predicted_close"]
                            df_display["predicted_volume_num"] = df_display["predicted_volume"]

                            stats_close = df_display["predicted_close_num"].describe(percentiles=[.25, .5, .75])
                            stats_vol = df_display["predicted_volume_num"].describe(percentiles=[.25, .5, .75])

                            def _find_best_idx(series, target, prefer_last=False):
                                s = series.dropna()
                                if s.empty:
                                    return None
                                if prefer_last:
                                    exact = s[s == target]
                                    if not exact.empty:
                                        return exact.index[-1]
                                return (s - target).abs().idxmin()

                            df_display["Thống kê giá"] = ""
                            df_display["Thống kê khối lượng"] = ""

                            mapping_close = {
                                "min": (stats_close["min"], True),
                                "25%": (stats_close["25%"], False),
                                "50%": (stats_close["50%"], False),
                                "75%": (stats_close["75%"], False),
                                "mean": (stats_close["mean"], False),
                                "max": (stats_close["max"], True),
                            }
                            for label, (val, prefer_last) in mapping_close.items():
                                if pd.isna(val): continue
                                idx = _find_best_idx(df_display["predicted_close_num"], val, prefer_last)
                                if idx is not None:
                                    df_display.at[idx, "Thống kê giá"] = label

                            mapping_vol = {
                                "min": (stats_vol["min"], True),
                                "25%": (stats_vol["25%"], False),
                                "50%": (stats_vol["50%"], False),
                                "75%": (stats_vol["75%"], False),
                                "mean": (stats_vol["mean"], False),
                                "max": (stats_vol["max"], True),
                            }
                            for label, (val, prefer_last) in mapping_vol.items():
                                if pd.isna(val): continue
                                idx = _find_best_idx(df_display["predicted_volume_num"], val, prefer_last)
                                if idx is not None:
                                    df_display.at[idx, "Thống kê khối lượng"] = label

                            # Thêm dòng count/std
                            stats_rows = pd.DataFrame([
                                {"date": "", "predicted_close": stats_close["count"], "predicted_volume": "", "Thống kê giá": "count", "Thống kê khối lượng": ""},
                                {"date": "", "predicted_close": stats_close["std"], "predicted_volume": "", "Thống kê giá": "std", "Thống kê khối lượng": ""},
                                {"date": "", "predicted_close": "", "predicted_volume": stats_vol["count"], "Thống kê giá": "", "Thống kê khối lượng": "count"},
                                {"date": "", "predicted_close": "", "predicted_volume": stats_vol["std"], "Thống kê giá": "", "Thống kê khối lượng": "std"},
                            ])
                            df_combined_display = pd.concat([df_display, stats_rows], ignore_index=True)

                            # Format chuẩn
                            df_combined_display["Ngày dự báo"] = df_combined_display["date"].apply(format_date)
                            df_combined_display["Giá dự báo"] = df_combined_display["predicted_close"].apply(format_currency)
                            df_combined_display["Khối lượng dự báo"] = df_combined_display["predicted_volume"].apply(format_volume)

                            # Hiển thị dataframe
                            st.write("#### 📅 Dữ liệu dự báo")
                            
                            st.dataframe(
                                df_combined_display[[
                                    "Ngày dự báo",
                                    "Giá dự báo",
                                    "Thống kê giá",
                                    "Khối lượng dự báo",
                                    "Thống kê khối lượng"
                                ]],
                                use_container_width=True  # ⚡ bổ sung tham số tại đây
                            )

                            # -----------------------------
                            # Biểu đồ biến động Giá & Volume trên cùng chart
                            # -----------------------------
                            df_plot = df_forecast.copy()
                            df_plot["date"] = pd.to_datetime(df_plot["date"], format="%d/%m/%Y", errors="coerce")
                            df_plot = df_plot.dropna(subset=["date"]).sort_values("date")

                            st.write("#### 📈 Biểu đồ biến động Giá & Khối lượng dự báo")

                            

                            fig, ax1 = plt.subplots(figsize=(10, 4))
                            ax1.plot(df_plot["date"], df_plot["predicted_close"], color="blue", marker="o", label="Giá dự báo")
                            ax1.set_xlabel("Ngày")
                            ax1.set_ylabel("Giá dự báo", color="blue")
                            ax1.tick_params(axis="y", labelcolor="blue")

                            ax2 = ax1.twinx()
                            ax2.plot(df_plot["date"], df_plot["predicted_volume"], color="orange", marker="x", label="Khối lượng dự báo")
                            ax2.set_ylabel("Khối lượng dự báo", color="orange")
                            ax2.tick_params(axis="y", labelcolor="orange")

                            fig.autofmt_xdate()
                            fig.tight_layout()
                            st.pyplot(fig)

                   
                
                # ================== DỰ BÁO HÀNH VI NHÀ ĐẦU TƯ ==================
                with sub_tabs[2]:

                    forecast_days = st.session_state.get("forecast_days_slider", 7)

                    for tk in valid_tickers:
                        if tk not in st.session_state.get("invalid_tickers", []):
                            if "forecast_result" in st.session_state:
                                df = st.session_state["forecast_result"].get(tk)
                                if df is not None:
                                    # Chuẩn hoá index/cột date
                                    if isinstance(df.index, pd.DatetimeIndex):
                                        df = df.reset_index().rename(columns={"index": "date"})

                                    df.columns = [col.strip().lower() for col in df.columns]

                                    if "date" not in df.columns:
                                        st.warning(f"⚠️ {tk}: DataFrame không có cột 'date'.")
                                        continue

                                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                                    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")

                                    # Bắt buộc phải có predicted_close
                                    if "predicted_close" not in df.columns:
                                        st.warning(f"⚠️ {tk}: Thiếu cột 'predicted_close'.")
                                        continue

                                    # Đảm bảo đủ cột
                                    for col in ["predicted_open", "predicted_high", "predicted_low", "predicted_volume"]:
                                        if col not in df.columns:
                                            df[col] = None

                                    # Giữ cột cần thiết
                                    df = df[[
                                        "date", "predicted_open", "predicted_high",
                                        "predicted_low", "predicted_close", "predicted_volume"
                                    ]]

                                    # ------------------ ĐỊNH DẠNG ------------------
                                    df["date"] = df["date"].apply(format_date)
                                    for col in ["predicted_open", "predicted_high", "predicted_low", "predicted_close"]:
                                        df[col] = df[col].apply(format_currency)
                                    df["predicted_volume"] = df["predicted_volume"].apply(format_volume)

                                    # Đổi tên cột
                                    df = df.rename(columns={
                                        "date": "Ngày",
                                        "predicted_open": "Giá mở cửa",
                                        "predicted_high": "Cao nhất",
                                        "predicted_low": "Thấp nhất",
                                        "predicted_close": "Giá đóng cửa",
                                        "predicted_volume": "Khối lượng"
                                    })

                                    # ====== Thêm phân loại hành vi nhà đầu tư ======
                                    df_numeric = st.session_state["forecast_result"][tk].copy()
                                    df_numeric["predicted_close"] = pd.to_numeric(df_numeric["predicted_close"], errors="coerce")
                                    df_numeric = df_numeric.sort_values("date")
                                    df_numeric["return"] = df_numeric["predicted_close"].pct_change()

                                    def classify_behavior(r):
                                        if pd.isna(r): return "Lưỡng lự / quan sát"
                                        if r > 0.02: return "Hưng phấn (mua mạnh)"
                                        if r > 0.005: return "Lạc quan thận trọng"
                                        if r < -0.02: return "Hoảng loạn (bán tháo)"
                                        if r < -0.005: return "Bi quan nhẹ"
                                        return "Lưỡng lự / quan sát"

                                    df_numeric["Hành vi NĐT"] = df_numeric["return"].apply(classify_behavior)
                                    df["Hành vi NĐT"] = df_numeric["Hành vi NĐT"].values

                                    # ====== Lưu vào session_state riêng cho tab này ======
                                    st.session_state[f"forecast_behavior_{tk}"] = df

                    # ================== HIỂN THỊ ==================
                    for tk in valid_tickers:
                        if tk in st.session_state.get("invalid_tickers", []):
                            continue

                        df_forecast = st.session_state.get(f"forecast_behavior_{tk}")
                        if df_forecast is None or df_forecast.empty:
                            st.warning(f"⚠️ {tk}: Chưa có dữ liệu dự báo.")
                            continue

                        with st.expander(f"🪙 Dự báo hành vi NĐT & giá của: {tk}"):
                            forecast_days = st.session_state.get("forecast_days_slider", 7)

                            # Hiển thị bảng
                            st.dataframe(df_forecast, use_container_width=True)

                            # Biểu đồ plotly
                            try:
                                df_chart = df_forecast.copy()
                                df_chart["Ngày"] = pd.to_datetime(df_chart["Ngày"], format="%d/%m/%Y", errors="coerce")
                                df_chart["Giá đóng cửa"] = df_chart["Giá đóng cửa"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                                df_chart["Giá đóng cửa"] = pd.to_numeric(df_chart["Giá đóng cửa"], errors="coerce")

                                # SMA
                                df_chart["SMA"] = df_chart["Giá đóng cửa"].rolling(window=forecast_days).mean()

                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=df_chart["Ngày"], y=df_chart["Giá đóng cửa"],
                                    mode="lines+markers", name="Giá đóng cửa",
                                    line=dict(color="orange", width=2)
                                ))

                                fig.add_trace(go.Scatter(
                                    x=df_chart["Ngày"], y=df_chart["SMA"],
                                    mode="lines", name=f"SMA {forecast_days} kỳ",
                                    line=dict(color="blue", width=2, dash="dot")
                                ))

                                fig.update_layout(
                                    title=f"📈 Xu hướng giá & SMA ({forecast_days} kỳ) - {tk}",
                                    xaxis_title="Ngày",
                                    yaxis_title="Giá (VNĐ)",
                                    legend=dict(x=0.95, y=0.05, xanchor="right", yanchor="bottom"),
                                    height=400
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # ====== Tóm tắt hành vi ======
                                summary = df_forecast["Hành vi NĐT"].value_counts()
                                colors = {
                                    "Hưng phấn (mua mạnh)": "green",
                                    "Hoảng loạn (bán tháo)": "red",
                                    "Lạc quan thận trọng": "blue",
                                    "Bi quan nhẹ": "orange",
                                    "Lưỡng lự / quan sát": "gray"
                                }

                                st.markdown("### 📌 Tóm tắt hành vi NĐT")
                                for behavior, count in summary.items():
                                    color = colors.get(behavior, "black")
                                    st.markdown(
                                        f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                                        f"<div style='width:15px;height:15px;background-color:{color};margin-right:8px;border-radius:3px;'></div>"
                                        f"<span>{behavior}: {count} ngày</span>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )

                            except Exception as e:
                                st.error(f"⚠️ Lỗi khi vẽ biểu đồ cho {tk}: {e}")

                    
                # ===============Giao dịch thị trường thực tế(khớp lệnh)============
                
                with sub_tabs[3]:
                    source = "yf"
                    st.markdown("### 📘 Giao dịch khớp lệnh thực tế (tham khảo)")

                    tickers = st.session_state.get("tickers", [])
                    forecast_days = st.session_state.get("forecast_days_slider", 7)

                    if not tickers:
                        st.warning("⚠️ Bạn chưa nhập mã nào trong sidebar.")
                    else:
                        def format_change(x):
                            try:
                                if pd.isna(x):
                                    return ""
                                s = f"{x:,.0f}"                      # "2,500" or "-2,500"
                                s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # "2.500" or "-2.500"
                                if x > 0:
                                    return f"+{s}"
                                return s
                            except:
                                return x
                        for ticker in tickers:
                            try:
                                df = load_data_dl(ticker, source=source)
                                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
                                df = df.dropna(subset=["Date", "Close", "Volume"])
                                df = df.sort_values("Date", ascending=False).reset_index(drop=True)

                                # Tính biến động & xu hướng
                                df["Mức thay đổi"] = df["Close"].diff(-1)
                                df["Xu hướng"] = df["Mức thay đổi"].apply(
                                    lambda x: "Tăng" if x > 0 else ("Giảm" if x < 0 else "Ổn định")
                                )

                                # SMA theo forecast_days
                                df["SMA"] = df["Close"].rolling(window=forecast_days).mean()

                                # Chuẩn hóa cột ngày
                                df["Ngày"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%d/%m/%Y")

                                # ------------------ ĐỊNH DẠNG ------------------
                                df["Giá đóng cửa"] = df["Close"].apply(format_currency)
                                df["Khối lượng"] = df["Volume"].apply(format_volume)
                                df["Mức thay đổi"] = df["Mức thay đổi"].apply(format_change)

                                # Chỉ giữ forecast_days dòng
                                df_display = df[["Ngày", "Giá đóng cửa", "Khối lượng", "Mức thay đổi", "Xu hướng"]].head(forecast_days).copy()

                                # Style cho cột Xu hướng
                                def highlight_trend(val):
                                    if val == "Tăng":
                                        return "color: green; font-weight: bold;"
                                    elif val == "Giảm":
                                        return "color: red; font-weight: bold;"
                                    elif val == "Ổn định":
                                        return "color: gray; font-weight: bold;"
                                    return ""

                                styled_df = df_display.style.applymap(highlight_trend, subset=["Xu hướng"])

                                with st.expander(f"📊 Khớp lệnh thực tế của {ticker} (tham khảo)"):
                                    st.dataframe(styled_df, use_container_width=True, height=forecast_days * 40)

                                    # ===== Biểu đồ =====
                                    try:
                                        df_chart = df.head(forecast_days).copy()
                                        df_chart["Ngày"] = pd.to_datetime(df_chart["Date"], errors="coerce")
                                        df_chart["Giá đóng cửa"] = pd.to_numeric(df_chart["Close"], errors="coerce")
                                        df_chart["Khối lượng"] = pd.to_numeric(df_chart["Volume"], errors="coerce")

                                        line_close = alt.Chart(df_chart).mark_line(point=True, color="blue").encode(
                                            x=alt.X("Ngày:T", title="Ngày"),
                                            y=alt.Y("Giá đóng cửa:Q", title="Giá đóng cửa (VNĐ)", axis=alt.Axis(format=",.3f")),
                                            tooltip=[
                                                alt.Tooltip("Ngày:T", title="Ngày"),
                                                alt.Tooltip("Giá đóng cửa:Q", title="Giá đóng cửa", format=",.3f")
                                            ]
                                        )

                                        line_volume = alt.Chart(df_chart).mark_line(point=True, color="orange").encode(
                                            x="Ngày:T",
                                            y=alt.Y("Khối lượng:Q", title="Khối lượng"),
                                            tooltip=[
                                                alt.Tooltip("Ngày:T", title="Ngày"),
                                                alt.Tooltip("Khối lượng:Q", title="Khối lượng", format=",")
                                            ]
                                        )

                                        sma_chart = alt.Chart(df_chart).mark_line(strokeDash=[4, 4], color="green").encode(
                                            x="Ngày:T",
                                            y=alt.Y("SMA:Q", title=f"SMA {forecast_days}"),
                                            tooltip=[alt.Tooltip("Ngày:T", title="Ngày"), alt.Tooltip("SMA:Q", title="SMA", format=",.3f")]
                                        )

                                        chart = (line_close + line_volume + sma_chart).properties(
                                            width="container",
                                            height=350,
                                            title=f"📈 Xu hướng giá & SMA ({forecast_days} kỳ) - {ticker}"
                                        )

                                        st.altair_chart(chart, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"⚠️ Lỗi khi vẽ biểu đồ cho {ticker}: {e}")

                            except Exception as e:
                                st.warning(f"⚠️ Không đọc được dữ liệu của {ticker}: {e}")

                    
        # ================TAB PHÂN TÍCH RỦI RO======================

        if "risk" in tab_map:
            with tabs[tab_map["risk"]]:
                
                invalid = st.session_state.get("invalid_tickers", [])
                valid_tk = [t for t in tickers if t not in invalid]

                risk_out: Dict[str, pd.DataFrame] = {}

                def _risk(df: pd.DataFrame, tk: str):
                    try:
                        risk_out[tk] = risk_metrics.calculate_risk_metrics(df)
                    except Exception as exc:
                        st.warning(f"{tk}: {exc}")

                # === TỰ ĐỘNG TÍNH KHI MỞ TAB NẾU CHƯA CÓ DỮ LIỆU ===
                if st.session_state.get("risk_result") is None:
                    for i in range(0, len(valid_tk), BATCH_SIZE if MEM_SAFE_MODE else len(valid_tk)):
                        batch = valid_tk[i : i + (BATCH_SIZE if MEM_SAFE_MODE else len(valid_tk))]
                        for tk in batch:
                            df_t = portfolio_data[portfolio_data["Ticker"] == tk]
                            _risk(df_t, tk)
                        gc.collect()
                        if tf:
                            tf.keras.backend.clear_session()
                    st.session_state["risk_result"] = risk_out

                # === HIỂN THỊ KẾT QUẢ ===
                if st.session_state.get("risk_result"):
                    for tk, df in st.session_state["risk_result"].items():
                        with st.expander(f"📉 Phân tích rủi ro theo mã - {tk}"):
                            st.plotly_chart(
                                px.bar(df, x="Metric", y="Value", text_auto=True),
                                use_container_width=True
                            )
                            st.dataframe(df)

                # === NÚT RERUN / TÍNH LẠI ===
                if st.button("🔍 Tính lại Risk Metrics"):
                    st.session_state["risk_result"] = None
                    st.rerun()  # reload tab để tự động tính lại


                # === NÚT XUẤT BÁO CÁO PDF ===
                st.markdown("---")

                if st.button("📥 Xuất báo cáo Phân tích Rủi ro"):
                    risk_result = st.session_state.get("risk_result", {})
                    if not risk_result:
                        st.warning("⚠️ Không có dữ liệu rủi ro để xuất.")
                    else:
                        today_str = st.session_state.get("today_str")
                        if not today_str:
                            today_str = datetime.today().strftime("%d%m%Y")
                            st.session_state["today_str"] = today_str

                        tickers = list(risk_result.keys())
                        ticker_part = "_".join(tickers)
                        filename = f"{ticker_part}_{today_str}_risk.pdf"
                        output_path = os.path.join("reports", filename)
                        os.makedirs("reports", exist_ok=True)

                        try:
                            pdf_file_path = export_report_pdf(
                                ticker=tickers,
                                bt_df=None,
                                perf_df=None,
                                risk_df=risk_result,   # ✅ đưa vào đây
                                weights_df=None,
                                output_path=output_path,
                                include_backtest=False,
                                include_perf=False,
                                include_risk=True,     # ✅ bật risk section
                                include_rebalance=False,
                                include_forecast=False,
                                include_optimization=False
                            )

                            with safe_open_pdf(pdf_file_path) as f:
                                st.download_button(
                                    label="📥 Tải về báo cáo",
                                    data=f,
                                    file_name=filename,
                                    mime="application/pdf"
                                )
                                st.success(f"✅ Đã xuất báo cáo: {output_path}")

                        except Exception as e:
                            st.error(f"❌ Lỗi khi xuất báo cáo: {e}")


                    
        # === TAB Backtest ==========================================================

        if "backtest_perf" in tab_map:
            with tabs[tab_map["backtest_perf"]]:

                st.subheader("🔄 Backtest & Hiệu suất")
              
                tab1, tab2 = st.tabs(["📊 Dữ liệu lịch sử", "🧪 Công cụ Backtest"])

         
                
                with tab1:

                    st.write("📊 Hành vi & Backtest quá khứ nhà đầu tư:")
                    def show_behavior_and_backtest(tk: str, df_backtest: pd.DataFrame):
                        # import plotly.graph_objects as go

                        with st.expander(f"📊 Hành vi & Backtest – {tk}"):

                            df = df_backtest.copy()
                            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                            df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)
                            df["Ngày"] = df["Date"].dt.strftime("%d/%m/%Y")

                            # --- Rút gọn tên cột Predicted_* ---
                            rename_pred_cols = {
                                "Predicted_Close": "PC",
                                "Predicted_Open": "PO",
                                "Predicted_High": "PH",
                                "Predicted_Low": "PL",
                                "Predicted_Volume": "PV"
                            }
                            df.rename(columns=rename_pred_cols, inplace=True)

                            # --- Tính các chỉ số hành vi ---
                            if all(c in df.columns for c in ["High","Low"]):
                                df["Độ biến động"] = df["High"] - df["Low"]
                            if all(c in df.columns for c in ["Close","Open","High","Low"]):
                                df["Tín hiệu xu hướng (float)"] = (df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-8)

                            # --- Format các cột tiền và volume ---
                            for col in ["Open","High","Low","Close","PO","PH","PL","PC"]:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors="coerce")
                                    df[col] = df[col].apply(format_currency)
                            if "PV" in df.columns:
                                df["PV"] = pd.to_numeric(df["PV"], errors="coerce").round(0)
                                df["PV"] = df["PV"].apply(format_volume)

                            # Format Độ biến động và Tín hiệu xu hướng cho bảng
                            if "Độ biến động" in df.columns:
                                df["Độ biến động"] = pd.to_numeric(df["Độ biến động"], errors="coerce")
                                df["Độ biến động"] = df["Độ biến động"].apply(format_currency)
                            if "Tín hiệu xu hướng (float)" in df.columns:
                                df["Tín hiệu xu hướng"] = df["Tín hiệu xu hướng (float)"].apply(
                                    lambda x: f"{x*100:.2f} %" if pd.notnull(x) else ""
                                )

                            # --- Hiển thị bảng hành vi ---
                            st.subheader("📈 Hành vi nhà đầu tư")
                            display_cols = ["Ngày","Open","High","Low","Close","Volume","PO","PH","PL","PC","PV","Độ biến động","Tín hiệu xu hướng"]
                            st.dataframe(df[display_cols], use_container_width=True)

                            # --- Biểu đồ hành vi (Volume + Tín hiệu xu hướng) ---
                            try:
                                fig = go.Figure()

                                # Volume
                                if "PV" in df.columns:
                                    y_volume = pd.to_numeric(df["PV"].str.replace(".",""), errors="coerce")
                                    fig.add_trace(go.Scatter(
                                        x=df["Date"][~y_volume.isna()],
                                        y=y_volume.dropna(),
                                        mode="lines+markers",
                                        name="Khối lượng"
                                    ))

                                # Tín hiệu xu hướng
                                if "Tín hiệu xu hướng (float)" in df.columns:
                                    y_signal = df["Tín hiệu xu hướng (float)"] * 100
                                    fig.add_trace(go.Scatter(
                                        x=df["Date"][~y_signal.isna()],
                                        y=y_signal.dropna(),
                                        mode="lines+markers",
                                        name="Tín hiệu xu hướng (%)"
                                    ))

                                fig.update_layout(
                                    title=f"Hành vi nhà đầu tư - {tk}",
                                    xaxis_title="Ngày",
                                    yaxis_title="Giá trị",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Lỗi vẽ biểu đồ hành vi: {e}")

                            # --- Bảng Backtest ---
                            df_bt = df.dropna(subset=["PC","Close"]).copy()
                            if not df_bt.empty:
                                df_bt["Close"] = pd.to_numeric(df_bt["Close"].str.replace(".","").str.replace(",","."))
                                df_bt["PC"] = pd.to_numeric(df_bt["PC"].str.replace(".","").str.replace(",","."))
                                df_bt["Sai số"] = df_bt["PC"] - df_bt["Close"]
                                df_bt["Sai số %"] = np.abs(df_bt["Sai số"] / df_bt["Close"]) * 100

                                df_bt = df_bt.sort_values("Date", ascending=False).reset_index(drop=True)
                                df_bt["Ngày"] = df_bt["Date"].dt.strftime("%d/%m/%Y")

                                df_bt["Giá thực tế"] = df_bt["Close"].apply(format_currency)
                                df_bt["Giá dự báo"] = df_bt["PC"].apply(format_currency)
                                df_bt["Sai số"] = df_bt["Sai số"].apply(format_currency)
                                df_bt["Sai số %"] = df_bt["Sai số %"].apply(lambda x: f"{x:.2f} %")

                                mae = np.mean(np.abs(df_bt["PC"] - df_bt["Close"]))
                                rmse = np.sqrt(np.mean((df_bt["PC"] - df_bt["Close"]) ** 2))
                                mape = np.mean(np.abs((df_bt["PC"] - df_bt["Close"]) / df_bt["Close"])) * 100

                                st.subheader("📉 Backtest dự báo")
                                st.markdown(f"- **MAE**: {format_currency(mae)}  \n- **RMSE**: {format_currency(rmse)}  \n- **MAPE**: {mape:.2f} %")
                                st.dataframe(df_bt[["Ngày","Giá thực tế","Giá dự báo","Sai số","Sai số %"]], use_container_width=True)

                                fig2 = go.Figure()
                                fig2.add_trace(go.Scatter(x=df_bt["Date"], y=df_bt["Close"], mode="lines+markers", name="Giá thực tế"))
                                fig2.add_trace(go.Scatter(x=df_bt["Date"], y=df_bt["PC"], mode="lines+markers", name="Giá dự báo"))
                                fig2.add_trace(go.Bar(x=df_bt["Date"], y=df_bt["PC"] - df_bt["Close"], name="Sai số", opacity=0.4))
                                fig2.update_layout(title=f"So sánh giá thực tế vs giá dự báo – {tk}",
                                                xaxis_title="Ngày", yaxis_title="Giá")
                                st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.warning("⚠️ Không có đủ dữ liệu để backtest.")

                    # --- Loop qua ticker ---
                    for tk in tickers:
                        df_backtest = db_manager.load_data(tk)
                        if not df_backtest.empty:
                            df_backtest["Date"] = pd.to_datetime(df_backtest["Date"], errors="coerce")
                            today = pd.Timestamp.today().normalize()
                            cutoff = today - pd.Timedelta(days=180)
                            df_backtest = df_backtest[(df_backtest["Date"] < today) & (df_backtest["Date"] >= cutoff)]
                            show_behavior_and_backtest(tk, df_backtest)
                        else:
                            st.warning(f"⚠️ {tk}: Không có dữ liệu backtest.")


                # =========Backtest============
                with tab2:
                    
                    days = st.slider(
                        "Số ngày backtest", 
                        30, 365, 180, 
                        key="bt_days", 
                        help="Chọn khoảng thời gian (tính từ ngày gần nhất) để thực hiện backtest chiến lược"
                    )
                    selected_strategies = st.multiselect(
                        "Chiến lược",
                        ["buy_and_hold", "ma_crossover", "momentum", "bollinger", "rsi"],
                        default=["buy_and_hold"],
                        key="bt_strategies",
                        help="Chọn một hoặc nhiều chiến lược để đánh giá hiệu suất"
                    )
                    threshold = st.number_input(
                        "Ngưỡng (momentum)", 
                        0.0, 1.0, 0.05, 0.01, 
                        key="bt_threshold", 
                        help="Ngưỡng xác định mức độ động lượng (áp dụng cho chiến lược momentum)"
                    )

                    invalid = st.session_state.get("invalid_tickers", [])
                    valid = [t for t in tickers if t not in invalid]

                    BATCH_SIZE = 5  # Có thể điều chỉnh theo khả năng máy
                    MEM_SAFE_MODE = True

                    # === TỰ ĐỘNG BACKTEST NẾU CHƯA CÓ DỮ LIỆU NHƯNG ĐỦ ĐIỀU KIỆN ===
                    auto_bt_ready = valid and selected_strategies and not st.session_state.get("backtest_result")
                    if auto_bt_ready:
                        last_bt = st.session_state.get("backtest_result", {})
                        last_perf = st.session_state.get("performance_result", {})
                        bt_result, perf_result, failed_bt = dict(last_bt), dict(last_perf), []

                        portfolio_data["Ticker"] = portfolio_data["Ticker"].astype(str).str.upper()
                        portfolio_data["Date"] = pd.to_datetime(portfolio_data["Date"], errors="coerce")
                        portfolio_data = portfolio_data.dropna(subset=["Date", "Close", "Ticker"])
                        if "Predicted_Close" not in portfolio_data.columns:
                            portfolio_data["Predicted_Close"] = pd.NA

                        valid = [t.upper() for t in valid if t.upper() in portfolio_data["Ticker"].unique()]

                        # --- Backtest theo batch để tiết kiệm bộ nhớ ---
                        for i in range(0, len(valid), BATCH_SIZE if MEM_SAFE_MODE else len(valid)):
                            batch = valid[i : i + (BATCH_SIZE if MEM_SAFE_MODE else len(valid))]
                            for t in batch:
                                for strategy in selected_strategies:
                                    df_t = portfolio_data[portfolio_data["Ticker"] == t].copy()
                                    key = (t, strategy)
                                    if key in last_bt and set(df_t["Date"]) == set(last_bt[key]["Date"]):
                                        continue
                                    try:
                                        bt = backtest.backtest_strategy(df_t, forecast_days=days, strategy=strategy, threshold=threshold)
                                        perf = performance.calculate_performance_metrics(bt)
                                        bt_result[key] = bt
                                        perf_result[key] = perf
                                    except Exception:
                                        failed_bt.append(t)
                            # Giải phóng bộ nhớ
                            gc.collect()
                            if 'tf' in globals() and tf:
                                tf.keras.backend.clear_session()

                        st.session_state["backtest_result"] = bt_result
                        st.session_state["performance_result"] = perf_result
                        st.session_state["invalid_tickers"] = list(set(st.session_state.get("invalid_tickers", [])) | set(failed_bt))
                        st.rerun()  # reload tab để hiển thị kết quả tự động

                    # === Lưu kết quả session để dùng báo cáo tổng hợp ===
                    bt_result = st.session_state.get("backtest_result", {})
                    perf_result = st.session_state.get("performance_result", {})
                    if bt_result:
                        st.session_state["bt_results"] = bt_result
                    if perf_result:
                        st.session_state["perf_results"] = perf_result

                    # === Hiển thị kết quả ===
                    if bt_result:
                        for (t, strategy), df in bt_result.items():
                            ticker = t.upper()
                            df["Ticker"] = ticker
                            df["Date"] = pd.to_datetime(df["Date"])
                            cutoff = datetime.now() - timedelta(days=180)
                            df = df[df["Date"] >= cutoff]

                            if "Predicted_Close" in df.columns:
                                df = df.drop(columns=["Predicted_Close"])

                            df_forecast = load_forecast(ticker)
                            if not df_forecast.empty:
                                df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
                                df = df.merge(
                                    df_forecast[["Date", "Ticker", "Predicted_Close"]],
                                    on=["Date", "Ticker"],
                                    how="left"
                                )

                            df_display = df.sort_values("Date", ascending=False).copy()
                            df_display["Date"] = df_display["Date"].dt.strftime("%d/%m/%Y")

                            viet_cols = {
                                "Date": "Ngày",
                                "Close": "Giá đóng cửa",
                                "Predicted_Close": "Dự báo đóng cửa",
                                "Portfolio_Value": "Giá trị danh mục",
                                "Ticker": "Mã CK"
                            }
                            df_display = df_display.rename(columns={k: viet_cols.get(k, k) for k in df_display.columns})

                            # Bỏ cột 'Ticker' / 'Mã CK' nếu có
                            drop_cols = ["Ticker", "Mã CK"]
                            df_display = df_display.drop(columns=[c for c in drop_cols if c in df_display.columns], errors="ignore")

                            # Định dạng cột Giá đóng cửa và Dự báo đóng cửa
                            for col in ["Giá đóng cửa", "Dự báo đóng cửa"]:
                                if col in df_display.columns:
                                    df_display[col] = df_display[col].apply(format_currency)

                            fig = px.line(
                                df.sort_values("Date", ascending=True),
                                x="Date",
                                y="Portfolio_Value",
                                title=f"Giá trị danh mục: {t} ({strategy})"
                            )
                            fig.update_layout(hovermode="x unified")

                            with st.expander(f"📈 {t} ({strategy}) - Portfolio Value", expanded=False):
                                st.plotly_chart(fig, use_container_width=True)
                                st.dataframe(df_display, use_container_width=True)

                    # === NÚT CHẠY BACKTEST THỦ CÔNG (ĐƯA XUỐNG DƯỚI EXPANDER) ===
                    if st.button("🧪 Chạy lại Backtest", help="Thực hiện kiểm tra hiệu suất trên các chiến lược đã chọn"):
                        last_bt = st.session_state.get("backtest_result", {})
                        last_perf = st.session_state.get("performance_result", {})
                        bt_result, perf_result, failed_bt = dict(last_bt), dict(last_perf), []

                        portfolio_data["Ticker"] = portfolio_data["Ticker"].astype(str).str.upper()
                        portfolio_data["Date"] = pd.to_datetime(portfolio_data["Date"], errors="coerce")
                        portfolio_data = portfolio_data.dropna(subset=["Date", "Close", "Ticker"])
                        if "Predicted_Close" not in portfolio_data.columns:
                            portfolio_data["Predicted_Close"] = pd.NA

                        valid = [t.upper() for t in valid if t.upper() in portfolio_data["Ticker"].unique()]

                        # --- Backtest theo batch ---
                        for i in range(0, len(valid), BATCH_SIZE if MEM_SAFE_MODE else len(valid)):
                            batch = valid[i : i + (BATCH_SIZE if MEM_SAFE_MODE else len(valid))]
                            for t in batch:
                                for strategy in selected_strategies:
                                    df_t = portfolio_data[portfolio_data["Ticker"] == t].copy()
                                    key = (t, strategy)
                                    if key in last_bt and set(df_t["Date"]) == set(last_bt[key]["Date"]):
                                        continue
                                    try:
                                        bt = backtest.backtest_strategy(df_t, forecast_days=days, strategy=strategy, threshold=threshold)
                                        perf = performance.calculate_performance_metrics(bt)
                                        bt_result[key] = bt
                                        perf_result[key] = perf
                                    except Exception as e:
                                        st.error(f"❌ Lỗi với mã {t} ({strategy}): {e}")
                                        failed_bt.append(t)
                            gc.collect()
                            if 'tf' in globals() and tf:
                                tf.keras.backend.clear_session()

                        st.session_state["backtest_result"] = bt_result
                        st.session_state["performance_result"] = perf_result
                        st.session_state["invalid_tickers"] = list(set(invalid) | set(failed_bt))

                        if bt_result:
                            st.success("✅ Đã backtest: " + ", ".join([f"{k[0]} ({k[1]})" for k in bt_result.keys()]))
                        if failed_bt:
                            st.warning("⚠️ Lỗi hoặc thiếu dữ liệu: " + ", ".join(set(failed_bt)))
                        if not bt_result:
                            st.warning("⚠️ Không có mã nào được backtest thành công.")


                             


                    if perf_result:
                        st.subheader("📌 Danh sách mã đạt yêu cầu")
                        table = []
                        for (t, strategy), df in perf_result.items():
                            row = {"Ticker": t, "Strategy": strategy}
                            for _, r in df.iterrows():
                                row[r["Metric"]] = r["Value"]
                            table.append(row)

                        table_df = pd.DataFrame(table)
                        st.session_state["filtered_tickers"] = table_df["Ticker"].tolist()

                        selected_metrics = st.multiselect(
                            "Lọc theo chỉ số",
                            table_df.columns.drop(["Ticker", "Strategy"]),
                            default=["Sharpe Ratio", "Max Drawdown"],
                            key="perf_filter_metrics",
                            help="Lọc cổ phiếu theo các chỉ số hiệu suất cụ thể"
                        )

                        for m in selected_metrics:
                            col1, col2 = st.columns(2)
                            min_v = col1.number_input(f"Tối thiểu {m}", value=float(table_df[m].min()), key=f"min_{m}")
                            max_v = col2.number_input(f"Tối đa {m}", value=float(table_df[m].max()), key=f"max_{m}")
                            table_df = table_df[(table_df[m] >= min_v) & (table_df[m] <= max_v)]

                        sort_metric = st.selectbox("🔽 Sắp xếp theo", ["Không sắp xếp"] + selected_metrics, help="Chọn chỉ số để sắp xếp danh sách")
                        if sort_metric != "Không sắp xếp":
                            ascending = st.checkbox("⬆️ Sắp xếp tăng dần", value=False)
                            table_df = table_df.sort_values(by=sort_metric, ascending=ascending)

                        st.dataframe(table_df, use_container_width=True)
                        st.caption("📌 Lọc theo hiệu suất và dữ liệu gần nhất")


                    # ================BÁO CÁO BACKTEST VÀ HIỆU XUẤT=================
                    st.markdown("---")
                    st.subheader("📄 Báo cáo Backtest")

                    
                    if st.button("📥 Xuất báo cáo PDF Backtest", help="Tạo báo cáo PDF từ dữ liệu đã backtest"):
                        # import os
                        bt_result = st.session_state.get("backtest_result", {})
                        performance_result = st.session_state.get("performance_result", {})
                        risk_result = st.session_state.get("risk_result", {})

                        ok_bt = isinstance(bt_result, dict) and len(bt_result) > 0
                        ok_perf = isinstance(performance_result, dict) and len(performance_result) > 0
                        ok_risk = isinstance(risk_result, dict) and len(risk_result) > 0

                        if not (ok_bt and ok_perf and ok_risk):
                            st.warning("⚠️ Cần chạy Backtest, Hiệu suất và Rủi ro trước.")
                        else:
                            today_str = st.session_state.get("today_str")
                            if not today_str:  # fallback nếu session chưa có giá trị
                                today_str = datetime.today().strftime("%d%m%Y")
                                st.session_state["today_str"] = today_str

                            valid_data = []

                            for (ticker, strategy), bt_df in bt_result.items():
                                if bt_df is None or bt_df.empty:
                                    continue

                                # Lấy performance
                                perf_df = performance_result.get((ticker, strategy))
                                if perf_df is None or perf_df.empty:
                                    perf_df = performance_result.get(ticker)
                                if perf_df is None or perf_df.empty:
                                    continue

                                # Lấy risk
                                risk_df = risk_result.get((ticker, strategy))
                                if risk_df is None or risk_df.empty:
                                    risk_df = risk_result.get(ticker)
                                if risk_df is None or risk_df.empty:
                                    continue

                                # ✅ Gắn thêm tên ticker vào từng bảng
                                bt_df = bt_df.copy()
                                bt_df["Ticker"] = ticker
                                perf_df = perf_df.copy()

                                perf_df["Ticker"] = ticker
                                # Bỏ cột Ticker khi xuất
                                perf_df = perf_df.drop(columns=["Ticker"], errors="ignore")

                                risk_df = risk_df.copy()
                                risk_df["Ticker"] = ticker

                                valid_data.append({
                                    "ticker": ticker,
                                    "bt_df": bt_df,
                                    "perf_df": perf_df,
                                    "risk_df": risk_df
                                })

                            if not valid_data:
                                st.error("❌ Không có mã nào đủ điều kiện để xuất báo cáo.")
                            else:
                                # ✅ Danh sách ticker
                                tickers = [d["ticker"] for d in valid_data]

                                # ✅ Đặt tên file theo quy tắc
                                ticker_part = "_".join(tickers)
                                filename = f"{ticker_part}_{today_str}_backtest.pdf"
                                output_path = os.path.join("reports", filename)
                                os.makedirs("reports", exist_ok=True)

                                # ✅ Gộp dữ liệu dạng dict
                                bt_df = {d["ticker"]: d["bt_df"] for d in valid_data}
                                perf_df = {d["ticker"]: d["perf_df"] for d in valid_data}
                                risk_df = {d["ticker"]: d["risk_df"] for d in valid_data}

                                try:
                                    pdf_file_path = export_report_pdf(
                                        ticker=tickers,        # list ticker
                                        bt_df=bt_df,
                                        perf_df=perf_df,
                                        risk_df=risk_df,
                                        weights_df=None,
                                        # allocation_df=None,
                                        output_path=output_path,
                                        include_backtest=True,
                                        include_perf=True,
                                        include_risk=True,
                                        include_rebalance=False,
                                        include_forecast=False,
                                        include_optimization=False
                                    )

                                    with safe_open_pdf(pdf_file_path) as f:
                                        st.download_button(
                                            label="📥 Tải về báo cáo",
                                            data=f,
                                            file_name=filename,
                                            mime="application/pdf"
                                        )
                                        st.success(f"✅ Đã xuất báo cáo backtest: {output_path}")

                                except Exception as e:
                                    st.error(f"❌ Lỗi khi xuất báo cáo: {e}")

        
        # =====================TAB TỐI ƯU HOÁ DANH MỤC==========================
       
        if "optimize" in tab_map:
            with tabs[tab_map["optimize"]]:
                
                filt = st.session_state.get("filtered_tickers", [])
                risk_res = st.session_state.get("risk_result", {})
                weights_df: pd.DataFrame | None = st.session_state.get("portfolio_weights")

                # === TỰ ĐỘNG TÍNH NẾU CHƯA CÓ weights_df nhưng có dữ liệu đủ điều kiện ===
                if weights_df is None and filt and risk_res:
                    try:
                        df_filt = portfolio_data[portfolio_data["Ticker"].isin(filt)]
                        weights_df = portfolio_optimizer.optimize_portfolio(df_filt)
                        st.session_state["portfolio_weights"] = weights_df
                        st.session_state["optimize_auto"] = True
                        # st.rerun()  # reload tab để hiển thị kết quả
                    except Exception as e:
                        st.error(f"❌ Lỗi tối ưu tự động: {e}")

                # === HIỂN THỊ KẾT QUẢ ===
                
                if weights_df is not None and not weights_df.empty:
                    # 🔹 Pie chart: Việt hoá nhãn cột
                    fig = px.pie(
                        weights_df,
                        names="Ticker",
                        values="Weight",
                        title="Tỷ trọng danh mục"
                    )
                    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{percent}")
                    fig.update_layout(legend_title_text="Mã cổ phiếu")
                    st.plotly_chart(fig, use_container_width=True)

                    # 🔹 Nhập tổng vốn
                    
                    total_cap_str = st.text_input(
                        "💰 Tổng vốn (VNĐ)",
                        value=f"{100_000_000:,}",  # format mặc định 100,000,000
                        key="total_cap_str"
                    )

                    # Parse lại về float/int để tính toán
                    try:
                        total_cap = float(total_cap_str.replace(",", ""))
                    except:
                        total_cap = 0.0



                    # --- Bảng phân bổ vốn ---
                    st.markdown("### 💸 Phân bổ vốn")
                    alloc = weights_df.copy()
                    alloc["Capital"] = alloc["Weight"] * total_cap

                    
                    # Việt hoá cột
                    alloc_display = alloc.rename(columns={
                        "Ticker": "Mã cổ phiếu",
                        "Weight": "Tỷ trọng",
                        "Capital": "Vốn phân bổ",
                        "Expected_Return": "Lợi nhuận kỳ vọng",
                        "Volatility": "Độ biến động"
                    })


                    # Format số liệu
                    alloc_display["Tỷ trọng"] = (alloc_display["Tỷ trọng"] * 100).round(2).astype(str) + " %"
                    alloc_display["Vốn phân bổ"] = alloc_display["Vốn phân bổ"].round(0).map("{:,.0f}".format)
                    # Format cột lợi nhuận kỳ vọng và độ biến động nếu là tỷ lệ
                    alloc_display["Lợi nhuận kỳ vọng"] = (alloc_display["Lợi nhuận kỳ vọng"] * 100).round(2).astype(str) + " %"
                    # alloc_display["Độ biến động"] = (alloc_display["Độ biến động"] * 100).round(2).astype(str) + " %"
                    # Chuyển sang số, lỗi sẽ thành NaN
                    numeric_vals = pd.to_numeric(alloc_display["Độ biến động"], errors="coerce")

                    # Áp dụng phép tính chỉ với giá trị số, giữ nguyên các giá trị không phải số
                    alloc_display["Độ biến động"] = [
                        f"{round(x * 100, 2)} %" if pd.notna(x) else orig
                        for x, orig in zip(numeric_vals, alloc_display["Độ biến động"])
                    ]



                    st.dataframe(alloc_display)
                    
                    # st.dataframe(alloc_display, use_container_width=True)
                    st.caption("💡 Đơn vị vốn phân bổ = đơn vị của Tổng vốn nhập vào")
                    

                    # --- Bảng phân bổ nâng cao ---
                    st.markdown("### ⚖️ Phân bổ nâng cao theo rủi ro")
                    vols, missing_vols = [], []

                    for tk in weights_df["Ticker"]:
                        try:
                            df_risk = risk_res.get(tk)
                            if df_risk is not None:
                                vol_row = df_risk[df_risk["Metric"] == "Volatility"]
                                if not vol_row.empty:
                                    vols.append(vol_row["Value"].iloc[0])
                                    continue
                            missing_vols.append(tk)
                            vols.append(None)
                        except Exception:
                            missing_vols.append(tk)
                            vols.append(None)

                    weights_df["Volatility"] = vols

                    if weights_df["Volatility"].notna().all():
                        adj = weights_df.copy()
                        adj["AdjWeight"] = 1 / adj["Volatility"]
                        adj["AdjWeight"] /= adj["AdjWeight"].sum()
                        adj["AdjCapital"] = adj["AdjWeight"] * total_cap

                        # Việt hoá cột
                        adj_display = adj.rename(columns={
                            "Ticker": "Mã cổ phiếu",
                            "Weight": "Tỷ trọng gốc",
                            "Expected_Return": "Lợi nhuận kỳ vọng",
                            "Volatility": "Độ biến động",
                            "AdjWeight": "Tỷ trọng điều chỉnh",
                            "AdjCapital": "Vốn phân bổ điều chỉnh"
                        })

                        # Format số liệu
                        adj_display["Tỷ trọng gốc"] = (adj_display["Tỷ trọng gốc"] * 100).round(2).astype(str) + " %"
                        adj_display["Độ biến động"] = (adj_display["Độ biến động"] * 100).round(2).astype(str) + " %"
                        adj_display["Tỷ trọng điều chỉnh"] = (adj_display["Tỷ trọng điều chỉnh"] * 100).round(2).astype(str) + " %"
                        adj_display["Vốn phân bổ điều chỉnh"] = adj_display["Vốn phân bổ điều chỉnh"].round(0).map("{:,.0f}".format)
                        adj_display["Lợi nhuận kỳ vọng"] = (adj_display["Lợi nhuận kỳ vọng"] * 100).round(2).astype(str) + " %"

                        st.dataframe(adj_display)
                        # st.dataframe(adj_display, use_container_width=True)
                        st.caption("💡 Đơn vị vốn phân bổ điều chỉnh = đơn vị của Tổng vốn nhập vào")
                        st.caption("💡 Lợi nhuận kỳ vọng tính theo 1 năm, dựa trên log return hàng ngày và annualized.")
                    else:
                        st.error(
                            f"❌ Thiếu dữ liệu Độ biến động cho: {', '.join(missing_vols)}.\n"
                            "Vui lòng chạy tab 📉 Phân tích rủi ro trước."
                        )

                    # 🔹 Lưu kết quả vẫn giữ nguyên cột English để không ảnh hưởng logic
                    allocation_result = {
                        row["Ticker"]: pd.DataFrame({
                            "Weight": [row["Weight"]],
                            "Capital": [row["Capital"]]
                        })
                        for idx, row in alloc.iterrows()
                    }
                    st.session_state["allocation_result"] = allocation_result


                # === NÚT CHẠY TỐI ƯU MANUAL ===
                if st.button("⚡ Tối ưu hóa lại danh mục"):
                    if not filt:
                        st.warning("⚠️ Chưa có danh sách mã đạt yêu cầu.")
                    else:
                        try:
                            df_filt = portfolio_data[portfolio_data["Ticker"].isin(filt)]
                            weights_df = portfolio_optimizer.optimize_portfolio(df_filt)
                            st.session_state["portfolio_weights"] = weights_df
                            st.success("✅ Đã tối ưu danh mục")
                        except Exception as e:
                            st.error(f"❌ Lỗi tối ưu: {e}")
                # === XUẤT PDF ===
                if st.button("📥 Xuất báo cáo PDF Tối ưu hóa danh mục"):
                    today_str = st.session_state.get("today_str")
                    if not today_str:
                        today_str = datetime.today().strftime("%d%m%Y")
                        st.session_state["today_str"] = today_str

                    allocation_result = st.session_state.get("allocation_result", {})
                    if weights_df is None or weights_df.empty:
                        st.warning("⚠️ Không có dữ liệu phân bổ vốn.")
                    else:
                        try:
                            ticker_list = weights_df["Ticker"].dropna().unique().tolist()
                            ticker_str = "_".join(ticker_list)

                            allocation_df = {ticker: allocation_result[ticker] for ticker in ticker_list if ticker in allocation_result}

                            output_path = os.path.join(REPORTS_DIR, f"{ticker_str}_{today_str}_optimization.pdf")

                            pdf_path = reporting.export_report_pdf(
                                ticker=ticker_list,
                                weights_df=weights_df,
                                # allocation_df=allocation_df,
                                output_path=output_path,
                                include_rebalance=False,
                                include_backtest=False,
                                include_perf=False,
                                alloc_display=alloc_display,
                                adj_display=adj_display if 'adj_display' in locals() else None,
                                include_risk=False,
                                include_forecast=False,
                                include_optimization=True
                                
                            )

                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="📥 Tải về báo cáo Tối ưu hóa",
                                    data=f,
                                    file_name=os.path.basename(output_path),
                                    mime="application/pdf"
                                )
                            st.success(f"✅ Đã xuất báo cáo Tối ưu hóa: `{output_path}`")
                        except Exception as e:
                            st.error(f"❌ Lỗi khi xuất báo cáo PDF: {e}")


        # =================TAB TÁI CÂN BẰNG=========================                
        if "rebalance" in tab_map:
            with tabs[tab_map["rebalance"]]:
                
                if portfolio_data is None:
                    st.warning("⚠️ Vui lòng tải dữ liệu danh mục trước.")
                    st.stop()

                latest_data = portfolio_data.groupby("Ticker").tail(1)

                # Lọc mã hợp lệ
                invalid = st.session_state.get("invalid_tickers", [])
                filtered_tickers = st.session_state.get("filtered_tickers", [])
                perf_tickers = [k[0] for k in st.session_state.get("performance_result", {}).keys()]

                tickers_available = [
                    t for t in latest_data["Ticker"].unique()
                    if t not in invalid and t in filtered_tickers and t in perf_tickers
                ]

                if not tickers_available:
                    st.warning("⚠️ Không có mã nào đủ điều kiện để tái cân bằng.")
                    st.stop()

                # --- Nhập số lượng đang nắm giữ ---
                st.markdown("### 💼 Danh mục hiện tại")
                st.caption("Nhập số lượng cổ phiếu đang nắm giữ hiện tại để tính toán tỷ trọng và lập kế hoạch tái cân bằng.")

                quantity_dict = {
                    t: st.number_input(
                        f"Số lượng `{t}` đang nắm giữ", 
                        min_value=0, 
                        value=10,
                        help="Số lượng cổ phiếu bạn đang sở hữu hiện tại đối với mã này."
                    )
                    for t in tickers_available
                }

                current_prices = dict(zip(latest_data["Ticker"], latest_data["Close"]))
                portfolio = {
                    t: {"price": current_prices[t], "quantity": quantity_dict[t]}
                    for t in tickers_available
                }

                total_value = sum(info["price"] * info["quantity"] for info in portfolio.values())

                current_weights = {
                    t: round((info["price"] * info["quantity"]) / total_value, 4)
                    for t, info in portfolio.items()
                }

                
                # --- Gợi ý tỷ trọng mục tiêu theo hiệu suất 1 tháng ---
                return_1m = {}
                for t in tickers_available:
                    df = portfolio_data[portfolio_data["Ticker"] == t].copy().sort_values("Date")
                    if len(df) >= 22:
                        return_1m[t] = (df["Close"].iloc[-1] / df["Close"].iloc[-22]) - 1
                    else:
                        return_1m[t] = 0.0

                # Tổng lợi nhuận dương để phân bổ trọng số
                total_rtn = sum(r for r in return_1m.values() if r > 0)
                suggested_weights = {
                    t: (r / total_rtn) if r > 0 and total_rtn > 0 else 0.0
                    for t, r in return_1m.items()
                }

                # --- Tạo DataFrame hiển thị ---
                df_suggested = pd.DataFrame({
                    "Ticker": list(return_1m.keys()),
                    "Hiệu suất 1M": [f"{r*100:.2f} %" for r in return_1m.values()],
                    "Tỷ trọng gợi ý": [f"{suggested_weights[t]*100:.2f} %" for t in return_1m.keys()],
                })

                # Nếu có total_cap thì thêm cột vốn phân bổ
                total_cap = st.session_state.get("total_cap", 0)
                if total_cap and total_cap > 0:
                    df_suggested["Vốn phân bổ"] = [
                        f"{(suggested_weights[t] * total_cap):,.0f}"
                        for t in return_1m.keys()
                    ]

                # --- Hiển thị ---
                st.markdown("### 📊 Bảng gợi ý tỷ trọng mục tiêu (theo hiệu suất 1 tháng gần nhất)")
                st.caption("Hệ thống sử dụng hiệu suất 1 tháng gần nhất để đưa ra tỷ trọng gợi ý cho các mã.")
                st.dataframe(df_suggested, use_container_width=True)


                # --- Nhập tỷ trọng mục tiêu ---
                st.markdown("### 🎯 Tỷ trọng mục tiêu")
                st.caption("Điều chỉnh tỷ trọng mục tiêu mà bạn mong muốn giữ cho từng mã sau khi tái cân bằng.")
                target_weights = {
                    t: st.number_input(
                        f"Tỷ trọng mục tiêu cho `{t}`", 
                        min_value=0.0, 
                        max_value=1.0,
                        value=suggested_weights.get(t, 0.0), 
                        step=0.01,
                        help="Tỷ trọng mong muốn của mã này trong danh mục. Tổng tất cả phải xấp xỉ 1.0."
                    )
                    for t in tickers_available
                }

                total_weight = sum(target_weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    st.error("⚠️ Tổng tỷ trọng mục tiêu phải xấp xỉ 1.0")
                    st.stop()

                # --- Kế hoạch tái cân bằng ---
                st.markdown("### 📋 Kế hoạch tái cân bằng")
                st.caption("Bảng chi tiết hành động mua/bán cần thực hiện để đạt được tỷ trọng mục tiêu.")

                rebalance_plan = []

                if tickers_available:  # chỉ khi có ticker
                    for t in tickers_available:
                        info = portfolio[t]
                        cur_val = info["price"] * info["quantity"]
                        tgt_val = total_value * target_weights[t]
                        diff_val = tgt_val - cur_val
                        qty_diff = int(round(diff_val / info["price"]))
                        action = "Giữ"
                        if qty_diff > 0:
                            action = "Mua thêm"
                        elif qty_diff < 0:
                            action = "Bán bớt"

                        rebalance_plan.append({
                            "Ticker": t,
                            "Giá hiện tại": info["price"],
                            "SL hiện tại": info["quantity"],
                            "Tỷ trọng hiện tại": current_weights[t],
                            "Tỷ trọng mục tiêu": target_weights[t],
                            "Chênh lệch SL": qty_diff,
                            "Hành động": action
                        })

                # Luôn tạo DataFrame, kể cả khi tickers_available rỗng
                df_plan = pd.DataFrame(rebalance_plan)

                # Format tỷ trọng thành %
                if not df_plan.empty:
                    df_plan["Tỷ trọng hiện tại"] = (df_plan["Tỷ trọng hiện tại"] * 100).round(2).astype(str) + " %"
                    df_plan["Tỷ trọng mục tiêu"] = (df_plan["Tỷ trọng mục tiêu"] * 100).round(2).astype(str) + " %"

                st.dataframe(df_plan)



                # Biểu đồ trước/sau tái cân bằng
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Trước khi tái cân bằng")
                    st.caption("Tỷ trọng hiện tại dựa trên giá trị thị trường.")
                    fig1 = px.pie(names=current_weights.keys(), values=current_weights.values(), title="Tỷ trọng hiện tại")
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.markdown("#### Sau khi tái cân bằng")
                    st.caption("Tỷ trọng mong muốn theo mục tiêu đã nhập.")
                    fig2 = px.pie(names=target_weights.keys(), values=target_weights.values(), title="Tỷ trọng mục tiêu")
                    st.plotly_chart(fig2, use_container_width=True)

                # Lưu lại dữ liệu phục vụ báo cáo
                
                st.session_state["rebalance_plan_df"] = df_plan
                st.session_state["rebalance_weights_df"] = pd.DataFrame({
                    "Ticker": tickers_available,
                    "Tỷ trọng mục tiêu (quá khứ)": [suggested_weights[t] for t in tickers_available],
                    "Tỷ trọng mục tiêu (chiến lược)": [target_weights[t] for t in tickers_available],
                    "Tỷ trọng hiện tại": [current_weights[t] for t in tickers_available],
                    "Giá trị danh mục": [portfolio[t]["price"] * portfolio[t]["quantity"] for t in tickers_available],
                    "Hành động": [
                        "Mua thêm" if df_plan[df_plan["Ticker"] == t]["Chênh lệch SL"].values[0] > 0 else
                        "Bán bớt" if df_plan[df_plan["Ticker"] == t]["Chênh lệch SL"].values[0] < 0 else
                        "Giữ"
                        for t in tickers_available
                    ]
                })
                # st.session_state["weights"] = weights_df  # ✅ Thêm dòng này để báo cáo tổng hợp đọc được
                st.session_state["weights_df"] = st.session_state["rebalance_weights_df"]
                st.session_state["weights"] = st.session_state["rebalance_weights_df"]

                # #==============Báo cáo kế hoạch tái cân bằng=============

                if st.button("📥 Xuất báo cáo PDF Tái cân bằng"):
                    # weights_df = st.session_state.get("weights_df", None)
                    weights_df = st.session_state.get("rebalance_weights_df", None)

                    if weights_df is None or weights_df.empty:
                        st.warning("⚠️ Không có dữ liệu phân bổ danh mục để xuất báo cáo.")
                    else:
                        # today_str = datetime.today().strftime("%d%m%Y")
                        today_str = st.session_state.get("today_str")
                        if not today_str:  # fallback nếu session chưa có giá trị
                            today_str = datetime.today().strftime("%d%m%Y")
                            st.session_state["today_str"] = today_str

                        # ✅ Chuẩn hóa ticker thành list để truyền đúng
                        tickers = weights_df["Ticker"].dropna().unique().tolist()
                        ticker_str = "_".join(tickers)
                        output_path = os.path.join(REPORTS_DIR, f"{ticker_str}_{today_str}_rebalance.pdf")

                        try:
                            # ✅ Gọi đúng chuẩn hàm export_report_pdf với ticker là list
                            pdf_path = reporting.export_report_pdf(
                                ticker=tickers,  # ✅ truyền đúng list
                                weights_df=weights_df,
                                
                                output_path=output_path,
                                include_rebalance=True,
                                include_backtest=False,
                                include_perf=False,
                                include_risk=False,
                                include_forecast=False,
                                include_optimization=False
                            )

                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="📥 Tải về báo cáo Tái cân bằng",
                                    data=f,
                                    file_name=os.path.basename(output_path),
                                    mime="application/pdf"
                                )
                            st.success(f"✅ Đã xuất báo cáo Tái cân bằng: `{output_path}`")

                        except Exception as e:
                            st.error(f"❌ Lỗi khi xuất báo cáo Tái cân bằng: {e}")


     
        # ---------------------------------------------------------------------------
        # ⚙️ TAB 1 – Huấn luyện mô hình (từng mã, bulk & tự động)
        # ---------------------------------------------------------------------------
       

        TRADING_DAYS_PER_YEAR = 252

        # -------------------------
        # Helpers
        # -------------------------
        def get_training_config_from_df(df_close: pd.DataFrame):
            df = df_close.dropna(subset=["Close"]).copy()
            n_valid = len(df)
            if n_valid < 60:
                raise ValueError("Không đủ dữ liệu (cần >= 60 dòng có Close).")
            years_avail = n_valid // TRADING_DAYS_PER_YEAR
            train_years = min(max(years_avail, 1), 3)
            lookback, forecast_days = (60, 90) if train_years == 1 else (60, 180) if train_years == 2 else (90, 180)
            train_size = min(n_valid, train_years * TRADING_DAYS_PER_YEAR)
            df_train = df.tail(train_size).copy()
            cfg = {
                "train_years": train_years,
                "lookback": lookback,
                "forecast_days": forecast_days,
                "train_size": len(df_train),
                "yesterday": pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
            }
            return df_train, cfg

        def quick_validate_lstm(df_sub: pd.DataFrame, lookback: int, horizon: int, quick_epochs: int = 5, batch_size: int = 16):
            
            df = df_sub.dropna(subset=["Close"]).copy()
            data = df["Close"].values.reshape(-1, 1)
            scaler = MinMaxScaler().fit(data)
            scaled = scaler.transform(data)
            if len(scaled) <= lookback:
                raise ValueError("Không đủ dữ liệu cho lookback này.")

            X = np.array([scaled[i-lookback:i,0] for i in range(lookback,len(scaled))])
            y = np.array([scaled[i,0] for i in range(lookback,len(scaled))])
            X = X.reshape((X.shape[0], X.shape[1], 1))

            model = Sequential([LSTM(16, input_shape=(X.shape[1],1)), Dense(1)])
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(X, y, epochs=quick_epochs, batch_size=batch_size, verbose=0,
                    callbacks=[EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)])

            n_eval = min(20, X.shape[0])
            preds = [scaler.inverse_transform([[model.predict(X[i:i+1], verbose=0)[0,0]]])[0,0] for i in range(X.shape[0]-n_eval,X.shape[0])]
            trues = [scaler.inverse_transform([[y[i]]])[0,0] for i in range(X.shape[0]-n_eval,X.shape[0])]
            if not preds:
                raise ValueError("Không đủ điểm đánh giá.")
            return float(np.sqrt(np.mean((np.array(trues)-np.array(preds))**2)))

        # =====LSTM=====
        #     return rmse, mape
        def evaluate_lstm_insample(model, scaler, df_train, lookback, n_eval=30):
            """
            Đánh giá LSTM trên dữ liệu insample với multi-output.
            Trả về dict RMSE và MAPE cho từng cột đã train.
            """
            # Lấy danh sách cột từ scaler
            if not hasattr(scaler, "feature_names_in_"):
                return None, None
            available_cols = list(scaler.feature_names_in_)

            data_vals = df_train[available_cols].values
            scaled = scaler.transform(data_vals)

            if len(scaled) <= lookback:
                return None, None

            max_preds = len(scaled) - lookback
            n_eval = min(n_eval, max_preds)

            preds_arr = {col: [] for col in available_cols}
            trues_arr = {col: [] for col in available_cols}

            for i in range(max_preds - n_eval, max_preds):
                X = scaled[i:i+lookback].reshape(1, lookback, len(available_cols))
                pred_scaled = model.predict(X, verbose=0)[0]

                # Kiểm tra số chiều dự báo
                if len(pred_scaled) != len(available_cols):
                    continue

                pred_real = scaler.inverse_transform([pred_scaled])[0]

                for j, col in enumerate(available_cols):
                    preds_arr[col].append(pred_real[j])
                    trues_arr[col].append(data_vals[i + lookback, j])

            rmse = {}
            mape = {}
            for col in available_cols:
                if preds_arr[col]:  # tránh chia 0 nếu không có dữ liệu
                    pred = np.array(preds_arr[col])
                    true = np.array(trues_arr[col])
                    rmse[col] = float(np.sqrt(np.mean((true - pred) ** 2)))
                    mape[col] = float(np.mean(np.abs((true - pred) / true))) * 100

            return rmse, mape
        
        # =====TCN=====
        def evaluate_tcn_insample(model, scaler, df_train, lookback, n_eval=30):
            """
            Đánh giá TCN trên dữ liệu insample với multi-output.
            Trả về dict RMSE và MAPE cho từng cột đã train.
            """
            # Lấy danh sách cột từ scaler
            if not hasattr(scaler, "feature_names_in_"):
                return None, None
            available_cols = list(scaler.feature_names_in_)

            data_vals = df_train[available_cols].values
            scaled = scaler.transform(data_vals)

            if len(scaled) <= lookback:
                return None, None

            max_preds = len(scaled) - lookback
            n_eval = min(n_eval, max_preds)

            preds_arr = {col: [] for col in available_cols}
            trues_arr = {col: [] for col in available_cols}

            for i in range(max_preds - n_eval, max_preds):
                X = scaled[i:i+lookback].reshape(1, lookback, len(available_cols))
                pred_scaled = model.predict(X, verbose=0)[0]

                # Kiểm tra số chiều dự báo
                if len(pred_scaled) != len(available_cols):
                    continue

                pred_real = scaler.inverse_transform([pred_scaled])[0]

                for j, col in enumerate(available_cols):
                    preds_arr[col].append(pred_real[j])
                    trues_arr[col].append(data_vals[i + lookback, j])

            rmse = {}
            mape = {}
            for col in available_cols:
                if preds_arr[col]:  # tránh chia 0 nếu không có dữ liệu
                    pred = np.array(preds_arr[col])
                    true = np.array(trues_arr[col])
                    rmse[col] = float(np.sqrt(np.mean((true - pred) ** 2)))
                    mape[col] = float(np.mean(np.abs((true - pred) / true))) * 100

            return rmse, mape

        # ===========

        def auto_optimize_training_params_wrapper(df_clean):
            best_cfg, min_rmse = {"years":1,"lookback":60,"horizon":90}, float("inf")
            for yrs in [1,2,3]:
                df_sub = df_clean[df_clean["Date"] >= df_clean["Date"].max() - pd.DateOffset(years=yrs)]
                if len(df_sub) < 60: continue
                for lb in [30,60,90]:
                    for hz in [90,120,180]:
                        try:
                            rmse = quick_validate_lstm(df_sub, lookback=lb, horizon=hz)
                            if rmse < min_rmse:
                                min_rmse, best_cfg = rmse, {"years":yrs,"lookback":lb,"horizon":hz}
                        except Exception:
                            continue
            return best_cfg, (min_rmse if min_rmse != float("inf") else None)
        

        def find_tickers_missing_predicted_close(ticker_list=None, min_days=60):
            """
            Quét toàn bộ DB, kiểm tra bảng nào có < min_days dòng Predicted_Close
            tính đến hết ngày hôm qua.
            """
            missing = []
            try:
                tables = db.list_tables()
            except Exception as e:
                print(f"⚠️ Không lấy được danh sách bảng: {e}")
                return []

            cutoff = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

            for tk in tables:
                try:
                    dfpf = db.load_forecast(tk)
                    if dfpf is None or dfpf.empty or "Predicted_Close" not in dfpf.columns:
                        missing.append(tk)
                        continue

                    # Lọc dữ liệu đến hết hôm qua
                    df_hist = dfpf[dfpf["Date"] <= cutoff]

                    # Đếm số dòng Predicted_Close có dữ liệu
                    count_pred = df_hist["Predicted_Close"].notna().sum()

                    if count_pred < min_days:
                        missing.append(tk)

                except Exception:
                    missing.append(tk)

            return missing

     
        # =====LSTM====    return preds
        def generate_lstm_preds(df_clean, model, scaler, lookback, horizon, ticker):
            """
            Sinh dự báo LSTM multi-output dựa trên các cột đã train.
            Trả về list tuple: (Date, Predicted_Open, Predicted_High, ..., Ticker)
            """
            df_full = df_clean.copy()

            # Lấy danh sách cột đã train từ scaler
            if not hasattr(scaler, "feature_names_in_"):
                return []  # scaler không có metadata => bỏ qua dự báo
            available_cols = list(scaler.feature_names_in_)

            # Chuẩn hóa dữ liệu input
            data_vals = df_full[available_cols].values
            scaled = scaler.transform(data_vals)

            # Dự báo trên horizon ngày cuối cùng
            candidate_dates = df_full["Date"].iloc[max(0, len(df_full) - horizon):].tolist()
            preds = []

            for day in candidate_dates:
                idxs = df_full.index[df_full["Date"] == day].tolist()
                if not idxs:
                    continue
                idx = idxs[0]

                # Bỏ qua weekend hoặc không đủ lookback
                if day.weekday() >= 5 or idx - lookback < 0:
                    continue

                window = scaled[idx - lookback: idx]
                if window.shape[0] < lookback:
                    continue

                # Predict
                X = window.reshape(1, lookback, len(available_cols))
                pred_scaled = model.predict(X, verbose=0)[0]  # shape: (n_features,)

                if len(pred_scaled) != len(available_cols):
                    continue  # tránh lỗi mismatch

                pred_real = scaler.inverse_transform([pred_scaled])[0]

                # Tạo tuple (Date, Predicted_*..., Ticker)
                row_tuple = [day.strftime("%Y-%m-%d")] + [float(v) for v in pred_real] + [ticker]
                preds.append(tuple(row_tuple))

            return preds

        # =====TCN====    return preds
        def generate_tcn_preds(df_clean, model, scaler, lookback, horizon, ticker):
            """
            Sinh dự báo TCN multi-output dựa trên các cột đã train.
            Trả về list tuple: (Date, Predicted_Open, Predicted_High, ..., Ticker)
            """
            df_full = df_clean.copy()

            # Lấy danh sách cột đã train từ scaler
            if not hasattr(scaler, "feature_names_in_"):
                return []  # scaler không có metadata => bỏ qua dự báo
            available_cols = list(scaler.feature_names_in_)

            # Chuẩn hóa dữ liệu input
            data_vals = df_full[available_cols].values
            scaled = scaler.transform(data_vals)

            # Dự báo trên horizon ngày cuối cùng
            candidate_dates = df_full["Date"].iloc[max(0, len(df_full) - horizon):].tolist()
            preds = []

            for day in candidate_dates:
                idxs = df_full.index[df_full["Date"] == day].tolist()
                if not idxs:
                    continue
                idx = idxs[0]

                # Bỏ qua weekend hoặc không đủ lookback
                if day.weekday() >= 5 or idx - lookback < 0:
                    continue

                window = scaled[idx - lookback: idx]
                if window.shape[0] < lookback:
                    continue

                # Predict
                X = window.reshape(1, lookback, len(available_cols))
                pred_scaled = model.predict(X, verbose=0)[0]  # shape: (n_features,)

                if len(pred_scaled) != len(available_cols):
                    continue  # tránh lỗi mismatch

                pred_real = scaler.inverse_transform([pred_scaled])[0]

                # Tạo tuple (Date, Predicted_*..., Ticker)
                row_tuple = [day.strftime("%Y-%m-%d")] + [float(v) for v in pred_real] + [ticker]
                preds.append(tuple(row_tuple))

            return preds

        # -------------------------
        # Streamlit Tabs
        # -------------------------
        # ----------------- Helpers -----------------
        def bulk_or_auto_train(ticker_list, model_type="LSTM", epochs=50, batch_size=16, auto_optimize=False, source="yf"):
            logs = []
            for ticker in ticker_list:
                try:
                    # df_raw = load_data_yf(ticker)
                    df_raw = load_data_dl(ticker, source=source)
                    df_clean = clean_dataframe(df_raw, ticker)
                    if auto_optimize:
                        try:
                            best_cfg, _ = auto_optimize_training_params_wrapper(df_clean)
                        except Exception:
                            df_train, cfg = get_training_config_from_df(df_clean)
                            best_cfg = {"years": cfg["train_years"], "lookback": cfg["lookback"], "horizon": cfg["forecast_days"]}
                        df_train = df_clean[df_clean["Date"] >= (df_clean["Date"].max() - pd.DateOffset(years=best_cfg["years"]))]
                        lookback, horizon = best_cfg["lookback"], best_cfg["horizon"]
                    else:
                        df_train, cfg = get_training_config_from_df(df_clean)
                        lookback, horizon = cfg["lookback"], cfg["forecast_days"]

                    
                    if model_type == "LSTM":
                        # Train multi-output model
                        model, scaler = train_lstm_model(df_train, ticker=ticker, epochs=epochs, batch_size=batch_size, look_back=cfg["lookback"])

                        # Lấy danh sách các cột đã train
                        if not hasattr(scaler, "feature_names_in_"):
                            logs.append({"Ticker": ticker, "Status": "⚠️ Scaler không có thông tin cột, cần train lại."})
                            continue
                        available_cols = list(scaler.feature_names_in_)
                        pred_cols = [f"Predicted_{col}" for col in available_cols]

                        # Đánh giá insample multi-output
                        rmse, mape = evaluate_lstm_insample(model, scaler, df_train, lookback)

                        # Dự báo multi-output
                        preds = generate_lstm_preds(df_clean, model, scaler, lookback, horizon, ticker)

                        if preds:
                            df_forecast = pd.DataFrame(preds, columns=["Date"] + pred_cols + ["Ticker"])
                            db.save_forecast_last(ticker, df_forecast)
                            logs.append({
                                "Ticker": ticker,
                                "Status": "✅ Success",
                                "RMSE": {col: round(rmse.get(col, None), 4) for col in available_cols} if rmse else None,
                                "MAPE": {col: round(mape.get(col, None), 2) for col in available_cols} if mape else None
                            })
                        else:
                            logs.append({"Ticker": ticker, "Status": "⚠️ No preds"})
                    # ====TCN====
                    elif model_type == "TCN":
                        # Train multi-output TCN
                        model, scaler = train_tcn_model(
                            df_train,
                            ticker=ticker,
                            epochs=epochs,
                            batch_size=batch_size,
                            look_back=cfg["lookback"]
                        )

                        # Lấy danh sách các cột đã train
                        if not hasattr(scaler, "feature_names_in_"):
                            logs.append({"Ticker": ticker, "Status": "⚠️ Scaler không có thông tin cột, cần train lại."})
                            continue
                        available_cols = list(scaler.feature_names_in_)
                        pred_cols = [f"Predicted_{col}" for col in available_cols]

                        # Đánh giá insample
                        rmse, mape = evaluate_tcn_insample(model, scaler, df_train, cfg["lookback"])

                        # Dự báo multi-output
                        preds = generate_tcn_preds(df_clean, model, scaler, cfg["lookback"], cfg["forecast_days"], ticker)

                        if preds:
                            df_forecast = pd.DataFrame(preds, columns=["Date"] + pred_cols + ["Ticker"])
                            db.save_forecast_last(ticker, df_forecast)
                            logs.append({
                                "Ticker": ticker,
                                "Status": "✅ Success",
                                "RMSE": {col: round(rmse.get(col, None), 4) for col in available_cols} if rmse else None,
                                "MAPE": {col: round(mape.get(col, None), 2) for col in available_cols} if mape else None
                            })
                        else:
                            logs.append({"Ticker": ticker, "Status": "⚠️ No preds"})

                    # ===========

                    # elif model_type=="Prophet":
                    #     train_prophet_model(df_train, ticker)
                    #     try:
                            
                    #         pred_df = forecast_prophet(df_clean, forecast_days=horizon, use_saved_model=True, ticker=ticker)
                    #         pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.strftime("%Y-%m-%d")
                    #         pred_df["Ticker"]=ticker
                    #         db.save_forecast_last(ticker, pred_df)
                    #         logs.append({"Ticker": ticker, "Status":"✅ Prophet saved"})
                    #     except Exception:
                    #         logs.append({"Ticker": ticker, "Status":"✅ Prophet trained (no auto forecast)"})
                    # elif model_type=="ARIMA":
                    #     train_arima_model(df_train, ticker)
                    #     try:
                            
                    #         pred_df = forecast_arima(df_clean, forecast_days=horizon, use_saved_model=True, ticker=ticker)
                    #         pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.strftime("%Y-%m-%d")
                    #         pred_df["Ticker"]=ticker
                    #         db.save_forecast_last(ticker, pred_df)
                    #         logs.append({"Ticker": ticker, "Status":"✅ ARIMA saved"})
                    #     except Exception:
                    #         logs.append({"Ticker": ticker, "Status":"✅ ARIMA trained (no auto forecast)"})
                except Exception as e:
                    logs.append({"Ticker": ticker, "Status": f"❌ {e}"})
            return logs

        # ----------------- Streamlit Tabs -----------------
        if "train" in tab_map:
            with tabs[tab_map["train"]]:
                st.header("🔧 Huấn luyện mô hình dự báo")
                tab1, tab2, tab3 = st.tabs(["📌 Huấn luyện từng mã", "🔁 Huấn luyện lại toàn bộ", "⚙️ Quản lý & huấn luyện tự động"])

                
                # -------------------------
                # TAB 1: Quản lý & huấn luyện từng mã
                # -------------------------
                with tab1:
                    # st.subheader("🔧 Huấn luyện theo từng mã")
                    source = "yf"
                    st.caption("Chọn mã cổ phiếu và mô hình để huấn luyện, kết quả sẽ lưu làm Backtest.")
                    with st.expander("ℹ️ Help: Huấn luyện từng mã"):
                        st.markdown(
                            "- **Mục đích**: Huấn luyện riêng lẻ 1 mã để cập nhật Backtest. Huấn luyện lại có thể làm cho mô hình học sâu hơn với các chỉ số dự báo OHLCV.\n"
                            "- **LSTM**: cần ít nhất 60 dòng OHLCV; sẽ huấn luyện học máy và lưu dự báo vào Backtest.\n"
                            "- **TCN**: cần ít nhất 60 dòng OHLCV; sẽ huấn luyện học máy và lưu dự báo vào Backtest.\n"
                            "- **Epochs/Batch**: chỉ áp dụng cho LSTM và TCN."
                        )


                    if not valid_tickers:
                        st.warning("⚠️ Không có mã cổ phiếu hợp lệ để chọn.")
                    else:
                        ticker = st.selectbox("Chọn mã cổ phiếu để huấn luyện:", valid_tickers,
                                            help="Chọn mã cổ phiếu hợp lệ để huấn luyện mô hình dự báo.",
                                            key="train_model_ticker_select")

                    # model_type = st.selectbox(
                    #     "Chọn mô hình:",
                    #     ["LSTM", "Prophet", "ARIMA"],
                    #     help=(
                    #         "LSTM – Mạng nơ-ron hồi quy cho chuỗi thời gian, cần ít nhất 60 dòng dữ liệu.\n"
                    #         "Prophet – Mô hình của Meta, giỏi xử lý seasonality.\n"
                    #         "ARIMA – Mô hình thống kê cổ điển, phù hợp chuỗi thời gian ổn định."
                    #     )
                    # )
                    model_type = st.selectbox(
                        "Chọn mô hình:",
                        ["LSTM", "TCN"],
                        help=(
                            "LSTM – (Long Short-Term Memory) là một kiến trúc mạng RNN (Recurrent Neural Network) đặc biệt, được thiết kế để học chuỗi thời gian hoặc dữ liệu tuần tự, cần ít nhất 60 dòng dữ liệu.\n"
                            "TCN – (Temporal Convolutional Network) là một mô hình học sâu cho chuỗi thời gian, dựa trên Convolutional Neural Network (CNN) nhưng được thiết kế đặc biệt để xử lý dữ liệu tuần tự, mạnh với chuỗi dài và quan hệ phức tạp."
                        )
                    )


                    epochs = st.slider(
                        "Số vòng lặp (Epochs)",
                        min_value=5,
                        max_value=200,
                        value=50,
                        step=5,
                        help="Số lần học trên toàn bộ tập dữ liệu (áp dụng cho LSTM và TCN)."
                    )

                    batch_size = st.slider(
                        "Batch size",
                        min_value=4,
                        max_value=128,
                        value=16,
                        step=4,
                        help="Kích thước tập con dữ liệu cho mỗi lần cập nhật trọng số (áp dụng cho LSTM và TCN)."
                    )

                    
                    if st.button("🚀 Bắt đầu huấn luyện"):
                        # run training for selected ticker
                        with st.spinner(f"Đang huấn luyện {model_type} cho {ticker}..."):
                            try:
                                # df_raw = load_data_yf(ticker)
                                df_raw = load_data_dl(ticker, source=source)
                                
                                df_clean = clean_dataframe(df_raw, ticker=ticker)
                                if df_clean is None:
                                    st.error(f"❌ Dữ liệu không hợp lệ hoặc không đủ cho {ticker}.")
                                else:
                                    # choose config automatically
                                    df_train, cfg = get_training_config_from_df(df_clean)
                                    
                                    if model_type == "LSTM":
                                        if len(df_train) < 60:
                                            st.error("⚠️ Không đủ dữ liệu để train LSTM (cần >= 60 ngày).")
                                        else:
                                            # Train LSTM multi-output
                                            # model, scaler = train_lstm_model(df_train, ticker=ticker, epochs=epochs, batch_size=batch_size)
                                            model, scaler = train_lstm_model(df_train, ticker=ticker, epochs=epochs, batch_size=batch_size, look_back=cfg["lookback"])
                                            # Tính RMSE/MAPE chỉ cho Close
                                            rmse, mape = evaluate_lstm_insample(model, scaler, df_train, cfg["lookback"])

                                            # Lấy giá trị Close nếu rmse/mape là dict
                                            if isinstance(rmse, dict):
                                                rmse_close = rmse.get("Close", None)
                                            else:
                                                rmse_close = rmse
                                            if isinstance(mape, dict):
                                                mape_close = mape.get("Close", None)
                                            else:
                                                mape_close = mape

                                            # Dự báo multi-output
                                            df_full = df_clean.copy()

                                            # Lấy danh sách cột đã train scaler
                                            if hasattr(scaler, "feature_names_in_"):
                                                available_cols = list(scaler.feature_names_in_)
                                            else:
                                                st.error("⚠️ Không xác định được các cột scaler đã fit. Vui lòng train lại LSTM.")
                                                available_cols = []

                                            if not available_cols:
                                                st.warning("⚠️ Không có cột hợp lệ để dự báo.")
                                            else:
                                                # Kiểm tra dữ liệu đầu vào có đầy đủ các cột đã huấn luyện không
                                                missing_cols = [col for col in available_cols if col not in df_full.columns]
                                                if missing_cols:
                                                    st.error(f"⚠️ Dữ liệu dự báo thiếu các cột quan trọng: {missing_cols}. Không thể dự báo.")
                                                else:
                                                    pred_cols = [f"Predicted_{col}" for col in available_cols]

                                                    # Chuẩn bị dữ liệu dự báo
                                                    data_vals = df_full[available_cols].values
                                                    scaled = scaler.transform(data_vals)

                                                    candidate_dates = df_full["Date"].iloc[max(0, len(df_full)-cfg["forecast_days"]):].tolist()

                                                    # Tạo batch windows dự báo
                                                    X_batch = []
                                                    date_batch = []
                                                    for idx, day in enumerate(candidate_dates):
                                                        global_idx = len(df_full) - cfg["forecast_days"] + idx
                                                        if day.weekday() >= 5:  # skip weekend
                                                            continue
                                                        if global_idx - cfg["lookback"] < 0:
                                                            continue

                                                        window = scaled[global_idx - cfg["lookback"]: global_idx]
                                                        if window.shape[0] == cfg["lookback"]:
                                                            X_batch.append(window)
                                                            date_batch.append(day)

                                                    if X_batch:
                                                        X_batch = np.array(X_batch).reshape(len(X_batch), cfg["lookback"], len(available_cols))
                                                        preds_scaled = model.predict(X_batch, verbose=0)

                                                        if preds_scaled.shape[1] != len(available_cols):
                                                            st.error("⚠️ Số cột dự báo không khớp với scaler, không thể inverse transform")
                                                        else:
                                                            preds_real = scaler.inverse_transform(preds_scaled)

                                                            df_forecast = pd.DataFrame(
                                                                [[d.strftime("%Y-%m-%d")] + list(map(float, p)) + [ticker] for d, p in zip(date_batch, preds_real)],
                                                                columns=["Date"] + pred_cols + ["Ticker"]
                                                            )

                                                            db.save_forecast_last(ticker, df_forecast)
                                                            st.success(f"✅ Đã huấn luyện và lưu dự báo multi-output cho {ticker}")

                                                            st.write(pd.DataFrame([{
                                                                "Ticker": ticker,
                                                                "Years_Used": cfg.get("train_years"),
                                                                "Lookback": cfg.get("lookback"),
                                                                "Forecast_Days": cfg.get("forecast_days"),
                                                                "Train_Samples": cfg.get("train_size"),
                                                                "RMSE_Close": round(rmse_close, 4) if rmse_close is not None else None,
                                                                "MAPE_Close (%)": round(mape_close, 2) if mape_close is not None else None
                                                            }]))
                                                    else:
                                                        st.warning("⚠️ Không tạo được dự báo hợp lệ (không đủ window/dữ liệu).")
                                    # # ====
                                    elif model_type == "TCN":
                                        if len(df_train) < 60:
                                            st.error("⚠️ Không đủ dữ liệu để train TCN (cần >= 60 ngày).")
                                        else:
                                            # Train TCN multi-output
                                            model, scaler = train_tcn_model(
                                                df_train, 
                                                ticker=ticker, 
                                                epochs=epochs, 
                                                batch_size=batch_size, 
                                                look_back=cfg["lookback"]
                                            )

                                            # Đánh giá nội mẫu
                                            rmse, mape = evaluate_tcn_insample(model, scaler, df_train, cfg["lookback"])

                                            # Lấy giá trị Close nếu rmse/mape là dict
                                            if isinstance(rmse, dict):
                                                rmse_close = rmse.get("Close", None)
                                            else:
                                                rmse_close = rmse
                                            if isinstance(mape, dict):
                                                mape_close = mape.get("Close", None)
                                            else:
                                                mape_close = mape

                                            # Dự báo multi-output
                                            df_full = df_clean.copy()

                                            if hasattr(scaler, "feature_names_in_"):
                                                available_cols = list(scaler.feature_names_in_)
                                            else:
                                                st.error("⚠️ Không xác định được các cột scaler đã fit. Vui lòng train lại TCN.")
                                                available_cols = []

                                            if not available_cols:
                                                st.warning("⚠️ Không có cột hợp lệ để dự báo.")
                                            else:
                                                # Kiểm tra dữ liệu đầu vào có đầy đủ các cột đã huấn luyện không
                                                missing_cols = [col for col in available_cols if col not in df_full.columns]
                                                if missing_cols:
                                                    st.error(f"⚠️ Dữ liệu dự báo thiếu các cột quan trọng: {missing_cols}. Không thể dự báo.")
                                                else:
                                                    pred_cols = [f"Predicted_{col}" for col in available_cols]

                                                    data_vals = df_full[available_cols].values
                                                    scaled = scaler.transform(data_vals)

                                                    candidate_dates = df_full["Date"].iloc[max(0, len(df_full)-cfg["forecast_days"]):].tolist()

                                                    X_batch = []
                                                    date_batch = []
                                                    for idx, day in enumerate(candidate_dates):
                                                        global_idx = len(df_full) - cfg["forecast_days"] + idx
                                                        if day.weekday() >= 5:  # skip weekend
                                                            continue
                                                        if global_idx - cfg["lookback"] < 0:
                                                            continue

                                                        window = scaled[global_idx - cfg["lookback"]: global_idx]
                                                        if window.shape[0] == cfg["lookback"]:
                                                            X_batch.append(window)
                                                            date_batch.append(day)

                                                    if X_batch:
                                                        X_batch = np.array(X_batch).reshape(len(X_batch), cfg["lookback"], len(available_cols))
                                                        preds_scaled = model.predict(X_batch, verbose=0)

                                                        if preds_scaled.shape[1] != len(available_cols):
                                                            st.error("⚠️ Số cột dự báo không khớp với scaler, không thể inverse transform")
                                                        else:
                                                            preds_real = scaler.inverse_transform(preds_scaled)

                                                            df_forecast = pd.DataFrame(
                                                                [[d.strftime("%Y-%m-%d")] + list(map(float, p)) + [ticker] for d, p in zip(date_batch, preds_real)],
                                                                columns=["Date"] + pred_cols + ["Ticker"]
                                                            )

                                                            db.save_forecast_last(ticker, df_forecast)
                                                            st.success(f"✅ Đã huấn luyện và lưu dự báo multi-output cho {ticker}")

                                                            st.write(pd.DataFrame([{
                                                                "Ticker": ticker,
                                                                "Years_Used": cfg.get("train_years"),
                                                                "Lookback": cfg.get("lookback"),
                                                                "Forecast_Days": cfg.get("forecast_days"),
                                                                "Train_Samples": cfg.get("train_size"),
                                                                "RMSE_Close": round(rmse_close, 4) if rmse_close is not None else None,
                                                                "MAPE_Close (%)": round(mape_close, 2) if mape_close is not None else None
                                                            }]))
                                                    else:
                                                        st.warning("⚠️ Không tạo được dự báo hợp lệ (không đủ window/dữ liệu).")


                                    # # ====            
                                    # elif model_type == "Prophet":
                                    #     # Train Prophet
                                        

                                    #     model = train_prophet(df_train, ticker)  # lưu tự động vào models/prophet/prophet_{ticker}.pkl

                                    #     # Dự báo và lưu Predicted_Close
                                    #     try:
                                    #         pred_df = forecast_prophet(model, forecast_days=cfg["forecast_days"])
                                    #         pred_df = pred_df.rename(columns={"ds": "Date", "yhat": "Predicted_Close"})
                                    #         pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.strftime("%Y-%m-%d")
                                    #         pred_df["Ticker"] = ticker
                                    #         db.save_forecast_last(ticker, pred_df)
                                    #         st.success(f"✅ Mô hình Prophet đã được huấn luyện và lưu dự báo cho: {ticker}")
                                    #     except Exception:
                                    #         st.success(f"✅ Mô hình Prophet đã được huấn luyện cho: {ticker}.")

                                    # # ====
                                    # elif model_type == "ARIMA":
                                    #     # Huấn luyện và lưu model vào models/arima/arima_{ticker}.pkl
                                    #     model = train_arima(df_train, ticker)  

                                    #     try:
                                    #         # Dự báo sử dụng model vừa lưu
                                    #         pred_df = forecast_arima(
                                    #             df_clean, 
                                    #             forecast_days=cfg["forecast_days"], 
                                    #             use_saved_model=True,  # đảm bảo dùng model đã lưu
                                    #             ticker=ticker          # truyền ticker đúng
                                    #         )
                                    #         pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.strftime("%Y-%m-%d")
                                    #         pred_df["Ticker"] = ticker
                                    #         db.save_forecast_last(ticker, pred_df)
                                    #         st.success(f"✅ Mô hình ARIMA đã được huấn luyện và lưu dự báo cho: {ticker}")
                                    #     except Exception:
                                    #         st.success(f"✅ Mô hình ARIMA đã được huấn luyện cho: {ticker}.")

                            except Exception as e:
                                st.error(f"❌ Lỗi huấn luyện {ticker}: {e}")

                
                # -------------------------
                # TAB 2: Quản lý & huấn luyện toàn bộ
                # -------------------------
                with tab2:
                    # st.subheader("🔁 Huấn luyện lại toàn bộ")
                    source = "yf"
                    st.caption("Huấn luyện lại tất cả các mã hợp lệ, kết quả lưu về DB.")
                    with st.expander("ℹ️ Help: Huấn luyện toàn bộ"):
                        st.markdown(
                            "- **Mục đích**: Huấn luyện lại toàn bộ danh mục có thể làm cho mô hình học sâu hơn với các chỉ số dự báo OHLCV.\n"
                            "- **Lưu ý**: Thời gian huấn luyện có thể lâu tuỳ thuộc vào khối lượng dữ liệu danh mục của bạn. Những mã thiếu dữ liệu sẽ bị bỏ qua.\n"
                            "- **Tùy chọn**: Bạn có thể chọn chỉ huấn luyện các ticker thiếu dự báo (auto_only_missing) ở tab Quản lý huấn luyện tự động."
                        )

                    model_type = st.selectbox("Chọn mô hình huấn luyện lại:", ["LSTM", "TCN"], key="bulk_model")
                    epochs = st.slider("Số vòng lặp (Epochs)", 5, 200, 50, 5, key="bulk_epochs")
                    batch_size = st.slider("Batch size", 4, 128, 16, 4, key="bulk_batch")

                    
                    if st.button("🚀 Bắt đầu huấn luyện lại toàn bộ"):
                        with st.spinner("Đang huấn luyện lại toàn bộ danh mục... Vui lòng chờ trong giây lát"):
                            logs = []

                            for ticker in valid_tickers:
                                try:
                                    # df_raw = load_data_yf(ticker)
                                    df_raw = load_data_dl(ticker, source=source)
                                    df_clean = clean_dataframe(df_raw, ticker=ticker)
                                    if df_clean is None:
                                        logs.append({"Ticker": ticker, "Status": "❌ Không có dữ liệu"})
                                        continue

                                    df_train, cfg = get_training_config_from_df(df_clean)

                                    # -------------------
                                    # LSTM
                                    # -------------------
                                    if model_type == "LSTM":
                                        if len(df_train) < 60:
                                            logs.append({"Ticker": ticker, "Status": "❌ Dữ liệu quá ít"})
                                            continue

                                        # Train multi-output LSTM
                                        model, scaler = train_lstm_model(
                                            df_train,
                                            ticker=ticker,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            look_back=cfg["lookback"]
                                        )

                                        # Đánh giá insample multi-output
                                        rmse, mape = evaluate_lstm_insample(model, scaler, df_train, cfg["lookback"])

                                        # Dự báo multi-output
                                        df_full = df_clean.copy()

                                        # Lấy danh sách các cột đã train
                                        if hasattr(scaler, "feature_names_in_"):
                                            available_cols = list(scaler.feature_names_in_)
                                        else:
                                            logs.append({"Ticker": ticker, "Status": "❌ Không xác định được cột scaler"})
                                            continue

                                        # Kiểm tra dữ liệu có đủ cột không
                                        missing_cols = [col for col in available_cols if col not in df_full.columns]
                                        if missing_cols:
                                            logs.append({"Ticker": ticker, "Status": f"❌ Thiếu cột {missing_cols}"})
                                            continue

                                        pred_cols = [f"Predicted_{col}" for col in available_cols]

                                        # Chuẩn hóa dữ liệu input
                                        data_vals = df_full[available_cols].values
                                        scaled = scaler.transform(data_vals)
                                        candidate_dates = df_full["Date"].iloc[max(0, len(df_full) - cfg["forecast_days"]):].tolist()

                                        preds = []
                                        for day in candidate_dates:
                                            idxs = df_full.index[df_full["Date"] == day].tolist()
                                            if not idxs:
                                                continue
                                            idx = idxs[0]
                                            if day.weekday() >= 5:  # skip weekends
                                                continue
                                            if idx - cfg["lookback"] < 0:
                                                continue

                                            window = scaled[idx - cfg["lookback"]: idx]
                                            if window.shape[0] < cfg["lookback"]:
                                                continue

                                            X = window.reshape(1, cfg["lookback"], len(available_cols))
                                            pred_scaled = model.predict(X, verbose=0)[0]

                                            if len(pred_scaled) != len(available_cols):
                                                logs.append({"Ticker": ticker, "Status": "❌ Output không khớp input"})
                                                continue

                                            pred_real = scaler.inverse_transform([pred_scaled])[0]
                                            row_tuple = [day.strftime("%Y-%m-%d")] + [float(v) for v in pred_real] + [ticker]
                                            preds.append(tuple(row_tuple))

                                        if preds:
                                            df_forecast = pd.DataFrame(preds, columns=["Date"] + pred_cols + ["Ticker"])
                                            db.save_forecast_last(ticker, df_forecast)

                                            # Chuẩn hóa RMSE/MAPE về dict float
                                            rmse_safe = {col: round(rmse[col], 4) for col in available_cols} if isinstance(rmse, dict) else None
                                            mape_safe = {col: round(mape[col], 2) for col in available_cols} if isinstance(mape, dict) else None

                                            logs.append({
                                                "Ticker": ticker,
                                                "Status": "✅ Thành công",
                                                "RMSE": rmse_safe,
                                                "MAPE": mape_safe
                                            })
                                        else:
                                            logs.append({"Ticker": ticker, "Status": "⚠️ Không có dự báo"})
                                    
                                    # -------------------
                                    # TCN
                                    # -------------------
                                    elif model_type == "TCN":
                                        if len(df_train) < 60:
                                            logs.append({"Ticker": ticker, "Status": "❌ Dữ liệu quá ít"})
                                            continue

                                        # Train multi-output TCN
                                        model, scaler = train_tcn_model(
                                            df_train,
                                            ticker=ticker,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            look_back=cfg["lookback"]
                                        )

                                        # Đánh giá insample multi-output
                                        rmse, mape = evaluate_tcn_insample(model, scaler, df_train, cfg["lookback"])

                                        # Dự báo multi-output
                                        df_full = df_clean.copy()

                                        # Lấy danh sách các cột đã train
                                        if hasattr(scaler, "feature_names_in_"):
                                            available_cols = list(scaler.feature_names_in_)
                                        else:
                                            logs.append({"Ticker": ticker, "Status": "❌ Không xác định được cột scaler"})
                                            continue

                                        # Kiểm tra dữ liệu có đủ cột không
                                        missing_cols = [col for col in available_cols if col not in df_full.columns]
                                        if missing_cols:
                                            logs.append({"Ticker": ticker, "Status": f"❌ Thiếu cột {missing_cols}"})
                                            continue

                                        pred_cols = [f"Predicted_{col}" for col in available_cols]

                                        # Chuẩn hóa dữ liệu input
                                        data_vals = df_full[available_cols].values
                                        scaled = scaler.transform(data_vals)
                                        candidate_dates = df_full["Date"].iloc[max(0, len(df_full) - cfg["forecast_days"]):].tolist()

                                        preds = []
                                        for day in candidate_dates:
                                            idxs = df_full.index[df_full["Date"] == day].tolist()
                                            if not idxs:
                                                continue
                                            idx = idxs[0]
                                            if day.weekday() >= 5:  # skip weekends
                                                continue
                                            if idx - cfg["lookback"] < 0:
                                                continue

                                            window = scaled[idx - cfg["lookback"]: idx]
                                            if window.shape[0] < cfg["lookback"]:
                                                continue

                                            X = window.reshape(1, cfg["lookback"], len(available_cols))
                                            pred_scaled = model.predict(X, verbose=0)[0]

                                            if len(pred_scaled) != len(available_cols):
                                                logs.append({"Ticker": ticker, "Status": "❌ Output không khớp input"})
                                                continue

                                            pred_real = scaler.inverse_transform([pred_scaled])[0]
                                            row_tuple = [day.strftime("%Y-%m-%d")] + [float(v) for v in pred_real] + [ticker]
                                            preds.append(tuple(row_tuple))

                                        if preds:
                                            df_forecast = pd.DataFrame(preds, columns=["Date"] + pred_cols + ["Ticker"])
                                            db.save_forecast_last(ticker, df_forecast)

                                            # Chuẩn hóa RMSE/MAPE về dict float
                                            rmse_safe = {col: round(rmse[col], 4) for col in available_cols} if isinstance(rmse, dict) else None
                                            mape_safe = {col: round(mape[col], 2) for col in available_cols} if isinstance(mape, dict) else None

                                            logs.append({
                                                "Ticker": ticker,
                                                "Status": "✅ Thành công",
                                                "RMSE": rmse_safe,
                                                "MAPE": mape_safe
                                            })
                                        else:
                                            logs.append({"Ticker": ticker, "Status": "⚠️ Không có dự báo"})


                                    # # -------------------
                                    # # Prophet
                                    # # -------------------
                                    # elif model_type == "Prophet":
                                    #     model = train_prophet(df_train, ticker)
                                    #     try:
                                    #         pred_df = forecast_prophet(model, forecast_days=cfg["forecast_days"])
                                    #         pred_df = pred_df.rename(columns={"ds": "Date", "yhat": "Predicted_Close"})
                                    #         pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.strftime("%Y-%m-%d")
                                    #         pred_df["Ticker"] = ticker
                                    #         db.save_forecast_last(ticker, pred_df)
                                    #         logs.append({"Ticker": ticker, "Status": "✅ Mô hình Prophet đã được huấn luyện và lưu dự báo."})
                                    #     except Exception:
                                    #         logs.append({"Ticker": ticker, "Status": "✅ Mô hình Prophet đã được huấn luyện."})

                                    # # -------------------
                                    # # ARIMA
                                    # # -------------------
                                    # elif model_type == "ARIMA":
                                    #     try:
                                    #         # Huấn luyện và lưu model
                                    #         model = train_arima(df_train, ticker)

                                    #         # Dự báo
                                    #         pred_df = forecast_arima(
                                    #             df_clean, 
                                    #             forecast_days=cfg["forecast_days"], 
                                    #             use_saved_model=True,
                                    #             ticker=ticker
                                    #         )
                                    #         pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.strftime("%Y-%m-%d")
                                    #         pred_df["Ticker"] = ticker
                                    #         db.save_forecast_last(ticker, pred_df)
                                    #         logs.append({"Ticker": ticker, "Status": "✅ Mô hình ARIMA đã được huấn luyện và lưu dự báo."})

                                    #     except Exception as e:
                                    #         logs.append({"Ticker": ticker, "Status": f"⚠️ ARIMA huấn luyện xong (dự báo thất bại: {str(e)})"})


                                    else:
                                        logs.append({"Ticker": ticker, "Status": f"❌ Mô hình không hợp lệ: {model_type}"})

                                except Exception as e:
                                    logs.append({"Ticker": ticker, "Status": f"❌ Lỗi: {str(e)}"})

                            # -------------------
                            # Hiển thị log trong Streamlit
                            # -------------------
                            st.subheader("📊 Kết quả huấn luyện (log)")
                            st.dataframe(pd.DataFrame(logs), use_container_width=True)

                # -------------------------
                # TAB 3: Quản lý & huấn luyện tự động
                # -------------------------
                with tab3:
                    # st.subheader("⚙️ Quản lý & huấn luyện tự động")
                    source = "yf"
                    st.caption("Quản lý các mã thiếu `Predicted_Close`, cho phép huấn luyện thủ công hoặc bật auto-train.")
                    with st.expander("ℹ️ Help: Quản lý & huấn luyện tự động"):
                        st.markdown(
                            "- **Auto-train**: nếu bật, khi mở tab này hệ thống sẽ tự động huấn luyện các ticker thiếu dữ liệu Backtest.\n"
                            "- **Tối ưu tham số**: hệ thống sẽ thử các cấu hình nhanh (quick_validate) và chọn cấu hình có RMSE nhỏ nhất.\n"
                            "- **Lưu**: Sau khi huấn luyện, dự báo được lưu vào CSDL để làm dữ liệu phục vụ Backtest.\n"
                        )

                    auto_train = st.checkbox("Huấn luyện tự động khi có ticker mới", value=False,
                                            help="Bật để hệ thống tự động huấn luyện các mã thiếu dữ liệu Backtest.")

                    # danh sách ticker thiếu dữ liệu dự báo
                    missing_pred = find_tickers_missing_predicted_close(valid_tickers)

                    st.write("Danh sách các mã cổ phiếu thiếu dữ liệu Backtest:")
                    if missing_pred:
                        st.warning(", ".join(missing_pred))
                    else:
                        st.info("✔️ Không có mã thiếu dữ liệu Backtest.")

                    
                    if st.button("🚀 Huấn luyện các mã thiếu dữ liệu"):
                        logs = []
                        with st.spinner("Đang huấn luyện các mã… Vui lòng chờ trong giây lát"):
                            for ticker in missing_pred:
                                try:
                                    df_raw = load_data_dl(ticker, source=source)
                                    df_clean = clean_dataframe(df_raw, ticker=ticker)
                                    if df_clean is None:
                                        logs.append({"Ticker": ticker, "Status": "❌ No raw data"})
                                        continue

                                    # Auto-optimize params
                                    best_cfg, best_rmse = None, None
                                    try:
                                        best_cfg, best_rmse = auto_optimize_training_params_wrapper(df_clean)
                                    except Exception:
                                        _, cfg = get_training_config_from_df(df_clean)
                                        best_cfg = {
                                            "years": cfg["train_years"],
                                            "lookback": cfg["lookback"],
                                            "horizon": cfg["forecast_days"]
                                        }
                                        best_rmse = None

                                    # Prepare df_train using best_cfg
                                    df_train = df_clean[df_clean["Date"] >= (df_clean["Date"].max() - pd.DateOffset(years=best_cfg["years"]))]
                                    if len(df_train) < 60:
                                        logs.append({"Ticker": ticker, "Status": "❌ Dữ liệu quá ít"})
                                        continue

                                    # Train multi-output LSTM
                                    model, scaler = train_lstm_model(df_train, ticker=ticker, epochs=50, batch_size=16, look_back=cfg["lookback"])

                                    # Evaluate multi-output
                                    rmse, mape = evaluate_lstm_insample(model, scaler, df_train, best_cfg["lookback"])

                                    # Produce multi-output predictions
                                    df_full = df_clean.copy()

                                    # Lấy danh sách cột đã train
                                    if not hasattr(scaler, "feature_names_in_"):
                                        logs.append({"Ticker": ticker, "Status": "⚠️ Scaler không có thông tin cột, cần train lại."})
                                        continue
                                    available_cols = list(scaler.feature_names_in_)
                                    pred_cols = [f"Predicted_{col}" for col in available_cols]

                                    # Chuẩn hóa dữ liệu input
                                    data_vals = df_full[available_cols].values
                                    scaled = scaler.transform(data_vals)
                                    candidate_dates = df_full["Date"].iloc[max(0, len(df_full)-best_cfg["horizon"]):].tolist()

                                    preds = []
                                    for day in candidate_dates:
                                        idxs = df_full.index[df_full["Date"] == day].tolist()
                                        if not idxs:
                                            continue
                                        idx = idxs[0]
                                        if day.weekday() >= 5:  # skip weekends
                                            continue
                                        if idx - best_cfg["lookback"] < 0:
                                            continue
                                        window = scaled[idx - best_cfg["lookback"]: idx]
                                        if window.shape[0] < best_cfg["lookback"]:
                                            continue

                                        X = window.reshape(1, best_cfg["lookback"], len(available_cols))
                                        pred_scaled = model.predict(X, verbose=0)[0]
                                        pred_real = scaler.inverse_transform([pred_scaled])[0]

                                        row_tuple = [day.strftime("%Y-%m-%d")] + [float(v) for v in pred_real] + [ticker]
                                        preds.append(tuple(row_tuple))

                                    if preds:
                                        df_forecast = pd.DataFrame(preds, columns=["Date"] + pred_cols + ["Ticker"])
                                        db.save_forecast_last(ticker, df_forecast)
                                        logs.append({
                                            "Ticker": ticker,
                                            "Status": "✅ Mô hình tự động huấn luyện và lưu dữ liệu đã hoàn thành.",
                                            "Years": best_cfg["years"],
                                            "Lookback": best_cfg["lookback"],
                                            "Horizon": best_cfg["horizon"],
                                            "RMSE": {col: round(rmse.get(col, None), 4) for col in available_cols} if rmse else None,
                                            "MAPE": {col: round(mape.get(col, None), 2) for col in available_cols} if mape else None
                                        })
                                    else:
                                        logs.append({"Ticker": ticker, "Status": "⚠️ Không tạo được dự báo"})

                                except Exception as e:
                                    logs.append({"Ticker": ticker, "Status": f"❌ {e}"})

                        st.subheader("📊 Kết quả tự động huấn luyện")
                        st.dataframe(pd.DataFrame(logs), use_container_width=True)

                    # auto_train on-open behavior
                    
                    # source = "yf"
                    if auto_train and missing_pred:
                        st.info("🔄 Auto-train đang chạy cho các mã thiếu...")
                        auto_logs = []
                        for ticker in missing_pred:
                            try:
                                df_raw = load_data_dl(ticker, source=source)
                                df_clean = clean_dataframe(df_raw, ticker=ticker)
                                if df_clean is None:
                                    auto_logs.append({"Ticker": ticker, "Status": "❌ No raw data"})
                                    continue

                                # Auto-optimize
                                try:
                                    best_cfg, best_rmse = auto_optimize_training_params_wrapper(df_clean)
                                except Exception:
                                    _, cfg = get_training_config_from_df(df_clean)
                                    best_cfg = {
                                        "years": cfg["train_years"],
                                        "lookback": cfg["lookback"],
                                        "horizon": cfg["forecast_days"]
                                    }
                                    best_rmse = None

                                # Chuẩn bị dữ liệu train
                                df_train = df_clean[df_clean["Date"] >= (df_clean["Date"].max() - pd.DateOffset(years=best_cfg["years"]))]
                                if len(df_train) < 60:
                                    auto_logs.append({"Ticker": ticker, "Status": "❌ Dữ liệu quá ít"})
                                    continue

                                # Train multi-output LSTM
                                model, scaler = train_lstm_model(df_train, ticker=ticker, epochs=50, batch_size=16, look_back=cfg["lookback"])

                                # Evaluate multi-output
                                rmse, mape = evaluate_lstm_insample(model, scaler, df_train, best_cfg["lookback"])

                                # Produce multi-output predictions
                                df_full = df_clean.copy()

                                # Lấy danh sách cột đã train
                                if not hasattr(scaler, "feature_names_in_"):
                                    auto_logs.append({"Ticker": ticker, "Status": "⚠️ Scaler không có thông tin cột, cần train lại."})
                                    continue
                                available_cols = list(scaler.feature_names_in_)
                                pred_cols = [f"Predicted_{col}" for col in available_cols]

                                # Chuẩn hóa dữ liệu input
                                data_vals = df_full[available_cols].values
                                scaled = scaler.transform(data_vals)
                                candidate_dates = df_full["Date"].iloc[max(0, len(df_full)-best_cfg["horizon"]):].tolist()

                                preds = []
                                for day in candidate_dates:
                                    idxs = df_full.index[df_full["Date"] == day].tolist()
                                    if not idxs:
                                        continue
                                    idx = idxs[0]
                                    if day.weekday() >= 5:  # skip weekends
                                        continue
                                    if idx - best_cfg["lookback"] < 0:
                                        continue
                                    window = scaled[idx - best_cfg["lookback"]: idx]
                                    if window.shape[0] < best_cfg["lookback"]:
                                        continue

                                    X = window.reshape(1, best_cfg["lookback"], len(available_cols))
                                    pred_scaled = model.predict(X, verbose=0)[0]
                                    pred_real = scaler.inverse_transform([pred_scaled])[0]

                                    row_tuple = [day.strftime("%Y-%m-%d")] + [float(v) for v in pred_real] + [ticker]
                                    preds.append(tuple(row_tuple))

                                if preds:
                                    df_forecast = pd.DataFrame(preds, columns=["Date"] + pred_cols + ["Ticker"])
                                    db.save_forecast_last(ticker, df_forecast)
                                    auto_logs.append({
                                        "Ticker": ticker,
                                        "Status": f"✅ Mô hình tự động huấn luyện và lưu dữ liệu đã hoàn thành cho {ticker}.",
                                        "Years": best_cfg["years"],
                                        "Lookback": best_cfg["lookback"],
                                        "Horizon": best_cfg["horizon"],
                                        "RMSE": {col: round(rmse.get(col, None), 4) for col in available_cols} if rmse else None,
                                        "MAPE": {col: round(mape.get(col, None), 2) for col in available_cols} if mape else None
                                    })
                                else:
                                    auto_logs.append({"Ticker": ticker, "Status": "⚠️ Không tạo được dự báo"})

                            except Exception as e:
                                auto_logs.append({"Ticker": ticker, "Status": f"❌ {e}"})

                        st.subheader("📊 Kết quả tự động huấn luyện")
                        st.dataframe(pd.DataFrame(auto_logs), use_container_width=True)

                        st.subheader("📊 Kết quả tự động huấn luyện (on-open)")
                        st.dataframe(pd.DataFrame(auto_logs), use_container_width=True)

                



        # ---------------------------------------------------------------------------
                                # Tab Báo cáo
        # ---------------------------------------------------------------------------
        if "report" in tab_map:
            with tabs[tab_map["report"]]:

                
                # Lấy dữ liệu từ session
                weights = st.session_state.get("portfolio_weights")
                risk_result = st.session_state.get("risk_result")
                bt_results = st.session_state.get("backtest_result")
                perf_results = st.session_state.get("performance_result")

                sub_tabs = st.tabs(["📈 Phân tích & báo cáo", "📧 Gửi Báo cáo qua Email"])

                # === 📈 SUB-TAB 1: Phân tích & Gợi ý ===
                with sub_tabs[0]:
  
                    # # ============XUẤT BÁO CÁO PDF=============

                    st.header("📄 Xuất báo cáo PDF")
                    # 1. Chọn Ticker cần xuất
                    tickers_all = list(set(t[0] for t in bt_results.keys()))
                    tickers_selected = st.multiselect("🔎 Chọn mã cổ phiếu cần xuất báo cáo", tickers_all, default=tickers_all)

                    # 2. Chọn nội dung cần xuất
                    st.markdown("### 📑 Chọn nội dung cần xuất vào báo cáo")

                    select_all = st.checkbox("🧾 Chọn tất cả")

                    col1, col2 = st.columns(2)
                    with col1:
                        include_forecast = st.checkbox("📈 Dự đoán tương lai", value=select_all, key="include_forecast")
                        include_backtest = st.checkbox("📊 Backtest", value=True if select_all else False, key="include_backtest")
                        include_perf = st.checkbox("📈 Hiệu suất", value=select_all, key="include_perf")

                    with col2:
                        include_risk = st.checkbox("⚠️ Rủi ro", value=select_all, key="include_risk")
                        include_optimization = st.checkbox("📈 Tối ưu danh mục", value=select_all, key="include_optimization")
                        include_rebalance = st.checkbox("🔄 Tái cân bằng", value=select_all, key="include_rebalance")

                    # 3. Thực thi xuất báo cáo PDF
                    if st.button("🚀 Xuất báo cáo PDF"):
                        
                        # today_str = datetime.today().strftime("%d%m%Y")
                        today_str = st.session_state.get("today_str")
                        if not today_str:  # fallback nếu session chưa có giá trị
                            today_str = datetime.today().strftime("%d%m%Y")
                            st.session_state["today_str"] = today_str
                        # Cũ
                        # allocation_result = globals().get("allocation_result", {})
                        # Mới
                        allocation_result = st.session_state.get("allocation_result", globals().get("allocation_result", {}))
                        allocation_df = None
                        valid_tickers = []

                        for ticker in tickers_selected:
                            bt_df = None
                            perf_df = None
                            risk_df = None

                            for (t, strategy), _bt_df in bt_results.items():
                                if t != ticker:
                                    continue
                                bt_df = _bt_df
                                perf_df = perf_results.get((t, strategy)) if perf_results.get((t, strategy)) is not None else perf_results.get(t)
                                risk_df = risk_result.get(t)
                                break

                            if include_backtest and (not isinstance(bt_df, pd.DataFrame) or bt_df.empty):
                                st.warning(f"⚠️ Dữ liệu backtest không hợp lệ cho {ticker}")
                                continue
                            if include_perf and (not isinstance(perf_df, pd.DataFrame) or perf_df.empty):
                                st.warning(f"⚠️ Dữ liệu hiệu suất không hợp lệ cho {ticker}")
                                continue
                            if include_risk and (not isinstance(risk_df, pd.DataFrame) or risk_df.empty):
                                st.warning(f"⚠️ Dữ liệu rủi ro không hợp lệ cho {ticker}")
                                continue

                            valid_tickers.append({
                                "ticker": ticker,
                                "bt_df": bt_df,
                                "perf_df": perf_df,
                                "risk_df": risk_df,
                            })

                        if not valid_tickers:
                            st.error("❌ Không có mã nào đủ điều kiện để xuất báo cáo.")
                        else:
                            ticker_str = "_".join([d["ticker"] for d in valid_tickers])
                            selected_parts = []
                            if include_backtest: selected_parts.append("backtest")
                            if include_perf: selected_parts.append("perf")
                            if include_risk: selected_parts.append("risk")
                            if include_rebalance: selected_parts.append("rebalance")
                            if include_forecast: selected_parts.append("forecast")
                            if include_optimization: selected_parts.append("optimization")
                            suffix = "all" if len(selected_parts) == 6 else "_".join(selected_parts)

                            filename = f"{ticker_str}_{today_str}_{suffix}.pdf"
                            output_path = os.path.join(REPORTS_DIR, filename)

                            try:
                                # Ghép DataFrame và chuẩn hóa thành dict[ticker]
                                bt_df_all = pd.concat([
                                    d["bt_df"].assign(Ticker=d["ticker"]) for d in valid_tickers
                                    if d["bt_df"] is not None and not d["bt_df"].empty
                                ]) if include_backtest else None

                                perf_df_all = pd.concat([
                                    d["perf_df"].assign(Ticker=d["ticker"]) for d in valid_tickers
                                    if d["perf_df"] is not None and not d["perf_df"].empty
                                ]) if include_perf else None

                                risk_df_all = pd.concat([
                                    d["risk_df"].assign(Ticker=d["ticker"]) for d in valid_tickers
                                    if d["risk_df"] is not None and not d["risk_df"].empty
                                ]) if include_risk else None

                                # Chuyển sang dict[ticker]
                                bt_dict = {ticker: df.drop(columns=["Ticker"], errors="ignore") for ticker, df in bt_df_all.groupby("Ticker")} if bt_df_all is not None else None
                                perf_dict = {ticker: df.drop(columns=["Ticker"], errors="ignore") for ticker, df in perf_df_all.groupby("Ticker")} if perf_df_all is not None else None
                                risk_dict = {ticker: df.drop(columns=["Ticker"], errors="ignore") for ticker, df in risk_df_all.groupby("Ticker")} if risk_df_all is not None else None

                                # Tối ưu danh mục
                                allocation_df_list = []
                                for d in valid_tickers:
                                    alloc = allocation_result.get(d["ticker"])
                                    if isinstance(alloc, list):
                                        alloc = pd.DataFrame(alloc)
                                    if isinstance(alloc, pd.DataFrame) and not alloc.empty:
                                        alloc["Ticker"] = d["ticker"]
                                        allocation_df_list.append(alloc)
                                allocation_df = pd.concat(allocation_df_list, ignore_index=True) if allocation_df_list else None
                                allocation_dict = {ticker: df.drop(columns=["Ticker"], errors="ignore") for ticker, df in allocation_df.groupby("Ticker")} if allocation_df is not None else None

                                
                                # ===== DỰ BÁO TƯƠNG LAI =====
                                
                                forecast_dict = {
                                    d["ticker"]: st.session_state["forecast_result"][d["ticker"]]
                                    for d in valid_tickers
                                    if d["ticker"] in st.session_state["forecast_result"]
                                    and isinstance(st.session_state["forecast_result"][d["ticker"]], pd.DataFrame)
                                    and not st.session_state["forecast_result"][d["ticker"]].empty
                                } if include_forecast else None

                                # Tái cân bằng

                                # ===== Lấy dữ liệu Tái cân bằng từ session =====
                                # weights_df = st.session_state.get("weights_df") if include_rebalance else None

                                # ===== Lấy dữ liệu Tái cân bằng từ session =====
                                weights_df = st.session_state.get("rebalance_weights_df") if include_rebalance else None


                                # Gọi hàm chuẩn đã thống nhất
                                pdf_path = reporting.export_report_pdf(
                                    
                                    ticker=[d["ticker"] for d in valid_tickers],
                                    
                                    bt_df=bt_dict,
                                    perf_df=perf_dict,
                                    risk_df=risk_dict,
                                    weights_df=weights_df if include_rebalance else None,
                                    
                                    alloc_display=alloc_display,                
                                    adj_display=adj_display if 'adj_display' in locals() else None,  
                                    forecast_df=forecast_dict,  
                                    output_path=output_path,
                                    include_backtest=include_backtest,
                                    include_perf=include_perf,
                                    include_risk=include_risk,
                                    include_rebalance=include_rebalance,
                                    include_forecast=include_forecast,
                                    include_optimization=include_optimization,
                                )
                                with safe_open_pdf(pdf_file_path) as f:
                                        st.download_button(
                                            label="📥 Tải về báo cáo",
                                            data=f,
                                            file_name=filename,
                                            mime="application/pdf"
                                        )

                                st.success(f"✅ Đã xuất báo cáo: `{pdf_path}`")
                                

                            except Exception as e:
                                st.error(f"❌ Lỗi khi xuất báo cáo PDF: {e}")

                        

                # # === 📤 SUB-TAB 2: Gửi Báo cáo qua Email ===
                with sub_tabs[1]:

                    # 🚨 Bắt buộc lấy từ session_state
                    forecast_result = st.session_state.get("forecast_result", {})
                    

                    # ✅ Nếu dữ liệu còn sống
                    if not isinstance(forecast_result, dict) or not forecast_result:
                        st.warning("⚠️ Không có dữ liệu dự báo từ session_state.")
                    else:
                        # ✅ Lọc dữ liệu sạch từ forecast_result
                        forecast_result = {
                            tk: df for tk, df in forecast_result.items()
                            if isinstance(df, pd.DataFrame) and not df.empty
                        }

                    st.markdown("### 📧 Gửi báo cáo qua email")

                    # 🔒 Lấy thông tin người đăng nhập để dùng trong gửi email
                    user = st.session_state.get("user", {})
                    user_group = user.get("role", "guest")    # Nhóm quyền
                    recipients = user.get("email", "")        # Email người nhận

                    st.text_input(
                        "🔐 Nhóm người dùng",
                        value=user_group,
                        disabled=True,
                        help="Nhóm người dùng tự động lấy từ thông tin đăng nhập"
                    )

                    st.text_input(
                        "📨 Email người nhận",
                        value=recipients,
                        disabled=True,
                        help="Email tự động lấy từ thông tin đăng nhập"
                    )


                    if user_group == "guest":
                        selected_report_types = ["forecast"]
                    elif user_group == "member":
                        selected_report_types = ["forecast", "risk", "backtest"]
                    elif user_group == "premium":
                        selected_report_types = ["forecast", "risk", "backtest", "perf", "optimize", "rebalance"]
                    else:
                        selected_report_types = []

                    group_roles = {
                        "guest": ["📈 Dự đoán"],
                        "member": ["📈 Dự đoán", "⚠️ Rủi ro", "🔄 Backtest"],
                        "premium": ["📈 Dự đoán", "⚠️ Rủi ro", "🔄 Backtest", "✅ Hiệu suất", "📊 Tối ưu danh mục", "♻️ Tái cân bằng"]
                    }

                    selected_labels = group_roles.get(user_group, [])

                    selected_tickers = st.multiselect(
                        "📈 Chọn mã cổ phiếu",
                        options=tickers,
                        default=tickers,
                        help="Chọn mã cổ phiếu muốn gửi"
                    )

                    report_date = st.date_input(
                        "🗓️ Ngày báo cáo",
                        value=date.today(),
                        help="Chọn ngày dùng làm tham chiếu trong báo cáo"
                    )

                    send_button = st.button(" 📤 Gửi báo cáo")

                    # ==========================
                    # 📌 Hàm tiện ích lấy DataFrame theo ticker
                    # ==========================
                    def _pick_df(store, tk):
                        # Key trực tiếp theo ticker
                        v = store.get(tk)
                        if isinstance(v, pd.DataFrame) and not v.empty:
                            return v
                        # Key dạng (ticker, strategy)
                        for k, df in store.items():
                            if isinstance(k, tuple) and len(k) >= 1 and k[0] == tk and isinstance(df, pd.DataFrame) and not df.empty:
                                return df
                        return None


                    if send_button:
                        # ===== LẤY DỮ LIỆU GỐC =====
                        bt_results = st.session_state.get("bt_results", globals().get("bt_results", {}))
                        perf_results = st.session_state.get("perf_results", globals().get("perf_results", {}))
                        risk_result = st.session_state.get("risk_result", globals().get("risk_result", {}))
                        forecast_result = st.session_state.get("forecast_result", {})
                        allocation_result = st.session_state.get("allocation_result", globals().get("allocation_result", {}))

                        email_list = [e.strip() for e in recipients.split(",") if e.strip()]
                        if not email_list:
                            st.warning("Vui lòng nhập ít nhất một email.")
                            st.stop()

                        suffix_list = []
                        if "📈 Dự đoán" in selected_labels: suffix_list.append("forecast")
                        if "🔄 Backtest" in selected_labels: suffix_list.append("backtest")
                        if "✅ Hiệu suất" in selected_labels: suffix_list.append("perf")
                        if "⚠️ Rủi ro" in selected_labels: suffix_list.append("risk")
                        if "📊 Tối ưu danh mục" in selected_labels: suffix_list.append("optimize")
                        if "♻️ Tái cân bằng" in selected_labels: suffix_list.append("rebalance")

                        include_forecast = "📈 Dự đoán" in selected_labels
                        include_backtest = "🔄 Backtest" in selected_labels
                        include_perf = "✅ Hiệu suất" in selected_labels
                        include_risk = "⚠️ Rủi ro" in selected_labels
                        include_optimization = "📊 Tối ưu danh mục" in selected_labels
                        include_rebalance = "♻️ Tái cân bằng" in selected_labels

                        # ===== LẤY DỮ LIỆU =====
                        tickers_sorted = sorted(set(selected_tickers))

                        bt_df, perf_df, risk_df, forecast_df, allocation_df = {}, {}, {}, {}, {}

                        # --- Backtest ---
                        for ticker in tickers_sorted:
                            bt_df[ticker] = _pick_df(bt_results, ticker)

                        # --- Hiệu suất ---
                        for ticker in tickers_sorted:
                            perf_df[ticker] = _pick_df(perf_results, ticker)

                        # --- Rủi ro ---
                        for ticker in tickers_sorted:
                            risk_df[ticker] = risk_result.get(ticker)

                        # --- Forecast ---
                        for ticker in tickers_sorted:
                            df = forecast_result.get(ticker)
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                forecast_df[ticker] = df

                        # --- Tối ưu danh mục ---
                        for ticker in tickers_sorted:
                            allocation_df[ticker] = allocation_result.get(ticker)

                        # ===== Trọng số Tái cân bằng
                        w1 = globals().get("weights")
                        w2 = st.session_state.get("weights")
                        if isinstance(w1, pd.DataFrame) and not w1.empty:
                            weights_df = w1.copy()
                        elif isinstance(w2, pd.DataFrame) and not w2.empty:
                            weights_df = w2.copy()
                        else:
                            weights_df = pd.DataFrame()

                        filename = generate_pdf_filename(tickers_sorted, report_date, suffix_list)
                        tmp_path = os.path.join("reports/tmp", filename)

                        # đảm bảo folder tạm tồn tại
                        os.makedirs("reports/tmp", exist_ok=True)

                        try:
                            pdf_path = export_report_pdf(
                                ticker=tickers_sorted,           # <-- truyền LIST
                                bt_df=bt_df,                     # dict
                                perf_df=perf_df,                 # dict
                                risk_df=risk_df,                 # dict
                                weights_df=weights_df if include_rebalance else None,
                                    
                                # weights_df=weights_df,           # DataFrame
                                # allocation_df=allocation_df,     # dict
                                alloc_display=alloc_display,                
                                adj_display=adj_display if 'adj_display' in locals() else None, 
                                forecast_df=forecast_df,         # dict
                                forecast_result=forecast_df,     # dict
                                include_forecast=include_forecast,
                                include_backtest=include_backtest,
                                include_perf=include_perf,
                                include_risk=include_risk,
                                include_rebalance=include_rebalance,
                                include_optimization=include_optimization,
                                output_path=tmp_path,
                            )

                            if not pdf_path or not os.path.exists(pdf_path):
                                st.error(f"❌ Không tạo được file PDF báo cáo. export_report_pdf trả về: {pdf_path}")
                                st.stop()

                        except Exception as e:
                            st.error(f"❌ Lỗi khi xuất báo cáo: {e}")
                            st.code(traceback.format_exc())
                            st.stop()

                        # Gửi email
                        for email in email_list:
                            try:
                                subject = f"Báo cáo đầu tư cho {', '.join(tickers_sorted)} ({report_date.strftime('%d%m%Y')})"
                                body = "Xin vui lòng xem file đính kèm để biết chi tiết."
                                send_report_via_email(
                                    to_email=email,
                                    subject=subject,
                                    body=body,
                                    attachments=[pdf_path]
                                )
                                st.success(f"✅ Đã gửi báo cáo đến {email}")

                                if os.getenv("EMAIL_SAVE_SENT", "false").lower() == "true":
                                    os.makedirs("reports/sent", exist_ok=True)
                                    shutil.copy(pdf_path, os.path.join("reports/sent", os.path.basename(pdf_path)))

                            except Exception as e:
                                st.error(f"❌ Lỗi khi gửi email cho {email}: {e}")
                                st.code(traceback.format_exc())

    else:
        st.warning("Bạn không có quyền truy cập tab nào.")

# ===== Router =====
def router():
    init_session_keys()
    page = st.session_state.get("page", "login")
    user = st.session_state.get("user", None)

    if not user and page not in ["login", "register"]:
        st.session_state["page"] = "login"
        page = "login"

    if page == "login":
        login_page()
    elif page == "register":
        register_page()
    elif page == "home":
        page_home()
    elif page == "upgrade":
        upgrade_page()
    elif page == "admin_activate":
        admin_activate_page()
    else:
        st.error(f"❌ Không tìm thấy trang: {page}")

# ===== Chạy app =====
if __name__ == "__main__":
    router()
