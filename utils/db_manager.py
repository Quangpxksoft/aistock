# utils/db_manager.py
import sqlite3
import pandas as pd
import os

import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

DB_PATH = os.path.join("database", "data.db")

# ---------------------------
# Kết nối tới database
# ---------------------------
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

# ---------------------------
# Tạo bảng mới (nếu chưa có)
# ---------------------------

def init_table_if_not_exists(ticker: str):
    """
    Tạo bảng cho 1 ticker nếu chưa có.
    Schema chuẩn:
    Date, Open, High, Low, Close, Volume,
    Predicted_Open, Predicted_High, Predicted_Low, Predicted_Close, Predicted_Volume, Ticker
    PRIMARY KEY = (Date, Ticker)
    """
    with get_connection() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{ticker}" (
                Date TEXT NOT NULL,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume REAL,
                Predicted_Open REAL,
                Predicted_High REAL,
                Predicted_Low REAL,
                Predicted_Close REAL,
                Predicted_Volume REAL,
                Ticker TEXT NOT NULL,
                PRIMARY KEY (Date, Ticker)
            )
        """)
        conn.commit()

# ---------------------------
# Load dữ liệu từ DB
# ---------------------------
def load_data(ticker: str) -> pd.DataFrame:
    with get_connection() as conn:
        try:
            return pd.read_sql_query(f'SELECT * FROM "{ticker}"', conn)
        except Exception:
            return pd.DataFrame()

# ---------------------------
# Ghi dữ liệu vào DB
# ---------------------------

def insert_new_rows(ticker: str, df: pd.DataFrame):
    """
    Chèn dữ liệu mới vào bảng SQLite.
    - Nếu bản ghi (Date, Ticker) chưa tồn tại -> insert mới
    - Nếu bản ghi đã tồn tại -> chỉ update các giá trị gốc (Open, High, Low, Close, Volume),
      không ảnh hưởng các cột Predicted_*
    """
    if df is None or df.empty:
        return

    ticker = ticker.upper()

    # Chuẩn hóa Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])  # bỏ dòng Date hoặc Close không hợp lệ
    if df.empty:
        return
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Giữ các cột OHLCV và Ticker
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()
    df["Ticker"] = ticker

    # Chuẩn hóa kiểu dữ liệu
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    init_table_if_not_exists(ticker)

    with get_connection() as conn:
        sql = f"""
            INSERT INTO "{ticker}" (Date, Open, High, Low, Close, Volume, Ticker)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(Date, Ticker) DO UPDATE SET
                Open=excluded.Open,
                High=excluded.High,
                Low=excluded.Low,
                Close=excluded.Close,
                Volume=excluded.Volume
        """
        values = df.values.tolist()
        conn.executemany(sql, values)
        conn.commit()

# ---------------------------
# Load dự đoán từ DB
# ---------------------------
def load_forecast(ticker: str) -> pd.DataFrame:
    df = load_data(ticker)
    if df.empty:
        return pd.DataFrame()
    return df[[
        "Date", "Predicted_Open", "Predicted_High",
        "Predicted_Low", "Predicted_Close", "Predicted_Volume", "Ticker"
    ]]

# ---------------------------
# Liệt kê tất cả các bảng
# ---------------------------
def list_tables():
    with get_connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]

# ---------------------------
# Debug: kiểm tra cấu trúc bảng
# ---------------------------
def inspect_table(ticker: str):
    with get_connection() as conn:
        cursor = conn.execute(f"PRAGMA table_info('{ticker}')")
        cols = cursor.fetchall()
        print(f"📌 Table: {ticker}")
        for col in cols:
            print(f"  - {col[1]} ({col[2]})")

# ---------------------------
# Lưu dự báo cho hôm qua và hôm nay
# ---------------------------

def save_forecast(ticker: str, df: pd.DataFrame):
    """
    Cập nhật Predicted_* cho hôm qua trong bảng ticker.
    - Chỉ update/insert nếu các cột gốc (Open, High, Low, Close, Volume) có giá trị.
    - Không ảnh hưởng hôm nay hoặc các ngày khác.
    """
    if df is None or df.empty:
        print(f"[{ticker}] DataFrame trống, bỏ qua.")
        return

    ticker = ticker.upper()
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    yesterday = datetime.now().date() - timedelta(days=1)

    # Lọc chỉ hôm qua
    df_yesterday = df[df["Date"] == yesterday]
    if df_yesterday.empty:
        print(f"[{ticker}] Không có dữ liệu cho hôm qua ({yesterday}), bỏ qua.")
        return

    init_table_if_not_exists(ticker)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        for _, row in df_yesterday.iterrows():
            cursor.execute(f"""
                SELECT Predicted_Open, Predicted_High, Predicted_Low, Predicted_Close, Predicted_Volume
                FROM "{ticker}"
                WHERE Date=? AND Ticker=?
            """, (row["Date"].isoformat(), ticker))
            existing = cursor.fetchone()

            if existing is None:
                cursor.execute(f"""
                    INSERT INTO "{ticker}" 
                    (Date, Open, High, Low, Close, Volume,
                     Predicted_Open, Predicted_High, Predicted_Low, Predicted_Close, Predicted_Volume, Ticker)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get("Date").isoformat(),
                    row.get("Open"), row.get("High"), row.get("Low"),
                    row.get("Close"), row.get("Volume"),
                    row.get("Predicted_Open"), row.get("Predicted_High"),
                    row.get("Predicted_Low"), row.get("Predicted_Close"),
                    row.get("Predicted_Volume"),
                    ticker
                ))
            else:
                cursor.execute(f"""
                    UPDATE "{ticker}"
                    SET Predicted_Open=?,
                        Predicted_High=?,
                        Predicted_Low=?,
                        Predicted_Close=?,
                        Predicted_Volume=?
                    WHERE Date=? AND Ticker=?
                """, (
                    row.get("Predicted_Open"), row.get("Predicted_High"),
                    row.get("Predicted_Low"), row.get("Predicted_Close"),
                    row.get("Predicted_Volume"),
                    row["Date"].isoformat(), ticker
                ))

        conn.commit()
        print(f"[{ticker}] Hoàn tất cập nhật dự báo hôm qua ({yesterday}).")

def save_forecast_last(ticker: str, forecast_df: pd.DataFrame, days: int = 180):
    """
    Lưu dự báo lịch sử (backtest) cho multi-output.
    - Chỉ lưu Predicted_* của `days` ngày gần nhất (tính đến hôm qua).
    - Chỉ điền vào chỗ trống nếu trong DB có dữ liệu gốc.
    - Không ghi đè dữ liệu đã có sẵn.
    - Đồng bộ với tất cả cột Predicted_*.
    """
    if forecast_df is None or forecast_df.empty:
        return

    ticker = ticker.upper()
    forecast_df = forecast_df.copy()

    # Chuẩn hóa Date
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors="coerce")
    forecast_df = forecast_df.dropna(subset=["Date"])
    if forecast_df.empty:
        return

    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=days)

    # Lọc chỉ những ngày cần lưu
    forecast_df = forecast_df[
        (forecast_df["Date"] < today) & (forecast_df["Date"] >= cutoff)
    ]
    if forecast_df.empty:
        return

    forecast_df["Date"] = forecast_df["Date"].dt.strftime("%Y-%m-%d")
    forecast_df["Ticker"] = ticker

    # Lấy tất cả cột Predicted_* hiện có trong DataFrame
    pred_cols = [col for col in forecast_df.columns if col.startswith("Predicted_")]

    # Tạo bảng nếu chưa tồn tại
    init_table_if_not_exists(ticker)

    with get_connection() as conn:
        for row in forecast_df.itertuples(index=False):
            date = row.Date
            preds = {col: getattr(row, col, None) for col in pred_cols}

            # Lấy bản ghi gốc từ DB
            cur = conn.execute(
                f'SELECT * FROM "{ticker}" WHERE Date=? AND Ticker=?',
                (date, ticker),
            )
            existing = cur.fetchone()
            if existing is None:
                continue  # bỏ qua nếu bản ghi gốc chưa có

            # Lấy tên cột từ cursor.description
            columns = [desc[0] for desc in cur.description]

            # Map Predicted_* với giá trị hiện có trong DB
            existing_map = {col: existing[idx] for idx, col in enumerate(columns) if col in pred_cols}

            # Chỉ update những cột Predicted_* còn trống và có giá trị dự báo
            updates = {col: val for col, val in preds.items() if val is not None and existing_map.get(col) is None}

            if updates:
                set_clause = ", ".join([f"{col}=?" for col in updates.keys()])
                values = list(updates.values()) + [date, ticker]
                conn.execute(
                    f'UPDATE "{ticker}" SET {set_clause} WHERE Date=? AND Ticker=?',
                    values,
                )
        conn.commit()
