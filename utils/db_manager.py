
# import psycopg
# import os
# import pandas as pd
# from datetime import datetime, timedelta
# from config import (
#     DB_HOST,
#     DB_NAME,
#     DB_USER,
#     DB_PASSWORD,
#     DB_PORT
# )

# # =========================
# # CONNECTION (SAFE + CLEAN)
# # =========================
# def get_connection():
#     return psycopg.connect(
#         host=DB_HOST,
#         dbname=DB_NAME,
#         user=DB_USER,
#         password=DB_PASSWORD,
#         port=DB_PORT,
#         sslmode="require",
#         sslrootcert=None,
#         connect_timeout=10,
#         keepalives=1,
#         keepalives_idle=30,
#         keepalives_interval=10,
#         keepalives_count=5
#     )
# # =========================
# # UTILS
# # =========================

# def to_date(x):
#     return pd.to_datetime(x).date()


# # =========================
# # INIT TABLE
# # =========================

# def init_table_if_not_exists(ticker: str):

#     with get_connection() as conn:
#         with conn.cursor() as c:

#             c.execute(f"""
#                 CREATE TABLE IF NOT EXISTS "{ticker}" (
#                     Date DATE NOT NULL,
#                     Open REAL,
#                     High REAL,
#                     Low REAL,
#                     Close REAL,
#                     Volume REAL,
#                     Predicted_Open REAL,
#                     Predicted_High REAL,
#                     Predicted_Low REAL,
#                     Predicted_Close REAL,
#                     Predicted_Volume REAL,
#                     Ticker TEXT NOT NULL,
#                     PRIMARY KEY (Date, Ticker)
#                 )
#             """)

#         conn.commit()


# # =========================
# # LOAD DATA
# # =========================

# def load_data(ticker: str):

#     try:
#         query = f'SELECT * FROM "{ticker}"'

#         with get_connection() as conn:
#             df = pd.read_sql_query(query, conn)

#         df.columns = df.columns.str.lower()

#         if "date" not in df.columns:
#             return pd.DataFrame()

#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         df = df.dropna(subset=["date"])
#         df = df.sort_values("date")

#         df = df.rename(columns={
#             "date": "Date",
#             "open": "Open",
#             "high": "High",
#             "low": "Low",
#             "close": "Close",
#             "volume": "Volume",
#             "ticker": "Ticker",
#             "predicted_open": "Predicted_Open",
#             "predicted_high": "Predicted_High",
#             "predicted_low": "Predicted_Low",
#             "predicted_close": "Predicted_Close",
#             "predicted_volume": "Predicted_Volume",
#         })

#         return df

#     except Exception as e:
#         print(f"[load_data] {ticker}: {e}")
#         return pd.DataFrame()


# # =========================
# # INSERT / UPDATE
# # =========================

# def insert_new_rows(ticker: str, df: pd.DataFrame):

#     if df is None or df.empty:
#         return

#     ticker = ticker.upper()

#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#     df = df.dropna(subset=["Date", "Close"])
#     df["Date"] = df["Date"].dt.date
#     df["Ticker"] = ticker

#     cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
#     df = df[[c for c in cols if c in df.columns]]

#     init_table_if_not_exists(ticker)

#     sql = f"""
#         INSERT INTO "{ticker}"
#         (Date, Open, High, Low, Close, Volume, Ticker)
#         VALUES (%s,%s,%s,%s,%s,%s,%s)
#         ON CONFLICT (Date, Ticker) DO UPDATE SET
#             Open=EXCLUDED.Open,
#             High=EXCLUDED.High,
#             Low=EXCLUDED.Low,
#             Close=EXCLUDED.Close,
#             Volume=EXCLUDED.Volume
#     """

#     with get_connection() as conn:
#         with conn.cursor() as c:
#             c.executemany(sql, df.values.tolist())

#         conn.commit()


# # =========================
# # LOAD FORECAST
# # =========================

# def load_forecast(ticker: str):

#     df = load_data(ticker)

#     if df.empty:
#         return pd.DataFrame()

#     return df[
#         [
#             "Date",
#             "Open", "High", "Low", "Close", "Volume",
#             "Predicted_Open",
#             "Predicted_High",
#             "Predicted_Low",
#             "Predicted_Close",
#             "Predicted_Volume",
#             "Ticker",
#         ]
#     ]


# # =========================
# # LIST TABLES
# # =========================

# def list_tables():

#     with get_connection() as conn:
#         with conn.cursor() as c:

#             c.execute("""
#                 SELECT tablename
#                 FROM pg_tables
#                 WHERE schemaname = 'public'
#             """)

#             tables = [r[0] for r in c.fetchall()]

#             stock_tables = []

#             for tb in tables:
#                 try:
#                     c.execute("""
#                         SELECT column_name
#                         FROM information_schema.columns
#                         WHERE table_name = %s
#                     """, (tb,))

#                     cols = [x[0].lower() for x in c.fetchall()]

#                     required = ["date", "open", "high", "low", "close", "ticker"]

#                     if all(col in cols for col in required):
#                         stock_tables.append(tb)

#                 except:
#                     pass

#             return stock_tables


# # =========================
# # INSPECT TABLE
# # =========================

# def inspect_table(ticker: str):

#     with get_connection() as conn:
#         with conn.cursor() as c:

#             c.execute("""
#                 SELECT column_name, data_type
#                 FROM information_schema.columns
#                 WHERE table_name = %s
#             """, (ticker,))

#             print(f"\n📌 {ticker}")
#             for col in c.fetchall():
#                 print(col)


# # =========================
# # SAVE FORECAST
# # =========================

# def save_forecast(ticker: str, df: pd.DataFrame):

#     if df is None or df.empty:
#         return

#     ticker = ticker.upper()

#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
#     df = df.dropna(subset=["Date"])

#     latest_date = max(df["Date"])
#     df = df[df["Date"] == latest_date]

#     if df.empty:
#         return

#     init_table_if_not_exists(ticker)

#     with get_connection() as conn:
#         with conn.cursor() as c:

#             for _, row in df.iterrows():

#                 date_val = row["Date"]

#                 c.execute(f"""
#                     SELECT 1
#                     FROM "{ticker}"
#                     WHERE Date=%s AND Ticker=%s
#                 """, (date_val, ticker))

#                 if c.fetchone() is None:
#                     c.execute(f"""
#                         INSERT INTO "{ticker}"
#                         (Date, Open, High, Low, Close, Volume,
#                          Predicted_Open, Predicted_High, Predicted_Low,
#                          Predicted_Close, Predicted_Volume, Ticker)
#                         VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
#                     """, (
#                         date_val,
#                         row.get("Open"),
#                         row.get("High"),
#                         row.get("Low"),
#                         row.get("Close"),
#                         row.get("Volume"),
#                         row.get("Predicted_Open"),
#                         row.get("Predicted_High"),
#                         row.get("Predicted_Low"),
#                         row.get("Predicted_Close"),
#                         row.get("Predicted_Volume"),
#                         ticker
#                     ))
#                 else:
#                     c.execute(f"""
#                         UPDATE "{ticker}"
#                         SET Predicted_Open=%s,
#                             Predicted_High=%s,
#                             Predicted_Low=%s,
#                             Predicted_Close=%s,
#                             Predicted_Volume=%s
#                         WHERE Date=%s AND Ticker=%s
#                     """, (
#                         row.get("Predicted_Open"),
#                         row.get("Predicted_High"),
#                         row.get("Predicted_Low"),
#                         row.get("Predicted_Close"),
#                         row.get("Predicted_Volume"),
#                         date_val,
#                         ticker
#                     ))

#         conn.commit()


# # =========================
# # SAVE BACKTEST
# # =========================

# def save_forecast_last(ticker: str, forecast_df: pd.DataFrame, days: int = 180):

#     if forecast_df is None or forecast_df.empty:
#         return

#     ticker = ticker.upper()

#     forecast_df = forecast_df.copy()
#     forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors="coerce")
#     forecast_df = forecast_df.dropna(subset=["Date"])

#     today = pd.Timestamp.today().normalize()
#     cutoff = today - pd.Timedelta(days=days)

#     forecast_df = forecast_df[
#         (forecast_df["Date"] < today) &
#         (forecast_df["Date"] >= cutoff)
#     ]

#     if forecast_df.empty:
#         return

#     forecast_df["Date"] = forecast_df["Date"].dt.date

#     pred_cols = [c for c in forecast_df.columns if c.startswith("Predicted_")]

#     init_table_if_not_exists(ticker)

#     with get_connection() as conn:
#         with conn.cursor() as c:

#             for row in forecast_df.itertuples(index=False):

#                 date_val = row.Date

#                 c.execute(f"""
#                     SELECT *
#                     FROM "{ticker}"
#                     WHERE Date = %s AND Ticker = %s
#                 """, (date_val, ticker))

#                 existing = c.fetchone()

#                 if not existing:
#                     continue

#                 colnames = [desc.name for desc in c.description]
#                 existing_map = dict(zip(colnames, existing))

#                 pred_values = {col: getattr(row, col) for col in pred_cols}

#                 updates = {
#                     k: v
#                     for k, v in pred_values.items()
#                     if v is not None and existing_map.get(k) is None
#                 }

#                 if updates:
#                     set_clause = ", ".join([f"{k}=%s" for k in updates.keys()])
#                     values = list(updates.values()) + [date_val, ticker]

#                     c.execute(f"""
#                         UPDATE "{ticker}"
#                         SET {set_clause}
#                         WHERE Date = %s AND Ticker = %s
#                     """, values)

#         conn.commit()

import psycopg
import os
import pandas as pd
from datetime import datetime, timedelta
from config import (
    DB_HOST,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_PORT
)

# =========================
# CONNECTION (SAFE + CLEAN)
# =========================
def get_connection():
    return psycopg.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        sslmode="require",
        sslrootcert=None,
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )

# =========================
# UTILS
# =========================
def to_date(x):
    return pd.to_datetime(x).date()


# =========================
# INIT TABLE
# =========================
def init_table_if_not_exists(ticker: str):

    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS "{ticker}" (
                    Date DATE NOT NULL,
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


# =========================
# LOAD DATA
# =========================
def load_data(ticker: str):

    try:
        query = f'SELECT * FROM "{ticker}"'

        with get_connection() as conn:
            df = pd.read_sql_query(query, conn)

        df.columns = df.columns.str.lower()

        if "date" not in df.columns:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")

        df = df.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "ticker": "Ticker",
            "predicted_open": "Predicted_Open",
            "predicted_high": "Predicted_High",
            "predicted_low": "Predicted_Low",
            "predicted_close": "Predicted_Close",
            "predicted_volume": "Predicted_Volume",
        })

        return df

    except Exception as e:
        print(f"[load_data] {ticker}: {e}")
        return pd.DataFrame()


# =========================
# INSERT / UPDATE
# =========================
def insert_new_rows(ticker: str, df: pd.DataFrame):

    if df is None or df.empty:
        return

    ticker = ticker.upper()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df["Date"] = df["Date"].dt.date
    df["Ticker"] = ticker

    cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    df = df[[c for c in cols if c in df.columns]]

    init_table_if_not_exists(ticker)

    sql = f"""
        INSERT INTO "{ticker}"
        (Date, Open, High, Low, Close, Volume, Ticker)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (Date, Ticker) DO UPDATE SET
            Open=EXCLUDED.Open,
            High=EXCLUDED.High,
            Low=EXCLUDED.Low,
            Close=EXCLUDED.Close,
            Volume=EXCLUDED.Volume
    """

    with get_connection() as conn:
        with conn.cursor() as c:
            c.executemany(sql, [tuple(x) for x in df.values.tolist()])
        conn.commit()


# =========================
# LOAD FORECAST
# =========================
def load_forecast(ticker: str):

    df = load_data(ticker)

    if df.empty:
        return pd.DataFrame()

    return df[
        [
            "Date",
            "Open", "High", "Low", "Close", "Volume",
            "Predicted_Open",
            "Predicted_High",
            "Predicted_Low",
            "Predicted_Close",
            "Predicted_Volume",
            "Ticker",
        ]
    ]


# =========================
# LIST TABLES
# =========================
def list_tables():

    with get_connection() as conn:
        with conn.cursor() as c:

            c.execute("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
            """)

            tables = [r[0] for r in c.fetchall()]

            stock_tables = []

            for tb in tables:
                try:
                    c.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = %s
                    """, (tb,))

                    cols = [x[0].lower() for x in c.fetchall()]

                    required = ["date", "open", "high", "low", "close", "ticker"]

                    if all(col in cols for col in required):
                        stock_tables.append(tb)

                except:
                    pass

            return stock_tables


# =========================
# INSPECT TABLE
# =========================
def inspect_table(ticker: str):

    with get_connection() as conn:
        with conn.cursor() as c:

            c.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
            """, (ticker,))

            print(f"\n📌 {ticker}")
            for col in c.fetchall():
                print(col)


# =========================
# SAVE FORECAST
# =========================
def save_forecast(ticker: str, df: pd.DataFrame):

    if df is None or df.empty:
        return

    ticker = ticker.upper()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.dropna(subset=["Date"])

    latest_date = max(df["Date"])
    df = df[df["Date"] == latest_date]

    if df.empty:
        return

    init_table_if_not_exists(ticker)

    with get_connection() as conn:
        with conn.cursor() as c:

            for _, row in df.iterrows():

                date_val = row["Date"]

                c.execute(f"""
                    SELECT 1
                    FROM "{ticker}"
                    WHERE Date=%s AND Ticker=%s
                """, (date_val, ticker))

                if c.fetchone() is None:
                    c.execute(f"""
                        INSERT INTO "{ticker}"
                        (Date, Open, High, Low, Close, Volume,
                         Predicted_Open, Predicted_High, Predicted_Low,
                         Predicted_Close, Predicted_Volume, Ticker)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        date_val,
                        row.get("Open"),
                        row.get("High"),
                        row.get("Low"),
                        row.get("Close"),
                        row.get("Volume"),
                        row.get("Predicted_Open"),
                        row.get("Predicted_High"),
                        row.get("Predicted_Low"),
                        row.get("Predicted_Close"),
                        row.get("Predicted_Volume"),
                        ticker
                    ))
                else:
                    c.execute(f"""
                        UPDATE "{ticker}"
                        SET Predicted_Open=%s,
                            Predicted_High=%s,
                            Predicted_Low=%s,
                            Predicted_Close=%s,
                            Predicted_Volume=%s
                        WHERE Date=%s AND Ticker=%s
                    """, (
                        row.get("Predicted_Open"),
                        row.get("Predicted_High"),
                        row.get("Predicted_Low"),
                        row.get("Predicted_Close"),
                        row.get("Predicted_Volume"),
                        date_val,
                        ticker
                    ))

        conn.commit()


# =========================
# SAVE BACKTEST
# =========================
def save_forecast_last(ticker: str, forecast_df: pd.DataFrame, days: int = 180):

    if forecast_df is None or forecast_df.empty:
        return

    ticker = ticker.upper()

    forecast_df = forecast_df.copy()
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors="coerce")
    forecast_df = forecast_df.dropna(subset=["Date"])

    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=days)

    forecast_df = forecast_df[
        (forecast_df["Date"] < today) &
        (forecast_df["Date"] >= cutoff)
    ]

    if forecast_df.empty:
        return

    forecast_df["Date"] = forecast_df["Date"].dt.date

    pred_cols = [c for c in forecast_df.columns if c.startswith("Predicted_")]

    init_table_if_not_exists(ticker)

    with get_connection() as conn:
        with conn.cursor() as c:

            for row in forecast_df.itertuples(index=False):

                date_val = row.Date

                c.execute(f"""
                    SELECT *
                    FROM "{ticker}"
                    WHERE Date = %s AND Ticker = %s
                """, (date_val, ticker))

                existing = c.fetchone()

                if not existing:
                    continue

                colnames = [desc.name for desc in c.description]
                existing_map = dict(zip(colnames, existing))

                pred_values = {col: getattr(row, col) for col in pred_cols}

                updates = {
                    k: v
                    for k, v in pred_values.items()
                    if v is not None and existing_map.get(k) is None
                }

                if updates:
                    set_clause = ", ".join([f"{k}=%s" for k in updates.keys()])
                    values = list(updates.values()) + [date_val, ticker]

                    c.execute(f"""
                        UPDATE "{ticker}"
                        SET {set_clause}
                        WHERE Date = %s AND Ticker = %s
                    """, values)

        conn.commit()