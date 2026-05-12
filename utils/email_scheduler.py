# import schedule
# import time
# import threading
# import os
# import sqlite3
# from datetime import datetime, date
# import pytz
# from utils.email_sender import send_report_via_email
# from utils.reporting import export_report_pdf, generate_pdf_filename

# DB_PATH = "database/users.db"
# TIMEZONE = pytz.timezone("Asia/Ho_Chi_Minh")

# # ===== LẤY DỮ LIỆU TỪ DB =====
# def get_users():
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
#         cursor.execute("SELECT id, full_name, username, email, role FROM users")
#         users = cursor.fetchall()
#         conn.close()
#         return [{"id": u[0], "full_name": u[1], "username": u[2], "email": u[3], "role": u[4]} for u in users]
#     except Exception as e:
#         print(f"❌ Lỗi lấy users từ DB: {e}")
#         return []

# def get_user_tickers(user_id):
#     """Lấy danh sách ticker mà user quan tâm (lấy tất cả ticker hiện có)"""
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT ticker FROM user_ticker_history
#             WHERE user_id = ?
#             ORDER BY last_viewed DESC
#         """, (user_id,))
#         tickers = [row[0] for row in cursor.fetchall()]
#         conn.close()
#         return tickers
#     except Exception as e:
#         print(f"❌ Lỗi lấy tickers cho user {user_id}: {e}")
#         return []

# # ===== GHI TICKER NGAY KHI USER NHẬP =====
# def record_user_ticker_view(user_id: int, ticker: str):
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()

#         today = datetime.now().date()
#         cursor.execute("""
#             SELECT id FROM user_ticker_history
#             WHERE user_id = ? AND ticker = ? AND DATE(last_viewed) = ?
#         """, (user_id, ticker, today))
#         row = cursor.fetchone()

#         if row:
#             cursor.execute(
#                 "UPDATE user_ticker_history SET last_viewed = ? WHERE id = ?",
#                 (datetime.now(), row[0])
#             )
#         else:
#             cursor.execute(
#                 "INSERT INTO user_ticker_history (user_id, ticker, last_viewed) VALUES (?, ?, ?)",
#                 (user_id, ticker, datetime.now())
#             )
#         conn.commit()
#         conn.close()
#         print(f"📝 User {user_id} quan tâm ticker {ticker} được ghi vào DB.")
#     except Exception as e:
#         print(f"❌ Lỗi ghi ticker cho user {user_id}: {e}")

# # ===== QUYỀN TAB =====
# ROLE_TAB_PERMISSIONS = {
#     "guest": ["forecast"],
#     "member": ["forecast", "risk", "backtest_perf"],
#     "premium": ["forecast", "risk", "backtest_perf", "optimize", "rebalance"],
#     "supervisor": ["forecast", "risk", "backtest_perf", "optimize", "rebalance", "train", "report"],
#     "admin": ["forecast", "risk", "backtest_perf", "optimize", "rebalance", "train", "report"],
# }

# def get_tab_permissions(role):
#     return ROLE_TAB_PERMISSIONS.get(role, [])

# # ===== LOẠI BÁO CÁO =====
# def get_report_types_by_role(role):
#     if role == "guest":
#         return ["forecast"]
#     elif role == "member":
#         return ["backtest", "perf", "risk", "forecast"]
#     elif role == "premium":
#         return ["forecast", "risk", "backtest", "perf", "optimize", "rebalance"]
#     elif role in ("supervisor", "admin"):
#         return ["forecast", "risk", "backtest", "perf", "optimize", "rebalance", "train", "report"]
#     return []

# # ===== GHI LOG EMAIL =====
# def log_email(user_id, email, pdf_file, status):
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO email_logs (role, recipients, pdf_file, status)
#             VALUES (?, ?, ?, ?)
#         """, (f"{user_id}", email, pdf_file, status))
#         conn.commit()
#         conn.close()
#         print(f"📝 Log đã lưu cho user {user_id} ({email}) với trạng thái {status}")
#     except Exception as e:
#         print(f"❌ Lỗi lưu log email cho user {user_id}: {e}")

# # ===== GỬI EMAIL BÁO CÁO =====
# def send_email_report():
#     now = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
#     print(f"📬 [{now}] Chạy job gửi báo cáo cho tất cả user")

#     users = get_users()
#     report_date = date.today()

#     for user in users:
#         user_id = user["id"]
#         email = user["email"]
#         role = user["role"]

#         if not email:
#             print(f"⚠️ User {user_id} ({user['username']}) không có email, bỏ qua")
#             continue

#         tickers = get_user_tickers(user_id)
#         report_types = get_report_types_by_role(role)

#         if not tickers or not report_types:
#             print(f"⚠️ User {user_id} ({email}) không có ticker hoặc loại báo cáo hợp lệ")
#             continue

#         if role == "supervisor":
#             print(f"ℹ️ Supervisor {email} không nhận email, chỉ log")
#             log_email(user_id, email, "N/A", "SKIPPED")
#             continue

#         filename = generate_pdf_filename(tickers, report_date, report_types)
#         tmp_folder = os.path.join("reports", "tmp")
#         os.makedirs(tmp_folder, exist_ok=True)
#         pdf_path = os.path.join(tmp_folder, filename)

#         try:
#             export_report_pdf(
#                 ticker=tickers if isinstance(tickers, str) else ", ".join(tickers),
#                 output_path=pdf_path,
#                 include_forecast="forecast" in report_types,
#                 include_backtest="backtest" in report_types,
#                 include_perf="perf" in report_types,
#                 include_risk="risk" in report_types,
#                 include_optimization="optimize" in report_types,
#                 include_rebalance="rebalance" in report_types,
#             )
#             print(f"⏳ Đã tạo báo cáo PDF cho user {email}: {pdf_path}")
#         except Exception as e:
#             print(f"❌ Lỗi tạo PDF cho user {email}: {e}")
#             log_email(user_id, email, pdf_path, "PDF ERROR")
#             continue

#         status = "SUCCESS"
#         try:
#             send_report_via_email(
#                 to_email=email,
#                 subject=f"Báo cáo đầu tư ngày {report_date.strftime('%d/%m/%Y')}",
#                 body="Xin vui lòng xem báo cáo đính kèm.",
#                 attachments=[pdf_path],
#             )
#             print(f"✅ Đã gửi email tới {email}")
#         except Exception as e:
#             print(f"❌ Lỗi gửi email tới {email}: {e}")
#             status = "FAILED"

#         log_email(user_id, email, pdf_path, status)

# # ===== SCHEDULER =====
# def run_email_scheduler():
#     schedule.every().day.at("08:00").do(send_email_report)
#     schedule.every().day.at("08:15").do(send_email_report)
#     schedule.every().day.at("08:30").do(send_email_report)
    

#     def loop():
#         while True:
#             schedule.run_pending()
#             time.sleep(60)

#     thread = threading.Thread(target=loop, daemon=True)
#     thread.start()
#     print("🚀 Email scheduler đang chạy...")

# # ===== CHẠY =====
# if __name__ == "__main__":
#     run_email_scheduler()
#     while True:
#         time.sleep(3600)


