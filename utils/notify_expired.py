from datetime import datetime
from utils.db_manager import get_connection   # ✅ FIX: unified DB layer
from utils.email_sender import send_report_via_email
import logging
import os
import schedule
import time


# ===== Cấu hình logging =====
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/notify_expired.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def notify_expired_subscriptions():
    try:
        today = datetime.today().date()

        with get_connection() as conn:
            with conn.cursor() as c:

                # ===== FIX: psycopg placeholder %s =====
                c.execute("""
                    SELECT s.id, u.full_name, u.email, s.role, s.end_date
                    FROM subscriptions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.end_date IS NOT NULL
                      AND DATE(s.end_date) < %s
                      AND s.status = 'active'
                """, (today,))

                expired = c.fetchall()

                logging.info(f"🔍 Tìm thấy {len(expired)} subscriptions hết hạn.")

                for sub_id, full_name, email, role, end_date in expired:

                    subject = "⚠️ Thông báo: Gói đăng ký của bạn đã hết hạn"

                    body = f"""
Xin chào {full_name},

Gói {role.upper()} của bạn đã hết hạn vào ngày {end_date}.
Vui lòng gia hạn để tiếp tục sử dụng các tính năng nâng cao.

👉 Liên hệ hỗ trợ hoặc đăng nhập để nâng cấp lại.

Trân trọng,
Đội ngũ hỗ trợ
"""

                    try:
                        send_report_via_email(email, subject, body)
                        logging.info(f"📧 Đã gửi email tới {email}")
                    except Exception as e:
                        logging.error(f"❌ Email fail {email}: {e}")

                    # ===== FIX: update query placeholder =====
                    c.execute("""
                        UPDATE subscriptions
                        SET status = 'expired',
                            active = 0
                        WHERE id = %s
                    """, (sub_id,))

                conn.commit()

        logging.info("✅ Hoàn tất xử lý subscriptions hết hạn.")

    except Exception as e:
        logging.error(f"❌ Lỗi notify_expired_subscriptions: {e}")


# ===== Scheduler Python =====
def run_scheduler():
    schedule.every().day.at("07:00").do(notify_expired_subscriptions)
    logging.info("🚀 Scheduler notify_expired đang chạy...")

    while True:
        schedule.run_pending()
        time.sleep(60)


# ===== ENTRY POINT =====
if __name__ == "__main__":
    run_scheduler()