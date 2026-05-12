# utils/expire_subscriptions.py
import sqlite3
from datetime import date
from config import DB_PATH

def expire_subscriptions():
    """Tự động hết hạn subscription khi quá hạn"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    today = date.today()

    # Tìm subscriptions active nhưng đã quá hạn
    c.execute("""
        SELECT s.id, s.user_id, s.role, s.end_date
        FROM subscriptions s
        WHERE s.status='active' AND s.end_date < ?
    """, (today,))
    expired = c.fetchall()

    for sub_id, user_id, role, end_date in expired:
        print(f"⏳ Subscription {sub_id} (User {user_id}) đã hết hạn {end_date}, hạ về guest...")

        # Cập nhật subscription
        c.execute("UPDATE subscriptions SET status='expired', active=0 WHERE id=?", (sub_id,))

        # Kiểm tra user có còn subscription active khác không
        c.execute("SELECT COUNT(*) FROM subscriptions WHERE user_id=? AND status='active'", (user_id,))
        active_count = c.fetchone()[0]

        if active_count == 0:
            # Nếu không còn active subscription → hạ user về guest
            c.execute("UPDATE users SET role='guest' WHERE id=?", (user_id,))

    conn.commit()
    conn.close()
    print("✅ Auto-expire check xong.")
