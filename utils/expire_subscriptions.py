from utils.db_manager import get_connection
from datetime import date


def expire_subscriptions():
    """Tự động hết hạn subscription khi quá hạn"""

    today = date.today()

    try:
        with get_connection() as conn:
            with conn.cursor() as c:

                # =========================
                # FIND EXPIRED SUBSCRIPTIONS
                # =========================
                c.execute("""
                    SELECT s.id, s.user_id, s.role, s.end_date
                    FROM subscriptions s
                    WHERE s.status = 'active'
                      AND s.end_date < %s
                """, (today,))

                expired = c.fetchall()

                for sub_id, user_id, role, end_date in expired:

                    print(f"⏳ Subscription {sub_id} (User {user_id}) đã hết hạn {end_date}, hạ về guest...")

                    # =========================
                    # UPDATE SUBSCRIPTION STATUS
                    # =========================
                    c.execute("""
                        UPDATE subscriptions
                        SET status = 'expired',
                            active = 0
                        WHERE id = %s
                    """, (sub_id,))

                    # =========================
                    # CHECK IF USER STILL HAS ACTIVE SUBSCRIPTIONS
                    # =========================
                    c.execute("""
                        SELECT COUNT(*)
                        FROM subscriptions
                        WHERE user_id = %s AND status = 'active'
                    """, (user_id,))

                    active_count = c.fetchone()[0]

                    # =========================
                    # DOWNGRADE USER ROLE IF NO ACTIVE SUB
                    # =========================
                    if active_count == 0:
                        c.execute("""
                            UPDATE users
                            SET role = 'guest'
                            WHERE id = %s
                        """, (user_id,))

                conn.commit()

        print("✅ Auto-expire check xong.")

    except Exception as e:
        print(f"❌ Lỗi expire_subscriptions: {e}")