#
import streamlit as st
from datetime import datetime
import psycopg
from config import DATABASE_URL

def admin_activate_page():
    user = st.session_state.get("user")
    if not user or user.get("role") != "admin":
        st.error("🚫 Bạn không có quyền truy cập trang này.")
        return

    st.title("🛠 Quản lý tài khoản pending")
    conn = psycopg.connect(DATABASE_URL)
    c = conn.cursor()

    if st.button("⬅️ Quay lại trang chủ"):
        st.session_state["page"] = "home"
        st.rerun()

    try:
        c.execute("""
            SELECT s.id, u.id, u.full_name, u.username, u.email, s.role, 
                   s.duration_months, s.amount_paid, s.start_date, s.end_date, s.status
            FROM subscriptions s
            JOIN users u ON s.user_id = u.id
            WHERE s.status = 'pending'
            AND s.id = (
                SELECT MAX(id) FROM subscriptions 
                WHERE user_id = u.id AND status = 'pending'
            )
            ORDER BY s.id DESC
        """)
        pending_subs = c.fetchall()
    except Exception as e:
        st.error(f"Lỗi khi truy vấn DB: {e}")
        conn.close()
        return

    if not pending_subs:
        st.info("✅ Không có tài khoản pending.")
        conn.close()
        return

    for sub in pending_subs:
        sub_id, user_id, full_name, username, email, role, duration, amount, start_date, end_date, status = sub
        with st.expander(f"👤 {full_name} ({username}) - {role.upper()} [{status}]"):
            st.write(f"- 📧 Email: {email}")
            st.write(f"- ⏳ Thời hạn: {duration} tháng")
            st.write(f"- 💰 Thanh toán: {amount:,} VNĐ")
            st.write(f"- 📅 Từ: {start_date} → {end_date}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Kích hoạt #{sub_id}", key=f"activate_{sub_id}"):
                    now = datetime.now()
                    c.execute("""
                        UPDATE subscriptions
                        SET status='active', received_by=?, received_at=?
                        WHERE id=?
                    """, (user["username"], now, sub_id))
                    c.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
                    conn.commit()
                    st.success(f"Đã kích hoạt tài khoản {username} ({role})")
                    st.rerun()
            with col2:
                if st.button(f"❌ Từ chối #{sub_id}", key=f"reject_{sub_id}"):
                    c.execute("UPDATE subscriptions SET status='rejected' WHERE id=?", (sub_id,))
                    conn.commit()
                    st.warning(f"Đã từ chối yêu cầu của {username}")
                    st.rerun()
    conn.close()

  

