# upgrade_page.py
import streamlit as st
from utils.user_manager import upgrade_user
from utils.user_subscription import calculate_amount

def upgrade_page():
    user = st.session_state.get("user")
    if not user:
        st.session_state["page"] = "login"
        st.experimental_rerun()
        return

    st.title("💳 Nâng cấp tài khoản")
    st.write(f"Bạn hiện tại là **{user['role']}**")

    # Lựa chọn role
    role_options = []
    if user["role"] == "guest":
        role_options = ["member", "premium"]
    elif user["role"] == "member":
        role_options = ["premium"]

    if not role_options:
        st.info("Bạn đang ở gói cao nhất hoặc không có quyền nâng cấp.")
        return

    new_role = st.selectbox("Chọn gói muốn nâng cấp", role_options)
    duration_months = st.radio("Chọn thời hạn:", [1, 6, 12], index=0)
    payment_method = st.selectbox("Phương thức thanh toán", ["Thẻ tín dụng", "PayPal", "Chuyển khoản"])
    payment_details = st.text_area("Chi tiết thanh toán")

    if st.button("Xác nhận nâng cấp"):
        amount_paid = calculate_amount(new_role, duration_months)
        upgrade_user(user["id"], new_role, payment_method, payment_details, duration_months, amount_paid)
        st.success(f"⏳ Yêu cầu nâng cấp lên **{new_role}** đã được gửi, chờ admin duyệt.")
