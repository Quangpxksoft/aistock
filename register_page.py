import streamlit as st
from utils.user_manager import register_user, login_user, upgrade_user
from utils.db_manager import get_connection   # ✅ FIX: dùng đúng module của bạn


# Quyền và lợi ích
FEATURES_INFO = {
    "forecast": ("📊 Dự báo cổ phiếu theo các mô hình học máy", "Dự đoán xu hướng giá, xu hướng của NĐT, dự báo thống kê mô tả phân vị theo mã cổ phiếu và khối lượng khớp lệnh, dự báo hành vi NĐT, hỗ trợ quyết định mua/bán, lập kế hoạch đầu tư. Hệ thống chatbot thông minh sẵn sàng hộ trợ người dùng trong phạm vi hệ thống."),
    "train": ("⚙️ Huấn luyện mô hình", "Huấn luyện mô hình LSTM/ARIMA/Prophet để dự báo chính xác hơn. Ưu điểm của Huấn luyện lại có thể làm cho mô hình dự báo sát giá hơn, người dùng có thể tự ý huấn luyện theo mã CP mình quan tâm."),
    "risk": ("📉 Phân tích rủi ro", "Phân tích drawdown, biến động, xác suất thua lỗ, quản lý rủi ro danh mục."),
    "backtest_perf": ("🔄 Backtest & Hiệu suất", "Thử nghiệm chiến lược trên dữ liệu lịch sử, phân tích hành vi NĐT trong quá khứ, lọc các mã CP an toàn, đánh giá hiệu suất dựa trên các chỉ số chuyên môn chi tiết."),
    "optimize": ("📈 Tối ưu danh mục", "Tối ưu tỷ trọng cổ phiếu, phân bổ vốn và phân bổ vốn nâng cao theo rủi ro để đạt hiệu quả tốt nhất."),
    "rebalance": ("♻️ Tái cân bằng", "Điều chỉnh danh mục và vốn theo thời gian để duy trì chiến lược tối ưu, tối đa hoá lợi nhuận mục tiêu"),
    "report": ("🎯 Báo cáo", "Xuất báo cáo PDF tổng hợp bằng biểu đồ trực quan, nội dung chi tiết bao gồm: backtest, hiệu suất, dự báo tương lai từ 1-30 ngày, rủi ro, tối ưu hoá danh mục, tái cân bằng danh mục."),
}

ROLE_FEATURES = {
    "guest": [],
    "member": ["forecast", "risk", "backtest_perf"],
    "premium": ["forecast", "risk", "backtest_perf", "optimize", "rebalance", "report"]
}

ROLE_PRICES = {
    "member": 390_000,
    "premium": 490_000
}

DISCOUNTS = {
    6: 0.8,
    12: 0.6
}


def register_page():
    st.title("📝 Đăng ký / Nâng cấp tài khoản")

    role = st.selectbox("Chọn loại tài khoản", ["guest", "member", "premium"], key="register_role")

    if role in ROLE_FEATURES and ROLE_FEATURES[role]:
        st.subheader(f"🎁 Quyền lợi khi đăng ký {role.upper()}:")
        for feature in ROLE_FEATURES[role]:
            title, desc = FEATURES_INFO[feature]
            st.markdown(f"- **{title}**: {desc}")

        monthly_price = ROLE_PRICES.get(role, 0)
        st.info(f"💰 Giá: {monthly_price:,} VNĐ / tháng")

        duration_text = st.radio(
            "Chọn thời hạn thanh toán:",
            ["1 tháng", "6 tháng (giảm 20%)", "12 tháng (giảm 40%)"],
            key="register_duration"
        )

        if "6 tháng" in duration_text:
            months = 6
            discount = DISCOUNTS[6]
        elif "12 tháng" in duration_text:
            months = 12
            discount = DISCOUNTS[12]
        else:
            months = 1
            discount = 1.0

        total_price = int(monthly_price * months * discount)
        st.success(f"Tổng chi phí: {total_price:,} VNĐ")
    else:
        months = 1
        total_price = 0
        discount = 1.0

    with st.form("register_form"):
        full_name = st.text_input("Họ và tên *")
        username = st.text_input("Tên đăng nhập *")
        phone_number = st.text_input("Số điện thoại")
        email = st.text_input("Email *")
        password = st.text_input("Mật khẩu *", type="password")
        confirm_password = st.text_input("Xác nhận mật khẩu *", type="password")

        payment_method = payment_details = None

        if role in ["member", "premium"]:
            st.subheader("💳 Thông tin thanh toán")

            st.text_input(
                "👤 Người nhận",
                value="Phạm Xuân Quang - Tài khoản: 42488888 - Ngân hàng thương mại cổ phàn Á Châu (ACB)",
                disabled=True
            )

            payment_method = st.selectbox(
                "Phương thức thanh toán",
                ["Thẻ tín dụng", "PayPal", "Ví điện tử"]
            )

            payment_details = st.text_input(
                "Gói đăng ký",
                value=f"Đăng ký {role.capitalize()}",
                disabled=True
            )

        submit = st.form_submit_button("Đăng ký")

    if submit:
        if not username or not password or not email:
            st.error("⚠️ Username, mật khẩu và email là bắt buộc.")
            return

        if password != confirm_password:
            st.error("⚠️ Mật khẩu nhập lại không khớp.")
            return

        # =========================
        # FIX: USE db_manager
        # =========================
        with get_connection() as conn:
            with conn.cursor() as c:

                # check email
                c.execute(
                    "SELECT id FROM users WHERE email=%s",
                    (email,)
                )
                if c.fetchone():
                    st.error("⚠️ Email này đã được sử dụng. Vui lòng nhập email khác.")
                    return

                # check username
                c.execute(
                    "SELECT id FROM users WHERE username=%s",
                    (username,)
                )
                if c.fetchone():
                    st.error("⚠️ Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.")
                    return

        # =========================
        # REGISTER
        # =========================
        success, msg = register_user(
            full_name, username, phone_number, email, password, role
        )

        if not success:
            st.error("❌ " + msg)
            return

        st.success("✅ " + msg)

        # =========================
        # SUBSCRIPTION
        # =========================
        if role in ["member", "premium"]:
            user = login_user(username, password)

            if user:
                upgrade_user(
                    user["id"],
                    role,
                    payment_method,
                    payment_details,
                    duration_months=months,
                    amount_paid=total_price
                )
                st.warning(f"Tài khoản {username} đang chờ xác nhận thanh toán ({role.upper()}).")

        st.session_state["page"] = "login"
        st.rerun()

    if st.button("⬅️ Quay lại đăng nhập"):
        st.session_state["page"] = "login"
        st.rerun()