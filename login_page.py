import streamlit as st
from utils.user_manager import login_user

def login_page():
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown("<h2 style='text-align: center;'>🔐 Đăng nhập</h2>", unsafe_allow_html=True)

        # Ô nhập căn giữa
        _, input_col, _ = st.columns([1, 4, 1])
        with input_col:
            username = st.text_input("Tên đăng nhập", key="login_username", max_chars=30)
            password = st.text_input("Mật khẩu", type="password", key="login_password")

        # Hàng nút đặt cùng cấp input_col (không lồng quá sâu)
        _, btn_left, btn_right, _ = st.columns([1, 2, 2, 1])

        with btn_left:
            if st.button("Đăng nhập", use_container_width=True):
                user = login_user(username, password)
                if user:
                    st.session_state["user"] = user
                    st.session_state["page"] = "home"
                    st.session_state["login_error"] = None
                    st.rerun()
                else:
                    st.session_state["login_error"] = "Tên đăng nhập hoặc mật khẩu không đúng"

        with btn_right:
            if st.button("Đăng ký", use_container_width=True):
                st.session_state["page"] = "register"

        # Cảnh báo đặt dưới hàng nút
        if st.session_state.get("login_error"):
            _, error_col, _ = st.columns([1, 4, 1])
            with error_col:
                st.error(st.session_state["login_error"])
