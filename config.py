# # import os

# # # Lấy đường dẫn gốc project (nơi đặt app.py)
# # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # # Thư mục chính của ứng dụng
# # # APP_DIR = os.path.join(BASE_DIR, "ai_investment_app")
# # APP_DIR = os.path.join(BASE_DIR, "ais")


# # # Các thư mục con bên trong app
# # DATABASE_DIR = os.path.join(APP_DIR, "database")
# # MODEL_DIR = os.path.join(APP_DIR, "models")
# # REPORT_DIR = os.path.join(APP_DIR, "reports")
# # LOG_DIR = os.path.join(APP_DIR, "logs")

# # # Đảm bảo các thư mục tồn tại
# # os.makedirs(DATABASE_DIR, exist_ok=True)
# # os.makedirs(MODEL_DIR, exist_ok=True)
# # os.makedirs(REPORT_DIR, exist_ok=True)
# # os.makedirs(LOG_DIR, exist_ok=True)

# # # Đường dẫn database
# # DB_PATH = os.path.join(DATABASE_DIR, "users.db")

# # # ===== NGƯỜI DÙNG =====
# # DEFAULT_ROLE = "guest"
# # ALLOWED_ROLES = ("guest", "member", "premium", "supervisor", "admin")
# # SUBSCRIPTION_ROLES = ("member", "premium")

# import os

# # =========================
# # BASE PROJECT PATH
# # =========================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Nếu bạn deploy Render, không nên hardcode "ais"
# # → dùng luôn BASE_DIR để tránh lệch môi trường
# APP_DIR = BASE_DIR


# # =========================
# # FOLDERS
# # =========================

# DATABASE_DIR = os.path.join(APP_DIR, "database")
# MODEL_DIR = os.path.join(APP_DIR, "models")
# REPORT_DIR = os.path.join(APP_DIR, "reports")
# LOG_DIR = os.path.join(APP_DIR, "logs")

# # đảm bảo tồn tại
# os.makedirs(DATABASE_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(REPORT_DIR, exist_ok=True)
# os.makedirs(LOG_DIR, exist_ok=True)


# # =========================
# # DATABASE
# # =========================

# DB_PATH = os.path.join(DATABASE_DIR, "users.db")


# # =========================
# # USER CONFIG
# # =========================

# DEFAULT_ROLE = "guest"

# ALLOWED_ROLES = (
#     "guest",
#     "member",
#     "premium",
#     "supervisor",
#     "admin"
# )

# SUBSCRIPTION_ROLES = (
#     "member",
#     "premium"
# )

# import os

# # =========================
# # BASE PROJECT PATH
# # =========================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# APP_DIR = BASE_DIR


# # =========================
# # FOLDERS
# # =========================

# DATABASE_DIR = os.path.join(APP_DIR, "database")
# MODEL_DIR = os.path.join(APP_DIR, "models")
# REPORT_DIR = os.path.join(APP_DIR, "reports")
# LOG_DIR = os.path.join(APP_DIR, "logs")


# # =========================
# # SAFE INIT FUNCTION (NO SIDE EFFECT ON IMPORT)
# # =========================

# def ensure_dirs():
#     """
#     Tạo toàn bộ thư mục cần thiết.
#     Chỉ gọi khi app start (không chạy lúc import).
#     """
#     os.makedirs(DATABASE_DIR, exist_ok=True)
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     os.makedirs(REPORT_DIR, exist_ok=True)
#     os.makedirs(LOG_DIR, exist_ok=True)


# # =========================
# # DATABASE
# # =========================

# DB_PATH = os.path.abspath(
#     os.path.join(DATABASE_DIR, "users.db")
# )


# # =========================
# # USER CONFIG
# # =========================

# DEFAULT_ROLE = "guest"

# ALLOWED_ROLES = (
#     "guest",
#     "member",
#     "premium",
#     "supervisor",
#     "admin"
# )

# SUBSCRIPTION_ROLES = (
#     "member",
#     "premium"
# )

import os
from dotenv import load_dotenv
load_dotenv()

# =========================
# BASE PROJECT PATH
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = BASE_DIR


# =========================
# FOLDERS
# =========================

DATABASE_DIR = os.path.join(APP_DIR, "database")
MODEL_DIR = os.path.join(APP_DIR, "models")
REPORT_DIR = os.path.join(APP_DIR, "reports")
LOG_DIR = os.path.join(APP_DIR, "logs")


# =========================
# SAFE INIT FUNCTION (NO SIDE EFFECT ON IMPORT)
# =========================

def ensure_dirs():
    """
    Tạo toàn bộ thư mục cần thiết.
    Chỉ gọi khi app start (không chạy lúc import).
    """
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


# =========================
# DATABASE (POSTGRESQL)
# =========================

# Render / Cloud DB connection string
DATABASE_URL = os.getenv("DATABASE_URL")


# =========================
# USER CONFIG
# =========================

DEFAULT_ROLE = "guest"

ALLOWED_ROLES = (
    "guest",
    "member",
    "premium",
    "supervisor",
    "admin"
)

SUBSCRIPTION_ROLES = (
    "member",
    "premium"
)