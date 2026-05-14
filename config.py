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

import os

# =========================
# BASE PROJECT PATH
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nếu bạn deploy Render, không nên hardcode "ais"
# → dùng luôn BASE_DIR để tránh lệch môi trường
APP_DIR = BASE_DIR


# =========================
# FOLDERS
# =========================

DATABASE_DIR = os.path.join(APP_DIR, "database")
MODEL_DIR = os.path.join(APP_DIR, "models")
REPORT_DIR = os.path.join(APP_DIR, "reports")
LOG_DIR = os.path.join(APP_DIR, "logs")

# đảm bảo tồn tại
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# =========================
# DATABASE
# =========================

DB_PATH = os.path.join(DATABASE_DIR, "users.db")


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

# import os

# # =========================
# # BASE PROJECT PATH
# # =========================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # =========================
# # ENV DETECTION
# # =========================

# # Render mount disk thường ở /data
# RENDER_DISK_PATH = "/data"

# IS_RENDER = os.path.exists(RENDER_DISK_PATH)

# # =========================
# # FOLDERS
# # =========================

# if IS_RENDER:
#     # 🔥 Production (Render)
#     DATABASE_DIR = RENDER_DISK_PATH
#     MODEL_DIR = os.path.join(RENDER_DISK_PATH, "models")
#     REPORT_DIR = os.path.join(RENDER_DISK_PATH, "reports")
#     LOG_DIR = os.path.join(RENDER_DISK_PATH, "logs")
# else:
#     # 💻 Local dev
#     DATABASE_DIR = os.path.join(BASE_DIR, "database")
#     MODEL_DIR = os.path.join(BASE_DIR, "models")
#     REPORT_DIR = os.path.join(BASE_DIR, "reports")
#     LOG_DIR = os.path.join(BASE_DIR, "logs")

# # đảm bảo tồn tại
# os.makedirs(DATABASE_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(REPORT_DIR, exist_ok=True)
# os.makedirs(LOG_DIR, exist_ok=True)

# # =========================
# # DATABASE
# # =========================

# DB_PATH = os.path.join(DATABASE_DIR, "users.db")

# # Debug log (rất hữu ích khi deploy)
# print(f"📂 DB PATH: {DB_PATH}")
# print(f"🌍 RUNNING ON RENDER: {IS_RENDER}")

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