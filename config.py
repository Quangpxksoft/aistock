
# # =========================
# # Server
# # =========================
# import os
# from dotenv import load_dotenv

# load_dotenv(override=False)

# # =========================
# # BASE PROJECT PATH
# # =========================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # =========================
# # FOLDERS
# # =========================

# MODEL_DIR = os.path.join(BASE_DIR, "models")
# REPORT_DIR = os.path.join(BASE_DIR, "reports")
# LOG_DIR = os.path.join(BASE_DIR, "logs")


# # =========================
# # SAFE INIT FUNCTION
# # =========================

# def ensure_dirs():
#     """Tạo toàn bộ thư mục cần thiết (chỉ gọi khi app start)."""
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     os.makedirs(REPORT_DIR, exist_ok=True)
#     os.makedirs(LOG_DIR, exist_ok=True)


# # =========================
# # DATABASE CONFIG
# # =========================

# # Primary (legacy-safe): DATABASE_URL
# DATABASE_URL = os.getenv("DATABASE_URL")

# # Optional fallback (compatibility layer nếu code cũ dùng DB_HOST)
# DB_HOST = os.getenv("DB_HOST")
# DB_NAME = os.getenv("DB_NAME")
# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_PORT = os.getenv("DB_PORT", "5432")


# # =========================
# # VALIDATION (fail fast but controlled)
# # =========================

# if not DATABASE_URL:
#     # fallback mode: allow DB_HOST style if URL missing
#     if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
#         raise RuntimeError(
#             "Missing DB config: either DATABASE_URL OR DB_HOST/DB_NAME/DB_USER/DB_PASSWORD must be set"
#         )


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
# =========================
# Local
# =========================
import os
from dotenv import load_dotenv

load_dotenv(override=False)

# =========================
# BASE PROJECT PATH
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# FOLDERS
# =========================

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
LOG_DIR = os.path.join(BASE_DIR, "logs")


# =========================
# SAFE INIT FUNCTION
# =========================

def ensure_dirs():
    """Tạo toàn bộ thư mục cần thiết (chỉ gọi khi app start)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


# =========================
# DATABASE CONFIG
# =========================

# Primary (legacy-safe): DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Optional fallback (compatibility layer nếu code cũ dùng DB_HOST)
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_ENV = os.getenv("DB_ENV", "production")
# =========================
# VALIDATION (fail fast but controlled)
# =========================

if not DATABASE_URL:
    # fallback mode: allow DB_HOST style if URL missing
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        raise RuntimeError(
            "Missing DB config: either DATABASE_URL OR DB_HOST/DB_NAME/DB_USER/DB_PASSWORD must be set"
        )


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