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

DATABASE_DIR = os.path.join(BASE_DIR, "database")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
LOG_DIR = os.path.join(BASE_DIR, "logs")


# =========================
# SAFE INIT FUNCTION
# =========================

def ensure_dirs():
    """Tạo toàn bộ thư mục cần thiết (chỉ gọi khi app start)."""

    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


# =========================
# DATABASE (POSTGRESQL)
# =========================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment variables")


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