from utils.db_manager import get_connection


# =========================
# ROLE PERMISSIONS (SINGLE SOURCE OF TRUTH)
# =========================
ROLE_PERMISSIONS = {
    "guest": ["forecast"],
    "member": ["forecast", "risk", "backtest_perf"],
    "premium": ["forecast", "risk", "backtest_perf", "optimize", "rebalance", "report"],
    "supervisor": ["forecast", "risk", "backtest_perf", "optimize", "rebalance", "train", "report"],
    "admin": ["forecast", "risk", "backtest_perf", "optimize", "rebalance", "train", "report"],
}


def get_permissions(role: str):
    """Trả về danh sách quyền theo role."""
    return ROLE_PERMISSIONS.get(role, [])


def can_access(role: str, feature: str) -> bool:
    """Kiểm tra user có quyền sử dụng một chức năng cụ thể không."""
    return feature in get_permissions(role)


# =========================
# DB QUERIES
# =========================

def get_role_by_user_id(user_id: int) -> str | None:
    """Lấy role của user từ DB bằng user_id."""

    with get_connection() as conn:
        with conn.cursor() as c:

            c.execute(
                "SELECT role FROM users WHERE id = %s",
                (user_id,)
            )

            row = c.fetchone()

    return row[0] if row else None


def get_role_by_username(username: str) -> str | None:
    """Lấy role của user từ DB bằng username."""

    with get_connection() as conn:
        with conn.cursor() as c:

            c.execute(
                "SELECT role FROM users WHERE username = %s",
                (username,)
            )

            row = c.fetchone()

    return row[0] if row else None


def get_permissions_by_user_id(user_id: int):
    """Lấy danh sách quyền dựa trên user_id."""
    role = get_role_by_user_id(user_id)
    return get_permissions(role) if role else []


def get_permissions_by_username(username: str):
    """Lấy danh sách quyền dựa trên username."""
    role = get_role_by_username(username)
    return get_permissions(role) if role else []


def get_permissions_by_role(role: str) -> list[str]:
    """Trả về danh sách tab/chức năng được phép truy cập theo role."""
    return ROLE_PERMISSIONS.get(role, [])