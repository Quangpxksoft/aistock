from utils.db_manager import get_connection
from datetime import date, timedelta

# Bảng giá
PRICE_TABLE = {
    "member": 390_000,
    "premium": 490_000
}

# Giảm giá
DISCOUNTS = {
    6: 0.8,
    12: 0.6
}


def calculate_amount(role: str, duration_months: int) -> int:
    base_price = PRICE_TABLE.get(role, 0)
    discount = DISCOUNTS.get(duration_months, 1)
    return int(base_price * duration_months * discount)


def create_subscription(
    user_id: int,
    role: str,
    duration_months: int = 1,
    payment_method: str = None,
    payment_details: str = None,
    amount_paid: int = 0,
    status: str = 'pending'
):
    """Tạo subscription mới."""

    start_date = date.today()
    end_date = start_date + timedelta(days=30 * duration_months)

    if amount_paid == 0:
        amount_paid = calculate_amount(role, duration_months)

    with get_connection() as conn:
        with conn.cursor() as c:

            c.execute("""
                INSERT INTO subscriptions
                (user_id, role, payment_method, payment_details,
                 duration_months, start_date, end_date,
                 active, amount_paid, status, received_by, received_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s, NULL, NULL)
            """, (
                user_id,
                role,
                payment_method,
                payment_details,
                duration_months,
                start_date,
                end_date,
                amount_paid,
                status
            ))

        conn.commit()


def get_user_subscription(user_id: int):
    """Lấy subscription gần nhất"""

    with get_connection() as conn:
        with conn.cursor() as c:

            c.execute("""
                SELECT role, amount_paid, status, duration_months,
                       start_date, end_date, active, created_at
                FROM subscriptions
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))

            row = c.fetchone()

    if row:
        return {
            "role": row[0],
            "amount_paid": row[1],
            "status": row[2],
            "duration_months": row[3],
            "start_date": row[4],
            "end_date": row[5],
            "active": bool(row[6]),
            "created_at": row[7]
        }

    return None