
# import psycopg
# import bcrypt
# import os
# from datetime import date
# from utils.user_subscription import create_subscription, get_user_subscription
# from config import DATABASE_URL, DEFAULT_ROLE, ALLOWED_ROLES


# # =========================
# # CONNECTION
# # =========================
# _conn = None

# def get_connection():
#     global _conn

#     try:
#         if _conn is None or _conn.closed:
#             _conn = psycopg.connect(
#                 DATABASE_URL,
#                 connect_timeout=10,
#                 keepalives=1,
#                 keepalives_idle=30,
#                 keepalives_interval=10,
#                 keepalives_count=5
#             )

#         return _conn

#     except Exception as e:
#         print("DATABASE CONNECTION ERROR:", e)
#         _conn = None   # 🔥 reset cực quan trọng
#         raise
# # =========================
# # INIT DB
# # =========================
# def init_db():

#     conn = get_connection()
#     c = conn.cursor()

#     # USERS
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS users (
#             id SERIAL PRIMARY KEY,
#             full_name TEXT NOT NULL,
#             username TEXT NOT NULL UNIQUE,
#             phone_number TEXT,
#             email TEXT NOT NULL UNIQUE,
#             password_hash TEXT NOT NULL,
#             role TEXT NOT NULL DEFAULT 'guest'
#                 CHECK(role IN ('guest','member','premium','supervisor','admin')),
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     """)

#     # SUBSCRIPTIONS
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS subscriptions (
#             id SERIAL PRIMARY KEY,
#             user_id INTEGER NOT NULL,
#             role TEXT NOT NULL CHECK(role IN ('member','premium')),
#             payment_method TEXT,
#             payment_details TEXT,
#             duration_months INTEGER DEFAULT 1,
#             start_date DATE DEFAULT CURRENT_DATE,
#             end_date DATE,
#             active BOOLEAN DEFAULT TRUE,
#             amount_paid REAL DEFAULT 0,
#             status TEXT DEFAULT 'pending',
#             received_by TEXT DEFAULT NULL,
#             received_at TIMESTAMP DEFAULT NULL,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY(user_id) REFERENCES users(id)
#         )
#     """)

#     # EMAIL LOGS
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS email_logs (
#             id SERIAL PRIMARY KEY,
#             role TEXT,
#             recipients TEXT,
#             pdf_file TEXT,
#             status TEXT,
#             timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     """)

#     # USER TICKER HISTORY
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS user_ticker_history (
#             id SERIAL PRIMARY KEY,
#             user_id INTEGER NOT NULL,
#             ticker TEXT NOT NULL,
#             last_viewed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY(user_id) REFERENCES users(id)
#         )
#     """)

#     # =========================
#     # SEED ADMIN USER
#     # =========================
#     c.execute(
#         "SELECT id FROM users WHERE username = %s",
#         ("admin",)
#     )

#     if not c.fetchone():

#         password_hash = bcrypt.hashpw(
#             "admin123".encode("utf-8"),
#             bcrypt.gensalt()
#         ).decode("utf-8")

#         c.execute("""
#             INSERT INTO users (
#                 full_name, username, phone_number, email, password_hash, role
#             )
#             VALUES (%s, %s, %s, %s, %s, %s)
#         """, (
#             "Admin User",
#             "admin",
#             "0000000000",
#             "admin@example.com",
#             password_hash,
#             "admin"
#         ))

#     conn.commit()
#     # conn.close()

#     print("🎯 Database initialized.")


# # =========================
# # REGISTER USER
# # =========================
# def register_user(full_name, username, phone_number, email, password,
#                   role="guest", payment_method=None,
#                   payment_details=None, duration_months=1, amount_paid=0):

#     conn = get_connection()
#     c = conn.cursor()

#     password_hash = bcrypt.hashpw(
#         password.encode("utf-8"),
#         bcrypt.gensalt()
#     ).decode("utf-8")

#     try:
#         c.execute(
#             "SELECT id FROM users WHERE username = %s",
#             (username,)
#         )

#         if c.fetchone():
#             return False, "Username này đã tồn tại!"

#         if email:
#             c.execute(
#                 "SELECT id FROM users WHERE email = %s",
#                 (email,)
#             )

#             if c.fetchone():
#                 return False, "Email này đã tồn tại!"

#         c.execute("""
#             INSERT INTO users (
#                 full_name,
#                 username,
#                 phone_number,
#                 email,
#                 password_hash,
#                 role
#             )
#             VALUES (%s, %s, %s, %s, %s, %s)
#             RETURNING id
#         """, (
#             full_name,
#             username,
#             phone_number,
#             email,
#             password_hash,
#             role
#         ))

#         user_id = c.fetchone()[0]
#         conn.commit()

#         if role in ("member", "premium"):
#             create_subscription(
#                 user_id,
#                 role,
#                 duration_months,
#                 payment_method,
#                 payment_details,
#                 amount_paid
#             )

#         return True, "Đăng ký thành công! Vui lòng chờ kích hoạt nếu bạn chọn gói trả phí."

#     finally:
#         # conn.close()
#         pass

# # =========================
# # LOGIN USER
# # =========================
# def login_user(username, password):

#     conn = get_connection()
#     c = conn.cursor()

#     c.execute("""
#         SELECT id, full_name, username, phone_number, email, password_hash, role
#         FROM users
#         WHERE username = %s
#     """, (username,))

#     user = c.fetchone()
#     # conn.close()

#     if user:

#         stored_hash = user[5]

#         if isinstance(stored_hash, str):
#             stored_hash = stored_hash.encode("utf-8")

#         if bcrypt.checkpw(
#             password.encode("utf-8"),
#             stored_hash
#         ):
#             return {
#                 "id": user[0],
#                 "full_name": user[1],
#                 "username": user[2],
#                 "phone_number": user[3],
#                 "email": user[4],
#                 "role": user[6]
#             }

#     return None


# # =========================
# # GET ROLE
# # =========================
# def get_role_by_username(username):

#     conn = get_connection()
#     c = conn.cursor()

#     c.execute(
#         "SELECT role FROM users WHERE username = %s",
#         (username,)
#     )

#     row = c.fetchone()
#     # conn.close()

#     return row[0] if row else "guest"


# # =========================
# # UPGRADE USER
# # =========================
# def upgrade_user(user_id, new_role,
#                  payment_method=None,
#                  payment_details=None,
#                  duration_months=1,
#                  amount_paid=0):

#     if new_role in ("member", "premium"):
#         create_subscription(
#             user_id,
#             new_role,
#             duration_months,
#             payment_method,
#             payment_details,
#             amount_paid
#         )

#         print(
#             f"⏳ User {user_id} requested upgrade to {new_role}, pending admin approval."
#         )

import psycopg
import bcrypt
from datetime import date
from utils.user_subscription import create_subscription, get_user_subscription
from config import DATABASE_URL, DEFAULT_ROLE, ALLOWED_ROLES
from utils.db_manager import get_connection
# =========================
# CONNECTION (SAFE VERSION)
# =========================

# def get_connection():
#     """
#     Always create a fresh connection if needed.
#     Avoid stale connection reuse (root cause of SSL issues).
#     """

#     try:
#         conn = psycopg.connect(
#             DATABASE_URL,
#             sslmode="require",
#             connect_timeout=10,
#             keepalives=True,
#             keepalives_idle=30,
#             keepalives_interval=10,
#             keepalives_count=5
#         )

#         return conn

#     except Exception as e:
#         print("DATABASE CONNECTION ERROR:", e)
#         raise


# =========================
# INIT DB
# =========================
def init_db():

    with get_connection() as conn:
        with conn.cursor() as c:

            # USERS
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    username TEXT NOT NULL UNIQUE,
                    phone_number TEXT,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'guest'
                        CHECK(role IN ('guest','member','premium','supervisor','admin')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # SUBSCRIPTIONS
            c.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('member','premium')),
                    payment_method TEXT,
                    payment_details TEXT,
                    duration_months INTEGER DEFAULT 1,
                    start_date DATE DEFAULT CURRENT_DATE,
                    end_date DATE,
                    active BOOLEAN DEFAULT TRUE,
                    amount_paid REAL DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    received_by TEXT DEFAULT NULL,
                    received_at TIMESTAMP DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            # EMAIL LOGS
            c.execute("""
                CREATE TABLE IF NOT EXISTS email_logs (
                    id SERIAL PRIMARY KEY,
                    role TEXT,
                    recipients TEXT,
                    pdf_file TEXT,
                    status TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # USER TICKER HISTORY
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_ticker_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    last_viewed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            # =========================
            # SEED ADMIN USER
            # =========================
            c.execute(
                "SELECT id FROM users WHERE username = %s",
                ("admin",)
            )

            if not c.fetchone():

                password_hash = bcrypt.hashpw(
                    "admin123".encode("utf-8"),
                    bcrypt.gensalt()
                ).decode("utf-8")

                c.execute("""
                    INSERT INTO users (
                        full_name, username, phone_number, email, password_hash, role
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    "Admin User",
                    "admin",
                    "0000000000",
                    "admin@example.com",
                    password_hash,
                    "admin"
                ))

        conn.commit()

    print("🎯 Database initialized.")


# =========================
# REGISTER USER
# =========================
def register_user(full_name, username, phone_number, email, password,
                  role="guest", payment_method=None,
                  payment_details=None, duration_months=1, amount_paid=0):

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    with get_connection() as conn:
        with conn.cursor() as c:

            c.execute(
                "SELECT id FROM users WHERE username = %s",
                (username,)
            )

            if c.fetchone():
                return False, "Username này đã tồn tại!"

            if email:
                c.execute(
                    "SELECT id FROM users WHERE email = %s",
                    (email,)
                )

                if c.fetchone():
                    return False, "Email này đã tồn tại!"

            c.execute("""
                INSERT INTO users (
                    full_name,
                    username,
                    phone_number,
                    email,
                    password_hash,
                    role
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                full_name,
                username,
                phone_number,
                email,
                password_hash,
                role
            ))

            user_id = c.fetchone()[0]
            conn.commit()

    if role in ("member", "premium"):
        create_subscription(
            user_id,
            role,
            duration_months,
            payment_method,
            payment_details,
            amount_paid
        )

    return True, "Đăng ký thành công!"


# =========================
# LOGIN USER
# =========================
def login_user(username, password):

    with get_connection() as conn:
        with conn.cursor() as c:

            c.execute("""
                SELECT id, full_name, username, phone_number, email, password_hash, role
                FROM users
                WHERE username = %s
            """, (username,))

            user = c.fetchone()

    if not user:
        return None

    stored_hash = user[5]
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")

    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        return {
            "id": user[0],
            "full_name": user[1],
            "username": user[2],
            "phone_number": user[3],
            "email": user[4],
            "role": user[6]
        }

    return None


# =========================
# GET ROLE
# =========================
def get_role_by_username(username):

    with get_connection() as conn:
        with conn.cursor() as c:

            c.execute(
                "SELECT role FROM users WHERE username = %s",
                (username,)
            )

            row = c.fetchone()

    return row[0] if row else "guest"


# =========================
# UPGRADE USER
# =========================
def upgrade_user(user_id, new_role,
                 payment_method=None,
                 payment_details=None,
                 duration_months=1,
                 amount_paid=0):

    if new_role in ("member", "premium"):
        create_subscription(
            user_id,
            new_role,
            duration_months,
            payment_method,
            payment_details,
            amount_paid
        )

        print(
            f"⏳ User {user_id} requested upgrade to {new_role}, pending admin approval."
        )