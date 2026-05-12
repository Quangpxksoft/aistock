import sqlite3
import bcrypt
import os
from datetime import date
from utils.user_subscription import create_subscription, get_user_subscription
from config import DB_PATH, DEFAULT_ROLE, ALLOWED_ROLES

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # USERS
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            phone_number TEXT,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'guest'
                CHECK(role IN ('guest','member','premium','supervisor','admin')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # SUBSCRIPTIONS
    c.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('member','premium')),
            payment_method TEXT,
            payment_details TEXT,
            duration_months INTEGER DEFAULT 1,
            start_date DATE DEFAULT CURRENT_DATE,
            end_date DATE,
            active BOOLEAN DEFAULT 1,
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            recipients TEXT,
            pdf_file TEXT,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # USER TICKER HISTORY
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_ticker_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            last_viewed DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()
    print("🎯 Database initialized.")

def register_user(full_name, username, phone_number, email, password, role="guest",
                  payment_method=None, payment_details=None, duration_months=1, amount_paid=0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        if c.fetchone():
            return False, "Username này đã tồn tại!"
        if email:
            c.execute("SELECT id FROM users WHERE email=?", (email,))
            if c.fetchone():
                return False, "Email này đã tồn tại!"

        # luôn tạo guest
        c.execute("""
            INSERT INTO users (full_name, username, phone_number, email, password_hash, role)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (full_name, username, phone_number, email, password_hash, role))
        user_id = c.lastrowid
        conn.commit()

        # nếu member/premium => tạo subscription pending
        if role in ("member", "premium"):
            create_subscription(user_id, role, duration_months, payment_method, payment_details, amount_paid)

        return True, "Đăng ký thành công! Vui lòng chờ kích hoạt nếu bạn chọn gói trả phí."

    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, full_name, username, phone_number, email, password_hash, role
        FROM users
        WHERE username=?
    """, (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[5].encode('utf-8')):
        return {
            "id": user[0],
            "full_name": user[1],
            "username": user[2],
            "phone_number": user[3],
            "email": user[4],
            "role": user[6]
        }
    return None

def upgrade_user(user_id, new_role, payment_method=None, payment_details=None, duration_months=1, amount_paid=0):
    if new_role in ("member", "premium"):
        create_subscription(user_id, new_role, duration_months, payment_method, payment_details, amount_paid)
        print(f"⏳ User {user_id} requested upgrade to {new_role}, pending admin approval.")
