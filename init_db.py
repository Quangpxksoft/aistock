import sqlite3
import bcrypt
import os

DB_PATH = "database/users.db"

def create_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # --- USERS ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

    # --- SUBSCRIPTIONS ---
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
            status TEXT DEFAULT 'pending',  -- pending / paid / active / expired / rejected
            received_by TEXT DEFAULT NULL,
            received_at TIMESTAMP DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # --- EMAIL LOGS - GỬI EMAIL NHẬN REPORT ---
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

    # --- USER TICKER HISTORY ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_ticker_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            last_viewed DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Thêm admin mặc định nếu chưa có
    c.execute("SELECT id FROM users WHERE username = 'admin'")
    if not c.fetchone():
        password_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        c.execute("""
            INSERT INTO users (full_name, username, phone_number, email, password_hash, role)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Admin User", "admin", "0000000000", "admin@example.com", password_hash, "admin"))
        print("✅ Created default admin user")

    conn.commit()
    conn.close()
    print("🎯 Database initialized.")


def get_role_by_username(username):
    """Trả về role của user theo username, mặc định 'guest' nếu không tìm thấy"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "guest"
    except Exception as e:
        print(f"❌ Lỗi lấy role cho username {username}: {e}")
        return "guest"


if __name__ == "__main__":
    create_db()
