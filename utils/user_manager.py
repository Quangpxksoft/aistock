import bcrypt
from utils.user_subscription import create_subscription
from utils.db_manager import get_connection


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
            # SEED ADMIN
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
                        full_name, username, phone_number,
                        email, password_hash, role
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

            # check username
            c.execute(
                "SELECT id FROM users WHERE username = %s",
                (username,)
            )
            if c.fetchone():
                return False, "Username này đã tồn tại!"

            # check email
            if email:
                c.execute(
                    "SELECT id FROM users WHERE email = %s",
                    (email,)
                )
                if c.fetchone():
                    return False, "Email này đã tồn tại!"

            # insert user
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

    # create subscription (non-atomic but OK by design)
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
                SELECT id, full_name, username,
                       phone_number, email, password_hash, role
                FROM users
                WHERE username = %s
            """, (username,))

            user = c.fetchone()

    if not user or not user[5]:
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

        print(f"⏳ User {user_id} upgrade request -> {new_role} (pending approval)")