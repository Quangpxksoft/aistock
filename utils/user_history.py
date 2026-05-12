import sqlite3
DB_PATH = "database/users.db"

def record_user_ticker_view(user_id, tickers):
    print("DEBUG record_user_ticker_view DB_PATH:", DB_PATH)
    """Lưu danh sách tickers mà user quan tâm"""
    if not tickers:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for ticker in tickers:
            ticker = ticker.strip().upper()
            if not ticker:
                continue

            # Kiểm tra ticker đã tồn tại chưa
            cursor.execute("""
                SELECT id FROM user_ticker_history
                WHERE user_id = ? AND ticker = ?
            """, (user_id, ticker))
            row = cursor.fetchone()
            if row:
                cursor.execute("""
                    UPDATE user_ticker_history
                    SET last_viewed = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (row[0],))
            else:
                cursor.execute("""
                    INSERT INTO user_ticker_history (user_id, ticker, last_viewed)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (user_id, ticker))

        conn.commit()
        conn.close()
        print(f"✔️ Lưu tickers {tickers} cho user {user_id}")
    except Exception as e:
        print(f"❌ Lỗi lưu tickers cho user {user_id}: {e}")
