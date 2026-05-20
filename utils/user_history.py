from utils.db_manager import get_connection


def record_user_ticker_view(user_id, tickers):

    if not tickers:
        return

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:

                for ticker in tickers:

                    ticker = ticker.strip().upper()

                    if not ticker:
                        continue

                    cursor.execute("""
                        SELECT id
                        FROM user_ticker_history
                        WHERE user_id = %s AND ticker = %s
                    """, (user_id, ticker))

                    row = cursor.fetchone()

                    if row:

                        cursor.execute("""
                            UPDATE user_ticker_history
                            SET last_viewed = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (row[0],))

                    else:

                        cursor.execute("""
                            INSERT INTO user_ticker_history
                            (user_id, ticker, last_viewed)
                            VALUES (%s, %s, CURRENT_TIMESTAMP)
                        """, (user_id, ticker))

            conn.commit()

    except Exception as e:
        print(f"❌ Lỗi lưu tickers cho user {user_id}: {e}")