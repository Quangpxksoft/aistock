def calculate_rebalance_orders(portfolio, target_weights, total_value):
    orders = []
    for _, row in target_weights.iterrows():
        ticker = row["Ticker"]
        target_val = row["Weight"] * total_value

        current = portfolio[portfolio["Ticker"] == ticker]
        if current.empty:
            raise ValueError(f"Ticker {ticker} không có trong portfolio hiện tại")

        price = current["Price"].iloc[0]
        if price <= 0:
            raise ValueError(f"Giá không hợp lệ cho {ticker}: {price}")

        cur_val = current["Shares"].iloc[0] * price
        diff = target_val - cur_val
        action = "BUY" if diff > 0 else "SELL"
        shares = abs(diff) // price
        orders.append({"Ticker": ticker, "Action": action, "Shares": int(shares)})
    return orders
