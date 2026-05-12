def calculate_rebalance_orders(portfolio, target_weights, total_value):
    orders = []
    for _, row in target_weights.iterrows():
        ticker = row["Ticker"]
        target_val = row["Weight"] * total_value
        current = portfolio[portfolio["Ticker"] == ticker]
        cur_val = current["Shares"].iloc[0] * current["Price"].iloc[0]
        diff = target_val - cur_val
        action = "BUY" if diff > 0 else "SELL"
        shares = abs(diff) // current["Price"].iloc[0]
        orders.append({"Ticker": ticker, "Action": action, "Shares": int(shares)})
    return orders
