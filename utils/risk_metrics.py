import pandas as pd
import numpy as np

def calculate_risk_metrics(df: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
    """
    Tính các chỉ số rủi ro: Volatility, VaR, CVaR, Sortino Ratio
    """
    returns = df["Close"].pct_change().dropna()

    # Tính các metric
    var = np.percentile(returns, 100 * alpha)
    cvar = returns[returns <= var].mean()
    downside = returns[returns < 0]
    sortino = returns.mean() / (downside.std() + 1e-6)

    # ✅ Thêm Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)

    # Kết quả trả về
    risk_df = pd.DataFrame({
        "Metric": ["Volatility", "VaR", "CVaR", "Sortino Ratio"],
        "Value": [volatility, var, cvar, sortino]
    })

    return risk_df
