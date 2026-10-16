"""Buy-and-hold $50k at the Oct 2007 market peak in 1x/2x/3x S&P 500 and Nasdaq-100 ETFs.

SPY, QQQ, SSO and QLD all traded in Oct 2007, so their real (dividend-adjusted)
prices are used. UPRO (launched Jun 2009) and TQQQ (Feb 2010) did not exist yet,
so they are simulated from SPY/QQQ daily returns until launch, then spliced
onto the real fund:
    daily_3x = 3 * r_underlying - 2 * cash_rate/252 - expense_ratio/252

Run: pip install yfinance pandas && python sim_2007_peak.py
"""
import pandas as pd
import yfinance as yf

START_CASH = 50_000
PEAK = "2007-10-09"  # S&P 500 closing high (1565.15); QQQ peaked a few weeks later
EXPENSE = 0.0091  # UPRO/TQQQ expense ratio

px = yf.download(["SPY", "QQQ", "SSO", "QLD", "UPRO", "TQQQ", "^IRX"],
                 start="2007-01-01", auto_adjust=True, progress=False)["Close"]
cash = (px["^IRX"].ffill() / 100).reindex(px.index).ffill()  # 13-week T-bill yield
rets = px.drop(columns="^IRX").pct_change()


def synth_3x(under, real):
    r = 3 * rets[under] - 2 * cash / 252 - EXPENSE / 252
    launch = rets[real].first_valid_index()
    r.loc[launch:] = rets[real].loc[launch:]  # real fund once it trades
    return r


series = {
    "1) S&P 500 (SPY)": rets["SPY"],
    "2) 2x S&P 500 (SSO)": rets["SSO"],
    "3) 3x S&P 500 (UPRO*)": synth_3x("SPY", "UPRO"),
    "4) Nasdaq-100 (QQQ)": rets["QQQ"],
    "5) 2x Nasdaq-100 (QLD)": rets["QLD"],
    "6) 3x Nasdaq-100 (TQQQ*)": synth_3x("QQQ", "TQQQ"),
}

rows = []
for name, r in series.items():
    r = r.loc[PEAK:].iloc[1:].fillna(0)  # buy at the PEAK close
    value = START_CASH * (1 + r).cumprod()
    trough = value.min()
    recovered = value[(value.index > value.idxmin()) & (value >= START_CASH)]
    years = (value.index[-1] - pd.Timestamp(PEAK)).days / 365.25
    rows.append({
        "Scenario": name,
        "Low point $": round(trough),
        "Max drawdown": f"{trough / START_CASH - 1:.1%}",
        "Low date": value.idxmin().date(),
        "Back to $50k": recovered.index[0].date() if len(recovered) else "never",
        f"Value {value.index[-1].date()} $": round(value.iloc[-1]),
        "CAGR": f"{(value.iloc[-1] / START_CASH) ** (1 / years) - 1:.1%}",
    })

pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
print("\n* simulated from the underlying until the real fund launched")
