"""Buy-and-hold $50k at a market peak in 1x/2x/3x S&P 500 and Nasdaq-100 ETFs.

Crashes: dot-com (2000), financial crisis (2007), COVID (2020). Each fund group is
bought at the close of its own index's closing high (S&P 500 for SPY/SSO/UPRO,
Nasdaq-100 for QQQ/QLD/TQQQ), found from the data.

Real dividend-adjusted fund prices are used wherever the fund traded. Before a
leveraged fund launched (SSO/QLD Jun 2006, UPRO Jun 2009, TQQQ Feb 2010) it is
simulated from SPY/QQQ daily total returns:
    daily_Lx = L * r_underlying - (L - 1) * tbill/252 - expense/252

Run: pip install yfinance pandas && python sim_crash_peaks.py
"""
import pandas as pd
import yfinance as yf

START_CASH = 50_000
CRASHES = {  # name: window searched for the pre-crash closing high
    "Dot-com 2000": ("1999-06-01", "2000-12-31"),
    "Financial crisis 2007": ("2007-01-01", "2008-06-30"),
    "COVID 2020": ("2019-06-01", "2020-03-31"),
}
FUNDS = [  # label, underlying, leverage, ticker, expense ratio, index
    ("S&P 500 (SPY)", "SPY", 1, "SPY", 0, "^GSPC"),
    ("2x S&P 500 (SSO)", "SPY", 2, "SSO", 0.0089, "^GSPC"),
    ("3x S&P 500 (UPRO)", "SPY", 3, "UPRO", 0.0091, "^GSPC"),
    ("Nasdaq-100 (QQQ)", "QQQ", 1, "QQQ", 0, "^NDX"),
    ("2x Nasdaq-100 (QLD)", "QQQ", 2, "QLD", 0.0095, "^NDX"),
    ("3x Nasdaq-100 (TQQQ)", "QQQ", 3, "TQQQ", 0.0084, "^NDX"),
]

px = yf.download(["SPY", "QQQ", "SSO", "QLD", "UPRO", "TQQQ", "^IRX", "^GSPC", "^NDX"],
                 start="1999-01-01", auto_adjust=True, progress=False)["Close"]
px = px[px["SPY"].notna()]
cash = (px["^IRX"] / 100).ffill().bfill()  # 13-week T-bill yield
rets = px.pct_change()


def fund_returns(under, lev, ticker, expense):
    if lev == 1:
        return rets[ticker]
    r = lev * rets[under] - (lev - 1) * cash / 252 - expense / 252
    launch = rets[ticker].first_valid_index()
    r.loc[launch:] = rets[ticker].loc[launch:]  # real fund once it trades
    return r, launch


def simulate(r, buy_date):
    r = r.loc[buy_date:].iloc[1:].fillna(0)  # buy at the buy_date close
    value = START_CASH * (1 + r).cumprod()
    low_date = value.idxmin()
    back = value[(value.index > low_date) & (value >= START_CASH)]
    years = (value.index[-1] - buy_date).days / 365.25
    return {
        "Low $": round(value.min()),
        "Max drop": f"{value.min() / START_CASH - 1:.1%}",
        "Low date": low_date.date(),
        "Back to $50k": back.index[0].date() if len(back) else "never",
        f"Value {value.index[-1].date()}": round(value.iloc[-1]),
        "CAGR": f"{(value.iloc[-1] / START_CASH) ** (1 / years) - 1:.1%}",
    }


pd.set_option("display.width", 250)
for crash, (lo, hi) in CRASHES.items():
    rows = []
    for label, under, lev, ticker, expense, index in FUNDS:
        buy = px[index].loc[lo:hi].idxmax()
        out = fund_returns(under, lev, ticker, expense)
        r, launch = out if lev > 1 else (out, None)
        sim_note = f"simulated until {launch.date()}" if launch is not None and launch > buy else "real"
        rows.append({"Fund": label, "Bought": buy.date(), **simulate(r, buy), "Prices": sim_note})
    print(f"\n=== {crash} peak ===")
    print(pd.DataFrame(rows).to_string(index=False))
