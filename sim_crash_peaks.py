"""Buy-and-hold $50k at a market peak in 1x/2x/3x S&P 500 and Nasdaq-100 ETFs.

Crashes: Black Monday (1987), dot-com (2000), financial crisis (2007), COVID (2020).
Each fund group is
bought at the close of its own index's closing high (S&P 500 for SPY/SSO/UPRO,
Nasdaq-100 for QQQ/QLD/TQQQ), found from the data.

Real dividend-adjusted fund prices are used wherever the fund traded. Before a
leveraged fund launched (SSO/QLD Jun 2006, UPRO Jun 2009, TQQQ Feb 2010) it is
simulated from SPY/QQQ daily total returns:
    daily_Lx = L * r_underlying - (L - 1) * tbill/252 - expense/252

Before SPY (Jan 1993) and QQQ (Mar 1999) existed, the 1x funds are rebuilt from
the indexes, net of the ETF's expense ratio:
  - S&P 500: ^SP500TR total return from 1988; before that ^GSPC plus an
    assumed 3%/yr dividend yield (1988-92 actual was ~3.4%).
  - Nasdaq-100: ^NDX price return, dividends assumed to offset fund expenses
    (QQQ vs ^NDX differed by ~0.03%/yr in 1999-2004).

Run: pip install yfinance pandas && python sim_crash_peaks.py
"""
import pandas as pd
import yfinance as yf

START_CASH = 50_000
CRASHES = {  # name: window searched for the pre-crash closing high
    "Black Monday 1987": ("1987-01-01", "1987-10-16"),
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

px = yf.download(["SPY", "QQQ", "SSO", "QLD", "UPRO", "TQQQ", "^IRX", "^GSPC", "^NDX", "^SP500TR"],
                 start="1985-10-01", auto_adjust=True, progress=False)["Close"]
px = px[px["^GSPC"].notna()]
cash = (px["^IRX"] / 100).ffill().bfill()  # 13-week T-bill yield
rets = px.pct_change()


def splice(base, ticker):
    """Real fund returns where they exist, `base` returns before launch."""
    r = base.copy()
    launch = rets[ticker].first_valid_index()
    r.loc[launch:] = rets[ticker].loc[launch:]
    return r, launch


# 1x total-return proxies before SPY/QQQ launched, net of SPY 0.0945% / QQQ 0.20% fees
sp_tr = (rets["^GSPC"] + 0.03 / 252).where(rets["^SP500TR"].isna(), rets["^SP500TR"])
UNDER = {
    "SPY": splice(sp_tr - 0.000945 / 252, "SPY"),
    "QQQ": splice(rets["^NDX"].fillna(0), "QQQ"),
}


def fund_returns(under, lev, ticker, expense):
    base, base_launch = UNDER[under]
    if lev == 1:
        return base, base_launch
    return splice(lev * base - (lev - 1) * cash / 252 - expense / 252, ticker)


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
        r, launch = fund_returns(under, lev, ticker, expense)
        sim_note = f"simulated until {launch.date()}" if launch > buy else "real"
        rows.append({"Fund": label, "Bought": buy.date(), **simulate(r, buy), "Prices": sim_note})
    print(f"\n=== {crash} peak ===")
    print(pd.DataFrame(rows).to_string(index=False))
