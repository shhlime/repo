"""Buy-and-hold $50k at a market peak in 1x/2x/3x index and semiconductor ETFs.

Crashes: Black Monday (1987), dot-com (2000), financial crisis (2007), COVID (2020).
Each fund group is bought at the close of its own index's closing high (S&P 500
for SPY/SSO/UPRO, Nasdaq-100 for QQQ/QLD/TQQQ, semiconductors for SOXX/USD/SOXL),
found from the data.

Assumptions: dividends reinvested, no taxes, nominal dollars, and a simulated
fund is assumed to stay open however far it falls.

Real dividend-adjusted fund prices are used wherever the fund traded. Before a
fund existed it is simulated from index total returns (index closes, not ETF
closes, which were noisy before ~2011):
  - S&P 500: ^SP500TR from 1988; before that ^GSPC plus an assumed 3%/yr
    dividend yield (1988-92 actual was ~3.4%).
  - Nasdaq-100: ^NDX plus QQQ's dividends; before QQQ (1999) dividends are
    assumed to be 0.2%/yr (QQQ's fee, which they roughly offset in 1999-2004).
  - Semiconductors: ^SOX plus SOXX's dividends (0.3%/yr assumed before SOXX,
    its 2001-10 yield). Before ^SOX starts (May 1994): an equal-weighted,
    monthly-rebalanced basket of chip stocks Yahoo has back to 1987, minus
    5.2%/yr for survivorship bias (its measured excess over ^SOX in 1994-2010;
    it tracked ^SOX with beta ~1.0, correlation 0.94-0.97). Ken French's
    "Chips" portfolio was rejected: ~40% less volatile than semis in 1987-94.
  - 1x before launch: index total return minus the ETF's expense ratio.
  - Lx before launch: L * index_tr - (L - 1) * (tbill + spread)/252 - 0.95%/252,
    where `spread` is fitted so the formula reproduces the real fund's own
    history. USD tracks a different, Nvidia-heavier semis index than ^SOX, so
    it reuses SOXL's spread. Crisis-era spreads ran higher, so simulated
    stretches through 2000-02 or 2008 are still somewhat optimistic.

Run: pip install yfinance pandas && python sim_crash_peaks.py
"""
import pandas as pd
import yfinance as yf

START_CASH = 50_000
LEV_EXPENSE = 0.0095  # 2x/3x fee for simulated stretches (period-appropriate)
CRASHES = {  # name: window searched for the pre-crash closing high
    "Black Monday 1987": ("1987-01-01", "1987-10-16"),
    "Dot-com 2000": ("1999-06-01", "2000-12-31"),
    "Financial crisis 2007": ("2007-01-01", "2008-06-30"),
    "COVID 2020": ("2019-06-01", "2020-03-31"),
}
FUNDS = [  # label, index group, leverage, ticker
    ("S&P 500 (SPY)", "SPX", 1, "SPY"),
    ("2x S&P 500 (SSO)", "SPX", 2, "SSO"),
    ("3x S&P 500 (UPRO)", "SPX", 3, "UPRO"),
    ("Nasdaq-100 (QQQ)", "NDX", 1, "QQQ"),
    ("2x Nasdaq-100 (QLD)", "NDX", 2, "QLD"),
    ("3x Nasdaq-100 (TQQQ)", "NDX", 3, "TQQQ"),
    ("Semiconductors (SOXX)", "SOX", 1, "SOXX"),
    ("2x Semiconductors (USD)", "SOX", 2, "USD"),
    ("3x Semiconductors (SOXL)", "SOX", 3, "SOXL"),
]
ETF_FEE = {"SPY": 0.000945, "QQQ": 0.0020, "SOXX": 0.0045}
CHIP_STOCKS = ["INTC", "TXN", "AMD", "MU", "AMAT", "KLAC", "LRCX", "ADI"]
SURVIVOR_HAIRCUT = 0.052

raw = yf.download(["SPY", "QQQ", "SSO", "QLD", "UPRO", "TQQQ", "SOXX", "USD", "SOXL",
                   "^IRX", "^GSPC", "^NDX", "^SP500TR", "^SOX", *CHIP_STOCKS],
                  start="1985-10-01", auto_adjust=False, progress=False)
keep = raw.index[raw["Adj Close"]["^GSPC"].notna() & (raw.index < pd.Timestamp.today().normalize())]
px = raw["Adj Close"].loc[keep].ffill()  # a missing day becomes 0%, not a lost move
close = raw["Close"].loc[keep].ffill()
cash = (px["^IRX"] / 100).bfill()  # 13-week T-bill yield
rets = px.pct_change()


def dividends(ticker, assumed):
    """Daily dividend return of an ETF; `assumed` annual yield before it existed."""
    div = (1 + rets[ticker]) / (1 + close[ticker].pct_change()) - 1
    return div.where(rets[ticker].notna(), assumed / 252)


def chip_basket():
    """Equal-weighted chip stocks, rebalanced monthly, net of the survivorship haircut."""
    stocks, prev = px[CHIP_STOCKS], px[CHIP_STOCKS].shift(1)
    months = []
    for _, block in stocks.groupby(stocks.index.to_period("M")):
        level = (block / prev.loc[block.index[0]]).mean(axis=1)  # weights reset at prior close
        months.append(level / level.shift(1, fill_value=1) - 1)
    return pd.concat(months) - SURVIVOR_HAIRCUT / 252


# Index total returns used to simulate funds before they existed.
INDEX_TR = {
    "SPX": rets["^SP500TR"].fillna(rets["^GSPC"] + 0.03 / 252),
    "NDX": rets["^NDX"].fillna(0) + dividends("QQQ", 0.0020),
    "SOX": (rets["^SOX"] + dividends("SOXX", 0.0030)).fillna(chip_basket()),
}
LEVEL = {  # what "the index's closing high" is measured on
    "SPX": px["^GSPC"],
    "NDX": px["^NDX"],
    "SOX": (1 + INDEX_TR["SOX"].fillna(0)).cumprod(),
}


def first_real(ticker):
    return rets[ticker].first_valid_index()  # first day with a real fund return


def lev_formula(group, lev, spread):
    return lev * INDEX_TR[group] - (lev - 1) * (cash + spread) / 252 - LEV_EXPENSE / 252


def fit_spread(group, lev, ticker):
    """Extra annual financing cost that makes the formula match the real fund."""
    real = (1 + rets[ticker].loc[first_real(ticker):]).prod()
    lo, hi = -0.05, 0.10
    for _ in range(60):
        mid = (lo + hi) / 2
        sim = (1 + lev_formula(group, lev, mid).loc[first_real(ticker):]).prod()
        lo, hi = (mid, hi) if sim > real else (lo, mid)
    return mid


SPREAD = {t: fit_spread(g, l, t) for _, g, l, t in FUNDS if l > 1 and t != "USD"}
SPREAD["USD"] = SPREAD["SOXL"]


def fund_returns(group, lev, ticker):
    if lev == 1:
        sim = INDEX_TR[group] - ETF_FEE[ticker] / 252
    else:
        sim = lev_formula(group, lev, SPREAD[ticker])
    launch = first_real(ticker)
    r = sim.copy()
    r.loc[launch:] = rets[ticker].loc[launch:]
    return r, px[ticker].first_valid_index()


def pct(x):
    return f"{x:.4%}" if x < -0.9999 else f"{x:.2%}" if x < -0.995 else f"{x:.1%}"


def simulate(r, buy):
    r = r.loc[buy:].iloc[1:].fillna(0)  # buy at the buy-date close
    value = START_CASH * (1 + r).cumprod()
    peak = value.cummax().clip(lower=START_CASH)
    worst = (value / peak - 1).idxmin()
    worst_from = value.loc[:worst].idxmax() if value.loc[:worst].max() > START_CASH else buy
    below = value < START_CASH
    if not below.any():
        first_back = for_good = "never fell below"
    else:
        after = value.loc[below.idxmax():]
        first_back = after[after >= START_CASH].index.min()
        first_back = first_back.date() if pd.notna(first_back) else "never"
        last_below = below[below].index[-1]
        for_good = "never" if below.iloc[-1] else value.index[value.index > last_below][0].date()
    years = (value.index[-1] - buy).days / 365.25
    low = value.min()
    return {
        "Lowest $": f"{low:,.{2 if low < 100 else 0}f} ({pct(low / START_CASH - 1)}, {value.idxmin().date()})",
        "Worst drop from a high": f"{pct(value[worst] / peak[worst] - 1)} ({worst_from.date()} to {worst.date()})",
        "First back to $50k": first_back,
        "Back for good": for_good,
        f"Value {value.index[-1].date()}": f"{value.iloc[-1]:,.0f}",
        "CAGR": f"{(value.iloc[-1] / START_CASH) ** (1 / years) - 1:.1%}",
    }


pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", 60)
print("Fitted extra financing over T-bills (per borrowed unit, per year):",
      ", ".join(f"{t} {s:+.2%}" for t, s in SPREAD.items()))
for crash, (lo, hi) in CRASHES.items():
    rows = []
    for label, group, lev, ticker in FUNDS:
        buy = LEVEL[group].loc[lo:hi].idxmax()
        r, launch = fund_returns(group, lev, ticker)
        note = f"simulated before {launch.date()}" if launch > buy else "real"
        rows.append({"Fund": label, "Bought": buy.date(), **simulate(r, buy), "Prices": note})
    print(f"\n=== {crash} peak ===")
    print(pd.DataFrame(rows).to_string(index=False))
