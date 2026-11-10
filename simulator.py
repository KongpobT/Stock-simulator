"""
simulator.py — the "engine" of the Investment Strategy Simulator.

This file contains no web-page code. It only does the maths:
  1. load_prices()      -> get price history (Yahoo Finance or your own CSV)
  2. STRATEGIES         -> the 7 investing strategies
  3. run_backtest()     -> replay a strategy day by day with fees and cash
  4. compute_metrics()  -> turn the result into returns, risk and win rate
  5. monte_carlo()      -> simulate thousands of possible futures

Rule used everywhere to keep the test honest ("no peeking into the future"):
a signal calculated from today's closing price is only acted on at the NEXT
trading day's close.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------

def load_prices_yahoo(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance.

    Returns a DataFrame indexed by date with columns:
      close      - the actual traded price (used for dividend yield)
      adj_close  - price adjusted for dividends & splits (used for returns,
                   so dividends are counted as if reinvested)
      dividends  - cash dividend paid per share on that day (0 most days)
    """
    import yfinance as yf  # imported here so the engine also works offline

    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=True)
    if hist is None or hist.empty:
        raise ValueError(
            f"No price data found for '{ticker}'. Check the symbol "
            "(Thai stocks need .BK at the end, e.g. PTT.BK)."
        )
    return _clean(hist.rename(columns={"Close": "close", "Adj Close": "adj_close",
                                       "Dividends": "dividends"}))


def load_prices_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV you downloaded yourself (e.g. from Yahoo, Investing.com, SET).

    Needs at least a Date column and a Close column. "Adj Close" and
    "Dividends" columns are used if present. Column names are matched loosely.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c: c.strip().lower().replace("*", "").replace(".", "") for c in df.columns}
    df = df.rename(columns=cols)

    date_col = next((c for c in df.columns if c in ("date", "datetime", "time")), None)
    if date_col is None:
        raise ValueError("CSV needs a 'Date' column.")
    close_col = next((c for c in df.columns if c in ("close", "price", "last", "close price")), None)
    if close_col is None:
        raise ValueError("CSV needs a 'Close' (or 'Price') column.")

    out = pd.DataFrame(index=pd.to_datetime(df[date_col], errors="coerce"))
    out["close"] = pd.to_numeric(df[close_col].astype(str).str.replace(",", ""), errors="coerce").values
    adj = next((c for c in df.columns if c in ("adj close", "adj_close", "adjusted close")), None)
    out["adj_close"] = (pd.to_numeric(df[adj].astype(str).str.replace(",", ""), errors="coerce").values
                        if adj else out["close"].values)
    out["dividends"] = pd.to_numeric(df["dividends"], errors="coerce").fillna(0).values \
        if "dividends" in df.columns else 0.0
    out = out[out.index.notna()]
    return _clean(out)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if "adj_close" not in df:
        df["adj_close"] = df["close"]
    if "dividends" not in df:
        df["dividends"] = 0.0
    df = df[["close", "adj_close", "dividends"]]
    df["dividends"] = df["dividends"].fillna(0.0)
    df = df.dropna(subset=["close", "adj_close"])
    df = df[(df["close"] > 0) & (df["adj_close"] > 0)]
    if len(df) < 60:
        raise ValueError("Not enough price history (need at least ~3 months of daily prices).")
    return df


def load_fundamentals(ticker: str) -> dict:
    """Today's valuation snapshot from Yahoo. Free sources only give the
    CURRENT figures, not history, so these are shown but not backtested."""
    import yfinance as yf

    info = yf.Ticker(ticker).info or {}
    keys = {
        "longName": "Company", "currency": "Currency", "sector": "Sector",
        "marketCap": "Market cap", "trailingPE": "P/E (trailing)",
        "forwardPE": "P/E (forward)", "priceToBook": "Price / Book",
        "returnOnEquity": "Return on equity",
        "profitMargins": "Profit margin", "debtToEquity": "Debt / Equity",
        "beta": "Beta",
    }
    return {label: info.get(k) for k, label in keys.items()}


# ---------------------------------------------------------------------------
# 1b. TECHNICAL-CHART DATA
#     Each chart range uses a candle size that gives a readable number of candles.
#     We download extra history ("fetch") so indicators like the 200-bar average
#     are already valid at the left edge of the chart, then crop to "show".
# ---------------------------------------------------------------------------

CHART_RANGES = {
    #  label   fetch   interval  candle label   how much to show
    "1D":  dict(fetch="5d",   interval="5m",  bar="5 minutes", show="session"),
    "7D":  dict(fetch="1mo",  interval="30m", bar="30 minutes", show=pd.DateOffset(days=7)),
    "1M":  dict(fetch="6mo",  interval="1h",  bar="1 hour",    show=pd.DateOffset(months=1)),
    "3M":  dict(fetch="2y",   interval="1d",  bar="1 day",     show=pd.DateOffset(months=3)),
    "6M":  dict(fetch="2y",   interval="1d",  bar="1 day",     show=pd.DateOffset(months=6)),
    "1Y":  dict(fetch="3y",   interval="1d",  bar="1 day",     show=pd.DateOffset(years=1)),
    "5Y":  dict(fetch="10y",  interval="1wk", bar="1 week",    show=pd.DateOffset(years=5)),
    "Max": dict(fetch="max",  interval="1wk", bar="1 week",    show=None),
}
INTRADAY = {"1D", "7D", "1M"}


def load_ohlc(ticker: str, range_label: str) -> pd.DataFrame:
    """Open/High/Low/Close/Volume candles from Yahoo for one chart range
    (un-cropped, so indicators can warm up). Times are in the exchange's local time."""
    import yfinance as yf

    cfg = CHART_RANGES[range_label]
    hist = yf.Ticker(ticker).history(period=cfg["fetch"], interval=cfg["interval"],
                                     auto_adjust=False, actions=False)
    if hist is None or hist.empty:
        raise ValueError(f"No {cfg['interval']} price data for '{ticker}' right now.")
    out = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)  # keep exchange-local clock time
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out[(out["Close"] > 0)]


def ohlc_from_closes(df: pd.DataFrame) -> pd.DataFrame:
    """For uploaded CSVs (closing prices only): a simple daily table the chart can draw."""
    c = df["close"]
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.nan})


def crop_to_range(ohlc: pd.DataFrame, range_label: str) -> pd.DataFrame:
    """Keep only the part of the (longer) download that the chosen range shows."""
    show = CHART_RANGES[range_label]["show"]
    if show is None or ohlc.empty:
        return ohlc
    last = ohlc.index[-1]
    if isinstance(show, str):  # "session": 1D = the most recent trading day
        return ohlc[ohlc.index.normalize() == last.normalize()]
    return ohlc[ohlc.index > last - show]


# ---------------------------------------------------------------------------
# 2. INDICATORS (used by the strategies and the technical chart)
# ---------------------------------------------------------------------------

def sma(price: pd.Series, n: int) -> pd.Series:
    """Simple moving average: plain average of the last n closes."""
    return price.rolling(n, min_periods=n).mean()


def ema(price: pd.Series, n: int) -> pd.Series:
    """Exponential moving average: like SMA but recent prices count more."""
    return price.ewm(span=n, adjust=False, min_periods=n).mean()


def bollinger(price: pd.Series, n: int = 20, k: float = 2.0):
    """Bollinger Bands: n-bar average ± k standard deviations."""
    mid = sma(price, n)
    sd = price.rolling(n, min_periods=n).std()
    return mid + k * sd, mid, mid - k * sd


def macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line and histogram (the gap between them)."""
    line = ema(price, fast) - ema(price, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig

def rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder). Below 30 = 'oversold', above 70 = 'overbought'."""
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def trailing_dividend_yield(df: pd.DataFrame) -> pd.Series:
    """Dividends paid over the last 365 days divided by today's price."""
    div = df["dividends"].copy()
    ttm = div.rolling("365D").sum()
    return ttm / df["close"]


def _state_machine(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """Turn 'buy now' / 'sell now' flags into a held(1) / not-held(0) series."""
    held, out = 0, np.zeros(len(enter))
    e, x = enter.to_numpy(), exit_.to_numpy()
    for i in range(len(e)):
        if held == 0 and e[i]:
            held = 1
        elif held == 1 and x[i]:
            held = 0
        out[i] = held
    return pd.Series(out, index=enter.index)


# ---------------------------------------------------------------------------
# 3. STRATEGIES
#    Each returns a 0/1 series: 1 = "I want to be fully invested",
#    0 = "I want to be in cash". Calculated from data up to that day only.
# ---------------------------------------------------------------------------

def sig_always_in(df: pd.DataFrame, p: dict) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sig_trend(df: pd.DataFrame, p: dict) -> pd.Series:
    """Moving-average crossover: own the stock while the fast average is above the slow one."""
    px = df["adj_close"]
    fast = px.rolling(p["sma_fast"]).mean()
    slow = px.rolling(p["sma_slow"]).mean()
    return (fast > slow).astype(float)


def sig_mean_reversion(df: pd.DataFrame, p: dict) -> pd.Series:
    """RSI: buy when the stock is oversold, sell when it becomes overbought."""
    r = rsi(df["adj_close"], p["rsi_period"])
    return _state_machine(r < p["rsi_buy"], r > p["rsi_sell"])


def sig_momentum(df: pd.DataFrame, p: dict) -> pd.Series:
    """Breakout: buy when price makes a new N-day high; sell if it falls X% from its peak since buying."""
    px = df["adj_close"].to_numpy()
    prior_high = df["adj_close"].rolling(p["breakout_days"]).max().shift(1).to_numpy()
    stop = p["trailing_stop"] / 100
    held, peak, out = 0, 0.0, np.zeros(len(px))
    for i in range(len(px)):
        if held == 0:
            if not np.isnan(prior_high[i]) and px[i] > prior_high[i]:
                held, peak = 1, px[i]
        else:
            peak = max(peak, px[i])
            if px[i] < peak * (1 - stop):
                held = 0
        out[i] = held
    return pd.Series(out, index=df.index)


def sig_value(df: pd.DataFrame, p: dict) -> tuple[pd.Series, str]:
    """Value investing, using only data a real investor had at the time.

    * If the stock pays dividends: buy when its dividend yield is unusually HIGH
      versus its own history (= the price is cheap relative to the cash it pays),
      sell when the yield becomes unusually LOW (= expensive).
    * If it pays no dividends: buy when the price is a 'margin of safety' below
      its long-run average, sell when it trades back above that average.
    """
    has_divs = (df["dividends"] > 0).sum() >= 3
    if has_divs:
        y = trailing_dividend_yield(df)
        y = y.where(df.index >= df.index[0] + pd.Timedelta(days=365))  # need a full year first
        hist = y.expanding(min_periods=TRADING_DAYS)
        cheap = y >= hist.quantile(p["value_buy_pct"] / 100)
        dear = y <= hist.quantile(p["value_sell_pct"] / 100)
        return _state_machine(cheap.fillna(False), dear.fillna(False)), "dividend-yield rule"
    px = df["adj_close"]
    fair = px.rolling(p["value_window"], min_periods=TRADING_DAYS).mean()
    cheap = px <= fair * (1 - p["value_margin"] / 100)
    dear = px >= fair
    return _state_machine(cheap.fillna(False), dear.fillna(False)), "price margin-of-safety rule"


DIP_EXITS = {
    "recover": "When it's back at the old high",
    "target": "At a profit target",
    "hold": "Never (keep holding)",
}


def sig_buy_dip(df: pd.DataFrame, p: dict) -> pd.Series:
    """Buy the dip: buy once the price has fallen `dip_pct`% below its highest
    price of the last `dip_lookback` trading days. Then exit by one of three rules:

      recover - sell when the price climbs back to the high it fell from
      target  - sell when the trade is up `dip_take_profit`%
      hold    - never sell; just keep the shares
    Optional `dip_stop` (%) sells if the price keeps falling that far below the buy price
    (0 = no stop-loss). After a stop-loss it won't buy again until the price has bounced
    back out of the dip zone, so it doesn't instantly re-buy the same falling stock.
    """
    px = df["adj_close"].to_numpy()
    high = df["adj_close"].rolling(int(p["dip_lookback"]), min_periods=1).max().to_numpy()
    dip = p["dip_pct"] / 100
    tp = p["dip_take_profit"] / 100
    stop = p.get("dip_stop", 0) / 100
    mode = p["dip_exit"]

    held, entry_px, ref_high = 0, 0.0, 0.0
    armed = True  # after a stop-loss, wait for the price to bounce out of the dip zone
    out = np.zeros(len(px))
    for i in range(len(px)):
        in_dip = px[i] <= high[i] * (1 - dip)
        if held == 0:
            if not armed and not in_dip:
                armed = True
            if armed and i >= 1 and in_dip:
                held, entry_px, ref_high = 1, px[i], high[i]
        else:
            if stop > 0 and px[i] <= entry_px * (1 - stop):
                held, armed = 0, False
            elif mode == "recover" and px[i] >= ref_high:
                held = 0
            elif mode == "target" and px[i] >= entry_px * (1 + tp):
                held = 0
        out[i] = held
    return pd.Series(out, index=df.index)


@dataclass
class StrategyInfo:
    key: str
    name: str
    short: str
    color: str


STRATEGIES = [
    StrategyInfo("buy_hold", "Buy & Hold", "Invest everything on day one and never sell.", "#2a78d6"),
    StrategyInfo("dca", "Dollar-Cost Averaging", "Split the starting money into equal monthly buys.", "#eb6834"),
    StrategyInfo("index", "Passive Index", "Buy & hold the market index/ETF instead of the stock.", "#1baf7a"),
    StrategyInfo("trend", "Trend Following", "Hold only while the fast moving average is above the slow one.", "#eda100"),
    StrategyInfo("mean_rev", "Mean Reversion (RSI)", "Buy when oversold, sell when overbought.", "#e87ba4"),
    StrategyInfo("momentum", "Momentum Breakout", "Buy new highs, exit on a trailing stop-loss.", "#008300"),
    StrategyInfo("value", "Value Investing", "Buy when cheap vs. its own history, sell when expensive.", "#6250d6"),
    StrategyInfo("dip", "Buy the Dip", "Buy after the price falls a set % from its recent high.", "#e34948"),
]
STRATEGY_BY_KEY = {s.key: s for s in STRATEGIES}

DEFAULT_PARAMS = dict(
    dca_months=12,
    sma_fast=50, sma_slow=200,
    rsi_period=14, rsi_buy=30, rsi_sell=70,
    breakout_days=50, trailing_stop=15,
    value_buy_pct=70, value_sell_pct=30, value_window=756, value_margin=15,
    dip_pct=10, dip_lookback=252, dip_exit="recover", dip_take_profit=20, dip_stop=0,
)


# ---------------------------------------------------------------------------
# 4. BACKTEST ENGINE
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    key: str
    equity: pd.Series             # portfolio value each day
    contributed: pd.Series        # total money put in so far
    position: pd.Series           # 1 = invested, 0 = cash
    trades: pd.DataFrame          # one row per completed buy -> sell round trip
    note: str = ""
    twr_returns: pd.Series = field(default=None)  # daily returns excluding deposits


def run_backtest(price: pd.Series, signal: pd.Series, start_cash: float, monthly: float,
                 fee_pct: float, dca_months: int = 0, key: str = "", note: str = "") -> BacktestResult:
    """Replay the strategy one trading day at a time.

    price       adjusted close (dividends reinvested)
    signal      1/0 wish to be invested, decided at that day's close
    start_cash  money available on day one
    monthly     extra money added on the first trading day of each month
    fee_pct     broker commission % charged on every buy and every sell
    dca_months  if > 0, starting cash is released in this many equal monthly slices
    """
    fee = fee_pct / 100
    pos_wanted = signal.reindex(price.index).shift(1).fillna(0).to_numpy()  # act next day
    if key in ("buy_hold", "dca", "index"):
        pos_wanted = np.ones(len(price))  # passive strategies are invested from day one

    px = price.to_numpy()
    idx = price.index
    month = idx.to_period("M")
    new_month = np.r_[True, month[1:] != month[:-1]]

    cash = 0.0 if dca_months > 0 else start_cash
    reserve = start_cash if dca_months > 0 else 0.0
    tranche = start_cash / dca_months if dca_months > 0 else 0.0
    shares = 0.0
    contributed_total = start_cash

    equity = np.zeros(len(px))
    contrib = np.zeros(len(px))
    flows = np.zeros(len(px))
    held = np.zeros(len(px))
    trades, entry = [], None

    for i in range(len(px)):
        p = px[i]
        if new_month[i]:
            if i > 0 and monthly > 0:
                cash += monthly
                contributed_total += monthly
                flows[i] += monthly
            if reserve > 0:
                take = min(tranche, reserve)
                reserve -= take
                cash += take

        if pos_wanted[i] >= 0.5 and cash > 1e-9:
            bought = cash / (p * (1 + fee))
            if shares == 0:
                entry = (idx[i], p)
            shares += bought
            cash = 0.0
        elif pos_wanted[i] < 0.5 and shares > 0:
            cash += shares * p * (1 - fee)
            if entry is not None:
                trades.append({"Bought": entry[0].date(), "Buy price": entry[1],
                               "Sold": idx[i].date(), "Sell price": p,
                               "Return %": (p * (1 - fee)) / (entry[1] * (1 + fee)) * 100 - 100,
                               "Days held": (idx[i] - entry[0]).days})
            shares, entry = 0.0, None

        equity[i] = cash + reserve + shares * p
        contrib[i] = contributed_total
        held[i] = 1.0 if shares > 0 else 0.0

    # open trade at the end is shown but marked as still open
    if shares > 0 and entry is not None:
        trades.append({"Bought": entry[0].date(), "Buy price": entry[1], "Sold": "still holding",
                       "Sell price": px[-1],
                       "Return %": (px[-1]) / (entry[1] * (1 + fee)) * 100 - 100,
                       "Days held": (idx[-1] - entry[0]).days})

    eq = pd.Series(equity, index=idx)
    prev = eq.shift(1)
    twr = ((eq - flows) / prev - 1).iloc[1:].replace([np.inf, -np.inf], np.nan).fillna(0)

    return BacktestResult(key=key, equity=eq, contributed=pd.Series(contrib, index=idx),
                          position=pd.Series(held, index=idx), trades=pd.DataFrame(trades),
                          note=note, twr_returns=twr)


def run_all(df: pd.DataFrame, bench: pd.DataFrame | None, keys: list[str], start_cash: float,
            monthly: float, fee_pct: float, params: dict) -> dict[str, BacktestResult]:
    px = df["adj_close"]
    out: dict[str, BacktestResult] = {}
    for k in keys:
        note = ""
        if k == "buy_hold":
            sig = sig_always_in(df, params)
            out[k] = run_backtest(px, sig, start_cash, monthly, fee_pct, key=k)
        elif k == "dca":
            sig = sig_always_in(df, params)
            out[k] = run_backtest(px, sig, start_cash, monthly, fee_pct,
                                  dca_months=int(params["dca_months"]), key=k)
        elif k == "index":
            if bench is None or bench.empty:
                continue
            bpx = bench["adj_close"].reindex(px.index).ffill().bfill()
            out[k] = run_backtest(bpx, sig_always_in(df, params), start_cash, monthly, fee_pct, key=k)
        elif k == "trend":
            out[k] = run_backtest(px, sig_trend(df, params), start_cash, monthly, fee_pct, key=k)
        elif k == "mean_rev":
            out[k] = run_backtest(px, sig_mean_reversion(df, params), start_cash, monthly, fee_pct, key=k)
        elif k == "momentum":
            out[k] = run_backtest(px, sig_momentum(df, params), start_cash, monthly, fee_pct, key=k)
        elif k == "value":
            sig, note = sig_value(df, params)
            out[k] = run_backtest(px, sig, start_cash, monthly, fee_pct, key=k, note=note)
        elif k == "dip":
            out[k] = run_backtest(px, sig_buy_dip(df, params), start_cash, monthly, fee_pct, key=k)
    return out


# ---------------------------------------------------------------------------
# 5. SCORECARD
# ---------------------------------------------------------------------------

def compute_metrics(res: BacktestResult, rf_pct: float = 0.0) -> dict:
    eq, twr = res.equity, res.twr_returns
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    growth = float((1 + twr).prod())
    cagr = growth ** (1 / years) - 1 if growth > 0 else -1.0
    vol = float(twr.std() * np.sqrt(TRADING_DAYS))
    rf_daily = (1 + rf_pct / 100) ** (1 / TRADING_DAYS) - 1
    excess = twr - rf_daily
    sharpe = float(excess.mean() / twr.std() * np.sqrt(TRADING_DAYS)) if twr.std() > 0 else np.nan
    curve = (1 + twr).cumprod()
    max_dd = float((curve / curve.cummax() - 1).min()) if len(curve) else 0.0

    t = res.trades
    closed = t[t["Sold"] != "still holding"] if len(t) else t
    win_rate = float((closed["Return %"] > 0).mean()) if len(closed) else np.nan

    final, put_in = float(eq.iloc[-1]), float(res.contributed.iloc[-1])
    return {
        "Final value": final,
        "Money put in": put_in,
        "Profit": final - put_in,
        "Total return %": (final / put_in - 1) * 100 if put_in else np.nan,
        "Annual return % (CAGR)": cagr * 100,
        "Volatility %": vol * 100,
        "Max drawdown %": max_dd * 100,
        "Sharpe ratio": sharpe,
        "Time invested %": float(res.position.mean() * 100),
        "Trades": int(len(closed)),
        "Win rate %": win_rate * 100 if not np.isnan(win_rate) else np.nan,
    }


def drawdown_series(res: BacktestResult) -> pd.Series:
    curve = (1 + res.twr_returns).cumprod()
    return (curve / curve.cummax() - 1) * 100


# ---------------------------------------------------------------------------
# 6. FORECAST
# ---------------------------------------------------------------------------

def monte_carlo(price: pd.Series, amount: float, monthly: float, years: float,
                n_paths: int = 2000, lookback_years: float = 5,
                drift_override_pct: float | None = None, seed: int = 42) -> dict:
    """Simulate many possible futures by re-shuffling the stock's real past daily moves.

    'Bootstrap' method: each simulated day picks a random real day from the last
    `lookback_years` of history. This keeps the stock's true volatility and its
    occasional crash days, without assuming returns follow a neat bell curve.

    drift_override_pct: if given, the average yearly return is replaced by this
    number while keeping the same day-to-day ups and downs.
    """
    rng = np.random.default_rng(seed)
    logret = np.log(price).diff().dropna()
    cutoff = logret.index[-1] - pd.Timedelta(days=int(lookback_years * 365.25))
    sample = logret[logret.index >= cutoff].to_numpy()
    if len(sample) < 60:
        sample = logret.to_numpy()

    hist_annual = float(np.exp(sample.mean() * TRADING_DAYS) - 1)
    if drift_override_pct is not None:
        target_daily = np.log(1 + drift_override_pct / 100) / TRADING_DAYS
        sample = sample - sample.mean() + target_daily

    n_days = int(round(years * TRADING_DAYS))
    draws = rng.choice(sample, size=(n_paths, n_days), replace=True)

    values = np.empty((n_paths, n_days + 1))
    values[:, 0] = amount
    growth = np.exp(draws)
    for d in range(n_days):
        v = values[:, d] * growth[:, d]
        if monthly > 0 and (d + 1) % 21 == 0:
            v = v + monthly
        values[:, d + 1] = v

    put_in = amount + monthly * (n_days // 21)
    pct = np.percentile(values, [5, 25, 50, 75, 95], axis=0)
    final = values[:, -1]
    return {
        "days": np.arange(n_days + 1),
        "p5": pct[0], "p25": pct[1], "p50": pct[2], "p75": pct[3], "p95": pct[4],
        "sample_paths": values[:30],
        "final": final,
        "put_in": put_in,
        "prob_loss": float((final < put_in).mean()),
        "prob_double": float((final >= 2 * put_in).mean()),
        "hist_annual_pct": hist_annual * 100,
        "years_used": len(sample) / TRADING_DAYS,
    }


def scenario_path(amount: float, monthly: float, years: float, annual_pct: float) -> np.ndarray:
    """A smooth 'what if it grows exactly X% a year' line for your own scenarios."""
    n_days = int(round(years * TRADING_DAYS))
    daily = (1 + annual_pct / 100) ** (1 / TRADING_DAYS)
    v = np.empty(n_days + 1)
    v[0] = amount
    for d in range(n_days):
        v[d + 1] = v[d] * daily + (monthly if (d + 1) % 21 == 0 else 0)
    return v
