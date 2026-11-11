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


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14,
               smooth: int = 3, d: int = 3):
    """Stochastic oscillator: where the close sits in the recent high–low range (0–100)."""
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    raw = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k_line = raw.rolling(smooth, min_periods=smooth).mean()
    return k_line, k_line.rolling(d, min_periods=d).mean()


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


def moving_avg(price: pd.Series, n: int, kind: str = "SMA") -> pd.Series:
    return ema(price, n) if kind == "EMA" else sma(price, n)


def sig_trend(df: pd.DataFrame, p: dict) -> pd.Series:
    """Moving-average crossover: own the stock while the fast average is above the slow one."""
    px = df["adj_close"]
    kind = p.get("ma_type", "SMA")
    fast = moving_avg(px, p["sma_fast"], kind)
    slow = moving_avg(px, p["sma_slow"], kind)
    return (fast > slow).astype(float)


def sig_mean_reversion(df: pd.DataFrame, p: dict) -> pd.Series:
    """RSI: buy when the stock is oversold, sell when it becomes overbought."""
    r = rsi(df["adj_close"], p["rsi_period"])
    return _state_machine(r < p["rsi_buy"], r > p["rsi_sell"])


def _momentum_run(df: pd.DataFrame, p: dict):
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
    return pd.Series(out, index=df.index), {"peak": peak}


def sig_momentum(df: pd.DataFrame, p: dict) -> pd.Series:
    """Breakout: buy when price makes a new N-day high; sell if it falls X% from its peak since buying."""
    return _momentum_run(df, p)[0]


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
    return _dip_run(df, p)[0]


def _dip_run(df: pd.DataFrame, p: dict):
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
    return pd.Series(out, index=df.index), {"entry_px": entry_px, "ref_high": ref_high,
                                            "armed": armed, "high": high[-1]}


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
    sma_fast=50, sma_slow=200, ma_type="SMA",
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


# ---------------------------------------------------------------------------
# 7. TODAY'S SIGNAL
#    Applies a strategy to the latest price and explains what it says, which
#    prices would trigger its next action, and how its past signals worked out.
# ---------------------------------------------------------------------------

PASSIVE = {"buy_hold", "dca", "index"}
HORIZONS = {"1 month": 21, "3 months": 63, "6 months": 126}


def strategy_position(df: pd.DataFrame, key: str, p: dict) -> pd.Series:
    """1 = the strategy wants to own the stock at that day's close, 0 = cash."""
    if key in PASSIVE:
        return sig_always_in(df, p)
    if key == "trend":
        return sig_trend(df, p)
    if key == "mean_rev":
        return sig_mean_reversion(df, p)
    if key == "momentum":
        return sig_momentum(df, p)
    if key == "value":
        return sig_value(df, p)[0]
    if key == "dip":
        return sig_buy_dip(df, p)
    raise KeyError(key)


def _ma_cross_price(px: np.ndarray, fast: float, slow: float, nf: int, ns: int, kind: str):
    """Closing price tomorrow at which the fast average would equal the slow one."""
    if kind == "EMA":
        af, a_s = 2 / (nf + 1), 2 / (ns + 1)
        return ((1 - a_s) * slow - (1 - af) * fast) / (af - a_s)
    if len(px) < ns:
        return np.nan
    out_f, out_s = px[-nf], px[-ns]  # prices that drop out of each window tomorrow
    return (slow - fast + out_f / nf - out_s / ns) / (1 / nf - 1 / ns)


def _rsi_trigger_price(price: pd.Series, n: int, level: float, direction: str) -> float:
    """Closing price tomorrow that would push RSI below (direction='down') or above ('up') level."""
    delta = price.diff()
    ag = float(delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1])
    al = float((-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1])
    rs = level / (100 - level)
    last = float(price.iloc[-1])
    if direction == "down":
        return last - max((n - 1) * (ag / rs - al), 0.0)
    return last + max((n - 1) * (rs * al - ag), 0.0)


def _level(label: str, price: float, kind: str, note: str = "") -> dict:
    return {"label": label, "price": float(price), "kind": kind, "note": note}


def todays_signal(df: pd.DataFrame, key: str, p: dict, bench: pd.DataFrame | None = None) -> dict:
    """What the strategy says at the latest close.

    action:  BUY  (fresh buy signal today)    HOLD (already in since an earlier signal)
             SELL (fresh sell signal today)   WAIT (out of the stock, no buy signal)
             INVEST (passive strategies — always invested, timing doesn't matter)
    """
    px = df["adj_close"]
    last = float(px.iloc[-1])
    pos = strategy_position(df, key, p)
    now, prev = int(pos.iloc[-1]), int(pos.iloc[-2]) if len(pos) > 1 else 0
    levels, readings, overlays = [], [], {}
    since_date = since_px = None

    if key in PASSIVE:
        action = "INVEST"
    elif now and not prev:
        action = "BUY"
    elif now:
        action = "HOLD"
    elif prev:
        action = "SELL"
    else:
        action = "WAIT"

    if now and key not in PASSIVE:
        starts = pos.index[(pos.diff() > 0) | ((pos > 0) & (pos.index == pos.index[0]))]
        since_date = starts[-1]
        since_px = float(px.loc[since_date])

    # --- common readings
    hi52 = float(px.iloc[-TRADING_DAYS:].max())
    lo52 = float(px.iloc[-TRADING_DAYS:].min())
    ma200 = float(px.rolling(200).mean().iloc[-1]) if len(px) >= 200 else np.nan

    if key in PASSIVE:
        what = {"buy_hold": "Buy & Hold invests regardless of price — it never waits for a signal.",
                "dca": "DCA buys a fixed amount every month regardless of price — this month's "
                       "instalment is due as normal.",
                "index": "Passive Index buys the whole market instead of this stock."}[key]
        headline = {"buy_hold": "Always invested", "dca": "Buy this month's instalment",
                    "index": "Invest in the index"}[key]
        reason = what
        readings += [("Price vs 52-week high", f"{(last / hi52 - 1) * 100:+.1f}%"),
                     ("Price vs 52-week low", f"{(last / lo52 - 1) * 100:+.1f}%")]
        if not np.isnan(ma200):
            readings.append(("Price vs 200-day average", f"{(last / ma200 - 1) * 100:+.1f}%"))
        if key == "index" and bench is not None and len(bench):
            b = bench["adj_close"]
            readings.append(("Index vs its 52-week high",
                             f"{(float(b.iloc[-1]) / float(b.iloc[-TRADING_DAYS:].max()) - 1) * 100:+.1f}%"))

    elif key == "trend":
        kind = p.get("ma_type", "SMA")
        nf, ns = int(p["sma_fast"]), int(p["sma_slow"])
        f_s, s_s = moving_avg(px, nf, kind), moving_avg(px, ns, kind)
        fast, slow = float(f_s.iloc[-1]), float(s_s.iloc[-1])
        overlays = {f"{kind} {nf}": f_s, f"{kind} {ns}": s_s}
        gap = (fast / slow - 1) * 100
        readings += [(f"{kind} {nf} (fast)", f"{fast:,.2f}"), (f"{kind} {ns} (slow)", f"{slow:,.2f}"),
                     ("Fast vs slow", f"{gap:+.2f}%")]
        cross = _ma_cross_price(px.to_numpy(), fast, slow, nf, ns, kind)
        reachable = np.isfinite(cross) and 0 < cross < last * 3
        if now:
            headline = "Uptrend — fast average is above the slow one"
            reason = (f"The {nf}-day {kind} ({fast:,.2f}) is {gap:.1f}% above the {ns}-day {kind} "
                      f"({slow:,.2f}).")
            if reachable and cross < last:
                levels.append(_level("Sell if it closes below", cross, "sell",
                                     "averages would cross down at the next close"))
        else:
            headline = "Downtrend — fast average is below the slow one"
            reason = (f"The {nf}-day {kind} ({fast:,.2f}) is {abs(gap):.1f}% below the {ns}-day "
                      f"{kind} ({slow:,.2f}).")
            if reachable and cross > last:
                levels.append(_level("Buy if it closes above", cross, "buy",
                                     "averages would cross up at the next close"))
        if not levels:
            reason += (f" The averages are too far apart for one day's price to flip them — "
                       f"it would take several days of {'gains' if not now else 'losses'}.")
        if action == "SELL":
            headline = "Sell signal — fast average just crossed below the slow one"
        if action == "BUY":
            headline = "Buy signal — fast average just crossed above the slow one"

    elif key == "mean_rev":
        n = int(p["rsi_period"])
        r_s = rsi(px, n)
        r = float(r_s.iloc[-1])
        overlays = {"RSI": r_s}
        readings += [(f"RSI {n}", f"{r:.1f}"), ("Buy below", f"{p['rsi_buy']}"),
                     ("Sell above", f"{p['rsi_sell']}")]
        if now:
            headline = f"Holding — waiting for RSI to rise above {p['rsi_sell']}"
            reason = f"RSI is {r:.0f}. The strategy sells once the stock becomes overbought."
            levels.append(_level(f"Sell if it closes above (RSI > {p['rsi_sell']})",
                                 _rsi_trigger_price(px, n, p["rsi_sell"], "up"), "sell"))
        else:
            headline = f"No buy — RSI {r:.0f} is not below {p['rsi_buy']}"
            reason = "The strategy only buys when the stock has fallen hard enough to look oversold."
            levels.append(_level(f"Buy if it closes below (RSI < {p['rsi_buy']})",
                                 _rsi_trigger_price(px, n, p["rsi_buy"], "down"), "buy"))
        if action == "BUY":
            headline = f"Buy signal — RSI {r:.0f} fell below {p['rsi_buy']} (oversold)"
        if action == "SELL":
            headline = f"Sell signal — RSI {r:.0f} rose above {p['rsi_sell']} (overbought)"

    elif key == "momentum":
        n = int(p["breakout_days"])
        _, st_ = _momentum_run(df, p)
        hi_n = float(px.iloc[-n:].max())
        overlays = {f"{n}-day high": px.rolling(n).max()}
        readings += [(f"{n}-day high", f"{hi_n:,.2f}"),
                     ("Price vs that high", f"{(last / hi_n - 1) * 100:+.1f}%")]
        if now:
            stop_px = st_["peak"] * (1 - p["trailing_stop"] / 100)
            headline = "Holding — riding the breakout"
            reason = (f"Peak since buying is {st_['peak']:,.2f}. The trailing stop sells if the price "
                      f"falls {p['trailing_stop']}% from that peak.")
            readings.append(("Peak since buying", f"{st_['peak']:,.2f}"))
            levels.append(_level("Sell if it closes below (trailing stop)", stop_px, "sell"))
        else:
            headline = f"No breakout — price is below its {n}-day high"
            reason = f"The strategy buys only when the price closes above its highest close of the last {n} days."
            levels.append(_level(f"Buy if it closes above ({n}-day high)", hi_n, "buy"))
        if action == "BUY":
            headline = f"Buy signal — price broke above its {n}-day high"
        if action == "SELL":
            headline = "Sell signal — price fell through the trailing stop"

    elif key == "value":
        _, mode = sig_value(df, p)
        if mode.startswith("dividend"):
            y = trailing_dividend_yield(df)
            hist = y.where(df.index >= df.index[0] + pd.Timedelta(days=365)).dropna()
            ttm_div = float(df["dividends"].rolling("365D").sum().iloc[-1])
            y_now = float(y.iloc[-1])
            y_buy = float(hist.quantile(p["value_buy_pct"] / 100)) if len(hist) else np.nan
            y_sell = float(hist.quantile(p["value_sell_pct"] / 100)) if len(hist) else np.nan
            readings += [("Dividend yield (12 months)", f"{y_now * 100:.2f}%"),
                         (f"'Cheap' yield ({p['value_buy_pct']}th pct)", f"{y_buy * 100:.2f}%"),
                         (f"'Expensive' yield ({p['value_sell_pct']}th pct)", f"{y_sell * 100:.2f}%")]
            if ttm_div > 0 and y_buy > 0:
                if now:
                    levels.append(_level("Sell at or above", last * y_now / y_sell, "sell",
                                         "yield would drop to the 'expensive' level"))
                else:
                    levels.append(_level("Buy at or below", last * y_now / y_buy, "buy",
                                         "yield would rise to the 'cheap' level"))
            fair_txt = "dividend yield versus its own history"
        else:
            fair_s = px.rolling(p["value_window"], min_periods=TRADING_DAYS).mean()
            fair = float(fair_s.iloc[-1])
            overlays = {"3-year average": fair_s}
            readings += [("3-year average price", f"{fair:,.2f}"),
                         ("Price vs average", f"{(last / fair - 1) * 100:+.1f}%")]
            if now:
                levels.append(_level("Sell at or above (back to average)", fair, "sell"))
            else:
                levels.append(_level(f"Buy at or below ({p['value_margin']}% under average)",
                                     fair * (1 - p["value_margin"] / 100), "buy"))
            fair_txt = "price versus its 3-year average"
        headline = ("Holding — still cheap enough to own" if now
                    else "Not cheap enough yet to buy")
        reason = f"Uses the {mode} ({fair_txt})."
        if action == "BUY":
            headline = "Buy signal — the stock just became 'cheap' versus its own history"
        if action == "SELL":
            headline = "Sell signal — the stock now looks 'expensive' versus its own history"

    elif key == "dip":
        _, st_ = _dip_run(df, p)
        lb = int(p["dip_lookback"])
        high = st_["high"]
        overlays = {"Recent high": px.rolling(lb, min_periods=1).max(),
                    f"Buy zone (−{p['dip_pct']}%)": px.rolling(lb, min_periods=1).max() * (1 - p["dip_pct"] / 100)}
        readings += [("Recent high", f"{high:,.2f}"),
                     ("Price vs recent high", f"{(last / high - 1) * 100:+.1f}%"),
                     ("Dip needed to buy", f"−{p['dip_pct']}%")]
        if now:
            headline = "Holding a dip you bought"
            reason = f"Bought at {st_['entry_px']:,.2f} after a {p['dip_pct']}% fall from {st_['ref_high']:,.2f}."
            if p["dip_exit"] == "recover":
                levels.append(_level("Sell at or above (back at old high)", st_["ref_high"], "sell"))
            elif p["dip_exit"] == "target":
                levels.append(_level(f"Sell at or above (+{p['dip_take_profit']}% target)",
                                     st_["entry_px"] * (1 + p["dip_take_profit"] / 100), "sell"))
            if p.get("dip_stop", 0) > 0:
                levels.append(_level(f"Stop-loss (−{p['dip_stop']}%)",
                                     st_["entry_px"] * (1 - p["dip_stop"] / 100), "stop"))
        else:
            trig = high * (1 - p["dip_pct"] / 100)
            if not st_["armed"]:
                headline = "Waiting — stopped out, needs a bounce before buying again"
                reason = (f"The last dip hit the stop-loss. It re-arms once the price climbs back above "
                          f"{trig:,.2f}.")
                levels.append(_level("Re-arms above", trig, "neutral"))
            else:
                headline = f"No dip yet — price is {abs(last / high - 1) * 100:.1f}% below its recent high"
                reason = f"The strategy buys once the price is {p['dip_pct']}% below its recent high."
                levels.append(_level(f"Buy at or below (−{p['dip_pct']}%)", trig, "buy"))
        if action == "BUY":
            headline = f"Buy signal — price is {p['dip_pct']}%+ below its recent high"
        if action == "SELL":
            headline = "Sell signal — exit rule reached"

    for lv in levels:
        lv["dist_pct"] = (lv["price"] / last - 1) * 100

    return {"key": key, "action": action, "headline": headline, "reason": reason,
            "levels": levels, "readings": readings, "overlays": overlays, "position": pos,
            "since_date": since_date, "since_px": since_px, "last": last,
            "date": df.index[-1]}


def signal_track_record(df: pd.DataFrame, pos: pd.Series, key: str) -> dict:
    """How did past buy signals work out, compared with buying on a random day?

    For each past buy signal, the trade is assumed to happen at the next day's close
    (same rule as the backtest), and the price change is measured 1, 3 and 6 months later.
    """
    px = df["adj_close"].to_numpy()
    n = len(px)
    if key in PASSIVE:
        entries = np.array([], dtype=int)
    else:
        p = pos.to_numpy()
        entries = np.where((p[1:] > 0) & (p[:-1] == 0))[0] + 1
    out = {"n_signals": int(len(entries)), "horizons": {}}
    for label, h in HORIZONS.items():
        base_r = px[h:] / px[:-h] - 1 if n > h else np.array([])
        buy_idx = entries + 1
        buy_idx = buy_idx[buy_idx + h < n]
        sig_r = px[buy_idx + h] / px[buy_idx] - 1 if len(buy_idx) else np.array([])
        out["horizons"][label] = {
            "n": int(len(sig_r)),
            "hit": float((sig_r > 0).mean()) if len(sig_r) else np.nan,
            "avg": float(sig_r.mean()) if len(sig_r) else np.nan,
            "base_hit": float((base_r > 0).mean()) if len(base_r) else np.nan,
            "base_avg": float(base_r.mean()) if len(base_r) else np.nan,
        }
    return out


# ---------------------------------------------------------------------------
# 8. YOUR POSITION
#    Applies the strategy's SELL rules to a real (or hypothetical) purchase.
#    Uses actual traded prices (not dividend-adjusted) so it matches your broker.
# ---------------------------------------------------------------------------

def position_check(df: pd.DataFrame, key: str, p: dict, buy_date, buy_price: float) -> dict:
    """Would this strategy sell a position bought on `buy_date` at `buy_price`?

    Returns sell_now (bool), why (text), exit levels, profit/loss, dividends since buying,
    and what the strategy itself was doing on the buy date.
    """
    buy_date = pd.Timestamp(buy_date).normalize()
    close, adj = df["close"], df["adj_close"]
    last = float(close.iloc[-1])
    since = close[close.index >= buy_date]
    divs = float(df["dividends"][df.index > buy_date].sum())
    levels: list[dict] = []
    sell_now, why = False, ""

    if key in PASSIVE:
        why = "A passive strategy never sells — it just holds."

    elif key == "trend":
        kind = p.get("ma_type", "SMA")
        nf, ns = int(p["sma_fast"]), int(p["sma_slow"])
        fast = float(moving_avg(adj, nf, kind).iloc[-1])
        slow = float(moving_avg(adj, ns, kind).iloc[-1])
        sell_now = fast <= slow
        why = (f"The {nf}-day {kind} ({fast:,.2f}) is {'below' if sell_now else 'above'} the "
               f"{ns}-day {kind} ({slow:,.2f}) — the trend rule says "
               f"{'get out' if sell_now else 'stay in'}.")
        if not sell_now:
            cross = _ma_cross_price(adj.to_numpy(), fast, slow, nf, ns, kind)
            if np.isfinite(cross) and 0 < cross < last:
                levels.append(_level("Sell if it closes below (averages cross down)", cross, "stop"))

    elif key == "mean_rev":
        n = int(p["rsi_period"])
        r = float(rsi(adj, n).iloc[-1])
        sell_now = r > p["rsi_sell"]
        why = (f"RSI is {r:.0f}. The rule sells once RSI rises above {p['rsi_sell']} "
               f"(overbought){' — it is there now' if sell_now else ''}. It has no stop-loss.")
        if not sell_now:
            levels.append(_level(f"Sell if it closes above (RSI > {p['rsi_sell']})",
                                 _rsi_trigger_price(adj, n, p["rsi_sell"], "up"), "target"))

    elif key == "momentum":
        peak = max(buy_price, float(since.max()) if len(since) else buy_price)
        stop_px = peak * (1 - p["trailing_stop"] / 100)
        sell_now = last < stop_px
        why = (f"Highest close since you bought: {peak:,.2f}. The trailing stop sells if the price "
               f"closes {p['trailing_stop']}% below that peak.")
        levels.append(_level(f"Trailing stop ({p['trailing_stop']}% below peak)", stop_px, "stop"))

    elif key == "value":
        _, mode = sig_value(df, p)
        if mode.startswith("dividend"):
            y = trailing_dividend_yield(df)
            hist = y.where(df.index >= df.index[0] + pd.Timedelta(days=365)).dropna()
            y_now = float(y.iloc[-1])
            y_sell = float(hist.quantile(p["value_sell_pct"] / 100)) if len(hist) else np.nan
            sell_now = bool(y_now <= y_sell)
            why = (f"Dividend yield is {y_now * 100:.2f}%. The rule sells when it falls to "
                   f"{y_sell * 100:.2f}% or lower (the stock looks expensive).")
            if not sell_now and y_sell > 0:
                levels.append(_level("Sell at or above (looks expensive)", last * y_now / y_sell, "target"))
        else:
            fair = float(adj.rolling(p["value_window"], min_periods=TRADING_DAYS).mean().iloc[-1])
            sell_now = last >= fair
            why = (f"The 3-year average price is {fair:,.2f}. The rule sells once the price is back "
                   "at that average.")
            if not sell_now:
                levels.append(_level("Sell at or above (back to average)", fair, "target"))
        why += " It has no stop-loss."

    elif key == "dip":
        lb = int(p["dip_lookback"])
        before = close[close.index <= buy_date].iloc[-lb:]
        ref_high = float(before.max()) if len(before) else buy_price
        mode = p["dip_exit"]
        if mode == "recover":
            sell_now = last >= ref_high
            levels.append(_level("Sell at or above (back at the old high)", ref_high, "target"))
            why = f"The high before your purchase was {ref_high:,.2f}. The rule sells when the price gets back there."
        elif mode == "target":
            tgt = buy_price * (1 + p["dip_take_profit"] / 100)
            sell_now = last >= tgt
            levels.append(_level(f"Sell at or above (+{p['dip_take_profit']}% target)", tgt, "target"))
            why = f"The rule sells once you're up {p['dip_take_profit']}%."
        else:
            why = "Your exit setting is 'never sell', so the rule just holds."
        if p.get("dip_stop", 0) > 0:
            stop_px = buy_price * (1 - p["dip_stop"] / 100)
            if last <= stop_px:
                sell_now = True
            levels.append(_level(f"Stop-loss ({p['dip_stop']}% below your price)", stop_px, "stop"))
            why += f" Stop-loss at −{p['dip_stop']}%."
        else:
            why += " No stop-loss is set."

    for lv in levels:
        lv["dist_pct"] = (lv["price"] / last - 1) * 100

    # what was the strategy doing when you bought?
    pos = strategy_position(df, key, p)
    on_buy = pos[pos.index <= buy_date]
    strat_in = bool(on_buy.iloc[-1]) if len(on_buy) else None
    starts = pos.index[pos.diff() > 0]
    prior = starts[starts <= buy_date]
    after = starts[starts > buy_date]

    return {
        "sell_now": bool(sell_now), "why": why, "levels": levels,
        "pl_pct": (last / buy_price - 1) * 100 if buy_price else np.nan,
        "pl_incl_div_pct": ((last + divs) / buy_price - 1) * 100 if buy_price else np.nan,
        "divs": divs, "days": int((df.index[-1] - buy_date).days), "last": last,
        "strat_in_on_buy": strat_in,
        "last_signal_before": prior[-1] if len(prior) else None,
        "next_signal_after": after[0] if len(after) else None,
        "in_range": buy_date >= df.index[0],
    }
