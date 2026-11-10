"""
app.py — the web page for the Investment Strategy Simulator.

Run on your own computer:   streamlit run app.py
All the maths lives in simulator.py; this file only draws the page.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import simulator as S

st.set_page_config(page_title="Strategy Simulator", page_icon="📈", layout="wide")

GRID = "rgba(128,128,128,0.18)"


# ---------------------------------------------------------------------------
# Cached data loaders (cache = don't re-download the same thing for 1 hour,
# which also keeps Yahoo from blocking us for asking too often)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    return S.load_prices_yahoo(ticker, start, end)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_fundamentals(ticker: str) -> dict:
    return S.load_fundamentals(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def get_ohlc(ticker: str, rng: str) -> pd.DataFrame:
    return S.load_ohlc(ticker, rng)


# ---------------------------------------------------------------------------
# Technical chart (runs as a "fragment": changing its buttons only redraws this
# chart instead of re-running the whole backtest)
# ---------------------------------------------------------------------------

TECH_INDICATORS = ["Volume", "SMA 20", "SMA 50", "SMA 200", "EMA 20", "Bollinger Bands",
                   "RSI", "MACD"]
TECH_HELP = (
    "- **Volume** — how many shares traded in each candle.\n"
    "- **SMA** — simple moving average of the last N candles (20 = short, 200 = long trend).\n"
    "- **EMA 20** — like SMA 20 but reacts faster to recent prices.\n"
    "- **Bollinger Bands** — 20-candle average ± 2 standard deviations; price near the top band "
    "is stretched high, near the bottom stretched low.\n"
    "- **RSI** — momentum from 0–100; above 70 often called overbought, below 30 oversold.\n"
    "- **MACD** — difference between a fast and slow average; lines crossing hint at a "
    "change in trend.")
UP, DOWN = "#1baf7a", "#e34948"
LINE_COLORS = {"SMA 20": "#2a78d6", "SMA 50": "#eb6834", "SMA 200": "#6250d6", "EMA 20": "#eda100"}


def panel_label(fig: go.Figure, row: int, text: str) -> None:
    """Small caption in the top-left corner of a lower chart panel."""
    yref = "y domain" if row == 1 else f"y{row} domain"
    fig.add_annotation(text=text, xref="paper", yref=yref, x=0, y=1, xanchor="left",
                       yanchor="top", showarrow=False, font=dict(size=11, color="gray"))


@st.fragment
def technical_chart(ticker: str, daily: pd.DataFrame, from_csv: bool, cur: str):
    from plotly.subplots import make_subplots

    c1, c2 = st.columns([3, 1])
    rng = c1.segmented_control("Range", list(S.CHART_RANGES), default="6M", key="tc_range")
    style = c2.segmented_control("Style", ["Candles", "Line"], default="Candles", key="tc_style")
    rng = rng or "6M"
    style = style or "Candles"
    inds = st.multiselect("Indicators", TECH_INDICATORS, default=["Volume", "SMA 50", "SMA 200"],
                          key="tc_inds", help=TECH_HELP)

    # ---- data
    if from_csv:
        if rng in S.INTRADAY:
            st.info("1D, 7D and 1M need live intraday data from Yahoo. With an uploaded CSV, "
                    "pick 3M or longer.")
            return
        full = S.ohlc_from_closes(daily)
        if style == "Candles":
            st.caption("Your CSV only has closing prices, so the chart is drawn as a line.")
            style = "Line"
    else:
        try:
            with st.spinner("Loading chart…"):
                full = get_ohlc(ticker, rng)
        except Exception as e:  # noqa: BLE001
            reason = _short_reason(e)
            st.warning(f"Couldn't load the {rng} chart ({reason}). Try another range, or wait "
                       "a few minutes if Yahoo is limiting requests.")
            return
    view = S.crop_to_range(full, rng)
    if len(view) < 2:
        st.info("Not enough candles in this range yet — the market may not have opened. "
                "Try 7D or longer.")
        return

    # ---- headline numbers (change is measured from the last close before the window)
    before = full["Close"][full.index < view.index[0]]
    base = float(before.iloc[-1]) if len(before) else float(view["Open"].iloc[0])
    last = float(view["Close"].iloc[-1])
    chg = (last / base - 1) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last price", f"{last:,.2f}", f"{chg:+.2f}% over {rng}")
    m2.metric(f"{rng} high", f"{view['High'].max():,.2f}")
    m3.metric(f"{rng} low", f"{view['Low'].min():,.2f}")
    vol = view["Volume"].sum()
    m4.metric(f"{rng} volume", f"{vol / 1e6:,.1f} M shares" if vol and not np.isnan(vol) else "–")

    # ---- indicators on the full download, then cropped to the window
    close = full["Close"]
    ind = {}
    for n in (20, 50, 200):
        if f"SMA {n}" in inds:
            ind[f"SMA {n}"] = S.sma(close, n)
    if "EMA 20" in inds:
        ind["EMA 20"] = S.ema(close, 20)
    show_vol = "Volume" in inds and view["Volume"].notna().any() and view["Volume"].sum() > 0
    panels = ["price"] + (["volume"] if show_vol else []) + \
             (["rsi"] if "RSI" in inds else []) + (["macd"] if "MACD" in inds else [])
    heights = {"price": 0.58, "volume": 0.14, "rsi": 0.16, "macd": 0.18}
    hs = [heights[p] for p in panels]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[h / sum(hs) for h in hs])

    intraday = rng in S.INTRADAY
    x = view.index.strftime("%d %b %H:%M") if intraday else view.index

    if style == "Candles":
        fig.add_trace(go.Candlestick(x=x, open=view["Open"], high=view["High"], low=view["Low"],
                                     close=view["Close"], name=ticker,
                                     increasing=dict(line=dict(color=UP), fillcolor=UP),
                                     decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN)),
                      row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=view["Close"], name=ticker, line=dict(color="#2a78d6", width=2),
                                 hovertemplate="%{y:,.2f}"), row=1, col=1)

    if "Bollinger Bands" in inds:
        up, mid, lo = (s.reindex(view.index) for s in S.bollinger(close))
        fig.add_trace(go.Scatter(x=x, y=up, line=dict(color="rgba(128,128,128,.6)", width=1),
                                 name="Bollinger Bands", legendgroup="bb",
                                 hovertemplate="upper %{y:,.2f}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=lo, line=dict(color="rgba(128,128,128,.6)", width=1),
                                 fill="tonexty", fillcolor="rgba(128,128,128,.10)",
                                 name="Bollinger lower", legendgroup="bb", showlegend=False,
                                 hovertemplate="%{y:,.2f}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=mid, line=dict(color="rgba(128,128,128,.8)", width=1, dash="dot"),
                                 name="Bollinger middle", legendgroup="bb", showlegend=False,
                                 hovertemplate="%{y:,.2f}"), row=1, col=1)
    for name, series in ind.items():
        fig.add_trace(go.Scatter(x=x, y=series.reindex(view.index), name=name,
                                 line=dict(color=LINE_COLORS[name], width=1.5),
                                 hovertemplate="%{y:,.2f}"), row=1, col=1)

    row = 2
    if show_vol:
        colors = np.where(view["Close"] >= view["Open"], UP, DOWN)
        fig.add_trace(go.Bar(x=x, y=view["Volume"], name="Volume", marker_color=colors,
                             opacity=0.6, showlegend=False, hovertemplate="%{y:,.0f}"),
                      row=row, col=1)
        fig.update_yaxes(title_text="Volume", row=row, col=1)
        row += 1
    if "RSI" in inds:
        r = S.rsi(close, 14).reindex(view.index)
        fig.add_trace(go.Scatter(x=x, y=r, name="RSI 14", line=dict(color="#6250d6", width=1.5),
                                 showlegend=False, hovertemplate="%{y:.0f}"), row=row, col=1)
        panel_label(fig, row, "RSI 14 — above 70 overbought, below 30 oversold")
        for lvl in (30, 70):
            fig.add_hline(y=lvl, line=dict(color="gray", width=1, dash="dash"), row=row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)
        row += 1
    if "MACD" in inds:
        line, sig, hist = (s.reindex(view.index) for s in S.macd(close))
        fig.add_trace(go.Bar(x=x, y=hist, name="MACD histogram", opacity=0.5,
                             marker_color=np.where(hist.fillna(0) >= 0, UP, DOWN),
                             showlegend=False, hovertemplate="%{y:.3f}"), row=row, col=1)
        fig.add_trace(go.Scatter(x=x, y=line, name="MACD", line=dict(color="#2a78d6", width=1.5),
                                 showlegend=False, hovertemplate="%{y:.3f}"), row=row, col=1)
        fig.add_trace(go.Scatter(x=x, y=sig, name="Signal", line=dict(color="#eb6834", width=1.5),
                                 showlegend=False, hovertemplate="%{y:.3f}"), row=row, col=1)
        panel_label(fig, row, "MACD (blue) vs signal (orange); bars = gap between them")
        fig.update_yaxes(title_text="MACD", row=row, col=1)

    fig.update_layout(height=420 + 130 * (len(panels) - 1), margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text=f"Price ({cur})", row=1, col=1)
    if intraday:
        fig.update_xaxes(type="category", nticks=8, showgrid=False)
    else:
        fig.update_xaxes(showgrid=False,
                         rangebreaks=[dict(bounds=["sat", "mon"])] if rng in ("3M", "6M", "1Y") else [])
    st.plotly_chart(fig, width="stretch", key="tc_chart")

    bar = S.CHART_RANGES[rng]["bar"]
    st.caption(f"Each candle = {bar} of trading. Indicator periods count candles "
               f"(e.g. SMA 50 = average of the last 50 candles). Prices are not dividend-adjusted."
               + (" Yahoo's intraday prices can be delayed ~15 minutes." if intraday else ""))


def default_benchmark(ticker: str) -> str:
    return "TDEX.BK" if ticker.upper().endswith(".BK") else "SPY"


# Backups tried in order if the chosen index can't be loaded
BENCH_FALLBACKS = {
    "thai": ["TDEX.BK", "^SET.BK", "^SET50.BK"],
    "us": ["SPY", "VOO", "^GSPC"],
}
BENCH_NAMES = {
    "TDEX.BK": "ThaiDEX SET50 ETF", "^SET.BK": "SET Index", "^SET50.BK": "SET50 Index",
    "SPY": "S&P 500 ETF", "VOO": "Vanguard S&P 500 ETF", "^GSPC": "S&P 500 Index",
}


def _short_reason(err: Exception) -> str:
    msg = str(err)
    if "Rate" in msg or "Too Many" in msg or "429" in msg:
        return "Yahoo is limiting requests"
    if "No price data" in msg or "delisted" in msg or "Not enough" in msg:
        return "no price history on Yahoo"
    return msg[:80] or type(err).__name__


def load_benchmark(symbol: str, stock: str, start: str, end: str):
    """Try the chosen index, then sensible backups. Retries once if Yahoo is busy.
    Returns (prices or None, symbol actually used, list of (symbol, reason) failures)."""
    import time
    market = "thai" if stock.endswith(".BK") else "us"
    candidates = [symbol] + [s for s in BENCH_FALLBACKS[market] if s != symbol]
    failures = []
    for sym in candidates:
        for attempt in range(2):
            try:
                return get_prices(sym, start, end), sym, failures
            except Exception as e:  # noqa: BLE001
                reason = _short_reason(e)
                if reason.startswith("Yahoo is limiting") and attempt == 0:
                    time.sleep(2)
                    continue
                failures.append((sym, reason))
                break
    return None, None, failures


# ---------------------------------------------------------------------------
# Strategy explanations (shown when you hover over a strategy name)
# ---------------------------------------------------------------------------

STRATEGY_GUIDE = {
    "buy_hold": ("Put all the money in on day one and never sell.",
                 "The simple baseline. If a clever strategy can't beat this after fees, "
                 "it isn't worth the effort."),
    "dca": ("Split the starting money into equal monthly purchases.",
            "Lowers the regret of buying everything right before a fall. In a rising market "
            "it usually trails Buy & Hold slightly because some cash waits on the sidelines."),
    "index": ("Same money, but in the whole market instead of one stock.",
              "Spreads risk over many companies. Tells you whether picking this stock was "
              "better than simply owning the market."),
    "trend": ("Own the stock only while its short-term average price is above its "
              "long-term average.",
              "Aims to ride long uptrends and step aside during long downtrends. Gets "
              "whipsawed (many small losses) in sideways markets."),
    "mean_rev": ("Buy after a sharp fall, sell after a sharp rise, using the RSI indicator.",
                 "Bets that prices snap back toward normal. Works in range-bound stocks; can "
                 "buy too early in a real crash."),
    "momentum": ("Buy when the price breaks above its recent high; sell after a set drop "
                 "from its peak.",
                 "Bets that strength keeps going. The trailing stop limits losses but can "
                 "sell on normal wobbles."),
    "value": ("Buy when the stock looks cheap versus its own history; sell when it looks "
              "expensive.",
              "For dividend payers 'cheap' means an unusually high dividend yield. For "
              "non-payers it means trading well below its 3-year average price."),
    "dip": ("Buy once the price has fallen a set % below its recent high.",
            "Bets that pullbacks in a good stock are buying chances. Works well in long "
            "uptrends; in a stock that keeps falling it buys too early, so consider a stop-loss."),
}


def strategy_settings(key: str, P: dict, bench: str) -> str:
    """One line describing the exact rule used with the current settings."""
    return {
        "buy_hold": "Buy on the first day, hold to the end.",
        "dca": f"Starting amount split into {P['dca_months']} equal monthly buys.",
        "index": f"Buy & hold {bench} with the same money.",
        "trend": f"Hold while the {P['sma_fast']}-day average is above the "
                 f"{P['sma_slow']}-day average; otherwise cash.",
        "mean_rev": f"Buy when {P['rsi_period']}-day RSI < {P['rsi_buy']}; "
                    f"sell when RSI > {P['rsi_sell']}.",
        "momentum": f"Buy on a new {P['breakout_days']}-day high; sell after a "
                    f"{P['trailing_stop']}% drop from the peak.",
        "value": f"Dividend payers: buy when yield ≥ its {P['value_buy_pct']}th percentile, "
                 f"sell when ≤ {P['value_sell_pct']}th. Non-payers: buy {P['value_margin']}% "
                 "below the 3-year average, sell back at the average.",
        "dip": dip_rule(P),
    }[key]


def dip_rule(P: dict) -> str:
    lookback = {21: "1-month", 63: "3-month", 126: "6-month", 252: "1-year"}.get(
        P["dip_lookback"], f"{P['dip_lookback']}-day")
    rule = f"Buy when the price is {P['dip_pct']}% below its {lookback} high; "
    rule += {"recover": "sell when it climbs back to that high.",
             "target": f"sell once the trade is up {P['dip_take_profit']}%.",
             "hold": "never sell after buying."}[P["dip_exit"]]
    if P.get("dip_stop", 0) > 0:
        rule += f" Stop-loss: sell if it falls {P['dip_stop']}% below the buy price."
    return rule


def strategy_tooltip(key: str, P: dict | None = None, bench: str = "") -> str:
    what, why = STRATEGY_GUIDE[key]
    text = f"{what} {why}"
    if P is not None:
        text += f" Rule used: {strategy_settings(key, P, bench)}"
    return text


SIDEBAR_HELP = "\n".join(
    f"- **{s.name}** — {STRATEGY_GUIDE[s.key][0]}" for s in S.STRATEGIES)


def strategy_chips(keys: list[str], P: dict, bench: str) -> str:
    """Coloured strategy labels that show an explanation box on hover."""
    from html import escape
    chips = []
    for k in keys:
        info = S.STRATEGY_BY_KEY[k]
        what, why = STRATEGY_GUIDE[k]
        chips.append(
            f'<span class="sg-chip" tabindex="0">'
            f'<span class="sg-dot" style="background:{info.color}"></span>{escape(info.name)}'
            f'<span class="sg-tip"><b>{escape(info.name)}</b><br>{escape(what)}'
            f'<br><span class="sg-why">{escape(why)}</span>'
            f'<br><span class="sg-rule">Rule used: {escape(strategy_settings(k, P, bench))}</span>'
            f'</span></span>')
    return f'<div class="sg-row">{"".join(chips)}</div>'


CHIP_CSS = """
<style>
.sg-row{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 6px}
.sg-chip{position:relative;display:inline-flex;align-items:center;gap:6px;padding:4px 10px;
  border:1px solid rgba(128,128,128,.35);border-radius:999px;font-size:.85rem;cursor:help;
  outline:none}
.sg-chip:hover,.sg-chip:focus{border-color:rgba(128,128,128,.8)}
.sg-dot{width:10px;height:10px;border-radius:50%;flex:none}
.sg-tip{visibility:hidden;opacity:0;transition:opacity .12s;position:absolute;z-index:1000;
  top:calc(100% + 8px);left:0;width:270px;max-width:calc(100vw - 48px);padding:10px 12px;border-radius:8px;
  background:#1f1f1e;color:#f0efec;font-size:.8rem;line-height:1.45;font-weight:400;
  box-shadow:0 6px 20px rgba(0,0,0,.25);pointer-events:none;white-space:normal}
.sg-chip:hover .sg-tip,.sg-chip:focus .sg-tip{visibility:visible;opacity:1}
.sg-why{color:#c3c2b7}
.sg-rule{display:inline-block;margin-top:4px;color:#9fc4f0}
</style>
"""


def money(x: float, cur: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{cur}{x:,.0f}"


def style_fig(fig: go.Figure, height: int = 420, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR — all the settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1 · Pick a stock")
    ticker = st.text_input("Ticker", value="PTT.BK",
                           help="Thai stocks: add .BK (e.g. CPALL.BK, AOT.BK). "
                                "US stocks/ETFs: plain symbol (AAPL, NVDA, SPY).").strip().upper()
    source = st.radio("Price data", ["Yahoo Finance (automatic)", "Upload my own CSV"],
                      help="If Yahoo is busy or blocks the request, download a CSV "
                           "and upload it here instead.")
    uploaded = None
    if source.startswith("Upload"):
        uploaded = st.file_uploader("CSV with Date and Close columns", type=["csv"])

    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=date.today() - timedelta(days=365 * 10),
                          min_value=date(1990, 1, 1), max_value=date.today())
    end = c2.date_input("To", value=date.today(), min_value=date(1990, 1, 1), max_value=date.today())

    st.header("2 · Your money")
    start_cash = st.number_input("Starting amount", min_value=0, value=100_000, step=10_000)
    monthly = st.number_input("Add every month", min_value=0, value=0, step=1_000,
                              help="Extra savings added on the first trading day of each month.")
    fee = st.number_input("Broker fee per trade (%)", min_value=0.0, max_value=2.0, value=0.15,
                          step=0.01, format="%.3f",
                          help="Set this to your broker's commission rate (incl. VAT if any).")
    rf = st.number_input("Risk-free rate (% per year)", min_value=0.0, max_value=15.0, value=2.0,
                         step=0.25, help="Roughly what a savings account / government bond pays. "
                                         "Used for the Sharpe ratio.")

    st.header("3 · Strategies to compare")
    names = {s.name: s.key for s in S.STRATEGIES}
    chosen_names = st.multiselect("Strategies", list(names), default=list(names),
                                  help="What each one does:\n\n" + SIDEBAR_HELP)
    chosen = [names[n] for n in chosen_names]
    bench_ticker = st.text_input("Index for 'Passive Index'", value=default_benchmark(ticker),
                                 help="TDEX.BK = SET50 ETF (Thai, dividends included). SPY = S&P 500 ETF. "
                                      "You can type any ETF or index symbol; if it fails, a backup is used automatically.").strip().upper()

    P = dict(S.DEFAULT_PARAMS)
    if "dip" in chosen:
        st.markdown("**Buy the Dip settings**")
        P["dip_pct"] = st.slider("Buy when the price drops this much (%)", 1, 60, P["dip_pct"],
                                 help="Measured from the highest price within the window below.")
        lookbacks = {"1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252}
        P["dip_lookback"] = lookbacks[st.select_slider(
            "…below its highest price of the last", list(lookbacks), value="1 year")]
        exits = {v: k for k, v in S.DIP_EXITS.items()}
        P["dip_exit"] = exits[st.selectbox("When to sell", list(exits))]
        if P["dip_exit"] == "target":
            P["dip_take_profit"] = st.slider("Profit target (%)", 2, 100, P["dip_take_profit"])
        P["dip_stop"] = st.slider("Stop-loss below buy price (%)", 0, 50, P["dip_stop"],
                                  help="0 = no stop-loss. Protects against a dip that turns "
                                       "into a long decline.")

    with st.expander("Strategy settings (optional)"):
        P["dca_months"] = st.slider("DCA: spread starting amount over (months)", 2, 36, P["dca_months"])
        st.caption("Trend following")
        P["sma_fast"] = st.slider("Fast moving average (days)", 5, 150, P["sma_fast"])
        P["sma_slow"] = st.slider("Slow moving average (days)", 20, 300, P["sma_slow"])
        st.caption("Mean reversion")
        P["rsi_period"] = st.slider("RSI period (days)", 5, 30, P["rsi_period"])
        P["rsi_buy"] = st.slider("Buy when RSI is below", 10, 45, P["rsi_buy"])
        P["rsi_sell"] = st.slider("Sell when RSI is above", 55, 90, P["rsi_sell"])
        st.caption("Momentum breakout")
        P["breakout_days"] = st.slider("Buy on a new high of the last (days)", 10, 252, P["breakout_days"])
        P["trailing_stop"] = st.slider("Sell if it drops from its peak by (%)", 3, 40, P["trailing_stop"])
        st.caption("Value investing")
        P["value_buy_pct"] = st.slider("Dividend payers: buy when yield is in the top (percentile)",
                                       50, 95, P["value_buy_pct"])
        P["value_sell_pct"] = st.slider("Dividend payers: sell when yield falls to (percentile)",
                                        5, 50, P["value_sell_pct"])
        P["value_margin"] = st.slider("Non-payers: buy when price is below its 3-year average by (%)",
                                      5, 40, P["value_margin"])

    run = st.button("Run simulation", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------------------------

st.title("📈 Investment Strategy Simulator")
st.caption("Test how different investing styles would have done on a stock, then see a range of "
           "possible futures. For learning and research only — past results don't guarantee "
           "future returns, and this is not financial advice.")

if "ready" not in st.session_state:
    st.session_state.ready = False
if run:
    st.session_state.ready = True
if not st.session_state.ready:
    st.info("👈 Choose a ticker and settings in the sidebar, then press **Run simulation**.")
    st.stop()

if P["sma_fast"] >= P["sma_slow"]:
    st.error("The fast moving average must be shorter than the slow one.")
    st.stop()
if start >= end:
    st.error("'From' date must be before 'To' date.")
    st.stop()

# ---- load data -----------------------------------------------------------
with st.spinner(f"Getting prices for {ticker}…"):
    try:
        if uploaded is not None:
            df = S.load_prices_csv(uploaded.getvalue())
            df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        elif source.startswith("Upload"):
            st.warning("Upload a CSV file in the sidebar first.")
            st.stop()
        else:
            df = get_prices(ticker, str(start), str(end + timedelta(days=1)))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "Rate" in msg or "Too Many" in msg or "429" in msg:
            st.error("Yahoo Finance is limiting requests right now. Wait a few minutes and try "
                     "again, or switch to **Upload my own CSV** in the sidebar.")
        else:
            st.error(f"Couldn't load prices: {msg}")
        st.stop()

if len(df) < 60:
    st.error("Less than 3 months of prices in this date range — pick an earlier 'From' date.")
    st.stop()

bench = None
if "index" in chosen and uploaded is None:
    bench, used, failed = load_benchmark(bench_ticker, ticker, str(df.index[0].date()),
                                         str(end + timedelta(days=1)))
    if bench is None:
        tried = "; ".join(f"{s} ({why})" for s, why in failed)
        st.warning(f"Passive Index is skipped — couldn't load any index. Tried: {tried}. "
                   "If Yahoo is limiting requests, wait a few minutes and press Run again, "
                   "or type a different ETF symbol in the sidebar.")
    else:
        if used != bench_ticker:
            st.info(f"Couldn't load **{bench_ticker}** ({failed[0][1]}), so Passive Index uses "
                    f"**{used}** ({BENCH_NAMES.get(used, 'backup index')}) instead.")
        bench_ticker = used
        if bench.index[0] > df.index[0] + pd.Timedelta(days=10):
            st.caption(f"Note: {used} prices only start on {bench.index[0]:%d %b %Y}, so Passive "
                       "Index stays flat until then.")
elif "index" in chosen:
    st.info("Passive Index needs online data, so it's skipped when you upload a CSV.")

cur = "฿" if ticker.endswith(".BK") else "$"

results = S.run_all(df, bench, chosen, start_cash, monthly, fee, P)
if not results:
    st.warning("Pick at least one strategy.")
    st.stop()
metrics = {k: S.compute_metrics(v, rf) for k, v in results.items()}
order = sorted(results, key=lambda k: metrics[k]["Final value"], reverse=True)

first, last = df.index[0].date(), df.index[-1].date()
st.subheader(f"{ticker}  ·  {first:%d %b %Y} → {last:%d %b %Y}")

tab_bt, tab_tc, tab_fc, tab_fund, tab_help = st.tabs(
    ["📊 Backtest", "📉 Technical chart", "🔮 Forecast", "🔎 Fundamentals", "❓ How to read this"])

with tab_tc:
    technical_chart(ticker, df, uploaded is not None, cur)

# ===========================================================================
# BACKTEST TAB
# ===========================================================================
with tab_bt:
    best, worst = order[0], order[-1]
    bh = metrics.get("buy_hold")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best strategy", S.STRATEGY_BY_KEY[best].name,
              f"{metrics[best]['Annual return % (CAGR)']:.1f}% a year",
              help=strategy_tooltip(best, P, bench_ticker))
    k2.metric("Its final value", money(metrics[best]["Final value"], cur),
              f"{money(metrics[best]['Profit'], cur)} profit")
    if bh:
        k3.metric("Buy & Hold for comparison", money(bh["Final value"], cur),
                  f"{bh['Annual return % (CAGR)']:.1f}% a year")
    safest = max(results, key=lambda k: metrics[k]["Max drawdown %"])
    k4.metric("Smallest worst-drop", S.STRATEGY_BY_KEY[safest].name,
              f"{metrics[safest]['Max drawdown %']:.1f}%", delta_color="off",
              help=strategy_tooltip(safest, P, bench_ticker))

    # --- growth of money chart
    st.markdown("#### How the money grew")
    st.markdown(CHIP_CSS, unsafe_allow_html=True)
    st.caption("Hover over (or tap) a strategy to see what it does.")
    st.markdown(strategy_chips(order, P, bench_ticker), unsafe_allow_html=True)
    fig = go.Figure()
    for k in order:
        info = S.STRATEGY_BY_KEY[k]
        fig.add_trace(go.Scatter(x=results[k].equity.index, y=results[k].equity, name=info.name,
                                 line=dict(color=info.color, width=2),
                                 hovertemplate=f"{cur}%{{y:,.0f}}"))
    fig.add_trace(go.Scatter(x=results[order[0]].contributed.index, y=results[order[0]].contributed,
                             name="Money put in", line=dict(color="gray", width=1.5, dash="dot"),
                             hovertemplate=f"{cur}%{{y:,.0f}}"))
    st.plotly_chart(style_fig(fig, 460, f"Portfolio value ({cur})"), width="stretch")

    # --- scorecard table
    st.markdown("#### Scorecard")
    st.caption("Hover over a column title (like **Strategy** or **Sharpe ratio**) for an explanation.")
    table = pd.DataFrame([metrics[k] for k in order])
    table.insert(0, "Strategy", [S.STRATEGY_BY_KEY[k].name for k in order])
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Strategy": st.column_config.TextColumn(
                help="\n".join(f"- **{S.STRATEGY_BY_KEY[k].name}** — {strategy_settings(k, P, bench_ticker)}"
                               for k in order)),
            "Final value": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
            "Money put in": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
            "Profit": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
            "Total return %": st.column_config.NumberColumn(format="%.1f%%"),
            "Annual return % (CAGR)": st.column_config.NumberColumn(
                format="%.2f%%", help="Steady yearly growth rate that gives the same end result."),
            "Volatility %": st.column_config.NumberColumn(
                format="%.1f%%", help="How bumpy the ride was."),
            "Max drawdown %": st.column_config.NumberColumn(
                format="%.1f%%", help="Worst fall from a previous peak."),
            "Sharpe ratio": st.column_config.NumberColumn(
                format="%.2f", help="Return per unit of risk, above the risk-free rate. Above 1 is good."),
            "Time invested %": st.column_config.NumberColumn(
                format="%.0f%%", help="How often it held the stock instead of cash."),
            "Trades": st.column_config.NumberColumn(format="%d", help="Completed buy→sell round trips."),
            "Win rate %": st.column_config.NumberColumn(
                format="%.0f%%", help="Share of completed trades that made money."),
        },
    )
    if "value" in results:
        st.caption(f"Value Investing used the **{results['value'].note}** for this stock "
                   "(see *How to read this*).")
    if "index" in results:
        st.caption(f"Passive Index = buying and holding **{bench_ticker}** with the same money.")

    # --- drawdown chart
    st.markdown("#### Worst drops along the way (drawdown)")
    st.caption("How far each strategy was below its own previous peak. Closer to 0 = smoother ride.")
    fig = go.Figure()
    for k in order:
        info = S.STRATEGY_BY_KEY[k]
        dd = S.drawdown_series(results[k])
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=info.name,
                                 line=dict(color=info.color, width=1.5),
                                 hovertemplate="%{y:.1f}%"))
    st.plotly_chart(style_fig(fig, 320, "% below peak"), width="stretch")

    # --- trades for one strategy
    st.markdown("#### When each strategy bought and sold")
    active = [k for k in order if k not in ("buy_hold", "dca", "index")]
    if active:
        pick = st.selectbox("Strategy", active, format_func=lambda k: S.STRATEGY_BY_KEY[k].name,
                            help="What each one does:\n\n" + "\n".join(
                                f"- **{S.STRATEGY_BY_KEY[k].name}** — {STRATEGY_GUIDE[k][0]}"
                                for k in active))
        st.caption(f"**Rule used:** {strategy_settings(pick, P, bench_ticker)}")
        r = results[pick]
        px = df["adj_close"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=px.index, y=px, name=f"{ticker} price",
                                 line=dict(color="gray", width=1.5), hovertemplate="%{y:,.2f}"))
        held = px.where(r.position > 0)
        fig.add_trace(go.Scatter(x=held.index, y=held, name="Invested",
                                 line=dict(color=S.STRATEGY_BY_KEY[pick].color, width=2.5),
                                 hovertemplate="%{y:,.2f}"))
        change = r.position.diff().fillna(r.position)
        buys, sells = px[change > 0], px[change < 0]
        fig.add_trace(go.Scatter(x=buys.index, y=buys, mode="markers", name="Buy",
                                 marker=dict(symbol="triangle-up", size=11, color="#1baf7a",
                                             line=dict(width=1, color="white"))))
        fig.add_trace(go.Scatter(x=sells.index, y=sells, mode="markers", name="Sell",
                                 marker=dict(symbol="triangle-down", size=11, color="#e34948",
                                             line=dict(width=1, color="white"))))
        st.plotly_chart(style_fig(fig, 380, "Price (dividend-adjusted)"), width="stretch")
        if len(r.trades):
            st.dataframe(r.trades, width="stretch", hide_index=True,
                         column_config={"Buy price": st.column_config.NumberColumn(format="%.2f"),
                                        "Sell price": st.column_config.NumberColumn(format="%.2f"),
                                        "Return %": st.column_config.NumberColumn(format="%.1f%%")})
        else:
            st.caption("This strategy never triggered a trade in this period.")

# ===========================================================================
# FORECAST TAB
# ===========================================================================
with tab_fc:
    st.markdown("#### What could happen next if you buy and hold this stock?")
    st.caption("The **shaded fan** comes from 2,000+ simulated futures built by randomly re-playing "
               "real daily moves from the stock's recent history. The **dashed lines** are your own "
               "what-if scenarios at a fixed yearly growth rate.")

    f1, f2, f3, f4 = st.columns(4)
    fc_amount = f1.number_input("Invest today", min_value=0, value=int(start_cash or 100_000),
                                step=10_000, key="fc_amt")
    fc_monthly = f2.number_input("Add every month", min_value=0, value=int(monthly),
                                 step=1_000, key="fc_mon")
    fc_years = f3.slider("Years ahead", 1, 20, 5)
    lookback = f4.slider("Learn from the last (years)", 1, 20, 5,
                         help="Which part of history the simulation copies its daily moves from.")

    g1, g2 = st.columns([1, 3])
    override = g1.toggle("Override the growth rate", value=False,
                         help="Keep the stock's real ups and downs, but set the typical yearly "
                              "growth yourself. Useful because a stock's past run-up often "
                              "doesn't repeat.")
    drift = g2.slider("Typical yearly growth (%)", -20.0, 40.0, 7.0, 0.5,
                      disabled=not override)

    st.markdown("**Your scenarios** (yearly growth %)")
    s1, s2, s3, s4 = st.columns(4)
    bear = s1.number_input("🐻 Bear", value=-3.0, step=1.0)
    base = s2.number_input("📊 Base", value=7.0, step=1.0)
    bull = s3.number_input("🐂 Bull", value=15.0, step=1.0)
    n_paths = s4.select_slider("Simulations", [500, 1000, 2000, 5000], value=2000)

    mc = S.monte_carlo(df["adj_close"], fc_amount, fc_monthly, fc_years, n_paths=n_paths,
                       lookback_years=lookback, drift_override_pct=drift if override else None)
    xs = [df.index[-1] + pd.offsets.BDay(int(d)) for d in mc["days"]]

    fig = go.Figure()
    fan = "42,120,214"
    fig.add_trace(go.Scatter(x=xs, y=mc["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=mc["p5"], fill="tonexty", fillcolor=f"rgba({fan},0.12)",
                             line=dict(width=0), name="5–95% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=mc["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=mc["p25"], fill="tonexty", fillcolor=f"rgba({fan},0.25)",
                             line=dict(width=0), name="25–75% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=mc["p50"], name="Median (middle outcome)",
                             line=dict(color=f"rgb({fan})", width=2.5),
                             hovertemplate=f"{cur}%{{y:,.0f}}"))
    for label, rate, color in [("Bear", bear, "#e34948"), ("Base", base, "#6250d6"),
                               ("Bull", bull, "#1baf7a")]:
        fig.add_trace(go.Scatter(x=xs, y=S.scenario_path(fc_amount, fc_monthly, fc_years, rate),
                                 name=f"{label} {rate:+.0f}%/yr",
                                 line=dict(color=color, width=1.5, dash="dash"),
                                 hovertemplate=f"{cur}%{{y:,.0f}}"))
    st.plotly_chart(style_fig(fig, 460, f"Value ({cur})"), width="stretch")

    put_in = mc["put_in"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Middle outcome", money(mc["p50"][-1], cur),
              f"{(mc['p50'][-1] / put_in - 1) * 100:+.0f}% vs money put in")
    m2.metric("Bad case (1 in 20)", money(mc["p5"][-1], cur),
              f"{(mc['p5'][-1] / put_in - 1) * 100:+.0f}%")
    m3.metric("Good case (1 in 20)", money(mc["p95"][-1], cur),
              f"{(mc['p95'][-1] / put_in - 1) * 100:+.0f}%")
    m4.metric("Chance of losing money", f"{mc['prob_loss'] * 100:.0f}%",
              f"{mc['prob_double'] * 100:.0f}% chance to double", delta_color="off")

    st.caption(f"Money put in over {fc_years} years: {money(put_in, cur)}. "
               f"The history used grew about {mc['hist_annual_pct']:.1f}% a year "
               f"(from {mc['years_used']:.1f} years of data, dividends included)"
               + (f"; you overrode the typical growth to {drift:.1f}%." if override else "."))
    if not override and mc["hist_annual_pct"] > 20:
        st.warning("This stock grew very fast in the period used. Simulations assume that "
                   "continues, which rarely happens — try **Override the growth rate** for a "
                   "more conservative view.")

    with st.expander("Outcome table"):
        yrs = list(range(1, fc_years + 1))
        idxs = [min(int(y * S.TRADING_DAYS), len(mc["p50"]) - 1) for y in yrs]
        st.dataframe(pd.DataFrame({
            "Year": yrs,
            "Bad (5%)": [mc["p5"][i] for i in idxs],
            "Below avg (25%)": [mc["p25"][i] for i in idxs],
            "Middle (50%)": [mc["p50"][i] for i in idxs],
            "Above avg (75%)": [mc["p75"][i] for i in idxs],
            "Good (95%)": [mc["p95"][i] for i in idxs],
        }).set_index("Year").style.format(f"{cur}{{:,.0f}}"), width="stretch")

# ===========================================================================
# FUNDAMENTALS TAB
# ===========================================================================
with tab_fund:
    st.markdown("#### Today's valuation snapshot")
    st.caption("Free data only gives *current* figures, so these are a reference point for the "
               "Value strategy rather than part of the backtest.")
    if uploaded is not None:
        st.info("Fundamentals need online data — switch to Yahoo Finance to see them.")
    else:
        try:
            fnd = get_fundamentals(ticker)
        except Exception:  # noqa: BLE001
            fnd = {}
        if not any(v is not None for v in fnd.values()):
            st.info("Yahoo didn't return fundamentals for this ticker right now. Try again later.")
        else:
            def fmt(k, v):
                if v is None:
                    return "–"
                if k == "Market cap":
                    return f"{v / 1e9:,.1f} B"
                if k in ("Return on equity", "Profit margin"):
                    return f"{v * 100:.1f}%"
                if isinstance(v, (int, float)):
                    return f"{v:,.2f}"
                return str(v)
            items = [(k, fmt(k, v)) for k, v in fnd.items()]
            ttm_now = S.trailing_dividend_yield(df).iloc[-1]
            items.append(("Dividend yield (12 mo)", f"{ttm_now * 100:.2f}%"))
            cols = st.columns(4)
            for i, (k, v) in enumerate(items):
                cols[i % 4].metric(k, v)

    ttm = S.trailing_dividend_yield(df).dropna()
    if len(ttm) and (df["dividends"] > 0).any():
        st.markdown("#### Dividend yield history (what the Value strategy watches)")
        fig = go.Figure(go.Scatter(x=ttm.index, y=ttm * 100, name="Trailing 12-month yield",
                                   line=dict(color="#6250d6", width=2), hovertemplate="%{y:.2f}%"))
        st.plotly_chart(style_fig(fig, 300, "Yield %"), width="stretch")

# ===========================================================================
# HELP TAB
# ===========================================================================
with tab_help:
    st.markdown("""
**The strategies**

| Strategy | In plain words |
|---|---|
| Buy & Hold | Put all the money in on day one, never sell. The baseline everything else must beat. |
| Dollar-Cost Averaging | Split the starting money into equal monthly buys (default 12 months). Lowers regret if you buy right before a drop. |
| Passive Index | Same money, but in the whole market (SET index or S&P 500 ETF) instead of one stock. |
| Trend Following | Own the stock only while its 50-day average price is above its 200-day average; otherwise sit in cash. |
| Mean Reversion | Buy when the RSI says it's fallen too far too fast (below 30); sell when it's risen too far (above 70). |
| Momentum Breakout | Buy when the price beats its highest level of the last 50 days; sell if it then drops 15% from its peak. |
| Buy the Dip | Buy once the price falls your chosen % below its recent high (e.g. 10% below the 1-year high). Then sell when it recovers to that high, at a profit target, or never — your choice. An optional stop-loss limits losses if the dip keeps going. |
| Value Investing | For dividend payers: buy when the dividend yield is high versus the stock's own history (price is cheap for the cash it pays) and sell when it's low. For non-payers: buy when the price is 15% below its 3-year average, sell when it recovers to the average. |

**The scorecard**

- **Annual return (CAGR)** – the steady yearly growth rate that would give the same result. Not affected by your monthly top-ups.
- **Max drawdown** – the worst fall from a peak. −40% means at one point you were 40% below your best value. Ask yourself whether you'd really have held on.
- **Volatility** – how bumpy the ride was. Higher = bigger daily swings.
- **Sharpe ratio** – return earned per unit of bumpiness, above the risk-free rate. Above 1 is good; below 0.5 is weak.
- **Time invested** – how often the strategy was in the stock rather than cash.
- **Win rate** – share of completed buy→sell trades that made money.

**Built-in honesty rules**

- Every signal is acted on the **next** trading day, so no strategy can "see the future".
- Your broker fee is charged on every buy and sell.
- Prices are dividend-adjusted, so dividends count as if reinvested.
- Cash waiting on the sidelines earns nothing (conservative).

**Limits to keep in mind**

- A strategy that won in the past can lose in the future — especially if you tweak the settings until it looks great on one stock ("overfitting"). Check that it also works on other stocks and other periods.
- Taxes, slippage and dividend withholding tax aren't included.
- The forecast is a range of possibilities, not a prediction.
""")
