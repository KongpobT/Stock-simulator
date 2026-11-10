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


def default_benchmark(ticker: str) -> str:
    return "^SET.BK" if ticker.upper().endswith(".BK") else "SPY"


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
    chosen_names = st.multiselect("Strategies", list(names), default=list(names))
    chosen = [names[n] for n in chosen_names]
    bench_ticker = st.text_input("Index for 'Passive Index'", value=default_benchmark(ticker),
                                 help="^SET.BK = Thai SET index (price only, excludes dividends). "
                                      "SPY = S&P 500 ETF. You can type any ETF symbol.").strip().upper()

    with st.expander("Strategy settings (optional)"):
        P = dict(S.DEFAULT_PARAMS)
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
    try:
        bench = get_prices(bench_ticker, str(df.index[0].date()), str(end + timedelta(days=1)))
    except Exception:  # noqa: BLE001
        st.warning(f"Couldn't load the index '{bench_ticker}', so Passive Index is skipped.")
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

tab_bt, tab_fc, tab_fund, tab_help = st.tabs(
    ["📊 Backtest", "🔮 Forecast", "🔎 Fundamentals", "❓ How to read this"])

# ===========================================================================
# BACKTEST TAB
# ===========================================================================
with tab_bt:
    best, worst = order[0], order[-1]
    bh = metrics.get("buy_hold")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best strategy", S.STRATEGY_BY_KEY[best].name,
              f"{metrics[best]['Annual return % (CAGR)']:.1f}% a year")
    k2.metric("Its final value", money(metrics[best]["Final value"], cur),
              f"{money(metrics[best]['Profit'], cur)} profit")
    if bh:
        k3.metric("Buy & Hold for comparison", money(bh["Final value"], cur),
                  f"{bh['Annual return % (CAGR)']:.1f}% a year")
    safest = max(results, key=lambda k: metrics[k]["Max drawdown %"])
    k4.metric("Smallest worst-drop", S.STRATEGY_BY_KEY[safest].name,
              f"{metrics[safest]['Max drawdown %']:.1f}%", delta_color="off")

    # --- growth of money chart
    st.markdown("#### How the money grew")
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
    table = pd.DataFrame({S.STRATEGY_BY_KEY[k].name: metrics[k] for k in order}).T
    st.dataframe(
        table, width="stretch",
        column_config={
            "Final value": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
            "Money put in": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
            "Profit": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
            "Total return %": st.column_config.NumberColumn(format="%.1f%%"),
            "Annual return % (CAGR)": st.column_config.NumberColumn(format="%.2f%%"),
            "Volatility %": st.column_config.NumberColumn(format="%.1f%%"),
            "Max drawdown %": st.column_config.NumberColumn(format="%.1f%%"),
            "Sharpe ratio": st.column_config.NumberColumn(format="%.2f"),
            "Time invested %": st.column_config.NumberColumn(format="%.0f%%"),
            "Trades": st.column_config.NumberColumn(format="%d"),
            "Win rate %": st.column_config.NumberColumn(format="%.0f%%"),
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
        pick = st.selectbox("Strategy", active, format_func=lambda k: S.STRATEGY_BY_KEY[k].name)
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
