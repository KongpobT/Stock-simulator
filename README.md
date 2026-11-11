# 📈 Investment Strategy Simulator

Type a stock ticker → see how 8 investing strategies would have performed on it → see a range of possible futures.

Works with **Thai SET stocks** (add `.BK`, e.g. `PTT.BK`, `CPALL.BK`) and **US stocks/ETFs** (`AAPL`, `NVDA`, `SPY`).

> For learning and research only. Past performance does not guarantee future results. Not financial advice.

---

## What's in this folder

| File | What it does | Do I need to edit it? |
|---|---|---|
| `app.py` | The web page (buttons, charts, tables) | No |
| `simulator.py` | The maths (strategies, backtest, forecast) | No — only if you want to add strategies |
| `requirements.txt` | The list of Python libraries the app needs | No |
| `README.md` | This guide | No |

---

## Part 1 — Put the files on GitHub (≈5 minutes, no coding)

You don't need to install anything for this part. Everything happens in your web browser.

1. **Create a GitHub account** at [github.com](https://github.com) if you don't have one.
2. Click the **+** button (top-right) → **New repository**.
3. Fill in:
   - **Repository name:** `stock-simulator` (any name is fine, no spaces)
   - Choose **Public** (Streamlit's free plan allows unlimited public apps; a private repo also works but you only get one).
   - Leave everything else as it is → click **Create repository**.
4. On the next page, click the link **"uploading an existing file"**.
5. **Drag all 4 files** (`app.py`, `simulator.py`, `requirements.txt`, `README.md`) into the box.
   ⚠️ Drag the files themselves, not a folder or a zip file — they must sit at the top level of the repository.
6. Scroll down and click **Commit changes**.

✅ You should now see your 4 files listed on the repository page.

---

## Part 2 — Turn it into a live website with a link (≈5 minutes)

We use **Streamlit Community Cloud**, which is free and reads your app straight from GitHub.

1. Go to [share.streamlit.io](https://share.streamlit.io) → **Continue with GitHub** → sign in and click **Authorize streamlit**.
2. Click **Create app** (top-right).
3. When it asks *"Do you already have an app?"* → choose **"Yup, I have an app"**.
4. Fill in:
   - **Repository:** `your-github-name/stock-simulator`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** pick a short name, e.g. `fu-stock-sim` → your link becomes `fu-stock-sim.streamlit.app`
5. Click **Advanced settings** → set **Python version** to **3.12** → **Save**.
6. Click **Deploy**.

The first start takes 2–5 minutes while it installs the libraries. After that, open your link and press **Run simulation**. 🎉

> **Updating later:** edit or re-upload a file on GitHub and commit. The website updates itself within a minute.
>
> **App went to sleep?** Free apps pause after a few days with no visitors. Just open the link and click **"Yes, get this app back up"**.

---

## How to use the app

1. **Sidebar → Pick a stock.** Type the ticker. Thai stocks need `.BK` at the end.
2. **Your money.** Starting amount, optional monthly top-up, your broker's fee %.
3. **Strategies.** All 8 are selected by default. With **Buy the Dip** selected, its settings appear right below the list: the drop % that triggers a buy, which recent high it's measured from, when to sell, and an optional stop-loss. Hover the ⓘ next to *Strategies* for a one-line summary of each, and open *Strategy settings* to tweak them.
4. Press **Run simulation**.
5. Look at the tabs:
   - **🚦 Today's signal** — pick one of your strategies and see what it says at the latest price: **BUY**, **HOLD**, **SELL** or **STAY OUT**, the exact prices that would trigger its next buy or sell, and how often its past buy signals on this stock beat buying on a random day. Under **Your position**, say whether you already own the stock: if not, you get a buy-now answer and where the rule would sell if you bought today; if you do, enter your buy date, price and shares to get KEEP or SELL for *your* purchase, your profit and dividends, and your exit prices. A table at the bottom shows all your strategies at once. Press **🔄 Latest price** to re-download prices.
   - **📊 Backtest** — who won, how the money grew, the scorecard, the worst drops, and every buy/sell. Hover over (or tap) any coloured strategy label above the chart to see what it does and the exact rule used with your settings.
   - **📉 Technical chart** — candlestick (or line) chart for 1D, 7D, 1M, 3M, 6M, 1Y, 5Y or Max. Open **⚙️ Indicators** to switch on and adjust: three moving averages (SMA or EMA, any period — default EMA 15/45/100), Bollinger Bands (period, width), RSI (period, overbought/oversold lines), MACD (fast/slow/signal), Stochastic (%K, smoothing, %D, levels) and volume with an average line.
   - **🔮 Forecast** — a fan of possible futures plus your own bear / base / bull lines.
   - **🔎 Fundamentals** — today's P/E, P/B, ROE, dividend yield.
   - **❓ How to read this** — plain-English explanations of every number.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| *"Yahoo Finance is limiting requests right now"* | Yahoo sometimes blocks shared cloud servers. Wait 10–30 minutes and try again — or use Option A or B below. |
| *"No price data found"* | Check the ticker spelling. Thai stocks must end in `.BK`. Try the same symbol on finance.yahoo.com. |
| App shows a red error box when starting | On share.streamlit.io, click your app → **Manage app** (bottom-right) to see the log. Most often a file was uploaded into a sub-folder instead of the top level. |
| Passive Index is missing or uses a different symbol | The app tries your index first, then backups automatically (Thai: `TDEX.BK` → `^SET.BK` → `^SET50.BK`; US: `SPY` → `VOO` → `^GSPC`). If all fail, the warning lists why — usually Yahoo limiting requests, so wait a few minutes and press Run again. |

**Option A — Upload your own price file.** In the sidebar choose **Upload my own CSV**. Any CSV with a `Date` column and a `Close` (or `Price`) column works — for example a historical-data export from Investing.com or your broker. If it has an `Adj Close` column, dividends will be included automatically.

**Option B — Run it on your own computer** (Yahoo almost never blocks home internet):

1. Install Python 3.12 from [python.org](https://www.python.org/downloads/) — on Windows, tick **"Add Python to PATH"** during install.
2. Put the 4 files in one folder, e.g. `Documents/stock-simulator`.
3. Open a terminal *in that folder*:
   - **Mac:** right-click the folder in Finder → *Services* → *New Terminal at Folder*
   - **Windows:** open the folder, click the address bar, type `cmd`, press Enter
4. Copy-paste these two lines, pressing Enter after each:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. Your browser opens the app automatically at `http://localhost:8501`. To stop it, press `Ctrl + C` in the terminal.

---

## How the simulator stays honest

- **No peeking into the future:** a signal from today's closing price is only traded at the next day's close.
- **Fees** are charged on every buy and sell.
- **Dividends** are included (prices are dividend-adjusted, as if dividends were reinvested).
- **Idle cash** earns 0% — slightly pessimistic for strategies that sit in cash.
- **Not included:** taxes, slippage on very illiquid stocks, dividend withholding tax.

**About "Value Investing":** free data sources don't keep years of historical P/E or book value, so a true fundamental backtest isn't possible for free. Instead the strategy uses information that *was* available at the time:
- For dividend payers → it buys when the 12-month dividend yield is unusually high versus the stock's own past (cheap for the cash it pays) and sells when unusually low.
- For non-payers → it buys when the price is a margin of safety below its 3-year average and sells when it recovers.

**About the forecast:** it re-shuffles real daily moves from recent history thousands of times. If the stock had a huge run-up, the forecast assumes that continues — use **Override the growth rate** for a more realistic view. Treat the fan as a range of possibilities, not a prediction.

**Beware of over-tuning:** if you adjust settings until one strategy looks amazing on one stock, it probably won't repeat. Test the same settings on several stocks and periods.
