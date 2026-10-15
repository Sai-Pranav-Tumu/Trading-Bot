# Master Blueprint: Indian Stock Trading Bot (`stock_bot_india`)
**Target Agent:** Gemini CLI / Autonomous Coding Assistant  
**Target Directory:** `C:\Users\saipr\Music\stock bot`  
**Market Focus:** India (NSE & BSE)  
**Timezone:** `Asia/Kolkata` (IST - Market Hours: 09:15 to 15:30)

---

## Agent Execution Rules (CRITICAL FOR GEMINI CLI)
You are acting as the lead software architect and quantitative developer for a completely new Python algorithmic trading bot for the Indian stock market. 

1. **Part-by-Part Execution:** This project is divided into **6 sequential parts**. You MUST execute **only one part at a time**.
2. **Stop & Confirm:** After completing the file creation, code implementation, and verification for a single part, **STOP** and ask the user: *"Part X is complete and verified. Shall I proceed to Part X+1?"* Do not jump ahead.
3. **Proactive Clarification:** Before starting Part 1, or at the beginning of any subsequent Part, if any user requirement, API credential, or risk parameter is ambiguous, **ask the user your doubts before writing the code**.
4. **Paper Trading Default:** Ensure the bot runs in `PAPER_TRADING = True` mode by default so no real money is risked until explicitly toggled by the user.

---

## Pre-Flight Questions (Ask the User Before Starting Part 1)
Before generating any files, check with the user on the following points (or use the stated defaults if the user tells you to proceed with defaults):
1. **Broker API Preference:** Which Indian broker API do you want to integrate for live/paper execution? (Options: *Zerodha Kite Connect, AngelOne SmartAPI, Upstox, Fyers, or Dhan*. **Default:** Paper Trading Simulator + Modular Broker Adapter with `yfinance` / `nselib` for data).
2. **Trading Timeframe:** Is this bot meant for **Intraday** (5-minute / 15-minute candles, square-off by 15:15 IST) or **Swing/Positional** (Daily candles)? (**Default:** Configurable, defaulting to 15-minute Intraday on NIFTY 50 stocks).
3. **Starting Virtual Capital:** What starting capital in INR should the simulator use? (**Default:** $\text{₹}1,00,000$).

---

## Mathematical & Algorithmic Core Specification

### 1. The "Fear & Pullback" Variable ($F_t$)
To protect capital when a stock or the broader market is dropping rapidly, the bot must compute a real-time **Composite Fear Score ($F_t \in [0.0, 1.0]$)** before every cycle.

$$F_t = w_1 \cdot \text{VIX}_{\text{norm}} + w_2 \cdot \text{Velocity}_{\text{drop}} + w_3 \cdot \text{ATR}_{\text{spike}} + w_4 \cdot \text{Drawdown}_{\text{factor}}$$

Where:
* **$\text{VIX}_{\text{norm}}$ (Macro Market Fear - Weight $w_1 = 0.25$):** Normalized value of **India VIX** (`^INDIAVIX`). If India VIX $> 20$ or spikes $> 8\%$ intraday, fear increases.
* **$\text{Velocity}_{\text{drop}}$ (Stock Drop Speed - Weight $w_2 = 0.35$):** Rate of negative price change over the last $N$ candles relative to volume:
  $$\text{Velocity}_{\text{drop}} = \min\left(1.0, \max\left(0.0, -\frac{P_t - P_{t-3}}{3 \cdot \text{ATR}_{14}}\right)\right)$$
* **$\text{ATR}_{\text{spike}}$ (Volatility Expansion - Weight $w_3 = 0.20$):** Ratio of current candle True Range to the 14-period Average True Range ($\text{ATR}_{14}$).
* **$\text{Drawdown}_{\text{factor}}$ (Portfolio Stress - Weight $w_4 = 0.20$):** Current open position unrealized loss or daily portfolio drawdown approaching the daily max loss limit.

#### Dynamic Pullback Action Matrix based on $F_t$:
* **$0.00 \le F_t < 0.40$ (Normal Regime):** Standard position sizing ($100\%$), standard trailing stop-loss ($2.0 \times \text{ATR}_{14}$).
* **$0.40 \le F_t < 0.65$ (Elevated Caution):** Reduce new position sizes by $50\%$. Tighten active trailing stop-losses to $1.2 \times \text{ATR}_{14}$.
* **$0.65 \le F_t < 0.80$ (High Fear - Pullback Mode):** **Halt all new BUY entries.** Scale out (sell $50\%$) of existing profitable or weakening long positions immediately. Tighten remaining stop-loss to $0.75 \times \text{ATR}_{14}$.
* **$F_t \ge 0.80$ (Extreme Panic / Circuit Breaker):** **Emergency Pullback.** Square off $100\%$ of open long positions at market price and enter a cooldown lock for $K$ candles.

---

### 2. Core Trading Strategies to Implement
1. **Strategy A: VWAP + EMA Trend Pullback (Intraday Momentum)**
   * **Entry Condition:** Price $> \text{VWAP}$, $\text{EMA}_{9} > \text{EMA}_{21}$, $\text{RSI}_{14} \in [45, 65]$, Volume $> 1.2 \times \text{SMA}(\text{Volume}, 20)$, and Fear Index $F_t < 0.65$.
2. **Strategy B: Mean-Reversion Oversold Bounce (Controlled Dip Buying)**
   * **Entry Condition:** Price touches lower Bollinger Band ($20, 2\sigma$), $\text{RSI}_{14} < 30$, bullish reversal candle (Hammer / Bullish Engulfing), **ONLY IF** $\text{Velocity}_{\text{drop}}$ is decelerating (second derivative of price is positive, avoiding "catching a falling knife") and $F_t < 0.60$.

---

## Target Project Architecture
```text
C:\Users\saipr\Music\stock bot\
├── .env.example
├── requirements.txt
├── config.py
├── main.py
├── data/
│   ├── __init__.py
│   ├── market_feed.py       # NSE/BSE & India VIX fetcher (yfinance / broker API)
│   └── database.py          # SQLite logger for trades, signals, and fear metrics
├── indicators/
│   ├── __init__.py
│   ├── technicals.py        # EMA, RSI, ATR, VWAP, Bollinger Bands
│   └── fear_engine.py       # Composite Fear Score (F_t) & Pullback logic
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── vwap_momentum.py
│   └── mean_reversion.py
├── risk_and_execution/
│   ├── __init__.py
│   ├── risk_manager.py      # Position sizing, stop-loss, & fear-based scaling
│   └── broker_adapter.py    # Paper trading engine + Indian Broker API template
└── backtest/
    ├── __init__.py
    └── backtester.py        # Historical simulation & performance metrics
```

---

## Step-by-Step Execution Plan

### PART 1: Project Scaffolding, Environment & Configuration
**Objective:** Initialize the directory structure, virtual environment dependencies, and Indian market configuration constants.
* **Tasks for Gemini CLI:**
  1. Create all folders and `__init__.py` files shown in the architecture tree inside `C:\Users\saipr\Music\stock bot`.
  2. Create `requirements.txt` with `pandas`, `numpy`, `yfinance`, `ta` (or `pandas-ta`), `pytz`, `python-dotenv`, `rich` (for terminal UI), and `pytest`.
  3. Create `.env.example` for broker API keys (`BROKER_API_KEY`, `BROKER_API_SECRET`, `PAPER_TRADING=True`).
  4. Create `config.py` containing:
     * Watchlist of liquid NSE tickers (e.g., `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ICICIBANK.NS`).
     * Benchmark and Fear tickers (`^NSEI` for NIFTY 50, `^INDIAVIX` for India VIX).
     * Indian market hours (`09:15` to `15:30` IST, auto-square-off at `15:15` IST).
     * Default Fear Engine weights ($w_1=0.25, w_2=0.35, w_3=0.20, w_4=0.20$) and thresholds.
* **Verification:** Run a quick syntax check on `config.py`.
* **STOP:** Ask the user for confirmation before moving to Part 2.

---

### PART 2: Indian Market Data Feed & Local SQLite Database
**Objective:** Build a reliable data pipeline for NSE stocks and India VIX, plus local storage for trade auditing.
* **Tasks for Gemini CLI:**
  1. Implement `data/market_feed.py`:
     * Fetch historical and live/delayed OHLCV data for `.NS` (NSE) symbols and `^INDIAVIX`.
     * Handle timezone conversion strictly to `Asia/Kolkata`.
     * Include data cleaning (handling missing bars or market holidays).
  2. Implement `data/database.py`:
     * Create an SQLite database (`trading_bot.db`) with tables for `orders`, `positions`, `fear_logs`, and `daily_pnl`.
* **Verification:** Write and run a quick test fetching the latest 5 days of 15-minute data for `RELIANCE.NS` and `^INDIAVIX`.
* **STOP:** Ask the user for confirmation before moving to Part 3.

---

### PART 3: Technical Indicators & The "Fear & Pullback" Engine
**Objective:** Implement the mathematical indicators and the core `FearEngine` class.
* **Tasks for Gemini CLI:**
  1. Implement `indicators/technicals.py`:
     * Calculate $\text{EMA}_9$, $\text{EMA}_{21}$, $\text{RSI}_{14}$, $\text{ATR}_{14}$, $\text{VWAP}$, and Bollinger Bands ($20, 2\sigma$).
  2. Implement `indicators/fear_engine.py`:
     * Build the `FearEngine` class that takes stock OHLCV, `^INDIAVIX` data, and current portfolio drawdown.
     * Compute $\text{VIX}_{\text{norm}}$, $\text{Velocity}_{\text{drop}}$ (price drop speed during sell-offs), $\text{ATR}_{\text{spike}}$, and $\text{Drawdown}_{\text{factor}}$.
     * Return a structured `FearState` object containing `fear_score` ($F_t \in [0, 1]$), `regime` (`NORMAL`, `CAUTION`, `HIGH_FEAR_PULLBACK`, `EXTREME_PANIC`), `position_size_multiplier`, and `stop_loss_atr_multiplier`.
* **Verification:** Run a unit test simulating a sharp $3.5\%$ stock drop over 3 candles and verify that `FearEngine` triggers `HIGH_FEAR_PULLBACK` or `EXTREME_PANIC`.
* **STOP:** Ask the user for confirmation before moving to Part 4.

---

### PART 4: Strategy Engine (Momentum & Dip-Buying with Knife-Catch Protection)
**Objective:** Code the entry/exit signal generators that integrate with the Fear Engine.
* **Tasks for Gemini CLI:**
  1. Implement `strategies/base_strategy.py` defining standard signal outputs (`BUY`, `SELL`, `HOLD`, `PARTIAL_PULLBACK`, `EMERGENCY_EXIT`).
  2. Implement `strategies/vwap_momentum.py` (Strategy A) and `strategies/mean_reversion.py` (Strategy B).
  3. Add a **"Falling Knife Filter"** inside the strategies: even if a stock looks oversold on RSI, block `BUY` signals if $\text{Velocity}_{\text{drop}} > 0.65$ or if the current candle closes near its absolute low with expanding red volume.
* **Verification:** Test strategy signal generation against sample DataFrame slices.
* **STOP:** Ask the user for confirmation before moving to Part 5.

---

### PART 5: Risk Management, Dynamic Pullback & Broker Execution Engine
**Objective:** Manage capital, Indian brokerage charges/taxes (STT, GST, SEBI charges, Stamp Duty), and dynamic stop-loss tightening when a stock starts dropping.
* **Tasks for Gemini CLI:**
  1. Implement `risk_and_execution/risk_manager.py`:
     * Calculate position quantity based on risk-per-trade (default $1\%$ of capital) scaled by `FearState.position_size_multiplier`.
     * Dynamically ratchet trailing stop-losses tighter whenever $F_t$ rises while in an open trade.
     * Enforce a daily max loss circuit breaker (e.g., $-2.5\%$ of total capital).
  2. Implement `risk_and_execution/broker_adapter.py`:
     * Build a full `PaperBroker` that simulates realistic slippage ($0.05\%$) and exact Indian statutory charges (STT, NSE transaction charges, GST, Stamp Duty).
     * Provide clean wrapper stubs for Zerodha Kite / AngelOne SmartAPI so live trading can be plugged in via `.env`.
* **Verification:** Simulate a buy order, a price drop that spikes $F_t$, an automatic $50\%$ pullback sell, and a trailing stop exit.
* **STOP:** Ask the user for confirmation before moving to Part 6.

---

### PART 6: Backtester, Live CLI Dashboard & Main Event Loop
**Objective:** Tie all modules together into an interactive CLI application with backtesting and live paper-trading modes.
* **Tasks for Gemini CLI:**
  1. Implement `backtest/backtester.py`:
     * Run historical simulations across the NSE watchlist.
     * Output key metrics: Total Return ($\%$), Win Rate ($\%$), Max Drawdown ($\%$), Sharpe Ratio, and ** capital saved by the Fear Pullback Engine**.
  2. Implement `main.py`:
     * Provide CLI arguments: `python main.py --mode backtest` and `python main.py --mode live`.
     * Use the `rich` library to display a clean live terminal dashboard showing Watchlist prices, India VIX, each stock's live **Fear Score ($F_t$) bar**, Open Positions, and Net PnL.
* **Verification:** Run a full end-to-end backtest on `RELIANCE.NS` and `TCS.NS` and print the performance summary table in the terminal.
* **Final Handover:** Present the completed project usage guide to the user.