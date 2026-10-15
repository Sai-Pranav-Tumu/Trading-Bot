# Project Blueprint: Indian Market Algorithmic Paper-Trading Bot (NSE/BSE)

## Role & Execution Protocol for Gemini CLI
You are an expert Quantitative Developer and Python Architect operating inside `C:\Users\saipr\Music\stock bot`. 
Your task is to build a production-grade **Paper Trading Bot** for the Indian Stock Market (NSE) with a built-in **Graduation Gatekeeper** and a dynamic **Fear & Pullback Index**. The system must run completely free using paper trading and historical/live data feeds (`yfinance` / `nselib`), logging every virtual trade into a local SQLite database. Only when the bot achieves specific profit, win-rate, and drawdown thresholds will it unlock the Broker Integration interface.

**CRITICAL EXECUTION RULE FOR GEMINI CLI:**
Do **NOT** generate all files at once. Execute this project strictly **Part by Part** (Parts 1 through 6). After completing the code and file creation for a Part:
1. Run syntax and verification tests on the files created.
2. Summarize what was built.
3. Explicitly ask the user: *"Part X is complete and verified. Shall I proceed to Part X+1?"* and wait for confirmation.

---

## Pre-Flight Configuration (Ask the User Before Starting Part 1)
Before writing any code, ask the user to confirm or customize these 4 baseline parameters (provide the defaults below so they can simply reply "use defaults"):
1. **Initial Virtual Capital:** Default `₹1,00,000`
2. **Graduation Profit Target:** Default `+15%` net return (`₹1,15,000` portfolio value after simulated Indian brokerage/STT taxes) over a minimum of `30` paper trades.
3. **Maximum Allowed Drawdown (Kill Switch):** Default `5%` (`₹95,000` hard floor).
4. **Target Stock Universe:** Default Nifty 50 liquid leaders (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ICICIBANK.NS`, `SBIN.NS`, `TATAMOTORS.NS`, `LT.NS`).

---

## Target Directory Structure
```text
C:\Users\saipr\Music\stock bot\
│
├── config.py                  # Capital limits, Graduation Gatekeeper rules, ticker list
├── requirements.txt           # yfinance, pandas, numpy, ta, sqlite3, rich
├── main.py                    # Master CLI controller & live paper-trading loop
│
├── data\
│   ├── market_feed.py         # Fetches NSE OHLCV data + India VIX (^INDIAVIX) + Nifty 50 (^NSEI)
│   └── paper_ledger.db        # Auto-generated SQLite DB for virtual wallet, orders, and PnL
│
├── strategies\
│   ├── technical_signals.py   # EMA Crossover, RSI, Volume Breakout, VWAP calculations
│   └── fear_pullback.py       # Dynamic Fear & Pullback Engine (0-100 score + position sizing)
│
├── engine\
│   ├── paper_broker.py        # Simulates order execution, slippage (0.05%), and Indian taxes (STT/GST/SEBI)
│   └── gatekeeper.py          # Evaluates performance against Graduation Limits to unlock live broker API
│
└── brokers\
    └── broker_adapter.py      # Abstract base class ready for Zerodha Kite / AngelOne SmartAPI / Dhan