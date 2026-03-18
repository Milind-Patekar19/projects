# Swing Bot

Python 3 technical analysis bot for Indian stocks (NSE symbols via `yfinance`).

## Features
- Reads symbols from `watchlist.csv`/`.txt` and auto-appends `.NS`
- Pulls 60-day data for `5m` and `15m` intervals
- Computes RSI, MACD, EMA(9/21), volume ratio, candle patterns, support/resistance
- Generates score-based `BULLISH` / `BEARISH` / `NEUTRAL` verdicts
- Prints structured terminal analysis and stores daily output under `logs/YYYY-MM-DD.txt`
- Optional portfolio P&L summary from `portfolio.csv` (CMP, unrealised P&L, tax flag)

## Setup
```bash
pip install yfinance pandas
```

## Run
```bash
python3 swing_bot.py
```

## Watchlist format
One symbol per line (with or without exchange suffix):

```text
RELIANCE
TCS
INFY.NS
```

## Optional portfolio format
Create `portfolio.csv` with columns:

`SYMBOL,AVG_BUY_PRICE,QUANTITY,BUY_DATE`

Example:

```csv
SYMBOL,AVG_BUY_PRICE,QUANTITY,BUY_DATE
RELIANCE,2800,10,2025-04-10
TCS,3900,4,2025-11-01
```
