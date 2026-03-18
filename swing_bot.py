"""Swing trading technical analysis bot for NSE/BSE symbols using yfinance.

Reads symbols from watchlist files, fetches OHLCV data for 5m/15m intervals,
computes indicators, detects patterns, scores bullish/bearish bias, and prints
terminal summaries. Optionally writes the full output to logs/YYYY-MM-DD.txt.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf


RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SUP_RES_LOOKBACK = 30
VOLUME_LOOKBACK = 20
DEFAULT_PERIOD = "60d"
DEFAULT_INTERVALS = ("5m", "15m")


@dataclass
class AnalysisResult:
    symbol: str
    interval: str
    last_close: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    ema_fast: float
    ema_slow: float
    volume_ratio: float
    candle_pattern: str
    support: float
    resistance: float
    bullish_score: int
    bearish_score: int
    verdict: str


def _normalize_symbol(symbol: str) -> str:
    clean = str(symbol).upper().replace(" ", "")
    if "." not in clean:
        clean = f"{clean}.NS"
    return clean


def load_watchlist(filepath: str | Path) -> List[str]:
    """Parse .csv or .txt watchlist and normalize symbols for NSE data."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")

    symbols: List[str] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    token = cell.strip()
                    if token:
                        symbols.append(token)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if token:
                    symbols.append(token)

    normalized: List[str] = []
    seen = set()
    for symbol in symbols:
        clean = _normalize_symbol(symbol)
        if clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    return normalized


def load_portfolio(filepath: str | Path) -> List[dict]:
    """Load portfolio rows from CSV and return normalized positions."""
    path = Path(filepath)
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    required = {"SYMBOL", "AVG_BUY_PRICE", "QUANTITY"}
    if not required.issubset(set(df.columns)):
        return []

    positions: List[dict] = []
    for _, row in df.iterrows():
        quantity = float(row.get("QUANTITY", 0) or 0)
        if quantity == 0:
            continue

        raw_date = row.get("BUY_DATE")
        buy_date = None
        if pd.notna(raw_date) and str(raw_date).strip():
            try:
                buy_date = datetime.strptime(str(raw_date).strip(), "%Y-%m-%d").date()
            except ValueError:
                buy_date = None

        positions.append(
            {
                "symbol": _normalize_symbol(row.get("SYMBOL", "")),
                "avg_buy_price": float(row.get("AVG_BUY_PRICE", 0)),
                "quantity": quantity,
                "buy_date": buy_date,
            }
        )
    return positions


def compute_pl(avg_buy_price: float, quantity: float, cmp: float) -> dict:
    invested = avg_buy_price * quantity
    current_value = cmp * quantity
    pl_abs = current_value - invested
    pl_pct = (pl_abs / invested) * 100 if invested else 0.0
    return {
        "invested": round(invested, 2),
        "current_value": round(current_value, 2),
        "pl_abs": round(pl_abs, 2),
        "pl_pct": round(pl_pct, 2),
    }


def tax_flag(buy_date: Optional[date]) -> str:
    if buy_date is None:
        return "Date unavailable"

    days_held = (date.today() - buy_date).days
    if days_held >= 365:
        return f"LTCG — held {days_held} days (10% tax on gains above ₹1L)"
    if 358 <= days_held < 365:
        days_left = 365 - days_held
        return f"⚠️ LTCG threshold in {days_left} days — consider waiting (10% tax after)"
    return f"STCG — held {days_held} days (< 1 year, 15% tax)"


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = compute_ema(series, fast) - compute_ema(series, slow)
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _body_size(row: pd.Series) -> float:
    return abs(float(row["Close"]) - float(row["Open"]))


def _range_size(row: pd.Series) -> float:
    return max(float(row["High"]) - float(row["Low"]), 1e-9)


def detect_candle_pattern(df: pd.DataFrame) -> str:
    """Detect common patterns from the latest 3 candles."""
    if len(df) < 3:
        return "NONE"

    c1, c2, c3 = (df.iloc[-3], df.iloc[-2], df.iloc[-1])

    def is_bullish(candle: pd.Series) -> bool:
        return float(candle["Close"]) > float(candle["Open"])

    def is_bearish(candle: pd.Series) -> bool:
        return float(candle["Close"]) < float(candle["Open"])

    # Single-candle patterns on latest candle
    body = _body_size(c3)
    candle_range = _range_size(c3)
    upper_wick = float(c3["High"]) - max(float(c3["Open"]), float(c3["Close"]))
    lower_wick = min(float(c3["Open"]), float(c3["Close"])) - float(c3["Low"])

    if body / candle_range <= 0.1:
        return "DOJI"
    if lower_wick > body * 2 and upper_wick <= body:
        return "HAMMER"
    if upper_wick > body * 2 and lower_wick <= body:
        return "SHOOTING_STAR"

    # Engulfing patterns on last two candles
    if is_bearish(c2) and is_bullish(c3):
        if float(c3["Open"]) <= float(c2["Close"]) and float(c3["Close"]) >= float(c2["Open"]):
            return "BULLISH_ENGULFING"
    if is_bullish(c2) and is_bearish(c3):
        if float(c3["Open"]) >= float(c2["Close"]) and float(c3["Close"]) <= float(c2["Open"]):
            return "BEARISH_ENGULFING"

    # Morning/Evening star (simple approximation)
    c2_body_small = _body_size(c2) <= _range_size(c2) * 0.35
    if is_bearish(c1) and c2_body_small and is_bullish(c3):
        mid_c1 = (float(c1["Open"]) + float(c1["Close"])) / 2
        if float(c3["Close"]) > mid_c1:
            return "MORNING_STAR"
    if is_bullish(c1) and c2_body_small and is_bearish(c3):
        mid_c1 = (float(c1["Open"]) + float(c1["Close"])) / 2
        if float(c3["Close"]) < mid_c1:
            return "EVENING_STAR"

    return "NONE"


def get_support_resistance(df: pd.DataFrame, lookback: int = SUP_RES_LOOKBACK) -> Tuple[float, float]:
    sample = df.tail(lookback)
    return float(sample["Low"].min()), float(sample["High"].max())


def generate_verdict(
    rsi: float,
    ema_fast: float,
    ema_slow: float,
    macd_hist: float,
    volume_ratio: float,
    candle_pattern: str,
) -> Tuple[int, int, str]:
    bull = 0
    bear = 0

    # RSI: zone-based up to 2 points
    if rsi < 35:
        bull += 2
    elif rsi > 65:
        bear += 2
    elif rsi < 45:
        bull += 1
    elif rsi > 55:
        bear += 1

    # EMA crossover up to 2 points
    if ema_fast > ema_slow:
        bull += 2
    elif ema_fast < ema_slow:
        bear += 2

    # MACD histogram (1 point)
    if macd_hist > 0:
        bull += 1
    elif macd_hist < 0:
        bear += 1

    # Candle pattern (1 point)
    bullish_patterns = {"HAMMER", "BULLISH_ENGULFING", "MORNING_STAR"}
    bearish_patterns = {"SHOOTING_STAR", "BEARISH_ENGULFING", "EVENING_STAR"}
    if candle_pattern in bullish_patterns:
        bull += 1
    elif candle_pattern in bearish_patterns:
        bear += 1

    # Volume spike (1 amplification point toward leader)
    if volume_ratio >= 1.5:
        if bull > bear:
            bull += 1
        elif bear > bull:
            bear += 1

    if bull - bear > 1:
        verdict = "BULLISH"
    elif bear - bull > 1:
        verdict = "BEARISH"
    else:
        verdict = "NEUTRAL"

    return bull, bear, verdict


def fetch_ohlcv(symbol: str, interval: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return multi-indexed columns in some versions
        df.columns = [col[0] for col in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    return df[required].dropna()


def analyze(symbol: str, interval: str) -> Optional[AnalysisResult]:
    df = fetch_ohlcv(symbol, interval)
    if len(df) < max(MACD_SLOW, RSI_PERIOD, SUP_RES_LOOKBACK):
        return None

    close = df["Close"]
    volume = df["Volume"]

    rsi_series = compute_rsi(close, RSI_PERIOD)
    ema_fast_series = compute_ema(close, EMA_FAST)
    ema_slow_series = compute_ema(close, EMA_SLOW)
    macd_line, signal_line, hist = compute_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    vol_avg = volume.rolling(VOLUME_LOOKBACK).mean()
    vol_ratio = float(volume.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] else 1.0

    candle_pattern = detect_candle_pattern(df)
    support, resistance = get_support_resistance(df, SUP_RES_LOOKBACK)

    bull, bear, verdict = generate_verdict(
        rsi=float(rsi_series.iloc[-1]),
        ema_fast=float(ema_fast_series.iloc[-1]),
        ema_slow=float(ema_slow_series.iloc[-1]),
        macd_hist=float(hist.iloc[-1]),
        volume_ratio=vol_ratio,
        candle_pattern=candle_pattern,
    )

    return AnalysisResult(
        symbol=symbol,
        interval=interval,
        last_close=float(close.iloc[-1]),
        rsi=float(rsi_series.iloc[-1]),
        macd=float(macd_line.iloc[-1]),
        macd_signal=float(signal_line.iloc[-1]),
        macd_hist=float(hist.iloc[-1]),
        ema_fast=float(ema_fast_series.iloc[-1]),
        ema_slow=float(ema_slow_series.iloc[-1]),
        volume_ratio=vol_ratio,
        candle_pattern=candle_pattern,
        support=support,
        resistance=resistance,
        bullish_score=bull,
        bearish_score=bear,
        verdict=verdict,
    )


def _render_analysis(result: AnalysisResult) -> str:
    return (
        f"[{result.symbol} | {result.interval}]\n"
        f"  Close              : {result.last_close:.2f}\n"
        f"  RSI(14)            : {result.rsi:.2f}\n"
        f"  EMA9 / EMA21       : {result.ema_fast:.2f} / {result.ema_slow:.2f}\n"
        f"  MACD / Signal / H  : {result.macd:.4f} / {result.macd_signal:.4f} / {result.macd_hist:.4f}\n"
        f"  Volume ratio (20)  : {result.volume_ratio:.2f}x\n"
        f"  Candle pattern     : {result.candle_pattern}\n"
        f"  Support / Resistance: {result.support:.2f} / {result.resistance:.2f}\n"
        f"  Scores (BULL/BEAR) : {result.bullish_score}/{result.bearish_score}\n"
        f"  Verdict            : {result.verdict}\n"
    )


def print_analysis(symbol: str, results: Iterable[Optional[AnalysisResult]]) -> str:
    lines = [f"\n{'=' * 72}\nANALYSIS: {symbol}\n{'=' * 72}"]
    for result in results:
        if result is None:
            lines.append("[WARN] Insufficient data for analysis.")
        else:
            lines.append(_render_analysis(result))
    output = "\n".join(lines)
    print(output)
    return output


def print_portfolio_summary(positions: List[dict]) -> str:
    if not positions:
        return ""

    today_text = date.today().strftime("%d %b %Y")
    lines = [f"\n{'=' * 72}\nPORTFOLIO SUMMARY — {today_text}\n{'=' * 72}"]

    total_invested = 0.0
    total_current = 0.0

    for position in positions:
        symbol = position["symbol"]
        avg_buy = float(position["avg_buy_price"])
        quantity = float(position["quantity"])
        buy_date = position["buy_date"]

        try:
            df = yf.download(symbol, period="1d", interval="1d", auto_adjust=False, progress=False)
            if df.empty:
                raise ValueError("empty price data")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            cmp = float(df["Close"].iloc[-1])
        except Exception:
            lines.append(f"{symbol}: Price unavailable")
            continue

        pl = compute_pl(avg_buy, quantity, cmp)
        indicator = "🟢" if pl["pl_abs"] >= 0 else "🔴"
        tax_note = tax_flag(buy_date)

        total_invested += pl["invested"]
        total_current += pl["current_value"]

        lines.extend(
            [
                f"Symbol: {symbol}",
                f"  Avg Buy: ₹{avg_buy:.2f} | CMP: ₹{cmp:.2f} | Qty: {quantity:.2f}",
                f"  Invested: ₹{pl['invested']:.2f}",
                f"  Unrealised P&L: {indicator} ₹{pl['pl_abs']:.2f} ({pl['pl_pct']:.2f}%)",
                f"  Tax Flag: {tax_note}",
            ]
        )

    net_pl_abs = total_current - total_invested
    net_pl_pct = (net_pl_abs / total_invested) * 100 if total_invested else 0.0
    net_indicator = "🟢" if net_pl_abs >= 0 else "🔴"

    lines.extend(
        [
            "-" * 72,
            f"Total Invested: ₹{total_invested:.2f}",
            f"Current Value: ₹{total_current:.2f}",
            f"Net Unrealised P&L: {net_indicator} ₹{net_pl_abs:.2f} ({net_pl_pct:.2f}%)",
        ]
    )

    output = "\n".join(lines)
    print(output)
    return output


def save_daily_log(contents: str, logs_dir: str | Path = "logs") -> Path:
    path = Path(logs_dir)
    path.mkdir(parents=True, exist_ok=True)
    logfile = path / f"{date.today().isoformat()}.txt"
    logfile.write_text(contents, encoding="utf-8")
    return logfile


def main(watchlist_file: str = "watchlist.csv", write_log: bool = True) -> None:
    symbols = load_watchlist(watchlist_file)
    positions = load_portfolio("portfolio.csv")
    if not symbols:
        print("No symbols found in watchlist.")
        return

    full_output: List[str] = []
    for symbol in symbols:
        interval_results: List[Optional[AnalysisResult]] = []
        for interval in DEFAULT_INTERVALS:
            try:
                interval_results.append(analyze(symbol, interval))
            except Exception as exc:  # network/data variability
                print(f"[ERROR] {symbol} ({interval}): {exc}")
                interval_results.append(None)

        block = print_analysis(symbol, interval_results)
        full_output.append(block)

    portfolio_block = print_portfolio_summary(positions)
    if portfolio_block:
        full_output.append(portfolio_block)

    if write_log and full_output:
        logfile = save_daily_log("\n".join(full_output))
        print(f"\nSaved log: {logfile}")


if __name__ == "__main__":
    main()
