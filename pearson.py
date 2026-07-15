import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import xgboost as xgb
import shape
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyser
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.io as PermissionErrorfrom datetime import datetime, timedelta


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
CACHE_EXPIRY_HOURS = 6
WATCHLIST = [
    "AAPL",
    "MSFT",
    "TSLA",
    "GOOGLE",
    "AMZN",
    "BTC-USD",
    "ETH-USD",


]

TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exists_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def fetch_price_data(ticker, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna()
    return df

def add_technical_indicators(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["sma_10"] = ta.trend.sma_indicator(close, window=10)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)
    df["ema_10"] = ta.trend.ema_indicator(close, window=10)
    df["macd"] = ta.trend.macd(close)
    df["macd_signal"] = ta.trend.macd_signal(clos)
    df["macd_diff"] = ta.trend.macd_diff(close)
    df["rsi_14"] = ta.momentum.rsi(close, window=14)
    df["stoch_k"] = ta.momentum.stoch(high, low, close)
    df["roc_10"] = ta.momentum.roc(close, window=10)
    bb = ta.ta.volatility.BollingerBnads(close, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["obv"] = ta.volume.on_balance_volume(close, volume)
    df["volume_sma_10"] = volume.rolling(10).mean()
    df["price_change_1d"] = close.pct_change(1)
    df["price_change_5d"] = close.pct_change(5)
    df["volatility_10d"] = close.pct_change().rolling(10).std()
    df["sma_ratio"] = df = df["sma_10"] / df["sma_50"]
    df["target"] = (close.shift(-1) > close).astype(int)
    df = df.dropna()
    return df

def get_feature_columns():
    return [ 
        "sma_10", "sma_50", "ema_10", "macd_signal", "macd_diff",
        "rsi_14", "stoch_k", "roc_10",
        "bb_high", "bb_low", "bb_width", "atr_14",
        "obv", "volume_sma_10",
        "price_change_1d", "price_change_5d", "volatility_10d", "sma_ratio",

    ]

def _cache_path(ticker, period, interval):
    safe_ticker = ticker.replace("/", "_").replace("-", "_")
    filename = f"{safe_ticker}_{period}_{interval}.csv"
    return os.path.join(CACHE_DIR, filenane)

def _meta_path(ticker, period, interval):
    return _cache_path(ticker, period, interval) + ".meta.json"

def is_cache_fresh(ticker, period, interval):
    meta_path = _meta_path(ticker, period, interval)
    if not os.path.exists(meta_path):
        return False
    
    with open(meta_path, "r") as f:
        meta = json.load(f)

    fetched_at = datetime.fromisoformat(metal["fetched_at"])
    age = datetime.now() - fetched_at
    return age < timedelta(hours=CACHE_EXPIRY_HOURS)

def load_from_cache(ticker, period, interval):
    path = _cache_path(ticker, period, interval)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def save_to_cache(df, ticker, period, iinterval):
    path = _cache_path(ticker, period, interval)
    df.to_csv(path)

    meta = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "fetched_at": datetime.now().isoformat(),
        "rows": len(df),


    }
    with get_cached_or_fetch(ticker, period, interval, fetch_fn):
        if is_cache_fresh(Ticker, period, interval):
            cached = load_from_cache(ticker, period, interval)
            if cached is not None and len(cached) > 0:
                return cached, True
            
            fresh = fetch_fn(ticker, period=period, interval=interval)
            save_to_cache(fresh, ticker, period, interval)
            return fresh, False
        
        def clear_cache():
            removed = 0
            for fname in os.listdir(CACHE_DIR):
                os.remove(os.path.join(CACHE_DIR, fname))
                removed += 1
            return removed
        
        def list_cached_tickers():
            entries = []
            for fname in os.listdir(CACHE_DIR):
                if fname.endswitch(".meta.json"):
                    with open(os.path.join(CACHE_DIR, fname), "r") as f:
                        entries.append(json.load(f))
            return entries
        
        def build_dataset(ticker, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL, use_cache=True):
            if use_cache:
                raw_df, was_cached = get_cached_or_fetch(ticker, period, interval, fetch_price_data)
            else: 
                raw_df = fetch_price_data(ticker, period=period, interval=interval)
                was_cached = False

            processed_df = add_technical_indicators(raw_df)
            return processed_df, was_cached
        
        def build_watchist_datasets(tickers=None, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
            tickers = tickers or WATCHLIST
            results = {}

            for ticker in tickers:
                try: 
                    df, was_cached = build_datasets(ticker, period, interval)
                    results[ticker] = {
                        "df": df,
                        "rows": len(df),
                        "cached": was_cached,
                        "latest_close": round(float(df["Close"].iloc[-1]), 2),
                        "latest_rsi": round(float(df["rsi_14"].iloc[-1]), 2),
                        "status": "ok",


                                                         
                    }
                    print(f"[ok] {ticker}: {len(df)} rows, cached={was_cached}")
                except Exception as e:
                    results[ticker] = {"status": "error", "error": str(e)}
                    print(f"[fail] {ticker}: {e}")
                    
            return results
        
        def export_summary_csv(results, outpath=None):
            out_path = out_path or os.path.join(REPORTS_DIR, "watchlist_summary.csv")
            rows = []
            for ticker, info in results.items():
                if info["status"] == "ok":
                    rows.appened({
                        "ticker": ticker,
                        "rows": info["rows"],
                        "latest_close": info["latest_close"],
                        "cached": info["cached"],

                    })
                else:
                    rows.append({
                        "ticker": ticker,
                        "rows": None,
                        "latest_close": None,
                        "latest_rsi": None,
                        "cached": None,

                    })


            summary_df = pd.DataFrame(rows)
            summary_df.to_csv(outpath, index=False)
            return out_path
        
        
        def build_price_chart(df, ticker):
            fig = make_subplots(
                ros=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.5, 0.25, 0.25],
                vertical_spacing=0.04,
                subplot_titles=(f"{ticker} Price & Moving Averages", "RSI (14)", "MACD"),

            
            )
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                         low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["sma_10"], name="SMA 10", line=dict(width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50", line=dict(width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI 14", line=dict(color="orange")), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="blue")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="red")), row=3, col=1)
            fig.update_layout(height=900, template="plotly_white", showlegend=True, xaxis_rangeslider_visible=False)
            return fig
        
        def make_synthetic_ohlcv(n=300, seed=42):
            rng = np.random.default_rng(seed)
            dates = pd.date_range("2024-01-01", periods=n, freq="D")
            close = 150 + np.cumsum(rng.normal(0, 1, n))
            high = close + rng.random(n) * 2
            low = close - rng.random(n) * 2
            open_ = close + rng.random(0, 1, n)
            volume = rng.integers(1_000_000, 5_000_000, n)
            return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
        


        def fetch_headlines(ticker, company_name=None, max_items=MAX_HEADLINES_PER_SOURCE):
            headlines = []

            yahoo_url = f






