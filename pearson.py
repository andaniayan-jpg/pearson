import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import feedparser
import xgboost as xgb
import shap
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
CACHE_EXPIRY_HOURS = 6
SENTIMENT_CACHE_EXPIRY_HOURS = 2
MAX_HEADLINES_PER_SOURCE = 25

WATCHLIST = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "BTC-USD", "ETH-USD"]

TICKER_TO_COMPANY = {
    "AAPL": "Apple", "MSFT": "Microsoft", "TSLA": "Tesla", "GOOGL": "Google",
    "AMZN": "Amazon", "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
}

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

_analyzer = SentimentIntensityAnalyzer()



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
    df["macd_signal"] = ta.trend.macd_signal(close)
    df["macd_diff"] = ta.trend.macd_diff(close)

    df["rsi_14"] = ta.momentum.rsi(close, window=14)
    df["stoch_k"] = ta.momentum.stoch(high, low, close)
    df["roc_10"] = ta.momentum.roc(close, window=10)

    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    df["obv"] = ta.volume.on_balance_volume(close, volume)
    df["volume_sma_10"] = volume.rolling(10).mean()

    df["price_change_1d"] = close.pct_change(1)
    df["price_change_5d"] = close.pct_change(5)
    df["volatility_10d"] = close.pct_change().rolling(10).std()
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]

    df["target"] = (close.shift(-1) > close).astype(int)

    df = df.dropna()
    return df


def get_feature_columns():
    return [
        "sma_10", "sma_50", "ema_10", "macd", "macd_signal", "macd_diff",
        "rsi_14", "stoch_k", "roc_10",
        "bb_high", "bb_low", "bb_width", "atr_14",
        "obv", "volume_sma_10",
        "price_change_1d", "price_change_5d", "volatility_10d", "sma_ratio",
    ]


def _cache_path(ticker, period, interval):
    safe_ticker = ticker.replace("/", "_").replace("-", "_")
    filename = f"{safe_ticker}_{period}_{interval}.csv"
    return os.path.join(CACHE_DIR, filename)


def _meta_path(ticker, period, interval):
    return _cache_path(ticker, period, interval) + ".meta.json"


def is_cache_fresh(ticker, period, interval):
    meta_path = _meta_path(ticker, period, interval)
    if not os.path.exists(meta_path):
        return False
    with open(meta_path, "r") as f:
        meta = json.load(f)
    fetched_at = datetime.fromisoformat(meta["fetched_at"])
    return datetime.now() - fetched_at < timedelta(hours=CACHE_EXPIRY_HOURS)


def load_from_cache(ticker, period, interval):
    path = _cache_path(ticker, period, interval)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def save_to_cache(df, ticker, period, interval):
    path = _cache_path(ticker, period, interval)
    df.to_csv(path)
    meta = {
        "ticker": ticker, "period": period, "interval": interval,
        "fetched_at": datetime.now().isoformat(), "rows": len(df),
    }
    with open(_meta_path(ticker, period, interval), "w") as f:
        json.dump(meta, f, indent=2)


def get_cached_or_fetch(ticker, period, interval, fetch_fn):
    if is_cache_fresh(ticker, period, interval):
        cached = load_from_cache(ticker, period, interval)
        if cached is not None and len(cached) > 0:
            return cached, True
    fresh = fetch_fn(ticker, period=period, interval=interval)
    save_to_cache(fresh, ticker, period, interval)
    return fresh, False


def build_dataset(ticker, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL, use_cache=True):
    if use_cache:
        raw_df, was_cached = get_cached_or_fetch(ticker, period, interval, fetch_price_data)
    else:
        raw_df = fetch_price_data(ticker, period=period, interval=interval)
        was_cached = False
    processed_df = add_technical_indicators(raw_df)
    return processed_df, was_cached


def build_watchlist_datasets(tickers=None, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    tickers = tickers or WATCHLIST
    results = {}
    for ticker in tickers:
        try:
            df, was_cached = build_dataset(ticker, period, interval)
            results[ticker] = {
                "df": df, "rows": len(df), "cached": was_cached,
                "latest_close": round(float(df["Close"].iloc[-1]), 2),
                "latest_rsi": round(float(df["rsi_14"].iloc[-1]), 2),
                "status": "ok",
            }
            print(f"[ok] {ticker}: {len(df)} rows, cached={was_cached}")
        except Exception as e:
            results[ticker] = {"status": "error", "error": str(e)}
            print(f"[fail] {ticker}: {e}")
    return results


def build_price_chart(df, ticker):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
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
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="red")), row=3, col=1)
    fig.update_layout(height=900, template="plotly_white", showlegend=True, xaxis_rangeslider_visible=False)
    return fig


def make_synthetic_ohlcv(n=300, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 150 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.random(n) * 2
    low = close - rng.random(n) * 2
    open_ = close + rng.normal(0, 1, n)
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)





def fetch_headlines(ticker, company_name=None, max_items=MAX_HEADLINES_PER_SOURCE):
    headlines = []

    yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(yahoo_url)
        for entry in feed.entries[:max_items]:
            headlines.append({
                "title": entry.title, "published": entry.get("published", ""),
                "source": "Yahoo Finance", "link": entry.get("link", ""),
            })
    except Exception:
        pass

    query = company_name if company_name else ticker
    google_url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(google_url)
        for entry in feed.entries[:max_items]:
            headlines.append({
                "title": entry.title, "published": entry.get("published", ""),
                "source": "Google News", "link": entry.get("link", ""),
            })
    except Exception:
        pass

    return headlines


def score_sentiment(headlines):
    rows = []
    for h in headlines:
        scores = _analyzer.polarity_scores(h["title"])
        rows.append({
            **h, "compound": scores["compound"], "positive": scores["pos"],
            "negative": scores["neg"], "neutral": scores["neu"],
        })
    return pd.DataFrame(rows)


def label_from_score(avg_compound):
    if avg_compound > 0.15:
        return "Positive"
    elif avg_compound < -0.15:
        return "Negative"
    return "Neutral"


def _sentiment_cache_path(ticker):
    safe_ticker = ticker.replace("/", "_").replace("-", "_")
    return os.path.join(CACHE_DIR, f"sentiment_{safe_ticker}.json")


def is_sentiment_cache_fresh(ticker):
    path = _sentiment_cache_path(ticker)
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        data = json.load(f)
    fetched_at = datetime.fromisoformat(data["fetched_at"])
    return datetime.now() - fetched_at < timedelta(hours=SENTIMENT_CACHE_EXPIRY_HOURS)


def load_sentiment_cache(ticker):
    path = _sentiment_cache_path(ticker)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_sentiment_cache(ticker, summary):
    path = _sentiment_cache_path(ticker)
    payload = {**summary, "fetched_at": datetime.now().isoformat()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def get_sentiment_summary(ticker, company_name=None, use_cache=True):
    if use_cache and is_sentiment_cache_fresh(ticker):
        cached = load_sentiment_cache(ticker)
        if cached is not None:
            cached["cached"] = True
            return cached

    headlines = fetch_headlines(ticker, company_name)

    if not headlines:
        return {
            "ticker": ticker, "avg_compound": 0.0, "label": "No data",
            "n_headlines": 0, "top_headlines": [], "cached": False,
        }

    df = score_sentiment(headlines)
    avg = df["compound"].mean()
    label = label_from_score(avg)
    top = df.reindex(df["compound"].abs().sort_values(ascending=False).index).head(5)

    summary = {
        "ticker": ticker, "avg_compound": round(float(avg), 4), "label": label,
        "n_headlines": len(df),
        "positive_count": int((df["compound"] > 0.15).sum()),
        "negative_count": int((df["compound"] < -0.15).sum()),
        "neutral_count": int(((df["compound"] >= -0.15) & (df["compound"] <= 0.15)).sum()),
        "top_headlines": top[["title", "compound", "source", "link"]].to_dict("records"),
        "cached": False,
    }

    save_sentiment_cache(ticker, summary)
    return summary


def sentiment_to_feature_score(summary):
    label_weight = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0, "No data": 0.0}
    base = label_weight.get(summary["label"], 0.0)
    confidence_scaler = min(summary["n_headlines"] / 15, 1.0)
    return round(summary["avg_compound"] * confidence_scaler, 4), round(base * confidence_scaler, 4)


def build_watchlist_sentiment(tickers, use_cache=True):
    results = {}
    for ticker in tickers:
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        try:
            summary = get_sentiment_summary(ticker, company, use_cache=use_cache)
            results[ticker] = summary
            print(f"[ok] {ticker}: {summary['label']} ({summary['avg_compound']}) from {summary['n_headlines']} headlines")
        except Exception as e:
            results[ticker] = {"ticker": ticker, "status": "error", "error": str(e)}
            print(f"[fail] {ticker}: {e}")
    return results


def make_synthetic_headlines(scenario="mixed"):
    positive_titles = [
        "Company beats earnings expectations by wide margin",
        "Analysts upgrade stock after strong quarterly guidance",
        "New product launch drives record demand",
        "Stock surges to all-time high on optimistic outlook",
        "Company announces major partnership deal, shares rally",
    ]
    negative_titles = [
        "Company misses revenue targets, shares tumble",
        "Regulators launch investigation into business practices",
        "CEO resigns amid controversy, stock drops sharply",
        "Analysts downgrade stock citing weak demand",
        "Company slashes guidance after disappointing sales",
    ]
    neutral_titles = [
        "Company to report quarterly earnings next week",
        "Stock trades flat in light volume session",
        "Company schedules annual shareholder meeting",
        "Analyst maintains hold rating ahead of earnings",
        "Company files routine regulatory paperwork",
    ]

    if scenario == "positive":
        titles = positive_titles + neutral_titles[:2]
    elif scenario == "negative":
        titles = negative_titles + neutral_titles[:2]
    elif scenario == "neutral":
        titles = neutral_titles * 2
    else:
        titles = positive_titles[:2] + negative_titles[:2] + neutral_titles[:2]

    return [{"title": t, "published": "", "source": "Synthetic", "link": ""} for t in titles]




MODEL_PATH = os.path.join(MODELS_DIR, "direction_model.json")
MODEL_META_PATH = os.path.join(MODELS_DIR, "direction_model_meta.json")

ALL_MODEL_FEATURES = get_feature_columns() + ["sentiment_score"]


def attach_sentiment_feature(df, ticker, company_name=None, use_cache=True):
    df = df.copy()
    summary = get_sentiment_summary(ticker, company_name, use_cache=use_cache)
    score, _ = sentiment_to_feature_score(summary)
    df["sentiment_score"] = score
    return df, summary


def train_direction_model(ticker, company_name=None, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL,
                           test_size=0.2, use_cache=True):
    df, _ = build_dataset(ticker, period=period, interval=interval, use_cache=use_cache)
    df, sentiment_summary = attach_sentiment_feature(df, ticker, company_name, use_cache=use_cache)

    X = df[ALL_MODEL_FEATURES]
    y = df["target"]

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    metrics = {
        "train_accuracy": round(float(accuracy_score(y_train, train_preds)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
        "test_precision": round(float(precision_score(y_test, test_preds, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, test_preds, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, test_preds, zero_division=0)), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "up_ratio_actual": round(float(y_test.mean()), 4),
        "baseline_accuracy": round(max(float(y_test.mean()), 1 - float(y_test.mean())), 4),
    }

    model.save_model(MODEL_PATH)
    meta = {
        "ticker": ticker,
        "trained_at": datetime.now().isoformat(),
        "features": ALL_MODEL_FEATURES,
        "metrics": metrics,
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    return model, metrics, sentiment_summary


def load_direction_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    with open(MODEL_META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


def predict_direction(ticker, company_name=None, model=None, period=DEFAULT_PERIOD,
                       interval=DEFAULT_INTERVAL, use_cache=True):
    if model is None:
        model, meta = load_direction_model()
        if model is None:
            raise RuntimeError("No trained model found. Call train_direction_model() first.")

    df, _ = build_dataset(ticker, period=period, interval=interval, use_cache=use_cache)
    df, sentiment_summary = attach_sentiment_feature(df, ticker, company_name, use_cache=use_cache)

    latest_row = df[ALL_MODEL_FEATURES].iloc[[-1]]
    pred_class = int(model.predict(latest_row)[0])
    pred_proba = model.predict_proba(latest_row)[0]

    confidence = round(float(max(pred_proba)) * 100, 2)
    direction = "UP" if pred_class == 1 else "DOWN"

    explanation = explain_prediction(model, latest_row)

    return {
        "ticker": ticker,
        "direction": direction,
        "confidence_pct": confidence,
        "prob_up": round(float(pred_proba[1]) * 100, 2),
        "prob_down": round(float(pred_proba[0]) * 100, 2),
        "sentiment_label": sentiment_summary["label"],
        "sentiment_score": round(float(latest_row["sentiment_score"].iloc[0]), 4),
        "n_headlines_used": sentiment_summary.get("n_headlines", 0),
        "top_contributing_features": explanation,
        "latest_close": round(float(df["Close"].iloc[-1]), 2),
        "as_of": str(df.index[-1]),
    }


def explain_prediction(model, row_df, top_n=5):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = shap_values[0]
    contributions = list(zip(row_df.columns, values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    result = []
    for feature, value in contributions[:top_n]:
        direction = "pushed UP" if value > 0 else "pushed DOWN"
        result.append({
            "feature": feature,
            "impact": round(float(value), 4),
            "direction": direction,
            "feature_value": round(float(row_df[feature].iloc[0]), 4),
        })
    return result


def evaluate_model_honestly(metrics):
    lines = []
    lines.append(f"Test accuracy: {metrics['test_accuracy']*100:.1f}%")
    lines.append(f"Naive baseline (always predict majority class): {metrics['baseline_accuracy']*100:.1f}%")

    edge = metrics["test_accuracy"] - metrics["baseline_accuracy"]
    if edge <= 0.02:
        lines.append("WARNING: model barely beats the naive baseline. Treat predictions as low-confidence.")
    elif edge <= 0.07:
        lines.append("Model shows a modest edge over baseline. Useful as one signal among many, not a standalone trading rule.")
    else:
        lines.append("Model shows a meaningful edge over baseline on this test window, but past performance on this window does not guarantee future accuracy.")

    lines.append("This model predicts short-term direction only, not magnitude, and should not be used as financial advice.")
    return lines


def make_synthetic_dataset_with_target(n=400, seed=7):
    df = make_synthetic_ohlcv(n=n, seed=seed)
    df = add_technical_indicators(df)
    df["sentiment_score"] = np.random.default_rng(seed).uniform(-0.3, 0.3, len(df))
    return df


def test_model_trains_and_predicts_on_synthetic_data():
    df = make_synthetic_dataset_with_target()
    X = df[ALL_MODEL_FEATURES]
    y = df["target"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_predict_proba_sums_to_one():
    df = make_synthetic_dataset_with_target()
    X = df[ALL_MODEL_FEATURES]
    y = df["target"]
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X.iloc[[-1]])[0]
    assert abs(sum(proba) - 1.0) < 1e-6


def test_explain_prediction_returns_features():
    df = make_synthetic_dataset_with_target()
    X = df[ALL_MODEL_FEATURES]
    y = df["target"]
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42)
    model.fit(X, y)
    explanation = explain_prediction(model, X.iloc[[-1]], top_n=5)
    assert len(explanation) == 5
    for item in explanation:
        assert "feature" in item and "impact" in item and "direction" in item


def test_evaluate_model_honestly_flags_weak_edge():
    weak_metrics = {"test_accuracy": 0.51, "baseline_accuracy": 0.50}
    lines = evaluate_model_honestly(weak_metrics)
    assert any("WARNING" in l for l in lines)


def run_phase3_tests():
    tests = [
        test_model_trains_and_predicts_on_synthetic_data,
        test_predict_proba_sums_to_one,
        test_explain_prediction_returns_features,
        test_evaluate_model_honestly_flags_weak_edge,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}  -  {e}")
    print(f"\n{passed}/{len(tests)} phase 3 tests passed")



def get_full_analysis(ticker, company_name=None, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL,
                       use_cache=True, retrain_if_missing=True):
    company_name = company_name or TICKER_TO_COMPANY.get(ticker, ticker)

    model, meta = load_direction_model()
    if model is None:
        if not retrain_if_missing:
            raise RuntimeError("No trained model available and retrain_if_missing=False")
        model, metrics, _ = train_direction_model(ticker, company_name, period, interval, use_cache=use_cache)
    else:
        metrics = meta["metrics"]

    prediction = predict_direction(ticker, company_name, model=model, period=period,
                                    interval=interval, use_cache=use_cache)
    honesty_notes = evaluate_model_honestly(metrics)

    df, _ = build_dataset(ticker, period=period, interval=interval, use_cache=use_cache)
    recent_prices = df[["Close"]].tail(60).reset_index()
    recent_prices.columns = ["date", "close"]

    return {
        "ticker": ticker,
        "company_name": company_name,
        "prediction": prediction,
        "model_metrics": metrics,
        "honesty_notes": honesty_notes,
        "recent_prices": recent_prices.to_dict("records"),
        "generated_at": datetime.now().isoformat(),
    }


def get_watchlist_analysis(tickers=None, use_cache=True):
    tickers = tickers or WATCHLIST
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = get_full_analysis(ticker, use_cache=use_cache)
            print(f"[ok] {ticker}: {results[ticker]['prediction']['direction']} "
                  f"({results[ticker]['prediction']['confidence_pct']}% confidence)")
        except Exception as e:
            results[ticker] = {"ticker": ticker, "status": "error", "error": str(e)}
            print(f"[fail] {ticker}: {e}")
    return results


def export_analysis_json(analysis, out_path=None):
    ticker = analysis.get("ticker", "unknown")
    out_path = out_path or os.path.join(REPORTS_DIR, f"{ticker.replace('-', '_')}_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    return out_path


def format_analysis_as_text(analysis):
    p = analysis["prediction"]
    lines = []
    lines.append(f"=== {analysis['company_name']} ({analysis['ticker']}) ===")
    lines.append(f"Latest close: {p['latest_close']}  (as of {p['as_of']})")
    lines.append(f"Predicted next-day direction: {p['direction']}  |  confidence: {p['confidence_pct']}%")
    lines.append(f"  prob UP: {p['prob_up']}%   prob DOWN: {p['prob_down']}%")
    lines.append(f"News sentiment: {p['sentiment_label']} (score {p['sentiment_score']}, "
                 f"from {p['n_headlines_used']} headlines)")
    lines.append("")
    lines.append("Top factors behind this prediction:")
    for f in p["top_contributing_features"]:
        lines.append(f"  - {f['feature']} ({f['feature_value']}) {f['direction']}  [impact {f['impact']}]")
    lines.append("")
    lines.append("Model honesty check:")
    for note in analysis["honesty_notes"]:
        lines.append(f"  - {note}")
    return "\n".join(lines)


def make_synthetic_full_analysis(ticker="TEST"):
    df = make_synthetic_dataset_with_target(n=400)
    X = df[ALL_MODEL_FEATURES]
    y = df["target"]

    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    metrics = {
        "test_accuracy": round(float(accuracy_score(y, preds)), 4),
        "baseline_accuracy": round(max(float(y.mean()), 1 - float(y.mean())), 4),
    }

    latest_row = X.iloc[[-1]]
    pred_class = int(model.predict(latest_row)[0])
    pred_proba = model.predict_proba(latest_row)[0]
    explanation = explain_prediction(model, latest_row)

    prediction = {
        "ticker": ticker,
        "direction": "UP" if pred_class == 1 else "DOWN",
        "confidence_pct": round(float(max(pred_proba)) * 100, 2),
        "prob_up": round(float(pred_proba[1]) * 100, 2),
        "prob_down": round(float(pred_proba[0]) * 100, 2),
        "sentiment_label": "Neutral",
        "sentiment_score": round(float(latest_row["sentiment_score"].iloc[0]), 4),
        "n_headlines_used": 10,
        "top_contributing_features": explanation,
        "latest_close": round(float(df["Close"].iloc[-1]), 2),
        "as_of": str(df.index[-1]),
    }

    honesty_notes = evaluate_model_honestly(metrics)

    return {
        "ticker": ticker,
        "company_name": ticker,
        "prediction": prediction,
        "model_metrics": metrics,
        "honesty_notes": honesty_notes,
        "recent_prices": df[["Close"]].tail(60).reset_index().rename(
            columns={"index": "date", "Close": "close"}).to_dict("records"),
        "generated_at": datetime.now().isoformat(),
    }


def test_full_analysis_has_required_keys():
    analysis = make_synthetic_full_analysis()
    for key in ["ticker", "company_name", "prediction", "model_metrics", "honesty_notes", "recent_prices"]:
        assert key in analysis


def test_format_analysis_as_text_runs_without_error():
    analysis = make_synthetic_full_analysis()
    text = format_analysis_as_text(analysis)
    assert isinstance(text, str)
    assert "Predicted next-day direction" in text


def test_export_analysis_json_writes_file():
    analysis = make_synthetic_full_analysis()
    path = export_analysis_json(analysis, out_path=os.path.join(REPORTS_DIR, "TEST_analysis.json"))
    assert os.path.exists(path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["ticker"] == "TEST"


def run_phase4_tests():
    tests = [
        test_full_analysis_has_required_keys,
        test_format_analysis_as_text_runs_without_error,
        test_export_analysis_json_writes_file,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}  -  {e}")
    print(f"\n{passed}/{len(tests)} phase 4 tests passed")



def test_feature_columns_all_present():
    df = make_synthetic_ohlcv()
    df = add_technical_indicators(df)
    for col in get_feature_columns():
        assert col in df.columns


def test_no_nan_after_processing():
    df = make_synthetic_ohlcv()
    df = add_technical_indicators(df)
    assert df[get_feature_columns()].isna().sum().sum() == 0


def test_target_is_binary():
    df = make_synthetic_ohlcv()
    df = add_technical_indicators(df)
    assert set(df["target"].unique()).issubset({0, 1})


def test_rsi_within_bounds():
    df = make_synthetic_ohlcv()
    df = add_technical_indicators(df)
    assert df["rsi_14"].min() >= 0 and df["rsi_14"].max() <= 100


def test_scoring_produces_expected_labels():
    for scenario, expected in [("positive", "Positive"), ("negative", "Negative"), ("neutral", "Neutral")]:
        df = score_sentiment(make_synthetic_headlines(scenario))
        assert label_from_score(df["compound"].mean()) == expected


def test_compound_scores_within_bounds():
    df = score_sentiment(make_synthetic_headlines("mixed"))
    assert df["compound"].between(-1.0, 1.0).all()


def make_synthetic_dataset_with_target(n=400, seed=7):
    df = make_synthetic_ohlcv(n=n, seed=seed)
    df = add_technical_indicators(df)
    df["sentiment_score"] = np.random.default_rng(seed).uniform(-0.3, 0.3, len(df))
    return df


def test_model_trains_and_predicts():
    df = make_synthetic_dataset_with_target()
    X, y = df[ALL_MODEL_FEATURES], df["target"]
    split = int(len(df) * 0.8)
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    preds = model.predict(X.iloc[split:])
    assert set(preds).issubset({0, 1})


def test_explain_prediction_returns_features():
    df = make_synthetic_dataset_with_target()
    X, y = df[ALL_MODEL_FEATURES], df["target"]
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42)
    model.fit(X, y)
    explanation = explain_prediction(model, X.iloc[[-1]], top_n=5)
    assert len(explanation) == 5


def test_full_analysis_has_required_keys():
    analysis = make_synthetic_full_analysis()
    for key in ["ticker", "company_name", "prediction", "model_metrics", "honesty_notes", "recent_prices"]:
        assert key in analysis


def run_all_tests():
    tests = [
        test_feature_columns_all_present,
        test_no_nan_after_processing,
        test_target_is_binary,
        test_rsi_within_bounds,
        test_scoring_produces_expected_labels,
        test_compound_scores_within_bounds,
        test_model_trains_and_predicts,
        test_explain_prediction_returns_features,
        test_full_analysis_has_required_keys,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}  -  {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}  -  {e}")
    print(f"\n{passed}/{len(tests)} tests passed")




def main():
    parser = argparse.ArgumentParser(description="Pearson — full pipeline (data, sentiment, model, backend)")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--company", type=str, default=None)
    parser.add_argument("--period", type=str, default=DEFAULT_PERIOD)
    parser.add_argument("--interval", type=str, default=DEFAULT_INTERVAL)
    parser.add_argument("--train", action="store_true", help="Train the model for --ticker")
    parser.add_argument("--predict", action="store_true", help="Run full analysis for --ticker")
    parser.add_argument("--watchlist", action="store_true", help="Run full analysis for the whole watchlist")
    parser.add_argument("--test", action="store_true", help="Run the offline test suite")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    use_cache = not args.no_cache

    if args.test:
        run_all_tests()
        return

    if args.train:
        if not args.ticker:
            print("Provide --ticker when using --train")
            return
        model, metrics, sentiment = train_direction_model(
            args.ticker, args.company, args.period, args.interval, use_cache=use_cache)
        print(json.dumps(metrics, indent=2))
        return

    if args.watchlist:
        results = get_watchlist_analysis(use_cache=use_cache)
        for ticker, analysis in results.items():
            if "prediction" in analysis:
                print(format_analysis_as_text(analysis))
                print()
        return

    if args.predict:
        if not args.ticker:
            print("Provide --ticker when using --predict")
            return
        analysis = get_full_analysis(args.ticker, args.company, args.period, args.interval, use_cache=use_cache)
        print(format_analysis_as_text(analysis))
        export_analysis_json(analysis)
        return

    parser.print_help()


if __name__ == "__main__":
    main()