import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List

from pearson_complete import (
    get_full_analysis,
    build_dataset,
    get_sentiment_summary,
    WATCHLIST,
    TICKER_TO_COMPANY,
    get_feature_columns,
)
from backtest import run_ticker_backtest, compute_backtest_metrics


app = FastAPI(
    title="Pearson API",
    description="Backend for the Pearson stock/crypto direction predictor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    ticker: str
    company_name: str
    direction: str
    confidence_pct: float
    prob_up: float
    prob_down: float
    latest_close: float
    as_of: str
    sentiment_label: str
    sentiment_score: float
    n_headlines_used: int
    top_contributing_features: list
    model_accuracy: float
    baseline_accuracy: float
    honesty_notes: List[str]


class PriceChartPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    sma_10: Optional[float] = None
    sma_50: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None


class SentimentHeadline(BaseModel):
    title: str
    source: str
    compound: float
    link: Optional[str] = ""


class SentimentResponse(BaseModel):
    ticker: str
    label: str
    avg_compound: float
    n_headlines: int
    positive_count: Optional[int] = 0
    negative_count: Optional[int] = 0
    neutral_count: Optional[int] = 0
    headlines: List[SentimentHeadline]


class BacktestPoint(BaseModel):
    date: str
    predicted: int
    actual: int
    correct: int
    confidence: float
    rolling_accuracy: Optional[float] = None


class BacktestResponse(BaseModel):
    ticker: str
    accuracy: float
    baseline_accuracy: float
    edge_over_baseline: float
    total_predictions: int
    cumulative_strategy_return_pct: float
    cumulative_buy_hold_return_pct: float
    high_confidence_accuracy: Optional[float] = None
    points: List[BacktestPoint]


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/watchlist")
def get_watchlist():
    return {
        "tickers": [
            {"ticker": t, "company": TICKER_TO_COMPANY.get(t, t)}
            for t in WATCHLIST
        ]
    }


@app.get("/api/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str, period: str = Query("2y"), use_cache: bool = Query(True)):
    ticker = ticker.upper()
    company = TICKER_TO_COMPANY.get(ticker)

    try:
        analysis = get_full_analysis(ticker, company_name=company, period=period, use_cache=use_cache)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not analyze '{ticker}': {str(e)}")

    p = analysis["prediction"]
    m = analysis["model_metrics"]

    return PredictionResponse(
        ticker=ticker,
        company_name=analysis["company_name"],
        direction=p["direction"],
        confidence_pct=p["confidence_pct"],
        prob_up=p["prob_up"],
        prob_down=p["prob_down"],
        latest_close=p["latest_close"],
        as_of=p["as_of"],
        sentiment_label=p["sentiment_label"],
        sentiment_score=p["sentiment_score"],
        n_headlines_used=p["n_headlines_used"],
        top_contributing_features=p["top_contributing_features"],
        model_accuracy=m["test_accuracy"],
        baseline_accuracy=m["baseline_accuracy"],
        honesty_notes=analysis["honesty_notes"],
    )


@app.get("/api/chart/{ticker}")
def get_chart_data(ticker: str, period: str = Query("1y"), use_cache: bool = Query(True)):
    ticker = ticker.upper()
    try:
        df, cached = build_dataset(ticker, period=period, use_cache=use_cache)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch data for '{ticker}': {str(e)}")

    points = []
    for idx, row in df.iterrows():
        points.append({
            "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": float(row["Volume"]),
            "sma_10": round(float(row["sma_10"]), 2) if pd_notna(row["sma_10"]) else None,
            "sma_50": round(float(row["sma_50"]), 2) if pd_notna(row["sma_50"]) else None,
            "rsi_14": round(float(row["rsi_14"]), 2) if pd_notna(row["rsi_14"]) else None,
            "macd": round(float(row["macd"]), 4) if pd_notna(row["macd"]) else None,
            "macd_signal": round(float(row["macd_signal"]), 4) if pd_notna(row["macd_signal"]) else None,
        })

    return {"ticker": ticker, "cached": cached, "points": points}


def pd_notna(value):
    try:
        return value == value
    except Exception:
        return value is not None


@app.get("/api/sentiment/{ticker}", response_model=SentimentResponse)
def sentiment(ticker: str, use_cache: bool = Query(True)):
    ticker = ticker.upper()
    company = TICKER_TO_COMPANY.get(ticker)

    try:
        summary = get_sentiment_summary(ticker, company, use_cache=use_cache)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch sentiment for '{ticker}': {str(e)}")

    headlines = [
        SentimentHeadline(
            title=h["title"], source=h["source"],
            compound=h["compound"], link=h.get("link", ""),
        )
        for h in summary.get("top_headlines", [])
    ]

    return SentimentResponse(
        ticker=ticker,
        label=summary["label"],
        avg_compound=summary["avg_compound"],
        n_headlines=summary["n_headlines"],
        positive_count=summary.get("positive_count", 0),
        negative_count=summary.get("negative_count", 0),
        neutral_count=summary.get("neutral_count", 0),
        headlines=headlines,
    )


@app.get("/api/backtest/{ticker}", response_model=BacktestResponse)
def backtest(ticker: str, period: str = Query("2y"), use_cache: bool = Query(True),
             min_train_size: int = Query(150), retrain_every: int = Query(20)):
    ticker = ticker.upper()

    try:
        output = run_ticker_backtest(
            ticker, period=period, use_cache=use_cache,
            min_train_size=min_train_size, retrain_every=retrain_every,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not backtest '{ticker}': {str(e)}")

    m = output["metrics"]
    results_df = output["results"]

    points = [
        BacktestPoint(
            date=row["date"], predicted=int(row["predicted"]), actual=int(row["actual"]),
            correct=int(row["correct"]), confidence=float(row["confidence"]),
            rolling_accuracy=float(row["rolling_accuracy"]) if pd_notna(row["rolling_accuracy"]) else None,
        )
        for _, row in results_df.iterrows()
    ]

    return BacktestResponse(
        ticker=ticker,
        accuracy=m["accuracy"],
        baseline_accuracy=m["baseline_accuracy"],
        edge_over_baseline=m["edge_over_baseline"],
        total_predictions=m["total_predictions"],
        cumulative_strategy_return_pct=m["cumulative_strategy_return_pct"],
        cumulative_buy_hold_return_pct=m["cumulative_buy_hold_return_pct"],
        high_confidence_accuracy=m["high_confidence_accuracy"],
        points=points,
    )


frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)