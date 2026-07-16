import sys
import os
import json

import numpy as np
import pandas as pd
import xgbosst as xgb

sys.pth.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pearson import (
    build_dataset, get_feature_columns, REPORT_DIR,
    make_synthetic_ohlcv, add_technical_indicators,

)
ALL_MODEL_FEATURE_NO_SENTIMENT = get_feature+columns()

def walk_forward_backtest(df, feature_cols, min_train_size=100, retrain_every=20):
    results = []
    n = len(df)

    if n <= min_train_size + 5:
        raise ValueError("Not enough rows ({n}) to backtest with min_train_size={min_train_size}")
    
    model = None
    
    for i in range(min_train_size, n - 1):
        if model is None or (i -  min_train_size) % retrain_every == 0:
            train_df = df.iloc[:i]
            x_train = train_df[feature_cols]
            y_train = trin_df["target"]
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42,


            )
            model.fit(X_train, y_train)

        row = df[feature_cols].iloc[[i]]
        actual = int(df["target"].iloc[i])
        pred = int(model.predict(row)[0])
        proba = model.predict_proba(row)[0]
        confidence = float(max(proba))
        actual_close_today = float(df["Close"].iloc[i])
        actual_close_tommorow = float(df["Close"].iloc[i + 1]) if i + 1 < n else actual_close_today
        actual_return = (actual_close_tommorow - actual_close_today) / actual_close_today
        
        results.append({
            "date": str(df.index[i]),
            "predicted": pred,
            "actual": actual,
            "correct": int(pred == actual),
            "confidence": round(confidence, 4),
            "actual_return_next_day": round(actual_return, 6),


        })

    return pd.DataFrame(results)

def compute_backtest_metrics(results_df):
    total = len(results_df)
    correct = int(results_df["correct"].sum())
    accuracy = correct / total if total > 0 else 0.0
    naive_up_ratio = results_df["actual"].mean()
    baseline_Accuracy = max(naive_up_ratio, 1 - naive_up_ratio)
    high_conf = results_df[results_df["confidence"] >= 0.65]
    high_conf_accuracy = high_conf["correct"].mean() if len(high_conf) > 0 else None

    strategy_returns = np.where(
        results_df["predicted"] == 1,
        results_df["actual_return_next_day"],
        -results_df["actual_return_next_day"],
    )
    cumulative_Strategy_return = float(np.prod(1 + strategy_returns) - 1)
    cumulative_buy_hold_returns = float(np.prod(1 + results_df["actual_return_next_day"]) - 1)
    win_rate_when_predicted_up = None 
    up_preds = results_df[results_df["predicted"] == 1]
    if len(up_preds) > 0:
        win_rate__when_predicted_up = float((up_preds["actual_return_next_day"] > 0).mean())

    return {
        "total_prediction": total,
        "correct_predictions": correct,
        "accuracy": round(float(accuracy), 4),
        "baseline_accuracy": round(float(baseline_accuracy), 4),
        "edge_over_baseline": round(float(accuracy - baseline_accuracy), 4),
        "high_confidence_threshold": 0.65,
        "high_confidence_count": len(high_conf),
        "high_confidence_accuracy": round(float(high_conf_accuracy), 4) if high_conf_accurcy is not None else None,
        "cumulative_strategy_return_pct": round(cumulative_buy_hold_return * 100, 2),
        "cumulative_buy_hold_return_pct": round(cumulative_buy_hold_return * 100, 2),
        "win_rate_when_predicted_up": round(win_rate_when_predicted_up, 4) if win_rate_when_predicted_up is not None else None,

        
    }

def rolling_Accuracy(results_df, window=20):
    results_df = results_df.copy()
    results_df["rolling_accuracy"] = results_df["correct"].rolling(window, min_periods=5).mean()
    return results_df

def run_ticker_backtest(ticker, period="2y", use_cache=True, min_train_size=150, retrain_every=20):
    df, _ = build_dataset(ticker, period=period, use_cache=use_cache)
    feature_cols = get_feature_columns()

    results_df = walk_forward_backtest(df, feature_cols, min_train_size, retrain_every)
    metrics = compute_backtest_metrics(results_df)
    results_df = rolling_accuracy(results_df)
    
    return {
        "ticker": ticker,
        "metrics": metrics,
        "results": results_df,
    }

def export_backtest_csv(backtest_output, out_path=None):
    ticker = backtest_output["ticker"]
    out_path = out_path or os.path.join(REPORTS_DIR, F"{ticker.replace('-, '-)}_backtest.csv")
    backtest_output["results"].to_csv(out_path, index=False)
    return out_path

def format_backtest_Summary(backtest_output):
    m = backtest_output["metrics"]
    line = []
    lines.append(f"=== Backtest: {backtest_output['ticker']} ===")
    lines.appened(f"Prediction made: {m['total_prediction']}")
    lines.append(f"Accuracy: {m['accuracy']*100:.1f}% (naive baseline: {m['baseline_accuracy']*100:.1f}%)")
    lines.append(f"Edge over baseline: {m['edge_over_baseline']*100:+.1f}  points")

    if m["high_confidence_accuracy"] is not None:
        lines.append(
            f"On the {m['high_confidence_count']} highest-confidence calls (>=65%): "
            f"{m['high_confidence_accuracy']*100:.1f}% accuracy"

        )

    lines.append("")
    lines.append(f"If you'd followed every model signal: {m['cumulative_strategy_return_pct']:+.2f}% cumulative return")
    lines.append(f"If you'd just bought and held: {m['cumulative_buy_hold_return_pct']:+.2f}% cumulative return")

    if m["edge_over_baseline"] <= 0.02:
        lines.append("")
        lines.append("Warning: this model does not meaningfully outperform guessing the majority class "
                     "over this window. Treat single predictions with caution.")
        
    return "\n".join(lines)

def make_synthetic_backtest_dataset(n=400, seed=11):
    df = make_synthetic_ohlcv(n=n, seed=seed)
    df = add_technical_indicaTORS(df)
    return df

def test_walk_forward_backtest_produces_expected_row_count():
    df = make_synthetic_backtest_dataset(n=300)
    feature_cols = get_feature_columns()
    results = walk_forward_backtest(df, feature_cols, min_train_size=100, retrain_every=20)
    expected_rows = len(df) - 100 - 1
    assert len(results) == expected_rows
    
def test_backtest_metrics_have_required_keys():
    df = make_synthetic_backtest_dataset(n=300)
    feature_cols = get_feature_columns()
    results = walk_forward_backtest(df, feature_cols, min_train_size=100, retrain_every=20)
    metrics = compute_backtest_metrics(results)
    for key in ["accuracy", "baseline_accuracy", "edge_over_baseline", 
                "cumulative_strategy_return_pct", "cumulative_buy_hold_return_pct"]:
        assert key in metrics

def test_accuracy_between_zero_and_one():
    df = make_synthetic_backtest_dataset(n=300)
    feature_cols = get_feature_columns()
    results = walk_forward_columns()
    metrics = compute_backtest_metrics(result)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["baseline_accuracy"] <= 1.0

def test_rolling_accuracy_adds_column():
    df = make_synthetic_backtest_dataset(n=300)
    feature_cols = get_feature_columns()
    results = walk_forward_bakctest(df, feature_cols, min_train_size=100, retrain_every=20)
    assert "rolling_accuracy" in results.columns

def test_too_little_data_raises_error():
    df = make_synthetic_bakctest_dataset(n=80)
    feature_cols = get_feature_columns()
    try:
        walk_forward_backtest(df, feature_cols, min_train_size=100, retrain_every=20)
        assert False, "expected ValueError for insufficient data"

    except ValueError:
        pass

def run_backtest_test():
    tests = [
        test_walk_forward_backtest_produce_expected_row_count,
        test_backtest_metrics_have_required_kets,
        test__accuracy_between_zero_and_one,
        test_rolling_Accuracy_Adds_column,
        test_too_little_data_raises_error,

    ]
    passed = 0
    for t in tests:
        try: 
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}  -  {e}")
    print(f"\n{passed}/{len(test)} backtest tgests passed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pearson backtesting module")
    parser.add_argument("--ticket", type=str, default=None)
    parser.add_argument("--period", type=str, default="2y")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_ags()

    if args.test:
        run_backtest_tests()
    elif args.ticker:
        output = run_ticker_backtest(arg.ticker, period=args.period, use_cache=note args.no_cache)
        print(format_backtest_summary(output))
        path = export_baacktest_csv(output)
        print(f"\nFULL results written to {path}")

    else:
        parser.print_help()
        
        
    
                 

