import sys
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pearson_complete import (
    get_full_analysis,
    build_dataset, 
    build_price_chart,
    get_sentiment_summary,
    TICKER_TO_COMPANY,
    WATCHLIST,
    load_direction_model,

)
st.set_page_config(page_title="Pearson", page_icon="📈", layout="wide")

st.markdown("""
<style>
.big-direction { font-size: 42px; font-weight: 700; }
.up { color: #16c784; }
.down { color: #ea3943; }
.confidence-box {
    background: #161922; padding: 20px; border-radius: 12px;
    text-align: center; margin-bottom: 10px;
}
.honesty-box {
    background: #1a1c24; border-left: 4px solid #f0b90b;
    padding: 14px 18px; border-radius: 6px; margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Pearson")
st.caption("Stock & crypto direction predictor - technical indicators + new sentiment + explainable ML")

with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_input("Ticker symbol", value="AAPL").strip().uper()
    company_input = st.text_input(
        "Company name (helps news search)",
        value=TICKER_TO_COMPANY.get(ticker_input, "")

    ).strip()
    period = st.selectbox("Historical period", ["6mo", "1y", "2y", "5y"], index=2)
    use_cache = st.checkbox("Use cached data when available", value=True)
    force_retrain = st.checkbox("Force retrain model", value=False)
    run_button = st.button("Run analysis", type="primary", use_container_width=True)

    st.divider()
    st.caption("Watchlist")
    for t in Watchlist:
        if st.button(t, key=f"watchlist_{t}", use_container_width=True):
      import sys
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pearson_complete import (
    get_full_analysis,
    build_dataset,
    build_price_chart,
    get_sentiment_summary,
    TICKER_TO_COMPANY,
    WATCHLIST,
    load_direction_model,
)

st.set_page_config(page_title="Pearson", page_icon="📈", layout="wide")

st.markdown("""
<style>
.big-direction { font-size: 42px; font-weight: 700; }
.up { color: #16c784; }
.down { color: #ea3943; }
.confidence-box {
    background: #161922; padding: 20px; border-radius: 12px;
    text-align: center; margin-bottom: 10px;
}
.honesty-box {
    background: #1a1c24; border-left: 4px solid #f0b90b;
    padding: 14px 18px; border-radius: 6px; margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Pearson")
st.caption("Stock & crypto direction predictor — technical indicators + news sentiment + explainable ML")

with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_input("Ticker symbol", value="AAPL").strip().upper()
    company_input = st.text_input(
        "Company name (helps news search)",
        value=TICKER_TO_COMPANY.get(ticker_input, "")
    ).strip()
    period = st.selectbox("Historical period", ["6mo", "1y", "2y", "5y"], index=2)
    use_cache = st.checkbox("Use cached data when available", value=True)
    force_retrain = st.checkbox("Force retrain model", value=False)
    run_button = st.button("Run analysis", type="primary", use_container_width=True)

    st.divider()
    st.caption("Watchlist")
    for t in WATCHLIST:
        if st.button(t, key=f"watchlist_{t}", use_container_width=True):
            ticker_input = t
            company_input = TICKER_TO_COMPANY.get(t, "")
            run_button = True

if force_retrain and os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "direction_model.json")):
    os.remove(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "direction_model.json"))

if run_button:
    if not ticker_input:
        st.warning("Enter a ticker symbol first.")
        st.stop()

    with st.spinner(f"Pulling data, scoring sentiment, and running the model for {ticker_input}..."):
        try:
            analysis = get_full_analysis(
                ticker_input,
                company_name=company_input or None,
                period=period,
                use_cache=use_cache,
            )
        except Exception as e:
            st.error(f"Couldn't complete analysis for {ticker_input}: {e}")
            st.stop()

    pred = analysis["prediction"]
    metrics = analysis["model_metrics"]

    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        direction_class = "up" if pred["direction"] == "UP" else "down"
        arrow = "▲" if pred["direction"] == "UP" else "▼"
        st.markdown(f"""
        <div class="confidence-box">
            <div style="color:#9aa0a6; font-size:14px;">PREDICTED NEXT-DAY DIRECTION</div>
            <div class="big-direction {direction_class}">{arrow} {pred['direction']}</div>
            <div style="color:#9aa0a6; margin-top:6px;">Confidence: <b>{pred['confidence_pct']}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Latest Close", f"${pred['latest_close']}")
        st.metric("Prob. UP", f"{pred['prob_up']}%")

    with col3:
        st.metric("News Sentiment", pred["sentiment_label"])
        st.metric("Headlines Used", pred["n_headlines_used"])

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Why This Prediction", "News", "Model Honesty"])

    with tab1:
        df, cached = build_dataset(ticker_input, period=period, use_cache=use_cache)
        fig = build_price_chart(df, ticker_input)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{'Loaded from cache' if cached else 'Freshly fetched'} · {len(df)} trading days")

    with tab2:
        st.subheader("Top factors behind this prediction")
        st.caption("SHAP values show how much each feature pushed the prediction up or down for this specific day.")

        for f in pred["top_contributing_features"]:
            impact = f["impact"]
            bar_color = "#16c784" if impact > 0 else "#ea3943"
            bar_width = min(abs(impact) * 100, 100)
            direction_word = "↑ pushed UP" if impact > 0 else "↓ pushed DOWN"

            st.markdown(f"**{f['feature']}** = {f['feature_value']}  &nbsp;&nbsp; *{direction_word}*")
            st.markdown(f"""
            <div style="background:#262a36; border-radius:4px; height:10px; width:100%; margin-bottom:16px;">
                <div style="background:{bar_color}; height:10px; border-radius:4px; width:{bar_width}%;"></div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.subheader("Recent headlines driving sentiment")
        sentiment_data = get_sentiment_summary(ticker_input, company_input or None, use_cache=use_cache)

        if sentiment_data["n_headlines"] == 0:
            st.info("No recent headlines found for this ticker.")
        else:
            for h in sentiment_data["top_headlines"]:
                score = h["compound"]
                color = "#16c784" if score > 0.15 else ("#ea3943" if score < -0.15 else "#9aa0a6")
                st.markdown(f"""
                <div style="border-left:3px solid {color}; padding-left:12px; margin-bottom:12px;">
                    <div>{h['title']}</div>
                    <div style="color:#9aa0a6; font-size:12px;">{h['source']} · sentiment score {round(score,3)}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab4:
        st.subheader("Be skeptical of this model — here's why")
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{metrics['test_accuracy']*100:.1f}%")
        c2.metric("Naive baseline", f"{metrics['baseline_accuracy']*100:.1f}%")

        for note in analysis["honesty_notes"]:
            st.markdown(f"""<div class="honesty-box">{note}</div>""", unsafe_allow_html=True)

        st.caption(
            "This tool predicts short-term price direction using historical patterns and public sentiment. "
            "It is a demonstration of applied ML, not financial advice."
        )

else:
    st.info("Enter a ticker in the sidebar and click **Run analysis** to get started.")
    st.markdown("""
    **What Pearson does:**
    - Pulls real price history and computes 19 technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Scores recent news sentiment using VADER on live headlines
    - Trains an XGBoost model combining both signals to predict next-day direction
    - Explains *why* using SHAP feature attribution
    - Reports its own accuracy against a naive baseline, honestly
    """)      ticker_input = t
            company_input = TICKER_TO_COMPANY.get(t, "")
            run_button = True
if force_retrain and os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "direction_model.json")):
    os.remove(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "direction_models.json"))

if run_button:
    if not ticker_input:
        st.warning("Enter a ticker symbol first.")
        st.stop()

    with st.spinner(f"Pulling data, scoring sentiment, and running the model for {ticker_input}..."):
        try:
            analysis = get_full_analysis(
                ticker_input,
                company_name=company_input or None,
                period=period,
                use_cache=use_cache,

            )
        except Exception as e: 
            st.error(f"Couldn't complete analysis for {ticker_input}: {e}")
            st.stop()

    pred = analysis["prediction"]
    metrics = analysis["models_metrics"]

    col1, col2, col3 = st.columns([1.2, 1. 1])

    with col1:
        direction_class = "up" if pred["direction"] == "UP" else "down"
        arrow = "▲" if pred["direction"] == "UP" else "▼"
        st.markdown(f"""
        <div class="confidence-box">
            <div style="color:#9aa0a6; font-size:14px;">PREDICTED NEXT-DAY DIRECTION</div>
            <div class="big-direction {direction_class}">{arrow} {pred['direction']}</div>
            <div style="color:#9aa0a6; margin-top:6px;">Confidence: <b>{pred['confidence_pct']}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col2: 
        st.metric("Latest Close", f"${pred['latest_close']}")
        st.metric("Prob. UP", f"{pred['prob_up']}%")
    with col3:
        st.metric("News Sentiment", pred["sentiment_label"])
        st.metric("Headlines Used", pred["n_headlines_used"])

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Why This Prediction", "News", "Model Honesty"])

    with tab1:
        df, cached = build_dataset(ticker_input, period=period, use_cache=use_cache)
        fig = build_price_chart(df, ticker_input)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{'Loaded from cache' if cached else 'Freshly fetched'} · {len(df)} trading days")

    with tab2: 
        st.subheader("Top factors behind this prediction")
        st.caption("SHAP values show how much each feature pushed the prediction up or down for this specific day.")

        for f in pred["top_contributing_features"]:
            impact = f["impact"]
            bar_color = "#16c784" if impact > 0 else "#ea3943"
            bar_width = min(abs(impact) * 100, 100) 
            direction_word = "↑ pushed UP" if impact > 0 else "↓ pushed DOWN"


            st.markdown(f"**{f['feature']}** = {f['feature_value']}  &nbsp;&nbsp; *{direction_word}*")
            st.markdown(f"""
            <div style="background:#262a36; border-radius:4px; height:10px; width:100%; margin-bottom:16px;">
                <div style="background:{bar_color}; height:10px; border-radius:4px; width:{bar_width}%;"></div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.subheader("Recent headlines driving sentiment")
        sentiment_data = get_sentiment_summary(ticker_input, company_input or None, use_cache=use_cache)

        if sentiment_data["n_headlines"] == 0:
            st.info("No recent headlines found for this ticker.")
        else:
            for h in sentiment_data["top_headlines"]:
                score = h["compound"]
                color = "#16c784" if score > 0.15 else ("#ea3943" if score < -0.15 else "#9aa0a6")
                st.markdown(f"""
                <div style="border-left:3px solid {color}; padding-left:12px; margin-bottom:12px;">
                    <div>{h['title']}</div>
                    <div style="color:#9aa0a6; font-size:12px;">{h['source']} · sentiment score {round(score,3)}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Be skeptical of this model - here's why")
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{metrics['test_accuracy']*100:.1f}%")
        c2.metric("Naive baseline", f"{metrics['baseline_accuracy']*100:.1f}%")

        for note in analysis["honesty_notes"]:
            st.markdown(f"""<div class="honesty-box">{note}</div>""", unsafe_allow_html=True)

        st.caption(
            "This tool predicts short-term price direction using historical patterns and public sentiment. "
            "It is a demonstration of applied ML, not financial advice."

        )
else:
   st.info("Enter a ticker in the sidebar and click **Run analysis** to get started.")
   st.markdown("""
    **What Pearson does:**
    - Pulls real price history and computes 19 technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Scores recent news sentiment using VADER on live headlines
    - Trains an XGBoost model combining both signals to predict next-day direction
    - Explains *why* using SHAP feature attribution
    - Reports its own accuracy against a naive baseline, honestly
    """)

                

                                 
                                 