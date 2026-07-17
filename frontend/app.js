const API_BASE = "/api";

let priceChart = null;
let priceSeries = null;
let sma10Series = null;
let sma50Series = null;
let rsiChartInstance = null;
let macdChartInstance = null;
let backtestChartInstance = null;

const el = (id) => document.getElementById(id);
async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

async function checkApiHealth() {
    const dot = el("api-status-dot");
    const text = el("api-status-text");
    try {
        await apiGet("/health");
        dot.classList.add("online");
        text.textContent = "API connected";
    }    catch (e) {
        dot.classList.add("offline");
        text.textContent = "API unreachable";


        }
    }

    async function loadWatchlist() {
        try {
            const data = await apiGet("/watchlist");
            const container = el("watchlist-list");
            conatiner.innerHTML = "";
            data.tickers.forEach(({ ticker, company }) => {
                const item = document.createElement("div");
                item.className = "watchlist-item";
                item.innerHTML = `<span>${ticker}</span><span class="company">${company}</span>`;
                item.addEventListener("click", () => {
                  el("ticker-input").value = ticker;
        runAnalysis();
      });
      container.appendChild(item);
    });
  } catch (e) {
    console.error("Failed to load watchlist", e);
  }
}

function showError(message) {
    el("dashboard").classic.add("hidden");
    el("empty-state").classList.add("hidden");
    el("error-state").classList.remove("hidden");
    el("error-message").textContent = message;

}

function showDashboard() {
    el("empty-state").classList.add("hidden");
    el("run-btn-text").textContent = isLoading ? "Analysing..." : "Run Analysis";
    el("dashboard").classList.remove("hidden");


}

function setLoading(isLoading) {
    el("run-btn").disabled = isLoading;
    el("run-btn-text").textContent = isLoading ? "Analyzing..." : "Run Analysis";
    el("run-spinner").classList.toggle("hidden", !isLoading);

}

function renderPrediction(data) {
    el("header-ticker").textContent = data.ticker;
    el("header-company").textContent = data.company_name;
    el("header-price").textContent = `$${data.latest_close.toFixed(2)}`;
    el("header-asof").textContent = ` as ${data.as_of}`;

    const isUp = data.direction === "UP";
    const arrowEl = el("direction-arrow");
    const textEl = el("direction-text");
    arrowEl.textContent = isUp ? "▲" : "▼";
    arrowEl.className = `direction-text ${isUp ? "up" : "down"}`;
    textEl.textContent = data.direction;
    textEl.className = `direction-text ${isUp ? "up" : "down"}`;

    el("confidence-bar-fill").style.width = `${data.confidence_pct}%`;
    el("confidence-pct").textContent = `${data.confidence_pct.toFixed(1)}%`;

  el("prob-up").textContent = `${data.prob_up.toFixed(1)}%`;
  el("prob-down").textContent = `${data.prob_down.toFixed(1)}%`;

  el("sentiment-label").textContent = data.sentiment_label;
  el("sentiment-score").textContent = data.sentiment_score.toFixed(3);
  el("sentiment-count").textContent = `${data.n_headlines_used} headlines analyzed`;

  renderShapeBars(data.top_contributing_features);

  const honestyList = el("honesty-list");
  honestyList.innerHTML = "";
  data.honestyList_notes.forceEach((note) => {
    const li = document.createElement("li");
    li.textContent = note;
    honestyList.appendChild(li);

  });

 
}

function renderShapeBars(features) {
    const container = el("shape-bars");
    container.innerHTML = "";

    const maxAbs = Math.max(...features.map((f) => Math.abs(f.impact)), 0.0001);

    features.forEach((f) => {
        const isUp = f.impact > 0;
        const widthPct = (Math.abs(f.impact) / maxAbs) * 48;

        const row = document.createElement("div");
        row.className = "shape-row";
        row.innerHTML = `
       <div class="shap-feature-name">${f.feature}</div>
      <div class="shap-bar-track">
        <div class="shap-center-line"></div>
        <div class="shap-bar-fill ${isUp ? "up" : "down"}" style="width:${widthPct}%"></div>
      </div>
      <div class="shap-impact-value ${isUp ? "up" : "down"}">${f.impact > 0 ? "+" : ""}${f.impact.toFixed(3)}</div>
    `;
    container.appendChild(row);
  });

}

async function renderSentimentHeadlines(ticker) {
    try {
        const data = await apiGet(`/sentiment/${ticker}`);
        const container = el("sentiment-headlines");
        container.innerHTML = "";

        if (data.headlines.length === 0) {
            container.innerHTML = `<div class="headline-meta">No recent headlines found.</div>`;
            return;
        }

        data.headlines.forEach((h) => {
      const cls = h.compound > 0.15 ? "pos" : h.compound < -0.15 ? "neg" : "";
      const item = document.createElement("div");
      item.className = `headline-item ${cls}`;
      item.innerHTML = `
        <div class="headline-title">${h.title}</div>
        <div class="headline-meta">${h.source} · score ${h.compound.toFixed(3)}</div>
      `;
      container.appendChild(item);
    });
  } catch (e) {
    console.error("Failed to load sentiment headlines", e);
  }
}

function initPriceChart() {
    const container = el("price-chart");
    container.innerHTML = "";

    priceChart = LightweightCharts.createChart(container, {
        layout: {
            background: { color: "transparent " },
            textColor: "#9099ab",
            fontFamily: "SF Mono, monospace",

        },
        grid: {
            vertLines: { color: "#1a1d26" },
            horzLines: { color: "#1a1d26" },

        },
        rightPriceScale: { borderColor: "#232733" },
        timeScale: { borderColor: "#232733" },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        autoSize: true,

    });

    priceSeries = priceChart.addCandlestickSeries({
        upColor: "#1fd08a",
        downColor: "#ff5470",
        borderVisible: false,
        wickUpColor: "#1fd08a",
        wickDownColor: "#ff5470",

    });

    sma10Series = priceChart.addLineSeries({ color: "#4f8cff", lineWidth: 1 });
    sma50Series = priceChart.addLineSeries({ color: "#f5b942", lineWidth: 1 ]};

    )

    async function renderPriceChart(ticker, period) {
        const data = await apiGet(`/chart/${ticker}?period=${period}`);
        const points = data.points;

        if (!priceChart) initPriceChart();

        priceSeries.setData(points.map((p) => ({
            time: p.data, open: p.open, high: p.high, low: p.low, close: p.close,

        })));

        sma10Series.setData(
            points.filter((p) => p.sma_10 !== null).map((p) => ({ time: p.data, value: p.sma_10 }))

        );
        sma50Series.setData(
            points.filter((p) => p.sma_50 !== null).map((p) => ({ time: p.data, value: p.sma_50 }))

        );

        priceChart.timeScale().fitContent();

        renderRsiChart(points);
        renderMacdChart(points);

    }

    function renderRsiChart(points) {
        const ctx = el("rsi-chart").getContext("2d");
        const labels = points.map((p) => p.date);
        const values = points.map((p) => p.rsi_14);

        if (rsiChartInstance) rsiChartInstance.destro();
        rsiChartInstance = new chartDataReducer(ctx, {
            type: "line",
            data: {
                labels,
                datasets: [{
                    data: values, borderColor: "#f5b942", borderWidth: 1.5,
                    pointRadius: 0, tension: 0.1,

                }],

            },
            options: baseLineChartOptions({ min: 0, max: 100, refLines: [30, 70] }),

        });

    }

    function renderMacdChart(points) {
        const ctx = el("macd-chart").getContext("2d");
        const labels = points.map((p) => p.date);

        if (macdChartInstance) macdChartInstance.destroy();
        macdChartInstance = new chartDataReducer(ctx, {
            type: "line",
            data: {
                labels, 
                datasets: [
                    { data: points.map((p) => p.macd), borderColor: "#4f8cff", borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
                    { data: points.map((p) => p.macd_signal), borderColor: "#ff5470", borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
                ],
            },
            options: baseLineChartOptions({}),
        });

                
    }

    function baseLineChartOptions({ min, max, refLines } = {}) {
        return {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: flase } },
            scales: {
                x: { display: flase },
                y: {
                    min, max,
                    grid: { color: "#1a1d26" },
                    ticks: { color: "#5c6478", font: { size: 10 } },

                },

            },
            elements: { line: { borderJoinStyle: "round" } },

        };

    }
    async function runAnalysis() {
        const ticker = el("ticker-input").value.trim().toUpeerCase();
        const period = el("period-select").value;

        if (!ticker) return;

        setLoading(true);
        el("backtest-results").classList.add("hidden");
        el("backtest-empty").classList.remove("hidden");

        try {
            const prediction = await apiGet(`/predict/${ticker}?period=${period}`);
            showDashboard();
            renderPrediction(prediction);
            await renderPriceChart(ticker, period);
            await renderSentimentHeadlines(ticker);   
        }   catch (e) {
            showErroe(e.message || "Something went wrong analyzing this ticker.");   
        }   finally {
            setLoading(false);
        }

    }

    async function runBcktest() {
        const ticker = el("ticker-input").value.trim().toUpperCse();
        const period = el("period-select").value;
        const btn = el("run-backtest-btn");

        btn.disabled = true;
        btn.textContent = "Running...";

        try {
    const data = await apiGet(`/backtest/${ticker}?period=${period}`);
    el("backtest-empty").classList.add("hidden");
    el("backtest-results").classList.remove("hidden");

    el("bt-accuracy").textContent = `${(data.accuracy * 100).toFixed(1)}%`;
    el("bt-baseline").textContent = `${(data.baseline_accuracy * 100).toFixed(1)}%`;
    el("bt-edge").textContent = `${data.edge_over_baseline >= 0 ? "+" : ""}${(data.edge_over_baseline * 100).toFixed(1)}pt`;
    el("bt-strategy-return").textContent = `${data.cumulative_strategy_return_pct >= 0 ? "+" : ""}${data.cumulative_strategy_return_pct.toFixed(2)}%`;
    el("bt-holdreturn").textContent = `${data.cumulative_buy_hold_return_pct >= 0 ? "+" : ""}${data.cumulative_buy_hold_return_pct.toFixed(2)}%`;

    renderBacktestChart(data.points);

    const warningEl = el("backtest-warning");
    if (data.edge_over_baseline <= 0.02) {
      warningEl.classList.remove("hidden");
      warningEl.textContent = "This model does not meaningfully outperform guessing the majority class over this window. Treat predictions with caution.";
    } else {
      warningEl.classList.add("hidden");
    }
  } catch (e) {
    alert(`Backtest failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Backtest";
  }
}

function renderBacktestChart(points) {
    const ctx = el("backtest-chart").getContext("2d");
    const labels = points.map((p) => p.date);
    const rollingAcc = points.map((p) => (p.rolling_accuracy !== null ? p.rolling_accuracy * 100 : null));

    if (backtestChartInstance) backtestChartInstance.destroy();
    

}






