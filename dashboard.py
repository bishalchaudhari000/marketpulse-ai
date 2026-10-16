
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import altair as alt
import html
from datetime import datetime
from zoneinfo import ZoneInfo

# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"
PACIFIC = ZoneInfo("America/Los_Angeles")

BLACK = "#0A0A0A"
CARD = "#171717"
BORDER = "#333333"
WHITE = "#FFFFFF"
GRAY = "#A3A3A3"
GREEN = "#22C55E"
RED = "#EF4444"

DAILY_MODELS = [
    "Linear Regression",
    "Random Forest",
    "LSTM",
    "Baseline"
]

HOURLY_MODELS = [
    "Linear Regression",
    "Random Forest",
    "Baseline"
]

# ==========================================
# STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0A0A0A;
    color: white;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #333333;
}

[data-testid="stSidebar"] * {
    color: white;
}

h1, h2, h3 {
    color: white !important;
    font-family: Arial, sans-serif;
    font-weight: 700;
}

p, label {
    color: #E5E5E5;
}

[data-testid="stMetric"] {
    background-color: #171717;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: #BDBDBD;
}

[data-testid="stMetricValue"] {
    color: white;
    font-weight: 700;
}

[data-testid="stMetricDelta"] {
    font-weight: 600;
}

div.stButton > button {
    background-color: white !important;
    color: black !important;
    border: 1px solid white !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    min-height: 44px !important;
}

div.stButton > button * {
    color: black !important;
    fill: black !important;
}

div.stButton > button:hover {
    background-color: #E5E5E5 !important;
    color: black !important;
}

div.stButton > button:hover * {
    color: black !important;
}

[data-testid="stTextInput"] input {
    background-color: #202020;
    color: white;
    border: 1px solid #444444;
    border-radius: 8px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #333333;
    border-radius: 10px;
}

.signal-card {
    background-color: #171717;
    border: 1px solid #333333;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 14px 0 22px 0;
}

.signal-title {
    font-size: 2.3rem;
    font-weight: 800;
    margin: 6px 0;
}

.signal-score {
    font-size: 1.25rem;
    font-weight: 600;
}

.section-label {
    color: #A3A3A3;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.market-card {
    background-color: #171717;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 14px;
}

.market-price {
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin: 6px 0;
}

.market-meta {
    color: #A3A3A3;
    font-size: 0.85rem;
    margin-top: 8px;
}

hr {
    border-color: #333333 !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# API FUNCTIONS
# ==========================================

def api_get(path, timeout=180):
    response = requests.get(
        API_URL + path,
        timeout=timeout
    )

    response.raise_for_status()
    result = response.json()

    if "error" in result:
        raise ValueError(result["error"])

    return result


def fetch_prediction(ticker):
    return api_get(f"/predict/{ticker}")


def fetch_hour_prediction(ticker):
    return api_get(f"/predict-hour/{ticker}")


def fetch_live_price(ticker):
    return api_get(f"/live/{ticker}", timeout=30)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker):
    return yf.download(
        ticker,
        period="1y",
        auto_adjust=True,
        progress=False
    )


# ==========================================
# CHART FUNCTIONS
# ==========================================

def style_chart(chart):
    return (
        chart.configure(background=BLACK)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=GRAY,
            titleColor=WHITE,
            gridColor=BORDER,
            domainColor=BORDER,
            tickColor=BORDER
        )
        .configure_legend(
            labelColor=WHITE,
            titleColor=WHITE
        )
        .configure_title(color=WHITE)
    )


def make_line_chart(df, columns, colors, height=350):
    plot = df[columns].reset_index()
    date_column = plot.columns[0]

    plot = plot.melt(
        id_vars=date_column,
        var_name="Series",
        value_name="Price"
    )

    chart = (
        alt.Chart(plot)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                f"{date_column}:T",
                title="Date"
            ),
            y=alt.Y(
                "Price:Q",
                title="Stock Price ($)",
                scale=alt.Scale(zero=False)
            ),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=columns,
                    range=colors
                ),
                legend=alt.Legend(
                    orient="bottom",
                    title=None
                )
            ),
            tooltip=[
                alt.Tooltip(
                    f"{date_column}:T",
                    title="Date",
                    format="%b %d, %Y"
                ),
                alt.Tooltip("Series:N"),
                alt.Tooltip(
                    "Price:Q",
                    format="$.2f"
                )
            ]
        )
        .properties(height=height)
        .interactive()
    )

    return style_chart(chart)


def show_performance(models, order, label):
    st.subheader(f"Model Performance{label}")

    performance = pd.DataFrame([
        {
            "Model": name,
            "Test MAE ($)": models.get(
                name, {}
            ).get("test_mae"),
            "Directional Accuracy (%)": models.get(
                name, {}
            ).get("directional_accuracy")
        }
        for name in order
    ])

    st.dataframe(
        performance,
        hide_index=True,
        width="stretch"
    )

    st.subheader(f"Model Error Comparison{label}")

    chart = (
        alt.Chart(performance)
        .mark_bar(
            color=WHITE,
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5
        )
        .encode(
            x=alt.X(
                "Model:N",
                title=None,
                sort=None
            ),
            y=alt.Y(
                "Test MAE ($):Q",
                title="Mean Absolute Error ($)"
            ),
            tooltip=[
                "Model:N",
                alt.Tooltip(
                    "Test MAE ($):Q",
                    format=".3f"
                )
            ]
        )
        .properties(height=320)
    )

    st.altair_chart(
        style_chart(chart),
        width="stretch"
    )


def format_time(timestamp):
    if not timestamp:
        return "Unavailable"

    parsed = pd.to_datetime(
        timestamp,
        utc=True,
        errors="coerce"
    )

    if pd.isna(parsed):
        return "Unavailable"

    return parsed.tz_convert(PACIFIC).strftime(
        "%b %d, %Y at %I:%M %p PT"
    )


# ==========================================
# SESSION STATE
# ==========================================

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "analyzed_ticker" not in st.session_state:
    st.session_state.analyzed_ticker = None

if "forecast_timeframe" not in st.session_state:
    st.session_state.forecast_timeframe = "Next Day"

if "hour_cache" not in st.session_state:
    st.session_state.hour_cache = {}


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("📈 MarketPulse AI")
    st.caption("Machine Learning Financial Analytics")

    st.divider()

    ticker = st.text_input(
        "Stock Symbol",
        value="AAPL",
        max_chars=15
    ).upper().strip()

    analyze = st.button(
        "Analyze Stock",
        width="stretch"
    )

    st.divider()

    st.subheader("Prediction Models")

    st.write("Linear Regression")
    st.write("Random Forest")
    st.write("LSTM Neural Network")
    st.write("Baseline")

    st.divider()

    st.caption(
        "Experimental financial analytics platform. "
        "Predictions are not guaranteed."
    )


# ==========================================
# MAIN HEADER
# ==========================================

st.title("Financial Market Prediction System")

st.caption(
    "Machine Learning Forecasting • "
    "Model Comparison • Trading Signal Analysis"
)

st.divider()


# ==========================================
# ANALYZE STOCK
# ==========================================

if analyze:
    valid = (
        bool(ticker)
        and ticker.replace(
            "-", ""
        ).replace(".", "").isalnum()
    )

    if not valid:
        st.warning("Please enter a valid stock symbol.")
        st.stop()

    with st.spinner("Analyzing stock and running models..."):
        try:
            data = fetch_prediction(ticker)

            st.session_state.prediction_data = data
            st.session_state.analyzed_ticker = ticker

            # Force a fresh hourly check after analysis.
            st.session_state.hour_cache.pop(
                ticker, None
            )

        except (requests.RequestException, ValueError) as error:
            st.error(
                f"Unable to analyze stock: {error}"
            )
            st.stop()


data = st.session_state.prediction_data
active_ticker = st.session_state.analyzed_ticker

if data is None or active_ticker is None:
    st.info(
        "Enter a stock symbol in the sidebar "
        "and click Analyze Stock to begin."
    )
    st.stop()


# ==========================================
# TIMEFRAME SELECTOR
# ==========================================

st.subheader(f"{active_ticker} Market Analysis")

st.markdown("### Forecast Timeframe")

timeframe = st.segmented_control(
    "Select prediction period",
    options=[
        "Next Hour",
        "End of Day",
        "Next Day"
    ],
    key="forecast_timeframe"
)

if timeframe == "Next Hour":
    st.caption(
        "Predicting the price 60 minutes after "
        "the latest completed market observation."
    )

elif timeframe == "End of Day":
    st.caption(
        "Predicting the current trading session's "
        "closing price."
    )

else:
    st.caption(
        "Predicting the next trading day's "
        "closing price."
    )


# ==========================================
# LIVE MARKET PANEL
# ==========================================

@st.fragment(run_every="60s")
def live_market_panel(stock_ticker):
    try:
        live = fetch_live_price(stock_ticker)

    except (requests.RequestException, ValueError) as error:
        st.warning(
            f"Market data unavailable: {error}"
        )
        return

    price = float(live["latest_price"])
    change = live.get("change_percent")

    status = live.get(
        "market_status", "UNKNOWN"
    )

    recorded = live.get(
        "price_time_display", "Unknown"
    )

    age = int(
        live.get("data_age_minutes", 0)
    )

    refreshed = datetime.now(PACIFIC).strftime(
        "%I:%M:%S %p PT"
    )

    if change is None:
        change_text = "Daily change unavailable"
        change_color = GRAY

    elif change > 0:
        change_text = f"▲ {change:+.2f}% today"
        change_color = GREEN

    elif change < 0:
        change_text = f"▼ {change:+.2f}% today"
        change_color = RED

    else:
        change_text = "0.00% today"
        change_color = GRAY

    if status == "OPEN":
        status_color = GREEN
        age_text = f"Price observed {age} minutes ago"

    else:
        status_color = GRAY
        age_text = "Last completed market session"

    safe_status = html.escape(str(status))
    safe_recorded = html.escape(str(recorded))
    safe_age = html.escape(age_text)

    # Keep HTML at the start of each line.
    market_html = f"""
<div class="market-card">
<div class="section-label">Latest Available Market Price</div>
<div class="market-price">${price:,.2f}</div>
<div style="color:{change_color};font-weight:700;">
{change_text}
</div>
<div class="market-meta">
{html.escape(stock_ticker)} • Market Status:
<span style="color:{status_color};font-weight:700;">
{safe_status}
</span>
</div>
<div class="market-meta">
Price recorded: {safe_recorded} • {safe_age}
</div>
<div class="market-meta">
Dashboard refreshed: {refreshed}
• Auto-refresh every 60 seconds
</div>
</div>
"""

    st.markdown(
        market_html.strip(),
        unsafe_allow_html=True
    )


live_market_panel(active_ticker)


# ==========================================
# AUTOMATIC NEXT-HOUR FORECAST
# ==========================================

@st.fragment(run_every="60s")
def hourly_forecast_panel(stock_ticker):

    try:
        live = fetch_live_price(stock_ticker)

    except (requests.RequestException, ValueError) as error:
        st.warning(
            f"Cannot check hourly forecast: {error}"
        )
        return

    status = live.get("market_status", "UNKNOWN")
    age = int(live.get("data_age_minutes", 9999))
    timestamp = live.get("price_timestamp")

    if not timestamp:
        st.warning("Market data has no observation timestamp.")
        return

    # ======================================
    # STALE QUOTE SAFEGUARD
    # ======================================

    # Allow a small buffer for a completed
    # 5-minute candle to reach Yahoo Finance.
    # This does not guarantee exchange-real-time data.
    if status == "OPEN" and age > 15:
        st.warning(
            f"Market data is {age} minutes old. "
            "Forecast updates are paused until "
            "fresher data becomes available."
        )
        return

    cache = st.session_state.hour_cache.get(
        stock_ticker, {}
    )

    cached_timestamp = cache.get("timestamp")
    hour_data = cache.get("data")

    # Fetch a new forecast when:
    # 1. There is no cached forecast.
    # 2. A new completed candle appears.
    # 3. The market transitions to/from open.
    needs_update = (
        hour_data is None
        or cached_timestamp != timestamp
        or cache.get("market_status") != status
    )

    if needs_update:
        with st.spinner(
            "Updating next-hour model predictions..."
        ):
            try:
                hour_data = fetch_hour_prediction(
                    stock_ticker
                )

                # Store the actual model input timestamp.
                # It can differ from the quote endpoint.
                model_timestamp = hour_data.get(
                    "price_timestamp"
                )

                st.session_state.hour_cache[
                    stock_ticker
                ] = {
                    "timestamp": timestamp,
                    "model_timestamp": model_timestamp,
                    "market_status": status,
                    "data": hour_data,
                    "updated_at": datetime.now(
                        PACIFIC
                    ).isoformat()
                }

            except (requests.RequestException, ValueError) as error:
                st.error(
                    f"Hourly prediction error: {error}"
                )
                return

    cache = st.session_state.hour_cache.get(
        stock_ticker, {}
    )

    hour_data = cache.get("data")

    if not hour_data:
        st.info("No hourly forecast data available.")
        return

    results = hour_data.get("models", {})

    reference = float(
        hour_data.get("reference_price", 0)
    )

    available = hour_data.get(
        "forecast_available", False
    )

    # ======================================
    # FORECAST STATUS
    # ======================================

    st.subheader("Next Hour Forecasts")

    if status != "OPEN":
        st.info(
            "Market Closed — Next Hour Forecast Unavailable. "
            "Historical model performance is shown below."
        )

    elif not available:
        st.info(
            "No valid next-hour forecast is available. "
            "This can happen near the end of the "
            "trading session or before enough "
            "completed candles are available."
        )

    # ======================================
    # TIMESTAMPS
    # ======================================

    st.caption(
        "Price observed: "
        + format_time(
            hour_data.get("price_timestamp")
        )
    )

    st.caption(
        "Prediction calculated: "
        + format_time(cache.get("updated_at"))
    )

    if available:
        st.caption(
            "Forecast target: "
            + format_time(
                hour_data.get("target_timestamp")
            )
        )

    st.caption(
        "Checking for new market candles every 60 seconds."
    )

    # ======================================
    # FORECAST CARDS
    # ======================================

    columns = st.columns(3)

    for col, name in zip(
        columns, HOURLY_MODELS
    ):
        model = results.get(name, {})

        predicted = model.get(
            "predicted_price"
        )

        with col:
            if predicted is None:
                st.metric(
                    name,
                    "Unavailable"
                )

            else:
                predicted = float(predicted)

                change = (
                    (predicted - reference)
                    / reference * 100
                    if reference else 0
                )

                st.metric(
                    name,
                    f"${predicted:,.2f}",
                    delta=(
                        None if name == "Baseline"
                        else f"{change:+.2f}%"
                    )
                )

    st.divider()

    # ======================================
    # PERFORMANCE
    # ======================================

    show_performance(
        results,
        HOURLY_MODELS,
        " — Next Hour"
    )

    evaluation = hour_data.get(
        "evaluation", {}
    )

    st.caption(
        f"Training: "
        f"{evaluation.get('training_days', '?')} days | "
        f"Testing: "
        f"{evaluation.get('testing_days', '?')} days | "
        f"Test period: "
        f"{evaluation.get('test_start', '?')} to "
        f"{evaluation.get('test_end', '?')}"
    )

    st.caption(
        "Historical model evaluation is experimental. "
        "Overlapping forecast windows can make "
        "accuracy estimates less reliable."
    )


# ==========================================
# NEXT HOUR TAB
# ==========================================

if timeframe == "Next Hour":
    hourly_forecast_panel(active_ticker)
    st.stop()


# ==========================================
# END OF DAY TAB
# ==========================================

if timeframe == "End of Day":
    st.info(
        "End-of-day forecasting is coming next. "
        "We'll build a separate model using intraday "
        "market data to predict the closing price."
    )
    st.stop()


# ==========================================
# NEXT DAY TAB
# ==========================================

results = data["models"]
current = float(data["latest_close"])

st.caption(
    f"Next-day models use historical daily data. "
    f"Last daily closing price: ${current:,.2f}. "
    "The latest intraday price is displayed above "
    "but is not yet used by these models."
)


# ==========================================
# BUY / SELL SIGNAL
# ==========================================

signal_data = data.get(
    "trading_signal", {}
)

signal = signal_data.get(
    "signal", "UNRELIABLE"
)

score = float(
    signal_data.get("score", 0)
)

if signal == "BUY":
    signal_color = GREEN
    symbol = "▲"

elif signal == "SELL":
    signal_color = RED
    symbol = "▼"

else:
    signal_color = GRAY
    symbol = "●"

safe_signal = html.escape(str(signal))

signal_html = f"""
<div class="signal-card" style="border-color:{signal_color};">
<div class="section-label">AI Trading Signal</div>
<div class="signal-title" style="color:{signal_color};">
{symbol} {safe_signal}
</div>
<div class="signal-score" style="color:{signal_color};">
{score:.1f}% Model Agreement
</div>
<p style="color:#A3A3A3;">
Experimental weighted model agreement.
Not a probability of profit.
</p>
</div>
"""

st.markdown(
    signal_html.strip(),
    unsafe_allow_html=True
)


# ==========================================
# NEXT DAY FORECAST CARDS
# ==========================================

st.subheader("Next Trading Day Forecasts")

columns = st.columns(4)

for col, name in zip(
    columns, DAILY_MODELS
):
    result = results[name]

    predicted = float(
        result["predicted_price"]
    )

    change = (
        (predicted - current) / current * 100
        if current else 0
    )

    with col:
        st.metric(
            label=name,
            value=f"${predicted:,.2f}",
            delta=(
                None if name == "Baseline"
                else f"{change:+.2f}%"
            )
        )

st.divider()


# ==========================================
# HISTORICAL STOCK CHART
# ==========================================

st.subheader("Historical Stock Price")

try:
    history = fetch_history(active_ticker)

    if not history.empty:
        historical = pd.DataFrame({
            "Closing Price": history[
                "Close"
            ].squeeze()
        })

        historical.index.name = "Date"

        chart = make_line_chart(
            historical,
            ["Closing Price"],
            [WHITE],
            height=350
        )

        st.altair_chart(
            chart,
            width="stretch"
        )

    else:
        st.warning(
            "Historical stock data unavailable."
        )

except Exception as error:
    st.warning(
        f"Unable to load historical chart: {error}"
    )


# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

st.subheader("Actual vs. Predicted Prices")

comparison = pd.DataFrame(
    data.get("chart_data", [])
)

if not comparison.empty:
    comparison["Date"] = pd.to_datetime(
        comparison["Date"]
    )

    comparison = comparison.set_index(
        "Date"
    )

    chart_columns = [
        "Actual",
        "Linear Regression",
        "Random Forest",
        "Baseline"
    ]

    available_columns = [
        name for name in chart_columns
        if name in comparison.columns
    ]

    if available_columns:
        colors = {
            "Actual": WHITE,
            "Linear Regression": GREEN,
            "Random Forest": RED,
            "Baseline": GRAY
        }

        chart = make_line_chart(
            comparison,
            available_columns,
            [
                colors[name]
                for name in available_columns
            ],
            height=380
        )

        st.altair_chart(
            chart,
            width="stretch"
        )

        st.caption(
            "Historical test predictions. "
            "LSTM historical predictions will be "
            "added after evaluation dates are aligned."
        )

else:
    st.info(
        "Historical prediction data unavailable."
    )

st.divider()


# ==========================================
# NEXT DAY MODEL PERFORMANCE
# ==========================================

show_performance(
    results,
    list(results.keys()),
    ""
)

st.caption(
    "Lower MAE indicates smaller historical "
    "prediction errors. Evaluation periods "
    "must be aligned for fair comparison."
)
