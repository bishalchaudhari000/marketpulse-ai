
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import altair as alt
import html

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BLACK = "#0A0A0A"
CARD = "#171717"
BORDER = "#333333"
WHITE = "#FFFFFF"
GRAY = "#A3A3A3"
GREEN = "#22C55E"
RED = "#EF4444"

# ==========================================
# DASHBOARD STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0A0A0A;
    color: #FFFFFF;
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
    color: #FFFFFF;
}

h1, h2, h3 {
    color: #FFFFFF !important;
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
    color: #FFFFFF;
    font-weight: 700;
}

[data-testid="stMetricDelta"] {
    font-weight: 600;
}

div.stButton > button {
    background-color: #FFFFFF;
    color: #000000;
    border: 1px solid #FFFFFF;
    border-radius: 8px;
    font-weight: 700;
    min-height: 44px;
}

div.stButton > button:hover {
    background-color: #E5E5E5;
    color: #000000;
    border-color: #E5E5E5;
}

[data-testid="stTextInput"] input {
    background-color: #202020;
    color: #FFFFFF;
    border: 1px solid #444444;
    border-radius: 8px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #333333;
    border-radius: 10px;
}

.signal-card {
    border-radius: 14px;
    padding: 18px 24px;
    border: 1px solid #333333;
    background-color: #171717;
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

.muted {
    color: #A3A3A3;
    font-size: 0.85rem;
}

hr {
    border-color: #333333 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CHART STYLING
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
# MAIN ANALYSIS
# ==========================================

if analyze:

    if not ticker or not ticker.isalnum():
        st.warning("Please enter a valid stock symbol.")
        st.stop()

    with st.spinner("Analyzing stock and running models..."):
        try:
            response = requests.get(
                f"http://127.0.0.1:8000/predict/{ticker}",
                timeout=180
            )

            response.raise_for_status()
            data = response.json()

        except requests.RequestException as error:
            st.error(f"Unable to connect to FastAPI: {error}")
            st.stop()

    if "error" in data:
        st.error(data["error"])
        st.stop()

    results = data["models"]
    current = float(data["latest_close"])

    # ======================================
    # STOCK OVERVIEW
    # ======================================

    st.subheader(f"{data['ticker']} Market Analysis")

    st.metric(
        "Latest Closing Price",
        f"${current:,.2f}"
    )

    # ======================================
    # BUY / SELL SIGNAL
    # ======================================

    signal_data = data.get("trading_signal", {})
    signal = signal_data.get("signal", "UNRELIABLE")
    score = float(signal_data.get("score", 0))

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

    st.markdown(
        f"""
<div class="signal-card" style="border-color:{signal_color};">
    <div class="section-label">AI Trading Signal</div>
    <div class="signal-title" style="color:{signal_color};">
        {symbol} {safe_signal}
    </div>
    <div class="signal-score" style="color:{signal_color};">
        {score:.1f}% Model Agreement
    </div>
    <p class="muted">
        Experimental weighted model agreement.
        Not a probability of profit.
    </p>
</div>
""",
        unsafe_allow_html=True
    )

    # ======================================
    # MODEL FORECAST CARDS
    # ======================================

    st.subheader("Next Trading Day Forecasts")

    model_order = [
        "Linear Regression",
        "Random Forest",
        "LSTM",
        "Baseline"
    ]

    columns = st.columns(4)

    for col, name in zip(columns, model_order):
        with col:
            result = results[name]
            predicted = float(result["predicted_price"])

            change = (
                (predicted - current) / current * 100
                if current != 0 else 0
            )

            st.metric(
                label=name,
                value=f"${predicted:,.2f}",
                delta=(
                    None if name == "Baseline"
                    else f"{change:+.2f}%"
                ),
                delta_color="normal"
            )

    st.divider()

    # ======================================
    # HISTORICAL STOCK CHART
    # ======================================

    st.subheader("Historical Stock Price")

    history = yf.download(
        ticker,
        period="1y",
        auto_adjust=True,
        progress=False
    )

    if not history.empty:
        historical = pd.DataFrame({
            "Closing Price": history["Close"].squeeze()
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
        st.warning("Historical stock data unavailable.")

    # ======================================
    # ACTUAL VS PREDICTED
    # ======================================

    st.subheader("Actual vs. Predicted Prices")

    comparison = pd.DataFrame(
        data.get("chart_data", [])
    )

    if not comparison.empty:
        comparison["Date"] = pd.to_datetime(
            comparison["Date"]
        )

        comparison = comparison.set_index("Date")

        chart_columns = [
            "Actual",
            "Linear Regression",
            "Random Forest",
            "Baseline"
        ]

        chart = make_line_chart(
            comparison,
            chart_columns,
            [WHITE, GREEN, RED, GRAY],
            height=380
        )

        st.altair_chart(
            chart,
            width="stretch"
        )

        st.caption(
            "Historical test predictions. "
            "LSTM historical predictions will be added "
            "after evaluation dates are aligned."
        )
    else:
        st.info("Historical prediction data unavailable.")

    st.divider()

    # ======================================
    # MODEL PERFORMANCE
    # ======================================

    st.subheader("Model Performance")

    performance = pd.DataFrame([
        {
            "Model": name,
            "Test MAE ($)": result.get("test_mae"),
            "Directional Accuracy (%)": result.get(
                "directional_accuracy"
            )
        }
        for name, result in results.items()
    ])

    st.dataframe(
        performance,
        hide_index=True,
        width="stretch"
    )

    # ======================================
    # ERROR COMPARISON CHART
    # ======================================

    st.subheader("Model Error Comparison")

    error_chart = (
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
                    format=".2f"
                )
            ]
        )
        .properties(height=320)
    )

    st.altair_chart(
        style_chart(error_chart),
        width="stretch"
    )

    st.caption(
        "Lower MAE indicates smaller historical prediction errors. "
        "Evaluation periods must be aligned for a fair comparison."
    )

else:
    st.info(
        "Enter a stock symbol in the sidebar "
        "and click Analyze Stock to begin."
    )
