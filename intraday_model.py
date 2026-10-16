
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ==========================================
# CONFIGURATION
# ==========================================

INTERVAL = "5m"
PERIOD = "60d"

LOOKBACK = 12
HORIZON = 12

TRAINING_DAYS = 20
TEST_DAYS = 5

MARKET_TZ = "America/New_York"
MAX_DATA_AGE_MINUTES = 15

NYSE = mcal.get_calendar("NYSE")


# ==========================================
# DOWNLOAD MARKET DATA
# ==========================================

def download_intraday(ticker):
    ticker = ticker.upper().strip()

    data = yf.Ticker(ticker).history(
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        prepost=False
    )

    if data.empty:
        raise ValueError("No intraday data available")

    required = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    data = data.dropna(
        subset=required
    ).sort_index()

    if data.index.tz is None:
        raise ValueError("Missing timezone information")

    data = data.tz_convert(MARKET_TZ)

    # Regular US trading hours
    data = data.between_time(
        "09:30", "15:55"
    )

    # Exclude unfinished 5-minute candles
    now = pd.Timestamp.now(tz=MARKET_TZ)

    candle_end = (
        data.index + pd.Timedelta(minutes=5)
    )

    data = data.loc[candle_end <= now]

    if data.empty:
        raise ValueError("No completed market candles")

    return data


# ==========================================
# TIMESTAMP VALIDATION
# ==========================================

def consecutive_candles(timestamps):
    if len(timestamps) < 2:
        return False

    gaps = (
        np.diff(timestamps)
        / np.timedelta64(1, "s")
    )

    return bool(np.all(gaps == 300))


# ==========================================
# ORIGINAL FEATURES
# ==========================================

def original_features(window):
    close = window["Close"].to_numpy(
        dtype=float
    )

    if len(close) != LOOKBACK:
        raise ValueError("Invalid price window")

    if not np.all(np.isfinite(close)):
        raise ValueError("Invalid closing prices")

    if np.any(close <= 0):
        raise ValueError("Closing prices must be positive")

    returns = np.diff(close) / close[:-1]

    return np.concatenate([
        close / close[-1] - 1.0,
        [
            returns.mean(),
            returns.std(),
            returns[-1]
        ]
    ])


# ==========================================
# ENHANCED FEATURES
# ==========================================

def enhanced_features(window):
    close = window["Close"].to_numpy(
        dtype=float
    )

    high = window["High"].to_numpy(
        dtype=float
    )

    low = window["Low"].to_numpy(
        dtype=float
    )

    volume = window["Volume"].to_numpy(
        dtype=float
    )

    if len(close) != LOOKBACK:
        raise ValueError("Invalid feature window")

    if not all(
        np.all(np.isfinite(values))
        for values in [close, high, low, volume]
    ):
        raise ValueError("Non-finite market data")

    if np.any(close <= 0):
        raise ValueError("Invalid closing prices")

    # ======================================
    # PRICE RETURNS
    # ======================================

    returns = np.diff(close) / close[:-1]

    normalized_prices = (
        close / close[-1] - 1.0
    )

    momentum = (
        close[-1] / close[0] - 1.0
    )

    # ======================================
    # VOLATILITY
    # ======================================

    volatility = float(
        np.std(returns)
    )

    price_range = (
        np.max(high) - np.min(low)
    ) / close[-1]

    candle_ranges = (
        high - low
    ) / close

    average_range = float(
        np.mean(candle_ranges)
    )

    # ======================================
    # VOLUME
    # ======================================

    mean_volume = max(
        float(np.mean(volume)), 1.0
    )

    relative_volume = (
        volume / mean_volume
    )

    volume_trend = (
        np.mean(volume[-4:])
        / mean_volume
    )

    # ======================================
    # RSI-LIKE MOMENTUM
    # ======================================

    # Uses available changes within the
    # 12-candle window (11 price changes).
    # This is not the standard 14-period RSI.

    differences = np.diff(close)

    gains = np.maximum(
        differences, 0
    )

    losses = np.maximum(
        -differences, 0
    )

    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))

    if avg_gain == 0 and avg_loss == 0:
        rsi = 50.0

    elif avg_loss == 0:
        rsi = 100.0

    elif avg_gain == 0:
        rsi = 0.0

    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)

    normalized_rsi = rsi / 100.0

    # ======================================
    # MOVING AVERAGES
    # ======================================

    sma_4 = float(
        np.mean(close[-4:])
    )

    sma_12 = float(
        np.mean(close)
    )

    sma_4_distance = (
        close[-1] / sma_4 - 1.0
    )

    sma_12_distance = (
        close[-1] / sma_12 - 1.0
    )

    # ======================================
    # TIME OF DAY
    # ======================================

    # Last candle's closing time
    observed_at = (
        window.index[-1]
        + pd.Timedelta(minutes=5)
    )

    minutes_since_open = (
        observed_at.hour * 60
        + observed_at.minute
        - (9 * 60 + 30)
    )

    time_fraction = np.clip(
        minutes_since_open / 390.0,
        0.0,
        1.0
    )

    # ======================================
    # COMBINE FEATURES
    # ======================================

    features = np.concatenate([
        normalized_prices,
        relative_volume,
        [
            returns.mean(),
            returns.std(),
            returns[-1],
            momentum,
            volatility,
            price_range,
            average_range,
            volume_trend,
            normalized_rsi,
            sma_4_distance,
            sma_12_distance,
            time_fraction
        ]
    ])

    return features.astype(float)


# ==========================================
# PREPARE TRAINING DATA
# ==========================================

def prepare_intraday_data(data):
    X_original = []
    X_enhanced = []

    y = []
    dates = []
    references = []

    for session_date, session in data.groupby(
        data.index.date
    ):
        times = session.index

        for i in range(
            LOOKBACK - 1,
            len(session) - HORIZON
        ):
            relevant_times = times[
                i - LOOKBACK + 1:
                i + HORIZON + 1
            ]

            if not consecutive_candles(
                relevant_times
            ):
                continue

            window = session.iloc[
                i - LOOKBACK + 1:i + 1
            ]

            reference = float(
                window["Close"].iloc[-1]
            )

            target = float(
                session["Close"].iloc[i + HORIZON]
            )

            if (
                reference <= 0
                or target <= 0
                or not np.isfinite(target)
            ):
                continue

            original = original_features(window)
            enhanced = enhanced_features(window)

            target_return = (
                target / reference - 1.0
            )

            X_original.append(original)
            X_enhanced.append(enhanced)

            y.append(target_return)
            dates.append(session_date)
            references.append(reference)

    if not X_original:
        raise ValueError(
            "Not enough valid training samples"
        )

    return (
        np.asarray(X_original),
        np.asarray(X_enhanced),
        np.asarray(y),
        np.asarray(dates),
        np.asarray(references)
    )


# ==========================================
# MACHINE LEARNING MODELS
# ==========================================

def build_models():
    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1
        )
    }


# ==========================================
# EVALUATION
# ==========================================

def evaluate_model(
    model,
    X_test,
    y_test,
    reference_prices
):
    predicted_returns = model.predict(X_test)

    predicted_prices = (
        reference_prices
        * (1.0 + predicted_returns)
    )

    actual_prices = (
        reference_prices
        * (1.0 + y_test)
    )

    mae = mean_absolute_error(
        actual_prices,
        predicted_prices
    )

    directional_accuracy = np.mean(
        np.sign(predicted_returns)
        == np.sign(y_test)
    ) * 100

    return {
        "test_mae": round(float(mae), 3),
        "directional_accuracy": round(
            float(directional_accuracy), 2
        )
    }


# ==========================================
# CACHED MODEL TRAINING
# ==========================================

@lru_cache(maxsize=32)
def train_cached_models(ticker, training_date):
    data = download_intraday(ticker)

    data = data.loc[
        data.index.date <= training_date
    ]

    (
        X_original,
        X_enhanced,
        y,
        dates,
        references
    ) = prepare_intraday_data(data)

    unique_days = sorted(set(dates))

    required_days = (
        TRAINING_DAYS + TEST_DAYS
    )

    if len(unique_days) < required_days:
        raise ValueError(
            "Not enough completed trading days"
        )

    test_days = unique_days[-TEST_DAYS:]

    train_days = unique_days[
        -required_days:-TEST_DAYS
    ]

    train_mask = np.isin(
        dates, train_days
    )

    test_mask = np.isin(
        dates, test_days
    )

    y_train = y[train_mask]
    y_test = y[test_mask]

    ref_test = references[test_mask]

    # ======================================
    # COMPARE ORIGINAL AND ENHANCED
    # ======================================

    feature_sets = {
        "Original": X_original,
        "Enhanced": X_enhanced
    }

    comparison = {}

    for version, X in feature_sets.items():
        models = build_models()
        metrics = {}

        for name, model in models.items():
            model.fit(
                X[train_mask],
                y_train
            )

            metrics[name] = evaluate_model(
                model,
                X[test_mask],
                y_test,
                ref_test
            )

        comparison[version] = metrics

    # ======================================
    # BASELINE
    # ======================================

    baseline_mae = mean_absolute_error(
        ref_test * (1.0 + y_test),
        ref_test
    )

    baseline_metrics = {
        "test_mae": round(
            float(baseline_mae), 3
        ),
        "directional_accuracy": None
    }

    # ======================================
    # FINAL MODELS
    # ======================================

    # Train enhanced models on the latest
    # 20 completed trading days.
    final_days = unique_days[
        -TRAINING_DAYS:
    ]

    final_mask = np.isin(
        dates, final_days
    )

    final_models = build_models()

    for model in final_models.values():
        model.fit(
            X_enhanced[final_mask],
            y[final_mask]
        )

    enhanced_metrics = comparison["Enhanced"]

    enhanced_metrics["Baseline"] = (
        baseline_metrics
    )

    evaluation = {
        "training_days": len(train_days),
        "testing_days": len(test_days),
        "training_samples": int(
            np.sum(train_mask)
        ),
        "testing_samples": int(
            np.sum(test_mask)
        ),
        "test_start": str(test_days[0]),
        "test_end": str(test_days[-1])
    }

    return (
        final_models,
        enhanced_metrics,
        comparison,
        evaluation
    )


# ==========================================
# NYSE MARKET SESSION
# ==========================================

def get_market_session(now):
    schedule = NYSE.schedule(
        start_date=now.date(),
        end_date=now.date()
    )

    if schedule.empty:
        return None

    session = schedule.iloc[0]

    market_open = session[
        "market_open"
    ].tz_convert(MARKET_TZ)

    market_close = session[
        "market_close"
    ].tz_convert(MARKET_TZ)

    return market_open, market_close


# ==========================================
# NEXT HOUR FORECAST
# ==========================================

def predict_next_hour(ticker):
    ticker = ticker.upper().strip()

    if not ticker:
        raise ValueError(
            "Ticker cannot be empty"
        )

    now = pd.Timestamp.now(
        tz=MARKET_TZ
    )

    data = download_intraday(ticker)

    completed_days = sorted({
        timestamp.date()
        for timestamp in data.index
        if timestamp.date() < now.date()
    })

    if not completed_days:
        raise ValueError(
            "No completed training sessions"
        )

    training_date = completed_days[-1]

    (
        models,
        metrics,
        comparison,
        evaluation
    ) = train_cached_models(
        ticker,
        training_date
    )

    latest_date = data.index[-1].date()

    latest_session = data.loc[
        data.index.date == latest_date
    ]

    if len(latest_session) < LOOKBACK:
        raise ValueError(
            "Not enough recent candles"
        )

    window = latest_session.iloc[
        -LOOKBACK:
    ]

    reference_price = float(
        window["Close"].iloc[-1]
    )

    observed_at = (
        window.index[-1]
        + pd.Timedelta(minutes=5)
    )

    target_time = (
        observed_at
        + pd.Timedelta(minutes=60)
    )

    data_age_minutes = (
        now - observed_at
    ).total_seconds() / 60.0

    session = get_market_session(now)

    forecast_available = False

    if session is not None:
        market_open, market_close = session

        forecast_available = (
            latest_date == now.date()
            and market_open <= now < market_close
            and market_open <= observed_at <= now
            and target_time <= market_close
            and 0 <= data_age_minutes
            <= MAX_DATA_AGE_MINUTES
            and consecutive_candles(window.index)
        )

    # ======================================
    # GENERATE PREDICTIONS
    # ======================================

    results = {}

    if forecast_available:
        features = enhanced_features(
            window
        ).reshape(1, -1)

        for name, model in models.items():
            predicted_return = float(
                model.predict(features)[0]
            )

            predicted_price = (
                reference_price
                * (1.0 + predicted_return)
            )

            results[name] = {
                **metrics[name],
                "predicted_price": round(
                    float(predicted_price), 2
                )
            }

        results["Baseline"] = {
            **metrics["Baseline"],
            "predicted_price": round(
                reference_price, 2
            )
        }

    else:
        for name, model_metrics in metrics.items():
            results[name] = {
                **model_metrics,
                "predicted_price": None
            }

    # ======================================
    # API RESPONSE
    # ======================================

    return {
        "ticker": ticker,
        "timeframe": "Next Hour",
        "reference_price": round(
            reference_price, 2
        ),
        "price_timestamp": observed_at.isoformat(),
        "target_timestamp": (
            target_time.isoformat()
            if forecast_available else None
        ),
        "forecast_available": forecast_available,
        "models": results,
        "feature_comparison": comparison,
        "data_interval": INTERVAL,
        "forecast_horizon_minutes": 60,
        "training_window_days": TRAINING_DAYS,
        "training_date": str(training_date),
        "data_age_minutes": round(
            data_age_minutes, 1
        ),
        "evaluation": evaluation
    }


# ==========================================
# RUN INDEPENDENTLY
# ==========================================

if __name__ == "__main__":
    result = predict_next_hour("AAPL")

    print(
        json.dumps(result, indent=2)
    )
