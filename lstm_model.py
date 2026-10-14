
import json
import random
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import yfinance as yf

from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

LOOKBACK = 5


def build_lstm():
    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def prepare_data(ticker):
    data = yf.download(
        ticker,
        period="2y",
        auto_adjust=True,
        progress=False
    )

    if data.empty or len(data) < 100:
        raise ValueError("Not enough historical stock data")

    close = data["Close"].squeeze().dropna()

    if len(close) < 100:
        raise ValueError("Not enough valid closing prices")

    prices = close.to_numpy().reshape(-1, 1)

    # 70% training, 10% validation, 20% testing
    train_end = int(len(prices) * 0.70)
    val_end = int(len(prices) * 0.80)

    scaler = MinMaxScaler()
    scaler.fit(prices[:train_end])

    scaled_prices = scaler.transform(prices)

    X = []
    y = []

    for i in range(LOOKBACK, len(scaled_prices)):
        X.append(scaled_prices[i - LOOKBACK:i])
        y.append(scaled_prices[i, 0])

    X = np.array(X)
    y = np.array(y)

    train_split = train_end - LOOKBACK
    val_split = val_end - LOOKBACK

    latest_sequence = scaled_prices[-LOOKBACK:].reshape(
        1, LOOKBACK, 1
    )

    return (
        X, y, scaler,
        train_split, val_split,
        latest_sequence,
        str(close.index[-1].date())
    )


def predict_lstm(ticker, show_results=False, force_retrain=False):
    ticker = ticker.upper().strip()

    if not ticker.isalnum():
        raise ValueError("Invalid ticker symbol")

    (
        X, y, scaler,
        train_split, val_split,
        latest_sequence,
        latest_date
    ) = prepare_data(ticker)

    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:val_split], y[train_split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]

    model_path = MODEL_DIR / f"{ticker}_lstm.keras"
    scaler_path = MODEL_DIR / f"{ticker}_scaler.joblib"
    metrics_path = MODEL_DIR / f"{ticker}_metrics.json"

    use_saved = False

    if (
        model_path.exists()
        and scaler_path.exists()
        and metrics_path.exists()
        and not force_retrain
    ):
        with open(metrics_path, "r") as file:
            saved_metrics = json.load(file)

        if (
            saved_metrics.get("latest_date") == latest_date
            and saved_metrics.get("config_version") == 1
        ):
            use_saved = True

    if use_saved:
        print(f"Loading saved LSTM for {ticker}")

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Transform current prices with the saved scaler
        # The scaler matches the trained model
        data = yf.download(
            ticker,
            period="10d",
            auto_adjust=True,
            progress=False
        )

        closes = data["Close"].squeeze().dropna()

        if len(closes) < LOOKBACK:
            raise ValueError("Not enough recent stock prices")

        recent_prices = closes.iloc[-LOOKBACK:].to_numpy()
        recent_scaled = scaler.transform(
            recent_prices.reshape(-1, 1)
        )

        latest_sequence = recent_scaled.reshape(
            1, LOOKBACK, 1
        )

        train_mae = saved_metrics["training_mae"]
        val_mae = saved_metrics["validation_mae"]
        test_mae = saved_metrics["test_mae"]

    else:
        print(f"Training new LSTM for {ticker}")

        random.seed(42)
        np.random.seed(42)
        tf.keras.utils.set_random_seed(42)

        model = build_lstm()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        def calculate_mae(X_data, y_data):
            predictions = model.predict(
                X_data,
                verbose=0
            )

            actual = scaler.inverse_transform(
                y_data.reshape(-1, 1)
            )

            predicted = scaler.inverse_transform(
                predictions
            )

            return float(
                mean_absolute_error(actual, predicted)
            )

        train_mae = calculate_mae(X_train, y_train)
        val_mae = calculate_mae(X_val, y_val)
        test_mae = calculate_mae(X_test, y_test)
        
        # Calculate LSTM directional accuracy
        test_predictions = model.predict(X_test, verbose=0)

        actual_prices = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        predicted_prices = scaler.inverse_transform(
            test_predictions
        ).flatten()

        # Previous day's actual price for each test sample
        previous_prices = scaler.inverse_transform(
            X_test[:, -1, :]
        ).flatten()

        actual_direction = np.sign(actual_prices - previous_prices)
        predicted_direction = np.sign(predicted_prices - previous_prices)

        directional_accuracy = np.mean(
            actual_direction == predicted_direction
        ) * 100


        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        with open(metrics_path, "w") as file:
            json.dump({
                "latest_date": latest_date,
                "config_version": 1,
                "training_mae": train_mae,
                "validation_mae": val_mae,
                "test_mae": test_mae
            }, file, indent=4)


    
    # Calculate directional accuracy on historical test data
    test_predictions = model.predict(X_test, verbose=0)

    actual_prices = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    predicted_prices = scaler.inverse_transform(
        test_predictions
    ).flatten()

    previous_prices = scaler.inverse_transform(
        X_test[:, -1, :]
    ).flatten()

    actual_direction = np.sign(actual_prices - previous_prices)
    predicted_direction = np.sign(predicted_prices - previous_prices)

    directional_accuracy = np.mean(
        actual_direction == predicted_direction
    ) * 100

    # Next trading day prediction
    prediction = model.predict(
        latest_sequence,
        verbose=0
    )

    next_price = scaler.inverse_transform(
        prediction
    )[0][0]

    results = {
        "ticker": ticker,
        "predicted_price": round(float(next_price), 2),
        "training_mae": round(float(train_mae), 2),
        "validation_mae": round(float(val_mae), 2),
        "test_mae": round(float(test_mae), 2),
        "directional_accuracy": round(float(directional_accuracy), 2)
    }

    if show_results:
        print("\n--- LSTM PERFORMANCE ---")
        print(f"Training MAE: ${train_mae:.2f}")
        print(f"Validation MAE: ${val_mae:.2f}")
        print(f"Testing MAE: ${test_mae:.2f}")
        print(f"Next-Day Prediction: ${next_price:.2f}")

    return results


if __name__ == "__main__":
    predict_lstm("AAPL", show_results=True)
