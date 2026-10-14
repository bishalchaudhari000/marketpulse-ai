from lstm_model import predict_lstm
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from signals import calculate_signal


def predict_stock(ticker):
    ticker = ticker.upper()

    # Download historical market data
    data = yf.download(
        ticker,
        period="2y",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No stock data found"}

    close = data["Close"].squeeze()

    # Build features using the previous 5 closing prices
    features = pd.DataFrame({
        f"lag_{lag}": close.shift(lag)
        for lag in range(1, 6)
    })

    features["target"] = close
    features = features.dropna()

    X = features.drop(columns=["target"])
    y = features["target"]

    # Chronological train/test split
    split = int(len(X) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Both models use identical training and testing data
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
    }

    # Latest 5 closing prices for the next prediction
    latest = pd.DataFrame([{
        f"lag_{lag}": float(close.iloc[-lag])
        for lag in range(1, 6)
    }])

    results = {}
    baseline_predictions = X_test["lag_1"]
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    results["Baseline"] = {
    "predicted_price": round(float(close.iloc[-1]), 2),
    "test_mae": round(float(baseline_mae), 2)
}
    
    
    
    
    chart_data = pd.DataFrame({
    "Date": y_test.index.strftime("%Y-%m-%d"),
    "Actual": y_test.to_numpy()
})
    chart_data["Baseline"] = baseline_predictions.to_numpy()

    for name, model in models.items():
        model.fit(X_train, y_train)

        test_predictions = model.predict(X_test)
        chart_data[name] = test_predictions
        mae = mean_absolute_error(y_test, test_predictions)
        
        previous_close = X_test["lag_1"].to_numpy()

        actual_direction = np.sign(
            y_test.to_numpy() - previous_close
        )

        predicted_direction = np.sign(
            test_predictions - previous_close
        )

        directional_accuracy = np.mean(
            actual_direction == predicted_direction
        ) * 100


        next_price = model.predict(latest)[0]
        
        
        results[name] = {
            "predicted_price": round(float(next_price), 2),
            "test_mae": round(float(mae), 2),
            "directional_accuracy": round(float(directional_accuracy), 2)
        }

        
    # Run the LSTM model
    lstm_results = predict_lstm(ticker)

    results["LSTM"] = {
    "predicted_price": lstm_results["predicted_price"],
    "test_mae": lstm_results["test_mae"],
    "directional_accuracy": lstm_results["directional_accuracy"]
}
    signal_result = calculate_signal(
    results,
    float(close.iloc[-1])
)
    return {
    "ticker": ticker,
    "latest_close": round(float(close.iloc[-1]), 2),
    "models": results,
    "trading_signal": signal_result,
    "chart_data": chart_data.round(2).to_dict(orient="records")
    
}
