from models import predict_stock
from fastapi import FastAPI
import yfinance as yf
from database import save_predictions
import pandas_market_calendars as mcal
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

app = FastAPI(title="Financial Market Predictor")

@app.get("/")
def home():
    return {"message": "Stock Market API is running!"}

@app.get("/stock/{ticker}")
def get_stock(ticker: str):
    stock = yf.Ticker(ticker.upper())
    data = stock.history(period="5d")

    if data.empty:
        return {"error": "Stock not found"}

    return {
        "ticker": ticker.upper(),
        "price": round(float(data["Close"].iloc[-1]), 2)
    }

@app.get("/predict/{ticker}")
def predict(ticker: str):
    ticker = ticker.upper()

    # Generate forecasts from all models
    data = predict_stock(ticker)

    if "error" in data:
        return data

    # Determine the next trading session
    market = mcal.get_calendar("NYSE")

    today = datetime.now(
        ZoneInfo("America/New_York")
    ).date()

    schedule = market.schedule(
        start_date=today,
        end_date=today + pd.Timedelta(days=14)
    )

    if schedule.empty:
        return {"error": "No upcoming trading sessions found"}

    # During a trading day, target the next session
    future_sessions = [
        day.date()
        for day in schedule.index
        if day.date() > today
    ]

    if not future_sessions:
        return {"error": "Next trading day not found"}

    target_date = str(future_sessions[0])

    # Save all four model forecasts and the signal
    save_result = save_predictions(
        ticker,
        target_date,
        data
    )

    data["target_date"] = target_date
    data["save_status"] = save_result

    return data
