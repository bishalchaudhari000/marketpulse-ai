from intraday_model import predict_next_hour
from models import predict_stock
from fastapi import FastAPI, HTTPException
import yfinance as yf
from database import save_predictions
import pandas_market_calendars as mcal
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

app = FastAPI(title="MarketPulse AI API")

MARKET_TZ = ZoneInfo("America/New_York")
DISPLAY_TZ = ZoneInfo("America/Los_Angeles")

market = mcal.get_calendar("NYSE")


@app.get("/")
def home():
    return {
        "message": "MarketPulse AI API is running!",
        "status": "online"
    }


# ==========================================
# ORIGINAL STOCK ENDPOINT
# ==========================================

@app.get("/stock/{ticker}")
def get_stock(ticker: str):
    ticker = ticker.upper().strip()

    stock = yf.Ticker(ticker)
    data = stock.history(period="5d")

    if data.empty:
        raise HTTPException(
            status_code=404,
            detail="Stock not found"
        )

    return {
        "ticker": ticker,
        "price": round(float(data["Close"].iloc[-1]), 2)
    }


# ==========================================
# LATEST AVAILABLE MARKET PRICE
# ==========================================

@app.get("/live/{ticker}")
def get_live_price(ticker: str):
    ticker = ticker.upper().strip()

    if not ticker or not ticker.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(
            status_code=400,
            detail="Invalid stock ticker"
        )

    try:
        # Download recent 5-minute market candles
        stock = yf.Ticker(ticker)

        data = stock.history(
            period="5d",
            interval="5m",
            auto_adjust=True,
            prepost=False
        )

        if data.empty:
            raise HTTPException(
                status_code=404,
                detail="No intraday stock data available"
            )

        data = data.dropna(subset=["Close"])

        if data.empty:
            raise HTTPException(
                status_code=404,
                detail="No valid stock prices available"
            )

        # Current time
        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(MARKET_TZ)

        # Find the most recent exchange session
        schedule = market.schedule(
            start_date=now_et.date() - timedelta(days=10),
            end_date=now_et.date() + timedelta(days=1)
        )

        if schedule.empty:
            raise HTTPException(
                status_code=503,
                detail="Market calendar unavailable"
            )

        # Determine whether regular trading is open
        market_open = False

        for _, session in schedule.iterrows():
            if (
                session["market_open"].to_pydatetime()
                <= now_utc
                < session["market_close"].to_pydatetime()
            ):
                market_open = True
                break

        # Use completed 5-minute candles only.
        # yfinance timestamps generally mark candle starts.
        data = data.sort_index()

        if data.index.tz is None:
            data.index = data.index.tz_localize(MARKET_TZ)

        candle_end = data.index + pd.Timedelta(minutes=5)

        completed = data.loc[
            candle_end <= pd.Timestamp(now_utc)
        ]

        if completed.empty:
            raise HTTPException(
                status_code=503,
                detail="No completed candles available"
            )

        latest = completed.iloc[-1]

        candle_start = completed.index[-1]
        observed_at = candle_start + pd.Timedelta(minutes=5)

        observed_at_utc = observed_at.tz_convert("UTC")
        observed_at_pt = observed_at.tz_convert(DISPLAY_TZ)

        # How old is the observation?
        age_seconds = max(
            0,
            (pd.Timestamp(now_utc) - observed_at_utc).total_seconds()
        )

        age_minutes = int(age_seconds // 60)

        latest_price = float(latest["Close"])

        # Previous completed daily close
        daily = stock.history(
            period="10d",
            interval="1d",
            auto_adjust=True
        )

        previous_close = None
        change_percent = None

        if not daily.empty:
            daily = daily.dropna(subset=["Close"])

            if not daily.empty:
                daily_dates = [
                    timestamp.date()
                    for timestamp in daily.index.tz_convert(MARKET_TZ)
                ] if daily.index.tz is not None else [
                    timestamp.date()
                    for timestamp in daily.index
                ]

                completed_daily = daily.loc[
                    [
                        date < now_et.date()
                        for date in daily_dates
                    ]
                ]

                if not completed_daily.empty:
                    previous_close = float(
                        completed_daily["Close"].iloc[-1]
                    )

                    if previous_close != 0:
                        change_percent = (
                            (latest_price - previous_close)
                            / previous_close * 100
                        )

        # Market status
        status = "OPEN" if market_open else "CLOSED"

        return {
            "ticker": ticker,
            "latest_price": round(latest_price, 2),
            "previous_close": (
                round(previous_close, 2)
                if previous_close is not None
                else None
            ),
            "change_percent": (
                round(change_percent, 2)
                if change_percent is not None
                else None
            ),
            "market_status": status,
            "price_timestamp": observed_at_pt.isoformat(),
            "price_time_display": observed_at_pt.strftime(
                "%I:%M %p PT"
            ),
            "data_age_minutes": age_minutes,
            "data_interval": "5m",
            "data_source": "Yahoo Finance",
            "is_delayed": age_minutes > 5,
            "last_refreshed": now_utc.astimezone(
                DISPLAY_TZ
            ).isoformat()
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Market data unavailable: {error}"
        )


# ==========================================
# EXISTING NEXT-DAY PREDICTIONS
# ==========================================

@app.get("/predict/{ticker}")
def predict(ticker: str):
    ticker = ticker.upper().strip()

    data = predict_stock(ticker)

    if "error" in data:
        return data

    # Preserve existing next-day prediction behavior
    today = datetime.now(MARKET_TZ).date()

    schedule = market.schedule(
        start_date=today,
        end_date=today + timedelta(days=14)
    )

    if schedule.empty:
        raise HTTPException(
            status_code=503,
            detail="No upcoming trading sessions found"
        )

    future_sessions = [
        day.date()
        for day in schedule.index
        if day.date() > today
    ]

    if not future_sessions:
        raise HTTPException(
            status_code=503,
            detail="Next trading day not found"
        )

    target_date = str(future_sessions[0])

    # Save all four model forecasts and trading signal
    save_result = save_predictions(
        ticker,
        target_date,
        data
    )

    data["target_date"] = target_date
    data["save_status"] = save_result

    return data


# ==========================================
# NEXT HOUR STOCK PREDICTIONS
# ==========================================

@app.get("/predict-hour/{ticker}")
def predict_hour(ticker: str):
    ticker = ticker.upper().strip()

    if not ticker or not ticker.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(
            status_code=400,
            detail="Invalid stock ticker"
        )

    try:
        results = predict_next_hour(ticker)
        return results

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Unable to generate next-hour forecast"
        )
