# MarketPulse AI 📈
### Machine Learning Financial Market Prediction & Analytics Platform

MarketPulse AI is a Python-based financial analytics application that uses machine learning to forecast stock prices, compare model performance, and generate experimental BUY/SELL trading signals.

The application combines a FastAPI backend with an interactive Streamlit dashboard, allowing users to analyze different stocks and visualize historical market data.

## Features

- **Stock Price Forecasting:** Generate next-trading-day closing-price predictions.
- **Multiple ML Models:** Compare Linear Regression, Random Forest, and LSTM neural networks against a simple baseline.
- **Trading Signal Analysis:** Generate experimental BUY/SELL signals using historical directional accuracy and weighted model agreement.
- **Interactive Dashboard:** Explore stock price charts, forecasts, and model performance.
- **Model Evaluation:** Measure Mean Absolute Error (MAE) and directional accuracy.
- **Prediction Tracking:** Store forecasts in SQLite for future performance evaluation. Historical result tracking is under development.
- **Model Persistence:** Save and reload trained LSTM models to avoid unnecessary retraining.

## Technology Stack

**Languages:** Python, SQL

**Machine Learning:** TensorFlow, Keras, Scikit-learn, NumPy, Pandas

**Backend:** FastAPI, Uvicorn, SQLite

**Frontend:** Streamlit, Altair

**Market Data:** Yahoo Finance (yfinance)

## Machine Learning Models

| Model | Description |
|---|---|
| Linear Regression | Predicts closing prices using historical price features |
| Random Forest | Uses an ensemble of decision trees |
| LSTM Neural Network | Processes sequences of historical stock prices |
| Baseline | Predicts that the next closing price equals the latest closing price |

## Installation

Clone the repository and navigate into the project folder.

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI backend:

```bash
python -m uvicorn main:app --reload
```

In a separate terminal, launch the Streamlit dashboard:

```bash
python -m streamlit run dashboard.py
```

Open the local URL displayed by Streamlit.

## Model Evaluation

MarketPulse AI evaluates models using historical testing data and reports MAE and directional accuracy.

The trading signal uses an experimental weighted agreement calculation. Model test periods must be aligned before definitive performance comparisons can be made.

## Future Improvements

- Complete automatic prediction tracking and actual-price comparisons.
- Align all model evaluation periods.
- Add selectable historical training windows.
- Explore intraday market data and updated forecasting.
- Deploy the application online.

## Disclaimer

This project is intended for educational and research purposes. Forecasts and trading signals are experimental and should not be interpreted as investment advice or probabilities of profitable trades.