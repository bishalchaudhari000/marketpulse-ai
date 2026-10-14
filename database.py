
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "predictions.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecast_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                target_date TEXT NOT NULL,
                reference_price REAL NOT NULL,
                trading_signal TEXT NOT NULL,
                agreement_score REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(ticker, target_date)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                absolute_error REAL,
                FOREIGN KEY (run_id)
                    REFERENCES forecast_runs(id),
                UNIQUE(run_id, model)
            )
        """)


def save_predictions(ticker, target_date, data):
    create_database()

    ticker = ticker.upper().strip()
    reference_price = float(data["latest_close"])

    signal_data = data.get("trading_signal", {})

    signal = signal_data.get("signal", "UNRELIABLE")
    score = float(signal_data.get("score", 0))

    created_at = datetime.now().astimezone().isoformat()
    prediction_date = created_at[:10]

    with get_connection() as conn:
        existing = conn.execute(
            """
            SELECT id
            FROM forecast_runs
            WHERE ticker = ? AND target_date = ?
            """,
            (ticker, target_date)
        ).fetchone()

        # Do not overwrite an existing forecast
        if existing:
            return {
                "saved": False,
                "message": "Prediction already saved for this date",
                "run_id": existing[0]
            }

        cursor = conn.execute(
            """
            INSERT INTO forecast_runs (
                ticker, prediction_date, target_date,
                reference_price, trading_signal,
                agreement_score, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                prediction_date,
                target_date,
                reference_price,
                signal,
                score,
                created_at
            )
        )

        run_id = cursor.lastrowid

        for model_name, result in data["models"].items():
            conn.execute(
                """
                INSERT INTO model_predictions (
                    run_id, model, predicted_price
                )
                VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    model_name,
                    float(result["predicted_price"])
                )
            )

    return {
        "saved": True,
        "message": "Prediction saved successfully",
        "run_id": run_id
    }


def update_actual_prices(ticker, target_date, actual_price):
    create_database()

    ticker = ticker.upper().strip()
    actual_price = float(actual_price)

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE model_predictions
            SET actual_price = ?,
                absolute_error = ABS(predicted_price - ?)
            WHERE run_id IN (
                SELECT id
                FROM forecast_runs
                WHERE ticker = ? AND target_date = ?
            )
            """,
            (
                actual_price,
                actual_price,
                ticker,
                target_date
            )
        )


def get_prediction_history(ticker):
    create_database()

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                f.prediction_date,
                f.target_date,
                f.ticker,
                f.reference_price,
                f.trading_signal,
                f.agreement_score,
                p.model,
                p.predicted_price,
                p.actual_price,
                p.absolute_error
            FROM forecast_runs f
            JOIN model_predictions p
                ON f.id = p.run_id
            WHERE f.ticker = ?
            ORDER BY f.target_date DESC, p.model
            """,
            (ticker.upper(),)
        ).fetchall()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    create_database()
    print("Prediction tracking database ready!")
